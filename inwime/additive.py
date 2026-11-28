"""Percorso additivo A (#41, ADR 0001): Qwen2 prefill in blocks of positions, native operators.

`AdditiveQwen2Runtime` runs the Qwen2 forward of the frozen `Qwen2Runtime` (tied head through
`qwen2_tied`, untied through the frozen catalog) with three differences, none of which changes a
bit of the results:

- **Blocks.** A prompt is processed `block` positions at a time. Each weight tile is read and
  expanded once per block and applied to every position of it, instead of once per position.
  Every output element keeps the frozen accumulation order (`inwime.ops`).
- **Logits only where needed.** The final norm and the head run on the last position of the
  call only, unless the runtime is built with `all_logits` (gates and tests), which runs them on
  every position of each block and hands them to the caller's trace.
- **Bounded temporary memory.** Activations hold `block` positions, the logits one position
  (or one block): nothing scales with the prompt. Residuals are not stored; a trace receives
  them as they are computed. The only arena that grows with the context is the KV, sized by
  `capacity`.

The KV has a single bank (#41 grill decision): a call writes its positions after the published
length, and the commit publishes the new length. Any error or cancellation leaves the published
length (nothing written past it is visible) and fails the session, as the frozen fail-stop
contract does. Truncation on a diverging prefix, the six rollback scenarios and the capacity plan
come with #42.

Weight I/O, leases and memory reservations stay with the frozen Gestore delle regioni
(`core.Coordinator`), unchanged; BF16 expansion is its `math_backend` (the pinned kernel or the
Python reference). Cancellation: the runtime checks `cancel_event` before every weight read, every
layer and every operator call; `linear` and `attention` also read the flag of a `CancelSignal`
inside the native loops (frequency in `inwime.ops`).
"""
from array import array
from concurrent.futures import CancelledError
from contextlib import closing
import ctypes
import threading

from inwime_runtime.core import CATEGORIES, Coordinator, Plan, deep_size

from . import ops as operators

DEFAULT_BLOCK = 32        # positions per block: 32 per expanded tile is the ADR 0002 measure


class CancelSignal(threading.Event):
    """A cancellation event whose state the native kernels can also read, through `flag`."""

    def __init__(self):
        super().__init__()
        self.flag = ctypes.c_int(0)

    def set(self):
        self.flag.value = 1
        super().set()

    def clear(self):
        super().clear()
        self.flag.value = 0


def additive_plan(catalog, capacity, block, all_logits, tile_elements, read_elements):
    c = catalog.spec
    if (type(capacity) is not int or not 1 <= capacity <= c.context
            or type(block) is not int or block < 1
            or type(tile_elements) is not int or tile_elements <= 0
            or type(read_elements) is not int or not 1 <= read_elements <= tile_elements):
        raise ValueError('invalid additive capacity/block/tile/read')
    arrays = {'weights': ('weights', tile_elements, 'd'), 'staging': ('staging', 2*read_elements, 'B')}
    for name in ('x', 'norm', 'q', 'attention', 'delta'):
        arrays[name] = ('activations', block*c.width, 'd')
    for name in ('gate', 'up'):
        arrays[name] = ('activations', block*c.hidden, 'd')
    arrays['denominators'] = ('accumulators', block, 'd')
    arrays['logits'] = ('output', (block if all_logits else 1)*c.vocab, 'd')
    arrays['kv'] = ('kv', c.layers*2*capacity*c.kv_width, 'd')
    required = dict.fromkeys(CATEGORIES, 0)
    for category, count, code in arrays.values():
        required[category] += count*array(code).itemsize
    required['metadata'] = catalog.bootstrap_bytes + deep_size(arrays) + 65536
    return Plan(arrays, required, capacity, tile_elements, read_elements)


class AdditiveResult:
    """Borrowed read-only views of the last call, valid until release; one result at a time."""

    def __init__(self, owner, version, row):
        self.owner, self.version, self.row = owner, version, row
        self.released = False

    def _check(self):
        self.owner.control.check_thread()
        if self.released or self.owner.closed or self.owner.version != self.version:
            raise RuntimeError('expired result reference')

    def _view(self, name, start, count):
        self._check()
        return memoryview(self.owner.control.buffers[name])[start:start+count].toreadonly()

    @property
    def logits(self):
        """The logits of the last position of the call."""
        vocab = self.owner.spec.vocab
        return self._view('logits', self.row*vocab, vocab)

    @property
    def kv(self):
        """Per layer, the K and V rows of every published position, in the frozen layout."""
        c, cap, length = self.owner.spec, self.owner.capacity, self.owner.length
        return tuple(tuple(self._view('kv', (layer*2+kind)*cap*c.kv_width, length*c.kv_width)
                           for kind in (0, 1)) for layer in range(c.layers))

    def release(self):
        self.owner.control.check_thread()
        if not self.released:
            self.owner.result_open = False
            self.released = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *exc):
        self.release()


class AdditiveQwen2Runtime:
    """One bounded fail-stop session over a Qwen2 catalog (tied or untied head)."""

    def __init__(self, catalog, capacity, *, integrity, math_backend, ops=None, block=DEFAULT_BLOCK,
                 all_logits=False, tile_elements=262144, read_elements=262144):
        """`ops`: operator name -> implementation (`inwime.ops`); None: the Python reference for all."""
        self.catalog = catalog
        self.spec, self.capacity = catalog.spec, capacity
        self.block = min(block, capacity) if type(block) is int and type(capacity) is int else block
        self.all_logits = bool(all_logits)
        self.plan = additive_plan(catalog, capacity, self.block, self.all_logits, tile_elements,
                                  read_elements)
        reference = operators.PythonOps()
        self.ops = {name: reference for name in operators.NAMES} | dict(ops or {})
        self.math_backend = math_backend
        self.control = Coordinator(catalog, self.plan, None, None, integrity=integrity,
                                   math_backend=math_backend)
        self._published = (0, 0)          # (length, version), one reference update
        self._flag = ctypes.c_int(0)      # for a cancel_event that is not a CancelSignal
        self.running = self.failed = self.result_open = self.closed = False

    @property
    def length(self):
        return self._published[0]

    @property
    def version(self):
        return self._published[1]

    # -- operators ------------------------------------------------------------------------------

    def _op(self, name, *args):
        code = getattr(self.ops[name], name)(*args)
        if code == operators.OK:
            return
        if code == operators.CANCELLED:
            raise CancelledError('inference cancelled')
        if code == operators.NONFINITE:
            raise ArithmeticError('nonfinite normalization denominator' if name == 'rms_denominators'
                                  else 'nonfinite linear output')
        raise ValueError('operator %s rejected its arguments (status %r)' % (name, code))

    def _tiles(self, name, start=0, count=None):
        """Leased FP64 tiles of a tensor (or of `count` elements from `start`)."""
        tile = self.plan.tile_elements
        end = start+(self.catalog.tensors[name].elements if count is None else count)
        return closing(self.control.stream((name, s, min(tile, end-s)) for s in range(start, end, tile)))

    def _weights(self, lease):
        lease.value(0)          # ownership, generation and liveness of the lease
        return self.control.buffers['weights']

    def _linear(self, name, x, xo, xs, y, yo, ys, positions, flag):
        columns = self.catalog.tensors[name].shape[1]
        with self._tiles(name) as stream:
            for start, count, lease in stream:
                self._op('linear', self._weights(lease), count, x, xo, xs, y, yo, ys, positions,
                         columns, start, flag)

    def _bias(self, name, y, yo, ys, positions):
        with self._tiles(name) as stream:
            for start, count, lease in stream:
                self._op('add', y, yo+start, ys, self._weights(lease), 0, 0, positions, count)

    def _norm(self, scale, x, xo, xs, y, yo, ys, positions):
        c, denominators = self.spec, self.control.buffers['denominators']
        with self.control.compute():
            self._op('rms_denominators', x, xo, xs, positions, c.width, float(c.rms_epsilon),
                     denominators)
        with self._tiles(scale) as stream:
            for start, count, lease in stream:
                self._op('rms_scale', x, xo, xs, denominators, self._weights(lease), count, start,
                         y, yo, ys, positions)

    def _embed(self, token, x, offset):
        width = self.spec.width
        with self._tiles('embedding', token*width, width) as stream:
            for start, count, lease in stream:
                at = offset+start-token*width
                memoryview(x)[at:at+count] = memoryview(self._weights(lease))[:count]

    # -- forward --------------------------------------------------------------------------------

    def _block(self, chunk, first, last, trace, flag):
        """Positions first .. first+len(chunk)-1 through every layer; returns the logits row."""
        b, c, cap = self.control.buffers, self.spec, self.capacity
        W, KW, H, P = c.width, c.kv_width, c.hidden, len(chunk)
        x, norm, kv = b['x'], b['norm'], b['kv']
        for p, token in enumerate(chunk):
            self._embed(token, x, p*W)
        for layer in range(c.layers):
            self.control.check_cancelled()
            pre = f'layers.{layer}.'
            keys = layer*2*cap*KW
            values = keys+cap*KW
            self._norm(pre+'norm1', x, 0, W, norm, 0, W, P)
            self._linear(pre+'q', norm, 0, W, b['q'], 0, W, P, flag)
            self._bias(pre+'q.bias', b['q'], 0, W, P)
            # Upstream adds the projection bias before RoPE; K is cached post-RoPE.
            self._linear(pre+'k', norm, 0, W, kv, keys+first*KW, KW, P, flag)
            self._bias(pre+'k.bias', kv, keys+first*KW, KW, P)
            self._linear(pre+'v', norm, 0, W, kv, values+first*KW, KW, P, flag)
            self._bias(pre+'v.bias', kv, values+first*KW, KW, P)
            with self.control.compute():
                theta = float(c.theta)
                self._op('rope', b['q'], 0, W, P, first, c.heads, c.dim, theta)
                self._op('rope', kv, keys+first*KW, KW, P, first, c.kv_heads, c.dim, theta)
                self._op('attention', b['q'], 0, W, P, first, kv, keys, values, KW, c.heads,
                         c.kv_heads, c.dim, b['attention'], 0, W, flag)
            self._linear(pre+'o', b['attention'], 0, W, b['delta'], 0, W, P, flag)
            with self.control.compute():
                self._op('add', x, 0, W, b['delta'], 0, W, P, W)
            self._norm(pre+'norm2', x, 0, W, norm, 0, W, P)
            self._linear(pre+'ffn.gate', norm, 0, W, b['gate'], 0, H, P, flag)
            self._linear(pre+'ffn.up', norm, 0, W, b['up'], 0, H, P, flag)
            with self.control.compute():
                self._op('silu_mul', b['gate'], b['up'], P*H)
            self._linear(pre+'ffn.down', b['gate'], 0, H, b['delta'], 0, W, P, flag)
            with self.control.compute():
                self._op('add', x, 0, W, b['delta'], 0, W, P, W)
            if trace is not None:
                for p in range(P):
                    trace.residual(layer, first+p, memoryview(x)[p*W:(p+1)*W].toreadonly())
        self.control.check_cancelled()
        if self.all_logits:
            self._norm('final_norm', x, 0, W, norm, 0, W, P)
            self._linear('lm_head', norm, 0, W, b['logits'], 0, c.vocab, P, flag)
            if trace is not None:
                for p in range(P):
                    trace.logit(first+p, memoryview(b['logits'])[p*c.vocab:(p+1)*c.vocab].toreadonly())
            return P-1
        if last:
            self._norm('final_norm', x, (P-1)*W, W, norm, 0, W, 1)
            self._linear('lm_head', norm, 0, W, b['logits'], 0, c.vocab, 1, flag)
            if trace is not None:
                trace.logit(first+P-1, memoryview(b['logits'])[:c.vocab].toreadonly())
        return 0

    def prefill(self, tokens, *, cancel_event=None, trace=None):
        """Append `tokens` to the session. `trace`: an object with `residual(layer, position,
        values)` and `logit(position, values)`, called with read-only views to copy at once."""
        self.control.check_thread()
        if self.closed or self.failed:
            raise RuntimeError('closed or failed session')
        if self.running or self.result_open or self.control.active:
            raise RuntimeError('release the previous result/region before executing')
        if (not isinstance(tokens, (tuple, list)) or not tokens
                or len(tokens)+self.length > self.capacity
                or any(type(t) is not int or not 0 <= t < self.spec.vocab for t in tokens)):
            raise ValueError('invalid token sequence or session capacity exceeded')
        tokens = tuple(tokens)
        flag = getattr(cancel_event, 'flag', None)
        flag = flag if isinstance(flag, ctypes.c_int) else self._flag
        length, version = self._published
        try:
            self.control.cancel_event = cancel_event
            self.control.check_cancelled()
            self.running = True
            row = 0
            for start in range(0, len(tokens), self.block):
                row = self._block(tokens[start:start+self.block], length+start,
                                  start+self.block >= len(tokens), trace, flag)
            self.control.check_cancelled()
            result = AdditiveResult(self, version+1, row)
            self.result_open = True
            # Every written position is ready; publishing the length is the single commit point.
            self._published = (length+len(tokens), version+1)
            self.running = False
            return result
        except BaseException:
            self._published = (length, version)
            self.result_open = False
            self.failed = True
            self.running = False
            self.control.abort()
            raise
        finally:
            self.control.cancel_event = None

    def decode(self, token, *, cancel_event=None, trace=None):
        if not self.length:
            raise ValueError('decode requires a prefilled session')
        return self.prefill([token], cancel_event=cancel_event, trace=trace)

    def stats(self):
        return self.control.stats() | {
            'session_length': self.length, 'session_failed': self.failed, 'block_positions': self.block,
            'math_backend': self.math_backend.name if self.math_backend else 'python-fp64-original',
            'operators': {name: self.ops[name].name for name in operators.NAMES}}

    def close(self):
        if self.running:
            raise RuntimeError('cannot close an executing session')
        self.control.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

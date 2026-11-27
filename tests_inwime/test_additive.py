"""Seam 3 of spec #32 for #41: the additive runtime's prefill/decode against the Percorso congelato.

The Percorso additivo prefills in blocks of positions and calls the operators of `inwime.ops`
(native C or their Python reference). Where the frozen `Qwen2Runtime` accepts the Checkpoint
(untied head) it must give the same bits; for a tied head the reference is the frozen runtime on
the twin that serializes the same values as an untied head (#38). Compared after the prompt and
after every decode: the logits of every position, every residual and the whole KV.
"""
import pytest

import tinyqwen
from inwime import ops as operators
from inwime.additive import AdditiveQwen2Runtime
from inwime.qwen2_tied import qwen2_catalog
from inwime_runtime.math_cpu import NativeMath, ScalarMath
from inwime_runtime.qwen2 import Qwen2Runtime
from inwime_runtime.safetensors_source import SafetensorsSource

FX = tinyqwen.generator()
TINY = FX.TINY_TOKEN
SEQUENCE = [3, TINY, 7, 30, 5, TINY, 12, 2]
KERNEL = tinyqwen.kernel_path()
OPS_KERNEL = operators.locate({}) or next(
    (p for p in [tinyqwen.TOOLS.parent/'build'/'libinwime_ops.so'] if p.is_file()), None)
# (BF16 expansion, operators): the Python reference, and the native kernels of the Motore.
MATHS = [('python', 'python')] + ([('native', 'native')] if KERNEL and OPS_KERNEL else [])


def expand(name):
    return NativeMath(str(KERNEL)) if name == 'native' else ScalarMath()


def ops(name):
    if name == 'python':
        return None
    native = operators.NativeOps(OPS_KERNEL)
    return {op: native for op in operators.NAMES}


@pytest.fixture(scope='module')
def heads(tmp_path_factory):
    root = tmp_path_factory.mktemp('heads')
    return {head: tinyqwen.generator(head).write(root/head) for head in ('untied', 'tied', 'duplicated')}


class Trace:
    """Collects the bytes the additive runtime reports for each position."""

    def __init__(self):
        self.logits, self.residuals = {}, {}

    def residual(self, layer, position, values):
        self.residuals[layer, position] = bytes(values)

    def logit(self, position, values):
        self.logits[position] = bytes(values)


def frozen_steps(directory, chunks, tile=7):
    """Per call: logits of each position, residual of each (layer, position), whole KV."""
    steps, start = [], 0
    with Qwen2Runtime(directory, capacity=len(SEQUENCE), tile_elements=tile, read_elements=tile,
                      integrity='full') as r:
        width, vocab = r.spec.width, r.spec.vocab
        for size in chunks:
            with r.prefill(SEQUENCE[start:start+size]) as out:
                logits = bytes(out.logits)
                steps.append(dict(
                    logits={start+i: logits[i*vocab*8:(i+1)*vocab*8] for i in range(size)},
                    residuals={(layer, start+i): bytes(h)[i*width*8:(i+1)*width*8]
                               for layer, h in enumerate(out.hidden) for i in range(size)},
                    kv=tuple(bytes(v) for pair in out.kv for v in pair)))
            start += size
    return steps


def additive_steps(directory, chunks, *, block, tile=7, maths=('python', 'python')):
    steps, start = [], 0
    catalog = qwen2_catalog(SafetensorsSource(directory))
    with AdditiveQwen2Runtime(catalog, len(SEQUENCE), block=block, all_logits=True, integrity='full',
                              math_backend=expand(maths[0]), ops=ops(maths[1]), tile_elements=tile,
                              read_elements=tile) as r:
        vocab = r.spec.vocab
        for size in chunks:
            trace = Trace()
            with r.prefill(SEQUENCE[start:start+size], trace=trace) as out:
                last = bytes(out.logits)
                assert len(last) == vocab*8
                assert last == trace.logits[start+size-1]
                steps.append(dict(logits=trace.logits, residuals=trace.residuals,
                                  kv=tuple(bytes(v) for pair in out.kv for v in pair)))
            start += size
    return steps


# Tiles of 7 elements cut every row of every tensor; 262144 is the Motore's tile.
@pytest.mark.parametrize('maths', MATHS, ids='-'.join)
@pytest.mark.parametrize('tile', [7, 262144])
@pytest.mark.parametrize('chunks,block', [([5, 1, 1, 1], 2), ([8], 3), ([8], 8), ([3, 5], 4)])
def test_block_prefill_gives_the_bits_of_the_frozen_path(heads, chunks, block, tile, maths):
    assert (additive_steps(heads['untied'], chunks, block=block, tile=tile, maths=maths)
            == frozen_steps(heads['untied'], chunks, tile=tile))


@pytest.mark.parametrize('maths', MATHS, ids='-'.join)
@pytest.mark.parametrize('chunks,block', [([5, 1, 1, 1], 2), ([8], 3)])
def test_tied_head_gives_the_bits_of_the_frozen_path_on_the_duplicated_twin(heads, chunks, block, maths):
    assert (additive_steps(heads['tied'], chunks, block=block, maths=maths)
            == frozen_steps(heads['duplicated'], chunks))


def plan(directory, capacity, block):
    catalog = qwen2_catalog(SafetensorsSource(directory))
    return AdditiveQwen2Runtime(catalog, capacity, block=block, integrity='metadata',
                                math_backend=ScalarMath())


def test_temporary_memory_depends_on_the_block_only_and_the_kv_on_the_capacity(heads):
    c = FX.CONFIG
    width, hidden, layers, vocab = c['hidden_size'], c['intermediate_size'], c['num_hidden_layers'], c['vocab_size']
    kv_width = c['num_key_value_heads']*width//c['num_attention_heads']
    sizes = {}
    for capacity in (8, 64):
        with plan(heads['untied'], capacity, 4) as r:
            sizes[capacity] = dict(r.plan.required)
            arenas = {name: len(a) for name, a in r.control.buffers.items()}
            with r.prefill([2+i % (vocab-2) for i in range(capacity-1)]):
                pass                                   # a prompt filling the whole capacity
            with r.decode(5):
                pass
            assert {name: len(a) for name, a in r.control.buffers.items()} == arenas
    small, large = sizes[8], sizes[64]
    # Activations: 4 positions of x, norm, q, attention and delta (width) and gate, up (hidden).
    assert small['activations'] == large['activations'] == 4*(5*width+2*hidden)*8
    assert small['accumulators'] == large['accumulators'] == 4*8          # RMSNorm denominators
    assert small['output'] == large['output'] == vocab*8                  # logits of one position
    assert small['workspace'] == large['workspace'] == 0                  # nothing quadratic
    assert (small['kv'], large['kv']) == (layers*2*8*kv_width*8, layers*2*64*kv_width*8)


def test_the_block_never_exceeds_the_capacity(heads):
    with plan(heads['untied'], 3, 32) as r:
        assert r.block == 3


class CancelAt(Trace):
    """Sets the cancellation signal when the given layer of the given position is reported."""

    def __init__(self, signal, layer, position, flag_only=False):
        super().__init__()
        self.signal, self.at, self.flag_only = signal, (layer, position), flag_only

    def residual(self, layer, position, values):
        if (layer, position) == self.at:
            if self.flag_only:
                self.signal.flag.value = 1     # seen only by the operators, not by the runtime
            else:
                self.signal.set()


@pytest.mark.parametrize('maths', MATHS, ids='-'.join)
@pytest.mark.parametrize('flag_only', [False, True], ids=['event', 'operator-flag'])
def test_a_cancelled_extension_publishes_nothing_and_fails_the_session(heads, maths, flag_only):
    from concurrent.futures import CancelledError
    from inwime.additive import CancelSignal
    catalog = qwen2_catalog(SafetensorsSource(heads['untied']))
    with AdditiveQwen2Runtime(catalog, len(SEQUENCE), block=2, integrity='metadata',
                              math_backend=expand(maths[0]), ops=ops(maths[1])) as r:
        with r.prefill(SEQUENCE[:5]) as out:
            kv = tuple(bytes(v) for pair in out.kv for v in pair)
        signal = CancelSignal()
        with pytest.raises(CancelledError):
            r.prefill(SEQUENCE[5:], cancel_event=signal, trace=CancelAt(signal, 0, 5, flag_only))
        assert (r.length, r.failed) == (5, True)
        assert tuple(bytes(memoryview(r.control.buffers['kv'])[(l*2+k)*len(SEQUENCE)*16:][:5*16])
                     for l in range(3) for k in (0, 1)) == kv   # published rows untouched
        with pytest.raises(RuntimeError):
            r.decode(5)


def test_a_cancel_signal_set_before_the_call_stops_it_before_any_read(heads):
    from concurrent.futures import CancelledError
    from inwime.additive import CancelSignal
    catalog = qwen2_catalog(SafetensorsSource(heads['untied']))
    with AdditiveQwen2Runtime(catalog, 4, integrity='metadata', math_backend=ScalarMath()) as r:
        signal = CancelSignal()
        signal.set()
        with pytest.raises(CancelledError):
            r.prefill([3, 4], cancel_event=signal)
        assert r.control.stats()['inference_read_bytes'] == 0 and r.length == 0
        signal.clear()
        assert signal.flag.value == 0 and not signal.is_set()


@pytest.mark.parametrize('maths', MATHS, ids='-'.join)
@pytest.mark.parametrize('head,twin', [('untied', 'untied'), ('tied', 'duplicated')])
@pytest.mark.parametrize('chunks,block', [([5, 1, 1, 1], 32), ([8], 3), ([7, 1], 2)])
def test_the_served_configuration_gives_the_frozen_last_logits_and_kv(heads, head, twin, chunks, block, maths):
    """As the Motore runs it: no trace, head and final norm on the last position of each call only."""
    frozen = frozen_steps(heads[twin], chunks, tile=262144)
    catalog = qwen2_catalog(SafetensorsSource(heads[head]))
    start = 0
    with AdditiveQwen2Runtime(catalog, len(SEQUENCE), block=block, integrity='full',
                              math_backend=expand(maths[0]), ops=ops(maths[1])) as r:
        assert r.plan.required['output'] == r.spec.vocab*8
        for size, step in zip(chunks, frozen, strict=True):
            with r.prefill(SEQUENCE[start:start+size]) as out:
                assert bytes(out.logits) == step['logits'][start+size-1]
                assert tuple(bytes(v) for pair in out.kv for v in pair) == step['kv']
            start += size

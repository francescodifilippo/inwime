"""Independent scalar/tiled CPU execution of the increment-1 decoder spec."""

import hashlib
import math
from contextlib import closing

from .core import Catalog, Coordinator, memory_plan
from .pipeline import PipelineCoordinator


class RuntimeResult:
    """Borrowed read-only views, valid until release; one result at a time."""

    def __init__(self, owner, count, version):
        self.owner, self.count, self.version = owner, count, version
        self.released = False

    def _check(self):
        self.owner.control.check_thread()
        if self.released or self.owner.closed or self.owner.version != self.version:
            raise RuntimeError("expired result reference")

    def _view(self, name, start, count):
        self._check()
        return memoryview(self.owner.control.buffers[name])[start:start+count].toreadonly()

    @property
    def logits(self):
        return self._view("logits", 0, self.count*self.owner.spec.vocab)

    @property
    def hidden(self):
        c, cap = self.owner.spec, self.owner.capacity
        return tuple(self._view("hidden", layer*cap*c.width, self.count*c.width)
                     for layer in range(c.layers))

    @property
    def kv(self):
        c, cap = self.owner.spec, self.owner.capacity
        name = "kv" + str(self.owner.active_bank)
        return tuple(tuple(self._view(name, (layer*2+kind)*cap*c.width,
                                      self.owner.length*c.width) for kind in (0, 1))
                     for layer in range(c.layers))

    @property
    def routing(self):
        if self.owner.catalog.variant == "dense":
            return ()
        c, cap = self.owner.spec, self.owner.capacity
        return tuple((self._view("route_ids", layer*cap*c.top_k, self.count*c.top_k),
                      self._view("route_probs", layer*cap*c.top_k, self.count*c.top_k))
                     for layer in range(c.layers))

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


class CPURuntime:
    """One session and one CPU phase at a time, bounded sync or pipelined I/O.

    Prefill processes tokens causally one at a time, retaining no weight cache.
    All numeric arrays are reserved at construction, including two KV banks.
    """

    @staticmethod
    def minimum_plan(directory, capacity=6, tile_elements=5, read_elements=2, **io_options):
        return memory_plan(Catalog(directory), capacity, tile_elements, read_elements, **io_options)

    def __init__(self, directory, capacity=6, tile_elements=5, read_elements=2,
                 budget=None, backend=None, *, io_mode='sync', in_flight=2, io_workers=1,
                 trace_capacity=0):
        self.catalog = Catalog(directory)
        self.spec, self.capacity = self.catalog.spec, capacity
        self.plan = memory_plan(self.catalog, capacity, tile_elements, read_elements,
                                io_mode=io_mode, in_flight=in_flight, io_workers=io_workers,
                                trace_capacity=trace_capacity)
        coordinator = Coordinator if io_mode == 'sync' else PipelineCoordinator
        self.control = coordinator(self.catalog, self.plan, budget, backend)
        # (bank, prefix length, generation) is published as one reference update.
        self._published = (0, 0, 0)
        self.running = False
        self.failed, self.result_open, self.closed = False, False, False

    @property
    def active_bank(self):
        return self._published[0]

    @property
    def length(self):
        return self._published[1]

    @property
    def version(self):
        return self._published[2]

    def _linear(self, tensor_name, source, destination, offset=0):
        tensor = self.catalog.tensors[tensor_name]
        rows, cols = tensor.shape
        acc = self.control.buffers["accumulator"]
        slot = self.spec.width//self.spec.heads
        requests = ((tensor_name, row*cols+start, min(self.plan.tile_elements, cols-start))
                    for row in range(rows) for start in range(0, cols, self.plan.tile_elements))
        with closing(self.control.stream(requests)) as stream:
            for flat, count, weights in stream:
                row, start = divmod(flat, cols)
                if start == 0:
                    acc[slot] = 0.
                for j in range(count):
                    acc[slot] += weights.value(j)*source[start+j]
                if start+count == cols:
                    destination[offset+row] = acc[slot]
                    if not math.isfinite(destination[offset+row]):
                        raise ArithmeticError("nonfinite linear output")

    def _embedding(self, name, row, destination, add=False):
        width = self.spec.width
        requests = ((name, row*width+start, min(self.plan.tile_elements, width-start))
                    for start in range(0, width, self.plan.tile_elements))
        with closing(self.control.stream(requests)) as stream:
            for flat, count, weights in stream:
                start = flat-row*width
                for i in range(count):
                    value = weights.value(i)
                    destination[start+i] = destination[start+i]+value if add else value

    def _norm(self, source, scale_name, destination):
        c = self.spec
        with self.control.compute():
            squared = 0.
            for v in source:
                squared += v*v
            denominator = math.sqrt(squared/c.width + c.rms_epsilon)
            if not math.isfinite(denominator):
                raise ArithmeticError("nonfinite normalization denominator")
        requests = ((scale_name, start, min(self.plan.tile_elements, c.width-start))
                    for start in range(0, c.width, self.plan.tile_elements))
        with closing(self.control.stream(requests)) as stream:
            for start, count, weights in stream:
                for i in range(count):
                    destination[start+i] = source[start+i]/denominator*weights.value(i)

    def _ffn(self, source, prefix, destination):
        b = self.control.buffers
        self._linear(prefix+"gate", source, b["gate"])
        self._linear(prefix+"up", source, b["up"])
        with self.control.compute():
            for i in range(self.spec.hidden):
                z = b["gate"][i]
                if z >= 0:
                    silu = z/(1+math.exp(-z))
                else:
                    e = math.exp(z)
                    silu = z*e/(1+e)
                b["gate"][i] = silu*b["up"][i]
        self._linear(prefix+"down", b["gate"], destination)

    def _route(self, prefix):
        """Mathematical selection, independent of region loading policy."""
        b, c = self.control.buffers, self.spec
        self._linear(prefix+"router", b["norm"], b["scores"])
        with self.control.compute():
            for slot in range(c.top_k):
                chosen, best = -1, -math.inf
                for expert in range(c.experts):
                    if any(b["selected"][j] == expert for j in range(slot)):
                        continue
                    # Strict comparison gives the lower expert index on ties.
                    if b["scores"][expert] > best:
                        chosen, best = expert, b["scores"][expert]
                b["selected"][slot] = chosen
            maximum = b["scores"][b["selected"][0]]
            total = 0.
            for slot in range(c.top_k):
                p = math.exp(b["scores"][b["selected"][slot]]-maximum)
                b["probabilities"][slot] = p
                total += p
            for slot in range(c.top_k):
                b["probabilities"][slot] /= total

    def _attention(self, layer, position, candidate):
        b, c = self.control.buffers, self.spec
        dim = c.width//c.heads
        with self.control.compute():
            for head in range(c.heads):
                def score(token):
                    total = 0.
                    base = layer*2*self.capacity*c.width + token*c.width + head*dim
                    for d in range(dim):
                        total += b["q"][head*dim+d]*candidate[base+d]
                    return total/math.sqrt(dim)
                maximum = -math.inf
                for token in range(position+1):
                    maximum = max(maximum, score(token))
                denominator = 0.
                for d in range(dim):
                    b["accumulator"][d] = 0.
                for token in range(position+1):
                    probability = math.exp(score(token)-maximum)
                    denominator += probability
                    base = (layer*2+1)*self.capacity*c.width + token*c.width + head*dim
                    for d in range(dim):
                        b["accumulator"][d] += probability*candidate[base+d]
                for d in range(dim):
                    b["attention"][head*dim+d] = b["accumulator"][d]/denominator

    def _token(self, token, position, output_index, candidate):
        b, c, cap = self.control.buffers, self.spec, self.capacity
        self._embedding("embedding", token, b["x"])
        self._embedding("position", position, b["x"], add=True)
        for layer in range(c.layers):
            p = f"layers.{layer}."
            self._norm(b["x"], p+"norm1", b["norm"])
            self._linear(p+"q", b["norm"], b["q"])
            self._linear(p+"k", b["norm"], candidate,
                         (layer*2*cap+position)*c.width)
            self._linear(p+"v", b["norm"], candidate,
                         ((layer*2+1)*cap+position)*c.width)
            self._attention(layer, position, candidate)
            self._linear(p+"o", b["attention"], b["delta"])
            with self.control.compute():
                for i in range(c.width):
                    b["x"][i] += b["delta"][i]
            self._norm(b["x"], p+"norm2", b["norm"])
            if self.catalog.variant == "dense":
                self._ffn(b["norm"], p+"ffn.", b["delta"])
            else:
                self._route(p)
                with self.control.compute():
                    for i in range(c.width):
                        b["delta"][i] = 0.
                for slot in range(c.top_k):
                    expert = b["selected"][slot]
                    probability = b["probabilities"][slot]
                    trace = (layer*cap+output_index)*c.top_k+slot
                    b["route_ids"][trace] = expert
                    b["route_probs"][trace] = probability
                    self._ffn(b["norm"], p+f"experts.{expert}.", b["expert"])
                    with self.control.compute():
                        for i in range(c.width):
                            b["delta"][i] += probability*b["expert"][i]
            with self.control.compute():
                for i in range(c.width):
                    b["x"][i] += b["delta"][i]
                    b["hidden"][(layer*cap+output_index)*c.width+i] = b["x"][i]
        self._norm(b["x"], "final_norm", b["norm"])
        self._linear("lm_head", b["norm"], b["logits"], output_index*c.vocab)

    def prefill(self, tokens, *, cancel_event=None):
        self.control.check_thread()
        if self.closed or self.failed:
            raise RuntimeError("closed or failed session")
        if self.running or self.result_open or self.control.active:
            raise RuntimeError("release the previous result/region before executing")
        if (not isinstance(tokens, (tuple, list)) or not tokens
                or len(tokens)+self.length > self.capacity
                or any(type(t) is not int or not 0 <= t < self.spec.vocab for t in tokens)):
            raise ValueError("invalid token sequence or session capacity exceeded")
        b = self.control.buffers
        active = b["kv"+str(self.active_bank)]
        candidate = b["kv"+str(1-self.active_bank)]
        previous = self._published
        try:
            self.control.cancel_event = cancel_event
            self.control.check_cancelled()
            self.running = True
            with self.control.compute():
                for i in range(len(active)):
                    candidate[i] = active[i]
            for output_index, token in enumerate(tokens):
                self._token(token, self.length+output_index, output_index, candidate)
            self.control.check_cancelled()
            result = RuntimeResult(self, len(tokens), self.version+1)
            published = (1-self.active_bank, self.length+len(tokens), self.version+1)
            self.result_open = True
            # Every candidate value is ready; this is the single commit point.
            self._published = published
            self.running = False
            return result
        except BaseException:
            # An interruption before returning the result must also undo a
            # publication already made inside this call. The old bank remains
            # untouched, so rollback only replaces the published descriptor.
            self._published = previous
            self.result_open = False
            self.failed = True
            self.running = False
            self.control.abort()
            raise
        finally:
            self.control.cancel_event = None

    def decode(self, token, *, cancel_event=None):
        if not self.length:
            raise ValueError("decode requires a prefilled session")
        return self.prefill([token], cancel_event=cancel_event)

    def state_digest(self):
        self.control.check_thread()
        if self.closed:
            raise RuntimeError("closed session")
        bank = self.control.buffers["kv"+str(self.active_bank)]
        return self.length, hashlib.sha256(memoryview(bank)).hexdigest()

    def stats(self):
        return self.control.stats() | {"session_length": self.length,
                                       "session_failed": self.failed}

    def close(self):
        if self.running:
            raise RuntimeError("cannot close an executing session")
        self.control.close()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

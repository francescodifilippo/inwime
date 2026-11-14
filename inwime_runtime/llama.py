"""Tied-embedding Llama adapter: split-half RoPE, compact GQA KV, CPU FP64.

Only storage/math/transaction primitives are shared with the synthetic decoder.
No Transformers, torch, NumPy, weight cache, or whole-matrix allocations.
"""
from array import array
from dataclasses import dataclass
from pathlib import Path
import math

from .core import (IndexedSource, CheckpointError, Coordinator, Plan, CATEGORIES,
                   deep_size)
from .safetensors_source import SafetensorsSource, bounded_json
from .decoder import CPURuntime, RuntimeResult
from .pipeline import PipelineCoordinator
from .math_cpu import ScalarMath


def read_config(path):
    with Path(path).open('rb') as f:
        raw = f.read(65537)
    if len(raw) > 65536:
        raise CheckpointError('model config exceeds 64 KiB')
    return bounded_json(raw)


@dataclass(frozen=True)
class LlamaSpec:
    width: int
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    vocab: int
    context: int
    rms_epsilon: float
    theta: float

    @property
    def dim(self):
        return self.width // self.heads

    @property
    def kv_width(self):
        return self.kv_heads * self.dim


class LlamaCatalog(IndexedSource):
    def __init__(self, source):
        self.source = source
        d = read_config(source.directory/'config.json')
        try:
            if (d['model_type'] != 'llama' or d['architectures'] != ['LlamaForCausalLM']
                    or d['hidden_act'] != 'silu' or d['tie_word_embeddings'] is not True
                    or d.get('attention_bias', False) is not False
                    or d.get('mlp_bias', False) is not False
                    or d.get('rope_interleaved', False) is not False
                    or d.get('rope_scaling') is not None
                    or d.get('pretraining_tp', 1) != 1):
                raise CheckpointError('unsupported Llama architecture/tying/RoPE')
            ints = [d[k] for k in ('hidden_size', 'intermediate_size', 'num_hidden_layers',
                                  'num_attention_heads', 'num_key_value_heads',
                                  'vocab_size', 'max_position_embeddings')]
            if any(type(v) is not int or v <= 0 for v in ints):
                raise CheckpointError('positive integer model dimensions required')
            eps, theta = d['rms_norm_eps'], d['rope_theta']
            if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                   for v in (eps, theta)):
                raise CheckpointError('invalid RMSNorm/RoPE parameters')
            c = self.spec = LlamaSpec(*ints, eps, theta)
            if (c.width % c.heads or c.heads % c.kv_heads or c.dim % 2
                    or d.get('head_dim', c.dim) != c.dim):
                raise CheckpointError('inconsistent RoPE/GQA geometry')
            if c.layers*9+2 != len(source.tensors):
                raise CheckpointError('tensor count incompatible with Llama layers/tying')
            mapping = {'embedding': ('model.embed_tokens.weight', (c.vocab, c.width)),
                       'final_norm': ('model.norm.weight', (c.width,))}
            for layer in range(c.layers):
                p, q = f'layers.{layer}.', f'model.layers.{layer}.'
                for name, physical, shape in (
                    ('norm1', 'input_layernorm', (c.width,)),
                    ('norm2', 'post_attention_layernorm', (c.width,)),
                    ('q', 'self_attn.q_proj', (c.width, c.width)),
                    ('k', 'self_attn.k_proj', (c.kv_width, c.width)),
                    ('v', 'self_attn.v_proj', (c.kv_width, c.width)),
                    ('o', 'self_attn.o_proj', (c.width, c.width)),
                    ('ffn.gate', 'mlp.gate_proj', (c.hidden, c.width)),
                    ('ffn.up', 'mlp.up_proj', (c.hidden, c.width)),
                    ('ffn.down', 'mlp.down_proj', (c.width, c.hidden))):
                    mapping[p+name] = (q+physical+'.weight', shape)
            # This first profile accepts tied heads serialized once. An explicit
            # head is rejected rather than assumed equal without checking bytes.
            if set(source.tensors) != {v[0] for v in mapping.values()}:
                raise CheckpointError('missing/extra tensors or invalid serialized weight tying')
            self.tensors = {}
            for name, (physical, shape) in mapping.items():
                tensor = source.tensors[physical]
                if tensor.shape != shape:
                    raise CheckpointError('Llama tensor shape mismatch: '+physical)
                self.tensors[name] = tensor
            self.tensors['lm_head'] = self.tensors['embedding']
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, CheckpointError):
                raise
            raise CheckpointError('invalid Llama config') from error
        self.config = d
        added = deep_size((d, self.spec, self.tensors, mapping)) + 65536
        self.retained_bytes = source.retained_bytes + added
        self.bootstrap_bytes = source.bootstrap_bytes + added

    def __getattr__(self, name):
        return getattr(self.source, name)


def llama_plan(catalog, capacity, tile_elements, read_elements, io_mode):
    c = catalog.spec
    if (type(capacity) is not int or not 1 <= capacity <= c.context
            or type(tile_elements) is not int or tile_elements <= 0
            or type(read_elements) is not int or not 1 <= read_elements <= tile_elements
            or io_mode not in ('sync', 'pipeline')):
        raise ValueError('invalid Llama capacity/tile/read/mode')
    slots = 2 if io_mode == 'pipeline' else 1
    arrays = {}
    def add(name, category, count, code='d'):
        arrays[name] = (category, count, code)
    for slot in range(slots):
        suffix = str(slot) if slot else ''
        add('weights'+suffix, 'weights', tile_elements)
        add('staging'+suffix, 'staging', 2*(tile_elements if slots > 1 else read_elements), 'B')
    for n in ('x', 'norm', 'q', 'attention', 'delta'):
        add(n, 'activations', c.width)
    for n in ('gate', 'up'):
        add(n, 'activations', c.hidden)
    add('logits', 'output', capacity*c.vocab)
    add('hidden', 'output', capacity*c.layers*c.width)
    for n in ('kv0', 'kv1'):
        add(n, 'kv', c.layers*2*capacity*c.kv_width)
    add('accumulator', 'accumulators', c.dim+1)
    required = dict.fromkeys(CATEGORIES, 0)
    for category, count, code in arrays.values():
        required[category] += count*array(code).itemsize
    required['metadata'] = catalog.bootstrap_bytes + deep_size(arrays) + 65536 + capacity*128
    required['requests'] = 24576 if slots > 1 else 0
    return Plan(arrays, required, capacity, tile_elements, read_elements,
                slots if slots > 1 else 0, 1 if slots > 1 else 0)


class LlamaResult(RuntimeResult):
    @property
    def routing(self):
        self._check()
        return ()

    @property
    def kv(self):
        c, cap = self.owner.spec, self.owner.capacity
        name = 'kv'+str(self.owner.active_bank)
        return tuple(tuple(self._view(name, (layer*2+kind)*cap*c.kv_width,
                                      self.owner.length*c.kv_width) for kind in (0, 1))
                     for layer in range(c.layers))


class LlamaRuntime(CPURuntime):
    """One bounded session. Inherits acquisition, RMSNorm, SwiGLU and lifecycle.

    Whole-call KV atomicity; failed/cancelled sessions remain inspectable but
    cannot resume, matching the established fail-stop decoder contract.
    """
    @staticmethod
    def minimum_plan(directory, capacity=16, tile_elements=262144, read_elements=262144,
                     *, io_mode='sync'):
        return llama_plan(LlamaCatalog(SafetensorsSource(directory)), capacity,
                          tile_elements, read_elements, io_mode)

    def __init__(self, directory, capacity=16, tile_elements=262144, read_elements=262144,
                 budget=None, backend=None, *, io_mode='sync', integrity='full',
                 math_backend=None):
        self.catalog = LlamaCatalog(SafetensorsSource(directory))
        self.spec, self.capacity = self.catalog.spec, capacity
        self.plan = llama_plan(self.catalog, capacity, tile_elements, read_elements, io_mode)
        self.math_backend = ScalarMath() if math_backend is None else math_backend
        coordinator = Coordinator if io_mode == 'sync' else PipelineCoordinator
        self.control = coordinator(self.catalog, self.plan, budget, backend,
                                   integrity=integrity, math_backend=self.math_backend)
        self._published = (0, 0, 0)
        self.running = self.failed = self.result_open = self.closed = False

    def _rope(self, vector, heads, position, offset=0):
        c = self.spec
        for head in range(heads):
            for j in range(c.dim//2):
                angle = position / c.theta**(2*j/c.dim)
                cosine, sine = math.cos(angle), math.sin(angle)
                a = offset+head*c.dim+j
                b = a+c.dim//2
                x, y = vector[a], vector[b]
                vector[a], vector[b] = x*cosine-y*sine, y*cosine+x*sine

    def _attention(self, layer, position, candidate):
        b, c, cap = self.control.buffers, self.spec, self.capacity
        with self.control.compute():
            for head in range(c.heads):
                kh = head//(c.heads//c.kv_heads)
                def score(token):
                    base = layer*2*cap*c.kv_width+token*c.kv_width+kh*c.dim
                    total = 0.
                    for d in range(c.dim):
                        total += b['q'][head*c.dim+d]*candidate[base+d]
                    return total/math.sqrt(c.dim)
                maximum = max(score(t) for t in range(position+1))
                denominator = 0.
                for d in range(c.dim):
                    b['accumulator'][d] = 0.
                for token in range(position+1):
                    probability = math.exp(score(token)-maximum)
                    denominator += probability
                    base = (layer*2+1)*cap*c.kv_width+token*c.kv_width+kh*c.dim
                    for d in range(c.dim):
                        b['accumulator'][d] += probability*candidate[base+d]
                for d in range(c.dim):
                    b['attention'][head*c.dim+d] = b['accumulator'][d]/denominator

    def _token(self, token, position, output_index, candidate):
        b, c, cap = self.control.buffers, self.spec, self.capacity
        self._embedding('embedding', token, b['x'])
        for layer in range(c.layers):
            self.control.check_cancelled()
            p = f'layers.{layer}.'
            self._norm(b['x'], p+'norm1', b['norm'])
            self._linear(p+'q', b['norm'], b['q'])
            ko = layer*2*cap*c.kv_width+position*c.kv_width
            vo = ko+cap*c.kv_width
            self._linear(p+'k', b['norm'], candidate, ko)
            self._linear(p+'v', b['norm'], candidate, vo)
            with self.control.compute():
                self._rope(b['q'], c.heads, position)
                self._rope(candidate, c.kv_heads, position, ko)
            self._attention(layer, position, candidate)
            self._linear(p+'o', b['attention'], b['delta'])
            with self.control.compute():
                for i in range(c.width):
                    b['x'][i] += b['delta'][i]
            self._norm(b['x'], p+'norm2', b['norm'])
            self._ffn(b['norm'], p+'ffn.', b['delta'])
            with self.control.compute():
                for i in range(c.width):
                    b['x'][i] += b['delta'][i]
                    b['hidden'][(layer*cap+output_index)*c.width+i] = b['x'][i]
        self._norm(b['x'], 'final_norm', b['norm'])
        self._linear('lm_head', b['norm'], b['logits'], output_index*c.vocab)

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
        tokens = tuple(tokens)  # Bounded snapshot before GIL-releasing I/O.
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
            result = LlamaResult(self, len(tokens), self.version+1)
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


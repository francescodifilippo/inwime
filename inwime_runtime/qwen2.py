"""Qwen2 dense decoder adapter: Q/K/V bias, untied head, split-half RoPE, GQA, CPU FP64.

Reuses the Llama session (RoPE, attention, transactional KV, prefill) and the
shared acquisition/RMSNorm/SwiGLU primitives. The safetensors source stays
family-independent; this module is only the semantic view and the bias/head
differences of Qwen2ForCausalLM. No Transformers, torch, NumPy or weight cache.
"""
from contextlib import closing
import math

from .core import CheckpointError, Coordinator, IndexedSource, deep_size
from .safetensors_source import SafetensorsSource
from .llama import LlamaRuntime, LlamaSpec, llama_plan, read_config
from .pipeline import PipelineCoordinator
from .math_cpu import ScalarMath


class Qwen2Catalog(IndexedSource):
    """Exact Qwen2ForCausalLM tensor set; anything else is rejected, not guessed."""

    def __init__(self, source):
        self.source = source
        d = read_config(source.directory/'config.json')
        try:
            if (d['model_type'] != 'qwen2' or d['architectures'] != ['Qwen2ForCausalLM']
                    or d['hidden_act'] != 'silu' or d['tie_word_embeddings'] is not False
                    or d.get('use_sliding_window', False) is not False
                    or d.get('use_mrope', False) is not False
                    or d.get('rope_scaling') is not None
                    or d.get('layer_types', None) not in (None, ['full_attention']*d['num_hidden_layers'])):
                raise CheckpointError('unsupported Qwen2 architecture/tying/RoPE/attention')
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
            if c.layers*12+3 != len(source.tensors):
                raise CheckpointError('tensor count incompatible with Qwen2 layers/head')
            mapping = {'embedding': ('model.embed_tokens.weight', (c.vocab, c.width)),
                       'final_norm': ('model.norm.weight', (c.width,)),
                       'lm_head': ('lm_head.weight', (c.vocab, c.width))}
            for layer in range(c.layers):
                p, q = f'layers.{layer}.', f'model.layers.{layer}.'
                for name, physical, shape in (
                    ('norm1', 'input_layernorm.weight', (c.width,)),
                    ('norm2', 'post_attention_layernorm.weight', (c.width,)),
                    ('q', 'self_attn.q_proj.weight', (c.width, c.width)),
                    ('q.bias', 'self_attn.q_proj.bias', (c.width,)),
                    ('k', 'self_attn.k_proj.weight', (c.kv_width, c.width)),
                    ('k.bias', 'self_attn.k_proj.bias', (c.kv_width,)),
                    ('v', 'self_attn.v_proj.weight', (c.kv_width, c.width)),
                    ('v.bias', 'self_attn.v_proj.bias', (c.kv_width,)),
                    ('o', 'self_attn.o_proj.weight', (c.width, c.width)),
                    ('ffn.gate', 'mlp.gate_proj.weight', (c.hidden, c.width)),
                    ('ffn.up', 'mlp.up_proj.weight', (c.hidden, c.width)),
                    ('ffn.down', 'mlp.down_proj.weight', (c.width, c.hidden))):
                    mapping[p+name] = (q+physical, shape)
            if set(source.tensors) != {v[0] for v in mapping.values()}:
                raise CheckpointError('missing/extra Qwen2 tensors (bias, head or projection)')
            self.tensors = {}
            for name, (physical, shape) in mapping.items():
                tensor = source.tensors[physical]
                if tensor.shape != shape:
                    raise CheckpointError('Qwen2 tensor shape mismatch: '+physical)
                self.tensors[name] = tensor
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, CheckpointError):
                raise
            raise CheckpointError('invalid Qwen2 config') from error
        self.config = d
        added = deep_size((d, self.spec, self.tensors, mapping)) + 65536
        self.retained_bytes = source.retained_bytes + added
        self.bootstrap_bytes = source.bootstrap_bytes + added

    def __getattr__(self, name):
        return getattr(self.source, name)


class Qwen2Runtime(LlamaRuntime):
    """Same fail-stop session and KV transaction as the Llama adapter."""

    @staticmethod
    def minimum_plan(directory, capacity=16, tile_elements=262144, read_elements=262144,
                     *, io_mode='sync'):
        return llama_plan(Qwen2Catalog(SafetensorsSource(directory)), capacity,
                          tile_elements, read_elements, io_mode)

    def __init__(self, directory, capacity=16, tile_elements=262144, read_elements=262144,
                 budget=None, backend=None, *, io_mode='sync', integrity='full',
                 math_backend=None):
        self.catalog = Qwen2Catalog(SafetensorsSource(directory))
        self.spec, self.capacity = self.catalog.spec, capacity
        self.plan = llama_plan(self.catalog, capacity, tile_elements, read_elements, io_mode)
        self.math_backend = ScalarMath() if math_backend is None else math_backend
        coordinator = Coordinator if io_mode == 'sync' else PipelineCoordinator
        self.control = coordinator(self.catalog, self.plan, budget, backend,
                                   integrity=integrity, math_backend=self.math_backend)
        self._published = (0, 0, 0)
        self.running = self.failed = self.result_open = self.closed = False

    def _bias(self, name, destination, offset=0):
        """destination[offset+i] += bias[i], bias streamed through bounded leases."""
        count = self.catalog.tensors[name].elements
        requests = ((name, start, min(self.plan.tile_elements, count-start))
                    for start in range(0, count, self.plan.tile_elements))
        with closing(self.control.stream(requests)) as stream:
            for start, size, weights in stream:
                for i in range(size):
                    destination[offset+start+i] += weights.value(i)

    def _token(self, token, position, output_index, candidate):
        b, c, cap = self.control.buffers, self.spec, self.capacity
        self._embedding('embedding', token, b['x'])
        for layer in range(c.layers):
            self.control.check_cancelled()
            p = f'layers.{layer}.'
            self._norm(b['x'], p+'norm1', b['norm'])
            self._linear(p+'q', b['norm'], b['q'])
            self._bias(p+'q.bias', b['q'])
            ko = layer*2*cap*c.kv_width+position*c.kv_width
            vo = ko+cap*c.kv_width
            self._linear(p+'k', b['norm'], candidate, ko)
            self._bias(p+'k.bias', candidate, ko)
            self._linear(p+'v', b['norm'], candidate, vo)
            self._bias(p+'v.bias', candidate, vo)
            # Upstream adds the projection bias before RoPE; K is cached post-RoPE.
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

"""Additive Qwen2 adapter for a tied head (`tie_word_embeddings=true`), #38.

The Percorso congelato (`inwime_runtime.qwen2.Qwen2Catalog`) accepts untied heads only and stays
unchanged. This catalog accepts the Qwen2ForCausalLM checkpoints that serialize the head once,
as `model.embed_tokens.weight` (Qwen2.5-0.5B and the other small Qwen2.5 models), and binds the
semantic tensor `lm_head` to the embedding's own descriptor, as `LlamaCatalog` does: the head
streams the embedding's bytes through the same bounded leases, with no second copy in memory or
on disk. The session, plan and computation are the frozen `Qwen2Runtime`'s, inherited unchanged.

The validation repeats `Qwen2Catalog`'s, except for the tying flag, the tensor count (12 per layer
plus embedding and final norm) and the absent head. A tied config that also serializes
`lm_head.weight` is refused: its bytes are never assumed equal to the embedding's.
"""
import math

from inwime_runtime.core import CheckpointError, IndexedSource, deep_size
from inwime_runtime.llama import LlamaSpec, read_config
from inwime_runtime.qwen2 import Qwen2Catalog

LAYER_TENSORS = (('norm1', 'input_layernorm.weight', 'width'),
                 ('norm2', 'post_attention_layernorm.weight', 'width'),
                 ('q', 'self_attn.q_proj.weight', 'width x width'),
                 ('q.bias', 'self_attn.q_proj.bias', 'width'),
                 ('k', 'self_attn.k_proj.weight', 'kv x width'),
                 ('k.bias', 'self_attn.k_proj.bias', 'kv'),
                 ('v', 'self_attn.v_proj.weight', 'kv x width'),
                 ('v.bias', 'self_attn.v_proj.bias', 'kv'),
                 ('o', 'self_attn.o_proj.weight', 'width x width'),
                 ('ffn.gate', 'mlp.gate_proj.weight', 'hidden x width'),
                 ('ffn.up', 'mlp.up_proj.weight', 'hidden x width'),
                 ('ffn.down', 'mlp.down_proj.weight', 'width x hidden'))


class TiedQwen2Catalog(IndexedSource):
    """Exact tied Qwen2ForCausalLM tensor set; anything else is rejected, not guessed."""

    def __init__(self, source):
        self.source = source
        d = read_config(source.directory/'config.json')
        try:
            if (d['model_type'] != 'qwen2' or d['architectures'] != ['Qwen2ForCausalLM']
                    or d['hidden_act'] != 'silu' or d['tie_word_embeddings'] is not True
                    or d.get('use_sliding_window', False) is not False
                    or d.get('use_mrope', False) is not False
                    or d.get('rope_scaling') is not None
                    or d.get('layer_types', None) not in (None, ['full_attention']*d['num_hidden_layers'])):
                raise CheckpointError('unsupported tied Qwen2 architecture/tying/RoPE/attention')
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
            if 'lm_head.weight' in source.tensors:
                raise CheckpointError('tie_word_embeddings=true with a serialized lm_head.weight: '
                                      'invalid serialized weight tying')
            if c.layers*12+2 != len(source.tensors):
                raise CheckpointError('tensor count incompatible with tied Qwen2 layers/head')
            size = {'width': (c.width,), 'kv': (c.kv_width,), 'width x width': (c.width, c.width),
                    'kv x width': (c.kv_width, c.width), 'hidden x width': (c.hidden, c.width),
                    'width x hidden': (c.width, c.hidden)}
            mapping = {'embedding': ('model.embed_tokens.weight', (c.vocab, c.width)),
                       'final_norm': ('model.norm.weight', (c.width,))}
            for layer in range(c.layers):
                for name, physical, shape in LAYER_TENSORS:
                    mapping[f'layers.{layer}.{name}'] = (f'model.layers.{layer}.{physical}', size[shape])
            if set(source.tensors) != {v[0] for v in mapping.values()}:
                raise CheckpointError('missing/extra tied Qwen2 tensors (bias or projection)')
            self.tensors = {}
            for name, (physical, shape) in mapping.items():
                tensor = source.tensors[physical]
                if tensor.shape != shape:
                    raise CheckpointError('Qwen2 tensor shape mismatch: '+physical)
                self.tensors[name] = tensor
            self.tensors['lm_head'] = self.tensors['embedding']
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


def is_tied(directory):
    """Whether the config of a Checkpoint directory declares a tied head (strictly `true`)."""
    return read_config(directory/'config.json').get('tie_word_embeddings') is True


def qwen2_catalog(source):
    """The Qwen2 adapter for a source: the frozen catalog for an untied head, this one for a tied one."""
    return (TiedQwen2Catalog if is_tied(source.directory) else Qwen2Catalog)(source)

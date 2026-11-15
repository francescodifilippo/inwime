"""Tiny deterministic Qwen2ForCausalLM-format BF16 fixture (stdlib only).

Same tensor names, bias set, untied head and config keys as Qwen2.5; dimensions
are small and values synthetic. Used by the CPU tests and by the torch-side
oracle check; it is not a trained model.
"""
import hashlib
import json
import math
from pathlib import Path
import struct

CONFIG = dict(architectures=['Qwen2ForCausalLM'], model_type='qwen2', hidden_act='silu',
              hidden_size=8, intermediate_size=12, num_hidden_layers=2, num_attention_heads=2,
              num_key_value_heads=1, vocab_size=16, max_position_embeddings=32,
              rms_norm_eps=1e-6, rope_theta=1000000.0, tie_word_embeddings=False,
              use_sliding_window=False, sliding_window=32, max_window_layers=2,
              use_mrope=False, attention_dropout=0.0, bos_token_id=0, eos_token_id=0,
              torch_dtype='bfloat16', initializer_range=0.02, use_cache=True)


def shapes(config):
    c = config
    width, hidden, kv = c['hidden_size'], c['intermediate_size'], \
        c['num_key_value_heads']*c['hidden_size']//c['num_attention_heads']
    result = {'model.embed_tokens.weight': [c['vocab_size'], width],
              'model.norm.weight': [width], 'lm_head.weight': [c['vocab_size'], width]}
    for layer in range(c['num_hidden_layers']):
        p = f'model.layers.{layer}.'
        for name, shape in (('input_layernorm.weight', [width]),
                            ('post_attention_layernorm.weight', [width]),
                            ('self_attn.q_proj.weight', [width, width]), ('self_attn.q_proj.bias', [width]),
                            ('self_attn.k_proj.weight', [kv, width]), ('self_attn.k_proj.bias', [kv]),
                            ('self_attn.v_proj.weight', [kv, width]), ('self_attn.v_proj.bias', [kv]),
                            ('self_attn.o_proj.weight', [width, width]),
                            ('mlp.gate_proj.weight', [hidden, width]), ('mlp.up_proj.weight', [hidden, width]),
                            ('mlp.down_proj.weight', [width, hidden])):
            result[p+name] = shape
    return result


def word(name, index):
    """Deterministic BF16 word: norms near 1, biases and matrices signed and varied."""
    h = int.from_bytes(hashlib.sha256(f'{name}:{index}'.encode()).digest()[:8], 'little')
    if name.endswith('layernorm.weight') or name == 'model.norm.weight':
        value = 0.75 + (h % 64)/128
    elif name.endswith('.bias'):
        value = ((h % 61)-30)/40
    else:
        value = ((h % 97)-48)/96
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    return bits >> 16  # values above have at most 8 significant bits: exact in BF16


def write(directory, config=None, *, extra=None, drop=(), dtype_override=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    config = dict(CONFIG if config is None else config)
    table = shapes(config)
    for name in drop:
        del table[name]
    table.update(extra or {})
    header, payload, offset = {}, bytearray(), 0
    for name, shape in table.items():
        n = math.prod(shape)
        header[name] = dict(dtype=(dtype_override or {}).get(name, 'BF16'), shape=shape,
                            data_offsets=[offset, offset+2*n])
        offset += 2*n
        for i in range(n):
            payload += struct.pack('<H', word(name, i))
    raw = json.dumps(header).encode()
    data = struct.pack('<Q', len(raw)) + raw + payload
    (directory/'model.safetensors').write_bytes(data)
    (directory/'safetensors.json').write_text(json.dumps(dict(
        format='inwime-safetensors-bf16-v1', index=None,
        files={'model.safetensors': dict(nbytes=len(data), sha256=hashlib.sha256(data).hexdigest())})))
    (directory/'config.json').write_text(json.dumps(config))
    return directory

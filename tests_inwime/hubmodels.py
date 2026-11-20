"""Tiny, real-format Hugging Face checkpoints for the fake Hub (config, tokenizer, safetensors)."""
import hashlib
import json

from fakehub import Repo, safetensors_bytes


def sha(n):
    """A deterministic full 40-hex revision."""
    return ('%040x' % (0x5eed0000 + n))


QWEN2 = dict(architectures=['Qwen2ForCausalLM'], model_type='qwen2', hidden_act='silu',
             hidden_size=8, intermediate_size=16, num_hidden_layers=2, num_attention_heads=2,
             num_key_value_heads=1, vocab_size=32, max_position_embeddings=32768,
             rms_norm_eps=1e-6, rope_theta=1000000.0, tie_word_embeddings=False,
             use_sliding_window=False, sliding_window=32768, rope_scaling=None,
             torch_dtype='bfloat16')

LLAMA = dict(architectures=['LlamaForCausalLM'], model_type='llama', hidden_act='silu',
             hidden_size=8, intermediate_size=16, num_hidden_layers=2, num_attention_heads=2,
             num_key_value_heads=1, vocab_size=32, max_position_embeddings=8192,
             rms_norm_eps=1e-5, rope_theta=100000.0, tie_word_embeddings=True,
             attention_bias=False, mlp_bias=False, rope_scaling=None, pretraining_tp=1,
             torch_dtype='bfloat16')

TOOL_TEMPLATE = ("{%- if tools %}<tools>{% for tool in tools %}{{ tool | tojson }}{% endfor %}</tools>"
                 "{%- endif %}{% for m in messages %}{{ m.content }}{% endfor %}")
PLAIN_TEMPLATE = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"


def tensors(config, *, tied=None):
    c = config
    width, hidden, vocab = c['hidden_size'], c['intermediate_size'], c['vocab_size']
    kv = c['num_key_value_heads']*width//c['num_attention_heads']
    bias = c['model_type'] == 'qwen2'
    tied = c['tie_word_embeddings'] if tied is None else tied
    shapes = {'model.embed_tokens.weight': (vocab, width), 'model.norm.weight': (width,)}
    if not tied:
        shapes['lm_head.weight'] = (vocab, width)
    for layer in range(c['num_hidden_layers']):
        p = 'model.layers.%d.' % layer
        shapes.update({p+'input_layernorm.weight': (width,), p+'post_attention_layernorm.weight': (width,),
                       p+'self_attn.q_proj.weight': (width, width), p+'self_attn.k_proj.weight': (kv, width),
                       p+'self_attn.v_proj.weight': (kv, width), p+'self_attn.o_proj.weight': (width, width),
                       p+'mlp.gate_proj.weight': (hidden, width), p+'mlp.up_proj.weight': (hidden, width),
                       p+'mlp.down_proj.weight': (width, hidden)})
        if bias:
            shapes.update({p+'self_attn.q_proj.bias': (width,), p+'self_attn.k_proj.bias': (kv,),
                           p+'self_attn.v_proj.bias': (kv,)})
    return shapes


def checkpoint(config, *, shapes=None, dtype='BF16', shards=1, template=TOOL_TEMPLATE,
               tokenizer=True, drop=(), extra=None, tokenizer_config=None):
    """Files of one revision. `shards` > 1 writes an index, as large checkpoints do."""
    shapes = tensors(config) if shapes is None else shapes
    files = {'config.json': json.dumps(config, indent=2).encode(),
             'generation_config.json': b'{"do_sample": false}', 'README.md': b'# model card\n'}
    if tokenizer:
        files['tokenizer.json'] = b'{"version": "1.0", "model": {"type": "BPE", "vocab": {}, "merges": []}}'
        tc = {'tokenizer_class': 'Qwen2Tokenizer', 'model_max_length': 32768}
        if template is not None:
            tc['chat_template'] = template
        tc.update(tokenizer_config or {})
        files['tokenizer_config.json'] = json.dumps(tc).encode()
    names = sorted(shapes)
    if shards == 1:
        files['model.safetensors'] = safetensors_bytes(shapes, dtype)
    else:
        weight_map, total = {}, 0
        for i in range(shards):
            part = {n: shapes[n] for n in names[i::shards]}
            shard = 'model-%05d-of-%05d.safetensors' % (i+1, shards)
            files[shard] = data = safetensors_bytes(part, dtype)
            weight_map.update({n: shard for n in part})
            total += len(data)
        files['model.safetensors.index.json'] = json.dumps(
            {'metadata': {'total_size': total}, 'weight_map': weight_map}).encode()
    files.update(extra or {})
    for name in drop:
        files.pop(name, None)
    return files


def repo(repo_id, files, *, revision=None, **kwargs):
    revision = revision or hashlib.sha1(repo_id.encode()).hexdigest()
    kwargs.setdefault('tags', ['transformers', 'safetensors', 'text-generation'])
    return Repo(id=repo_id, revisions={revision: files}, branches={'main': revision}, **kwargs)

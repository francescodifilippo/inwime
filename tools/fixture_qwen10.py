"""Small deterministic Qwen2.5-14B-shaped BF16 fixture (stdlib only), increment 10.

Preserves the mathematical properties of Qwen/Qwen2.5-14B that the 7B never
exercised, at a few hundred KB instead of 29.5 GB:

  * GQA 5:1 (10 query heads, 2 KV heads; the 14B has 40/8), split-half RoPE,
    rope_theta 1e6, Q/K/V bias, no O/MLP bias, untied head;
  * rms_norm_eps 1e-5, made observable: token TINY_TOKEN has an embedding row of
    magnitude ~2^-10, so its mean square (~1e-6) is comparable to eps and a
    wrong epsilon changes the result far beyond the FP64 thresholds;
  * dimensions that are not multiples of the tile sizes used in the tests
    (width 80, intermediate 54, vocab 37, KV width 16);
  * several shards with an index, a decoder layer split across two shards (as
    layers 4, 11, 18, 33, 40, 47 of the 14B) and the head in the last shard;
    headers carry __metadata__ {"format": "pt"} and are padded to 8 bytes.

Values are synthetic and exactly representable in BF16; it is not a trained model.
"""
import hashlib
import json
import math
from pathlib import Path
import struct

TINY_TOKEN = 36

CONFIG = dict(architectures=['Qwen2ForCausalLM'], model_type='qwen2', hidden_act='silu',
              hidden_size=80, intermediate_size=54, num_hidden_layers=3, num_attention_heads=10,
              num_key_value_heads=2, vocab_size=37, max_position_embeddings=64,
              rms_norm_eps=1e-05, rope_theta=1000000.0, tie_word_embeddings=False,
              use_sliding_window=False, sliding_window=64, max_window_layers=3,
              attention_dropout=0.0, bos_token_id=0, eos_token_id=1,
              torch_dtype='bfloat16', initializer_range=0.02, use_cache=True)

LAYER_TENSORS = (('input_layernorm.weight', 'w'), ('mlp.down_proj.weight', 'down'),
                 ('mlp.gate_proj.weight', 'mlp'), ('mlp.up_proj.weight', 'mlp'),
                 ('post_attention_layernorm.weight', 'w'),
                 ('self_attn.k_proj.bias', 'kv'), ('self_attn.k_proj.weight', 'kvw'),
                 ('self_attn.o_proj.weight', 'ww'), ('self_attn.q_proj.bias', 'w'),
                 ('self_attn.q_proj.weight', 'ww'), ('self_attn.v_proj.bias', 'kv'),
                 ('self_attn.v_proj.weight', 'kvw'))


def shapes(config=CONFIG):
    c = config
    width, hidden = c['hidden_size'], c['intermediate_size']
    kv = c['num_key_value_heads']*width//c['num_attention_heads']
    table = dict(w=[width], ww=[width, width], kv=[kv], kvw=[kv, width], mlp=[hidden, width], down=[width, hidden])
    result = {'model.embed_tokens.weight': [c['vocab_size'], width]}
    for layer in range(c['num_hidden_layers']):
        for name, kind in LAYER_TENSORS:
            result[f'model.layers.{layer}.{name}'] = table[kind]
    result['model.norm.weight'] = [width]
    result['lm_head.weight'] = [c['vocab_size'], width]
    return result


def value(name, index, width):
    h = int.from_bytes(hashlib.sha256(f'q10:{name}:{index}'.encode()).digest()[:8], 'little')
    if name.endswith('layernorm.weight') or name == 'model.norm.weight':
        return 0.75 + (h % 64)/128
    if name.endswith('.bias'):
        return ((h % 61)-30)/32
    if name == 'model.embed_tokens.weight' and index // width == TINY_TOKEN:
        return ((h % 15)-7 or 1) * 2.0**-10 / 4          # |v| in [2^-12, 7*2^-12]: mean square ~1e-6
    return ((h % 97)-48)/128


def bf16(v):
    bits = struct.unpack('<I', struct.pack('<f', v))[0]
    if bits & 0xFFFF:
        raise ValueError('value not exactly representable in BF16')
    return bits >> 16


def default_split(config=CONFIG):
    """Three shards; layer 1 is split between shard 1 and 2; the head is in the last shard."""
    names = list(shapes(config))
    layer1 = [n for n in names if n.startswith('model.layers.1.')]
    cut1 = names.index(layer1[0]) + len(layer1)//2
    cut2 = names.index('model.layers.2.self_attn.k_proj.bias')
    return [names[:cut1], names[cut1:cut2], names[cut2:]]


def write(directory, config=None, split=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    config = dict(CONFIG if config is None else config)
    table = shapes(config)
    groups = split or default_split(config)
    if sorted(n for g in groups for n in g) != sorted(table):
        raise ValueError('split must cover every tensor exactly once')
    files, weight_map, total = {}, {}, 0
    for i, group in enumerate(groups):
        filename = f'model-{i+1:05d}-of-{len(groups):05d}.safetensors'
        header, payload, offset = {'__metadata__': {'format': 'pt'}}, bytearray(), 0
        for name in sorted(group):
            shape = table[name]
            n = math.prod(shape)
            header[name] = dict(dtype='BF16', shape=shape, data_offsets=[offset, offset+2*n])
            for k in range(n):
                payload += struct.pack('<H', bf16(value(name, k, config['hidden_size'])))
            offset += 2*n
            weight_map[name] = filename
        raw = json.dumps(header, separators=(',', ':')).encode()
        raw += b' '*(-len(raw) % 8)
        data = struct.pack('<Q', len(raw)) + raw + bytes(payload)
        (directory/filename).write_bytes(data)
        files[filename] = dict(nbytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        total += len(payload)
    index = (json.dumps(dict(metadata=dict(total_size=total), weight_map=dict(sorted(weight_map.items()))),
                        indent=2)+'\n').encode()
    (directory/'model.safetensors.index.json').write_bytes(index)
    (directory/'safetensors.json').write_text(json.dumps(dict(
        format='inwime-safetensors-bf16-v1', files=files, index='model.safetensors.index.json',
        index_sha256=hashlib.sha256(index).hexdigest()), indent=2)+'\n')
    (directory/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (directory/'provenance.json').write_text(json.dumps(dict(revision='fixture-qwen10', model='synthetic'))+'\n')
    return directory


def load(directory):
    """Independent decode of every tensor: header JSON + struct BF16 -> float (no runtime import)."""
    directory = Path(directory)
    index = json.loads((directory/'model.safetensors.index.json').read_text())
    out = {}
    for filename in sorted(set(index['weight_map'].values())):
        raw = (directory/filename).read_bytes()
        length, = struct.unpack('<Q', raw[:8])
        header = json.loads(raw[8:8+length])
        header.pop('__metadata__', None)
        base = 8+length
        for name, d in header.items():
            begin, end = d['data_offsets']
            words = struct.unpack(f'<{(end-begin)//2}H', raw[base+begin:base+end])
            values = [struct.unpack('<f', struct.pack('<I', w << 16))[0] for w in words]
            shape = d['shape']
            out[name] = values if len(shape) == 1 else [values[r*shape[1]:(r+1)*shape[1]] for r in range(shape[0])]
    return out


def sparse_checkpoint(metadata, directory):
    """Structure-only copy of the real 14B checkpoint: original headers, payload as a filesystem hole.

    metadata: a directory laid out as results/increment10/preflight-metadata. Allocates
    a few KB per shard; the apparent size equals the official file size. No weights.
    """
    metadata, directory = Path(metadata), Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    api = json.loads((metadata/'api-revision-blobs.json').read_text())
    sizes = {e['rfilename']: (e['size'], (e.get('lfs') or {}).get('sha256')) for e in api['siblings']}
    files = {}
    for header in sorted(metadata.glob('header-*.safetensors.json')):
        name = header.name.removeprefix('header-').removesuffix('.json')
        raw = header.read_bytes()
        with (directory/name).open('xb') as f:
            f.write(struct.pack('<Q', len(raw)) + raw)
            f.truncate(sizes[name][0])
        files[name] = dict(nbytes=sizes[name][0], sha256=sizes[name][1])
    index = (metadata/'file-model.safetensors.index.json').read_bytes()
    (directory/'model.safetensors.index.json').write_bytes(index)
    (directory/'config.json').write_bytes((metadata/'file-config.json').read_bytes())
    (directory/'safetensors.json').write_text(json.dumps(dict(
        format='inwime-safetensors-bf16-v1', files=files, index='model.safetensors.index.json',
        index_sha256=hashlib.sha256(index).hexdigest())))
    return directory

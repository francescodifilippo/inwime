"""Independent stdlib header/byte/coverage audit; no runtime imports."""
import hashlib
import json
import math
from pathlib import Path
import re
import struct

root = Path('results/increment8')
original = Path('checkpoints/smollm2-1.7b-d4c18cc')
sharded = Path('checkpoints/smollm2-1.7b-d4c18cc-sharded')
expected = '1193528982f4ac0c0b707ce36fd7dc03a0ef6f3e1a432deb886dce2e90c300c0'

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def tensors(path):
    with path.open('rb') as f:
        size = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(size))
    return {name: (entry, size+8) for name, entry in header.items() if name != '__metadata__'}

def tensor_digest(path, entry, base):
    digest = hashlib.sha256()
    begin, end = entry['data_offsets']
    with path.open('rb') as f:
        f.seek(base+begin)
        left = end-begin
        while left:
            data = f.read(min(left, 2**20))
            assert data
            digest.update(data); left -= len(data)
    return digest.hexdigest()

source = original/'model.safetensors'
assert sha(source) == expected
entries = tensors(source)
assert len(entries) == 218 and all(e['dtype'] == 'BF16' for e, _ in entries.values())
original_tensor_hashes = {name: tensor_digest(source, *entry) for name, entry in entries.items()}
repacked = json.loads((root/'repack.json').read_text())
assert original_tensor_hashes == repacked['tensor_sha256']
index = json.loads((sharded/'model.safetensors.index.json').read_text())['weight_map']
shard_hashes = {}
for filename, descriptor in repacked['descriptor']['files'].items():
    path = sharded/filename
    assert sha(path) == descriptor['sha256'] and path.stat().st_size == descriptor['nbytes']
    for name, (entry, base) in tensors(path).items():
        assert index[name] == filename and entry['dtype'] == 'BF16'
        assert entry['shape'] == entries[name][0]['shape']
        shard_hashes[name] = tensor_digest(path, entry, base)
assert shard_hashes == original_tensor_hashes
config = json.loads((original/'config.json').read_text())
w, h, layers = config['hidden_size'], config['intermediate_size'], config['num_hidden_layers']
kv_width = config['num_key_value_heads']*(w//config['num_attention_heads'])
linear = config['vocab_size']*w + layers*(2*w*w+2*kv_width*w+3*h*w)
norms = (2*layers+1)*w
per_token_bytes = 2*(linear+norms+w)
payload = sum(2*math.prod(e['shape']) for e, _ in entries.values())
assert payload == 3422752768
mapping = {'input_layernorm': 'norm1', 'post_attention_layernorm': 'norm2',
    'self_attn.q_proj': 'q', 'self_attn.k_proj': 'k', 'self_attn.v_proj': 'v',
    'self_attn.o_proj': 'o', 'mlp.gate_proj': 'ffn.gate',
    'mlp.up_proj': 'ffn.up', 'mlp.down_proj': 'ffn.down'}
logical = {'embedding': 2*w}
for name, (entry, _) in entries.items():
    if name == 'model.embed_tokens.weight':
        semantic = 'lm_head'
    elif name == 'model.norm.weight':
        semantic = 'final_norm'
    else:
        match = re.fullmatch(r'model.layers.(\d+).(.+).weight', name)
        semantic = 'layers.'+match[1]+'.'+mapping[match[2]]
    logical[semantic] = 2*math.prod(entry['shape'])
assert sum(logical.values()) == per_token_bytes
checks = {}
files = list(root.glob('measured-*.json'))+list(root.glob('strict-*.json'))+[root/'smoke-sync-1024.json', root/'sharded-sync.json']
files += [root/'review/memory-sync.json', root/'review/memory-pipeline.json']
for path in files:
    data = json.loads(path.read_text())
    count = sum(len(p['ids']) for p in data['phases'])
    assert data['stats']['tensor_read_bytes'] == {n: b*count for n, b in logical.items()}
    assert data['stats']['gemv_weight_elements'] == count*linear
    assert data['stats']['inference_read_bytes'] == count*per_token_bytes
    assert data['unique_active_bf16_bytes'] == payload
    checks[str(path)] = True
old_source = Path('checkpoints/smollm2-135m-93efa2f/model.safetensors')
assert sha(old_source) == '80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1'
report = dict(original_sha256=expected, unique_active_bf16_bytes=payload,
    tensor_count=218, all_original_and_sharded_bf16_bytes_equal=True,
    original_and_sharded_tensor_sha256=original_tensor_hashes,
    independently_derived_gemv_elements_per_token=linear,
    independently_derived_logical_bytes_per_token=per_token_bytes,
    exact_counter_checks=checks, previous_135m_sha256_unchanged=True)
with (root/'review/integrity-and-coverage.json').open('x') as f:
    json.dump(report, f, indent=2); f.write('\n')
print('Original/shard hashes and independently derived actual coverage PASS')

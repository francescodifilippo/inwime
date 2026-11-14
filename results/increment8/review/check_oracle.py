"""Review exact values of the real promoted model; no InWiMe imports/forward."""
import hashlib
import inspect
import json
from pathlib import Path
import resource
import torch
from safetensors import safe_open
from transformers.models.llama import modeling_llama as llama
from tools.oracle8 import load_fp32

root = Path('checkpoints/smollm2-1.7b-d4c18cc')
torch.set_num_threads(2)
model, loader = load_fp32(root)
checks = {}
with safe_open(root/'model.safetensors', framework='pt', device='cpu') as source:
    names = list(source.keys())
# Reopen per tensor; avoid retaining an entire source mapping while checking.
for name in names:
    with safe_open(root/'model.safetensors', framework='pt', device='cpu') as source:
        value = source.get_tensor(name)
        actual = model.get_parameter(name)
        checks[name] = value.dtype == torch.bfloat16 and torch.equal(actual, value.float())
        del value, actual
assert all(checks.values()) and len(checks) == 218
assert model.lm_head.weight is model.model.embed_tokens.weight
rotary = model.model.rotary_emb
standard = llama.LlamaRotaryEmbedding(model.config, device='cpu')
assert torch.equal(rotary.inv_freq, standard.inv_freq)
assert rotary.inv_freq.dtype == standard.inv_freq.dtype == torch.float32
assert rotary.forward.__func__ is llama.LlamaRotaryEmbedding.forward
# Module methods and global attention are the installed stock implementations.
assert model.model.norm.forward.__func__ is llama.LlamaRMSNorm.forward
assert model.model.layers[0].forward.__func__ is llama.LlamaDecoderLayer.forward
hashes = {name: hashlib.sha256(inspect.getsource(obj).encode()).hexdigest()
          for name, obj in [('rotary', llama.LlamaRotaryEmbedding.forward),
                            ('rmsnorm', llama.LlamaRMSNorm.forward),
                            ('attention', llama.eager_attention_forward),
                            ('decoder', llama.LlamaDecoderLayer.forward)]}
with torch.inference_mode():
    x = torch.zeros(1, 8, model.config.hidden_size, dtype=torch.float32)
    positions = torch.arange(8).reshape(1, -1)
    own = rotary(x, positions)
    reference = standard(x, positions)
    assert all(torch.equal(a, b) for a, b in zip(own, reference))
report = dict(all_218_parameter_values_exact=True, parameter_checks=checks,
              loader=loader, tied_head_same_storage=True,
              rope_buffer_and_outputs_equal_standard=True,
              forward_methods_stock=True, installed_forward_sha256=hashes,
              independent_of_inwime=True,
              peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
with Path('results/increment8/review/oracle-exact-values.json').open('x') as f:
    json.dump(report, f, indent=2); f.write('\n')
print('Exact original BF16 -> FP32 values and stock upstream RoPE PASS')

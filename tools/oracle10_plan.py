"""Explicit peak-memory plan for tools/oracle10.py, computed from safetensors headers only.

Stdlib only (runs in the offline CI). The oracle refuses to start when the plan
exceeds the declared budget: there is no silent fallback to a smaller coverage,
a coarser dtype or a different head mode.

Components (bytes; s = itemsize of the oracle dtype, 4 or 8):
  overhead     interpreter + torch + transformers + meta model (declared estimate:
               768 MiB, about 2.3x the 342 MB measured for the 7B on 2026-11-18)
  layer        largest decoder layer promoted to s, plus its largest tensor in
               BF16 alive during promotion (one module resident at a time)
  head         blocked: block_rows*width*(2+s) + T*block_rows*s (F.linear output)
               resident: vocab*width*s + block_rows*width*2 (FP32 only by policy)
  embedding    T*width*(2+s): rows gathered from the shard, never the matrix
  outputs      T*vocab*s logits in the model + FP64 copies of logits, residuals and
               K/V, plus the DynamicCache in s
  activations  conservative: 16*T*max(width, intermediate)*s + heads*T*T*s*4
  mapped       BF16 source pages touched through safetensors' mmap while one module
               is read: max(largest layer, full head) in BF16. File-backed and
               reclaimable, but counted in RSS and charged to the cgroup as page cache.

planned_peak_bytes (anonymous) is checked against the explicit budget;
planned_rss_upper_bytes = planned_peak_bytes + mapped must fit the scope MemoryMax.
Calibration (2026-11-18, 7B component runs of tools/equivalence10.py): measured
peak RSS 3.78 GB (FP32 resident head) and 1.90 GB (FP64 blocked head); the first
planner revision omitted the mapped term and under-estimated RSS by 10-11 %.
"""
import json
from pathlib import Path
import struct

MiB, GiB = 2**20, 2**30
ITEMSIZE = {'float32': 4, 'float64': 8}
DEFAULT_OVERHEAD = 768*MiB


def read_headers(directory):
    """name -> (shape, nbytes) from every shard header (stdlib, header bytes only)."""
    directory = Path(directory)
    index = directory/'model.safetensors.index.json'
    shards = sorted(set(json.loads(index.read_text())['weight_map'].values())) if index.exists() else ['model.safetensors']
    out = {}
    for shard in shards:
        with (directory/shard).open('rb') as f:
            length, = struct.unpack('<Q', f.read(8))
            if not 2 <= length <= 64*MiB:
                raise ValueError('implausible safetensors header length')
            header = json.loads(f.read(length))
        header.pop('__metadata__', None)
        for name, d in header.items():
            if d['dtype'] != 'BF16':
                raise ValueError('original tensor is not BF16: '+name)
            begin, end = d['data_offsets']
            out[name] = (tuple(d['shape']), end-begin)
    return out


def plan(directory, config, dtype, tokens, head_mode='blocked', block_rows=16384, overhead=DEFAULT_OVERHEAD):
    if dtype not in ITEMSIZE:
        raise ValueError('unsupported oracle dtype')
    if head_mode not in ('blocked', 'resident'):
        raise ValueError('unknown head mode')
    if head_mode == 'resident' and dtype != 'float32':
        raise ValueError('a resident FP64 head is excluded by policy (C5)')
    s = ITEMSIZE[dtype]
    headers = read_headers(directory)
    width, vocab = config['hidden_size'], config['vocab_size']
    inter, layers = config['intermediate_size'], config['num_hidden_layers']
    heads, kvh = config['num_attention_heads'], config['num_key_value_heads']
    kvw = kvh*width//heads
    if type(block_rows) is not int or not 1 <= block_rows:
        raise ValueError('block_rows must be a positive integer')
    block_rows = min(block_rows, vocab)
    layer_peak = layer_bf16 = 0
    for layer in range(layers):
        prefix = f'model.layers.{layer}.'
        members = [v for k, v in headers.items() if k.startswith(prefix)]
        promoted = sum(n//2*s for _, n in members)
        layer_peak = max(layer_peak, promoted + max(n for _, n in members))
        layer_bf16 = max(layer_bf16, sum(n for _, n in members))
    if headers['lm_head.weight'][0] != (vocab, width) or headers['model.embed_tokens.weight'][0] != (vocab, width):
        raise ValueError('unexpected embedding/head shape')
    T = tokens
    if head_mode == 'blocked':
        head = block_rows*width*(2+s) + T*block_rows*s
    else:
        head = vocab*width*s + block_rows*width*2
    components = dict(
        overhead=overhead, layer=layer_peak, head=head, embedding=T*width*(2+s),
        outputs=T*vocab*s + 8*(T*vocab + layers*T*width + layers*2*T*kvw) + layers*2*T*kvw*s,
        activations=16*T*max(width, inter)*s + heads*T*T*s*4)
    peak = components['overhead'] + max(components['layer'], components['head']) + sum(
        components[k] for k in ('embedding', 'outputs', 'activations'))
    mapped = max(layer_bf16, vocab*width*2)
    largest_matrix_promoted = max(n//2*s for k, (shape, n) in headers.items() if k.startswith('model.layers.'))
    return dict(dtype=dtype, head_mode=head_mode, block_rows=block_rows, tokens=T, components=components,
                planned_peak_bytes=peak, mapped_source_bytes=mapped, planned_rss_upper_bytes=peak+mapped, full_head_in_dtype_bytes=vocab*width*s,
                full_embedding_in_dtype_bytes=vocab*width*s, largest_layer_matrix_in_dtype_bytes=largest_matrix_promoted,
                note='estimate; measured peak RSS is reported next to it')


def check_budget(p, budget, scope=None):
    if type(budget) is not int or budget <= 0:
        raise ValueError('explicit positive memory budget required')
    if p['planned_peak_bytes'] > budget:
        raise MemoryError(f"planned oracle peak {p['planned_peak_bytes']} B exceeds the budget {budget} B; "
                          'refusing to start (no fallback)')
    if scope is not None and p['planned_rss_upper_bytes'] > scope:
        raise MemoryError(f"planned RSS upper bound {p['planned_rss_upper_bytes']} B exceeds the scope "
                          f'MemoryMax {scope} B; refusing to start (no fallback)')
    return dict(budget_bytes=budget, scope_memory_max_bytes=scope, planned_peak_bytes=p['planned_peak_bytes'],
                planned_rss_upper_bytes=p['planned_rss_upper_bytes'], headroom_bytes=budget-p['planned_peak_bytes'])

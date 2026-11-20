"""Bounded independent Qwen2 oracle for increment 10 (stock Transformers, explicit memory plan).

Differences from tools/oracle9.py (which stays unchanged):
  * the embedding matrix is never materialized: the rows of the requested token
    IDs are read from the original shard (safe_open().get_slice) and promoted.
    A gather has no arithmetic, so it is exact by construction;
  * the LM head is never materialized in FP64. 'blocked' mode reads block_rows
    rows at a time, promotes them and writes F.linear(hidden, block) into a
    preallocated logits tensor; 'resident' (FP32 only) promotes the head block by
    block into one preallocated FP32 matrix and calls the same F.linear as
    nn.Linear (bit-identical to stock by construction, 3.11 GB for the 14B);
  * decoder layers and the final norm stream one module at a time as in oracle9;
  * an explicit memory budget is checked against tools.oracle10_plan before any
    tensor is read. Exceeding it is a refusal, never a fallback.

Independence: no InWiMe import (enforced by a static test); the only input taken
from the runtime is the token sequence. Attention, RoPE, RMSNorm, SwiGLU, the
causal mask and DynamicCache are the unmodified upstream modules. full64 reuses
the increment-7/9 precision profile (tools.oracle_precision9) unchanged.

Every full logits row (all positions x full vocabulary), every decoder residual
and every K/V entry is produced; the output layout equals oracle9 ('full.logits',
'full.hidden', 'full.kv') so captures can be compared directly.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
from safetensors import safe_open
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

from .oracle_precision9 import precision_profile
from . import oracle10_plan

CONTRACTS = {'stock_fp32': (torch.float32, 'stock'), 'stock_fp64': (torch.float64, 'stock'),
             'full64': (torch.float64, 'full64')}
# Preregistered head mode per contract (protocol.numerical_contract.oracle10). A blocked
# F.linear is not bit-identical to the full one in general (BLAS tail kernels differ:
# measured up to 1.9e-6 in FP32 and 4.4e-15 in FP64 on the fixture), so stock_fp32 keeps
# the full FP32 head (bit-identical by construction) and the FP64 contracts use blocks.
DEFAULT_HEAD_MODE = {'stock_fp32': 'resident', 'stock_fp64': 'blocked', 'full64': 'blocked'}
EMBED, HEAD = 'model.embed_tokens.weight', 'lm_head.weight'


def weight_map(directory):
    index = Path(directory)/'model.safetensors.index.json'
    if index.exists():
        return json.loads(index.read_text())['weight_map']
    with safe_open(Path(directory)/'model.safetensors', framework='pt', device='cpu') as f:
        return {name: 'model.safetensors' for name in f.keys()}


class Counters:
    def __init__(self):
        self.loads, self.source_bytes, self.largest_module_bytes = {}, 0, 0
        self.embedding_rows, self.head_blocks, self.head_rows, self.largest_head_block_bytes = 0, 0, 0, 0


class RowGatherEmbedding(torch.nn.Module):
    def __init__(self, path, dtype, vocab, width, counters):
        super().__init__()
        self.path, self.dtype, self.vocab, self.width, self.counters = path, dtype, vocab, width, counters

    def forward(self, input_ids):
        ids = input_ids.reshape(-1).tolist()
        if any(type(t) is not int or not 0 <= t < self.vocab for t in ids):
            raise ValueError('token id outside the vocabulary')
        with safe_open(self.path, framework='pt', device='cpu') as f:
            rows = f.get_slice(EMBED)
            gathered = torch.cat([rows[t:t+1] for t in ids])
        if gathered.dtype != torch.bfloat16 or tuple(gathered.shape) != (len(ids), self.width):
            raise ValueError('embedding rows are not BF16 rows of the expected width')
        self.counters.embedding_rows += len(ids)
        self.counters.source_bytes += gathered.numel()*2
        self.counters.loads[EMBED] = self.counters.loads.get(EMBED, 0)+1
        return gathered.to(self.dtype).reshape(*input_ids.shape, self.width)


class BlockedHead(torch.nn.Module):
    def __init__(self, path, dtype, vocab, width, block_rows, mode, counters):
        super().__init__()
        if mode == 'resident' and dtype != torch.float32:
            raise ValueError('resident head is FP32 only')
        self.path, self.dtype, self.vocab, self.width = path, dtype, vocab, width
        self.block_rows, self.mode, self.counters = block_rows, mode, counters

    def forward(self, hidden):
        c = self.counters
        with safe_open(self.path, framework='pt', device='cpu') as f:
            rows = f.get_slice(HEAD)
            if tuple(rows.get_shape()) != (self.vocab, self.width):
                raise ValueError('unexpected head shape')
            if self.mode == 'resident':
                weight = torch.empty(self.vocab, self.width, dtype=self.dtype)
                for start in range(0, self.vocab, self.block_rows):
                    block = rows[start:start+self.block_rows]
                    if block.dtype != torch.bfloat16:
                        raise ValueError('head is not BF16')
                    weight[start:start+block.shape[0]] = block.to(self.dtype)
                    c.head_blocks += 1; c.head_rows += block.shape[0]; c.source_bytes += block.numel()*2
                c.largest_head_block_bytes = max(c.largest_head_block_bytes, weight.numel()*weight.element_size())
                out = torch.nn.functional.linear(hidden, weight)
                del weight
            else:
                out = torch.empty(*hidden.shape[:-1], self.vocab, dtype=self.dtype)
                for start in range(0, self.vocab, self.block_rows):
                    block = rows[start:start+self.block_rows]
                    if block.dtype != torch.bfloat16:
                        raise ValueError('head is not BF16')
                    promoted = block.to(self.dtype)
                    out[..., start:start+promoted.shape[0]] = torch.nn.functional.linear(hidden, promoted)
                    c.head_blocks += 1; c.head_rows += promoted.shape[0]; c.source_bytes += block.numel()*2
                    c.largest_head_block_bytes = max(c.largest_head_block_bytes,
                                                     promoted.numel()*(promoted.element_size()+2))
                    del block, promoted
        c.loads[HEAD] = c.loads.get(HEAD, 0)+1
        return out


class LayerStreamer:
    """Decoder layers and final norm: one top-level module materialized at a time (as oracle9)."""

    def __init__(self, model, directory, wmap, dtype, counters):
        self.model, self.directory, self.wmap, self.dtype, self.c = model, Path(directory), wmap, dtype, counters
        self.handles = []
        for unit in [*model.model.layers, model.model.norm]:
            prefix = next(n for n, m in model.named_modules() if m is unit)
            names = sorted(n for n in wmap if n.startswith(prefix+'.'))
            self.handles.append(unit.register_forward_pre_hook(self._loader(names)))
            self.handles.append(unit.register_forward_hook(self._releaser(names)))

    def _loader(self, names):
        def hook(module, args):
            groups = {}
            for name in names:
                groups.setdefault(self.wmap[name], []).append(name)
            size = 0
            for shard, group in groups.items():
                with safe_open(self.directory/shard, framework='pt', device='cpu') as f:
                    for name in group:
                        original = f.get_tensor(name)
                        if original.dtype != torch.bfloat16:
                            raise ValueError('original tensor is not BF16: '+name)
                        parent, field = name.rsplit('.', 1)
                        owner = self.model.get_submodule(parent)
                        if tuple(original.shape) != tuple(getattr(owner, field).shape):
                            raise ValueError('shape mismatch: '+name)
                        promoted = original.to(dtype=self.dtype)
                        setattr(owner, field, torch.nn.Parameter(promoted, requires_grad=False))
                        self.c.loads[name] = self.c.loads.get(name, 0)+1
                        self.c.source_bytes += original.numel()*2
                        size += promoted.numel()*promoted.element_size()
                        del original
            self.c.largest_module_bytes = max(self.c.largest_module_bytes, size)
        return hook

    def _releaser(self, names):
        def hook(module, args, output):
            for name in names:
                parent, field = name.rsplit('.', 1)
                owner = self.model.get_submodule(parent)
                old = getattr(owner, field)
                setattr(owner, field, torch.nn.Parameter(
                    torch.empty(old.shape, dtype=self.dtype, device='meta'), requires_grad=False))
        return hook


def build(directory, dtype, head_mode, block_rows, counters):
    config = AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    if config.model_type != 'qwen2' or config.tie_word_embeddings:
        raise ValueError('this oracle is for untied Qwen2ForCausalLM checkpoints')
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, attn_implementation='eager',
                                                trust_remote_code=False, dtype=dtype)
    rotary = model.model.rotary_emb
    model.model.rotary_emb = type(rotary)(config=config, device='cpu')
    wmap = weight_map(directory)
    directory = Path(directory)
    model.model.embed_tokens = RowGatherEmbedding(directory/wmap[EMBED], dtype, config.vocab_size,
                                                  config.hidden_size, counters)
    model.lm_head = BlockedHead(directory/wmap[HEAD], dtype, config.vocab_size, config.hidden_size,
                                block_rows, head_mode, counters)
    if set(model.state_dict()) | {EMBED, HEAD} != set(wmap):
        raise ValueError('upstream parameter set differs from the checkpoint tensor set')
    if any(b.is_meta for b in model.buffers()):
        raise ValueError('unmaterialized buffer')
    return model.eval(), config, wmap


def run_oracle(directory, contract, sequence, chunks, *, head_mode=None, block_rows=16384,
               memory_budget=None, scope_memory_max=None, overhead=oracle10_plan.DEFAULT_OVERHEAD):
    dtype, profile = CONTRACTS[contract]
    head_mode = DEFAULT_HEAD_MODE[contract] if head_mode is None else head_mode
    if sum(chunks) != len(sequence) or any(type(c) is not int or c <= 0 for c in chunks):
        raise ValueError('chunks must partition the sequence')
    raw_config = json.loads((Path(directory)/'config.json').read_text())
    memory = oracle10_plan.plan(directory, raw_config, str(dtype).removeprefix('torch.'), len(sequence),
                                head_mode, block_rows, overhead)
    budget = oracle10_plan.check_budget(memory, memory_budget, scope_memory_max)
    counters = Counters()
    model, config, wmap = build(directory, dtype, head_mode, block_rows, counters)
    streamer = LayerStreamer(model, directory, wmap, dtype, counters)
    residuals = []
    capture = [layer.register_forward_hook(
        lambda module, inputs, output: residuals.append(
            (output[0] if isinstance(output, tuple) else output).detach()[0].clone()))
        for layer in model.model.layers]
    logits, hidden, calls, cache, start = [], [], [], None, 0
    width = config.num_key_value_heads*config.hidden_size//config.num_attention_heads
    began = time.monotonic()
    try:
        with torch.inference_mode(), torch.autocast('cpu', enabled=False), precision_profile(model, profile):
            for size in chunks:
                residuals.clear()
                ids = sequence[start:start+size]
                t = time.monotonic()
                out = model(torch.tensor([ids]), past_key_values=cache, use_cache=True, return_dict=True)
                cache = out.past_key_values
                if tuple(out.logits.shape) != (1, size, config.vocab_size):
                    raise ValueError('incomplete logits: every position and the full vocabulary are required')
                logits.append(out.logits[0].double().numpy().copy())
                hidden.append(np.stack([r.double().numpy() for r in residuals]))
                calls.append(dict(tokens=ids, wall_s=time.monotonic()-t, argmax=out.logits[0].argmax(-1).tolist()))
                start += size
            kv = np.stack([np.stack([z[0].transpose(0, 1).contiguous().double().numpy().reshape(-1, width)
                                     for z in (layer.keys, layer.values)]) for layer in cache.layers])
    finally:
        for h in capture+streamer.handles:
            h.remove()
    if set(counters.loads) != set(wmap) or set(counters.loads.values()) != {len(chunks)}:
        raise ValueError('not every original tensor was read once per forward call')
    if counters.embedding_rows != len(sequence) or counters.head_rows != len(chunks)*config.vocab_size:
        raise ValueError('embedding rows or head rows not covered exactly')
    arrays = dict(logits=np.concatenate(logits), hidden=np.concatenate(hidden, axis=1), kv=kv)
    return arrays, dict(
        contract=contract, dtype=str(dtype).removeprefix('torch.'), precision_profile=profile,
        attention_implementation=model.config._attn_implementation, chunks=list(chunks), calls=calls,
        head_mode=head_mode, head_mode_is_preregistered=head_mode == DEFAULT_HEAD_MODE[contract],
        block_rows=block_rows, memory_plan=memory, memory_budget=budget,
        oracle_wall_s=time.monotonic()-began,
        loader=dict(one_top_level_module_resident=True, embedding_materialized=False,
                    fp64_head_materialized=False, largest_module_bytes=counters.largest_module_bytes,
                    largest_head_block_bytes=counters.largest_head_block_bytes,
                    embedding_rows_read=counters.embedding_rows, head_blocks=counters.head_blocks,
                    head_rows_read=counters.head_rows, source_bf16_bytes_read=counters.source_bytes,
                    tensors=len(counters.loads), loads_per_tensor=len(chunks)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--contract', choices=sorted(CONTRACTS), required=True)
    p.add_argument('--sequence', required=True, help='comma-separated token IDs (prompt + continuation)')
    p.add_argument('--prompt-length', type=int, required=True)
    p.add_argument('--chunks', required=True, help='e.g. 5,1,1,1')
    p.add_argument('--head-mode', choices=('blocked', 'resident'), default=None,
                   help='default: preregistered mode of the contract (DEFAULT_HEAD_MODE)')
    p.add_argument('--block-rows', type=int, default=16384)
    p.add_argument('--memory-budget', type=int, required=True, help='bytes of anonymous memory; refusal above the plan')
    p.add_argument('--scope-memory-max', type=int, required=True,
                   help='MemoryMax of the enclosing scope; refusal if the planned RSS upper bound exceeds it')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.with_suffix('.json').exists():
        raise FileExistsError(a.output)
    sequence = [int(x) for x in a.sequence.split(',')]
    chunks = [int(x) for x in a.chunks.split(',')]
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    began = time.monotonic()
    arrays, meta = run_oracle(a.checkpoint, a.contract, sequence, chunks, head_mode=a.head_mode,
                              block_rows=a.block_rows, memory_budget=a.memory_budget,
                              scope_memory_max=a.scope_memory_max)
    argmax = arrays['logits'].argmax(-1).tolist()
    greedy = argmax[a.prompt_length-1:len(sequence)-1] == sequence[a.prompt_length:]
    with a.output.open('xb') as f:
        np.savez_compressed(f, **{f'full.{k}': v for k, v in arrays.items()})
    with a.output.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    provenance = a.checkpoint/'provenance.json'
    meta |= dict(versions=dict(torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__),
                 revision=json.loads(provenance.read_text()).get('revision') if provenance.exists() else None,
                 sequence=sequence, prompt_length=a.prompt_length, argmax=argmax,
                 upstream_greedy_matches_sequence=greedy, shapes={k: list(v.shape) for k, v in arrays.items()},
                 elapsed_s=time.monotonic()-began,
                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 capture_sha256=digest, threads=torch.get_num_threads())
    with a.output.with_suffix('.json').open('x') as f:
        json.dump(meta, f, indent=2); f.write('\n')
    print(json.dumps({k: meta[k] for k in ('contract', 'head_mode', 'oracle_wall_s', 'peak_rss_bytes',
                                           'upstream_greedy_matches_sequence')}
                     | dict(planned_peak=meta['memory_plan']['planned_peak_bytes']), indent=1))


if __name__ == '__main__':
    main()

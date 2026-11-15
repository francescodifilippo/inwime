"""Independent bounded Qwen2 oracle: stock Transformers modules, one module resident at a time.

No InWiMe import and no runtime-provided intermediates (only the token sequence).
The model is built on the meta device; forward pre-hooks materialize the
parameters of exactly one top-level module (embedding, one decoder layer, final
norm, head) by reading the original BF16 tensors and promoting them in RAM; forward
hooks release them again. The forward itself is the unmodified upstream
Qwen2ForCausalLM.forward (mask, rotary, DynamicCache, GQA, eager attention, head).
Peak memory is about one module in the oracle dtype, not the whole model.
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
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .oracle_precision9 import precision_profile

CONTRACTS = {'stock_fp32': (torch.float32, 'stock'), 'stock_fp64': (torch.float64, 'stock'),
             'full64': (torch.float64, 'full64')}


class Streamer:
    def __init__(self, model, directory, dtype):
        self.model, self.directory, self.dtype = model, Path(directory), dtype
        index = self.directory/'model.safetensors.index.json'
        if index.exists():
            self.weight_map = json.loads(index.read_text())['weight_map']
        else:
            with safe_open(self.directory/'model.safetensors', framework='pt', device='cpu') as f:
                self.weight_map = {name: 'model.safetensors' for name in f.keys()}
        expected = set(model.state_dict())
        if expected != set(self.weight_map):
            raise ValueError('upstream parameter set differs from the checkpoint tensor set')
        self.loads, self.source_bytes, self.largest_module_bytes, self.resident = {}, 0, 0, 0
        self.handles = []
        units = [model.model.embed_tokens, *model.model.layers, model.model.norm, model.lm_head]
        for unit in units:
            prefix = next(n for n, m in model.named_modules() if m is unit)
            names = sorted(n for n in self.weight_map if n.startswith(prefix+'.'))
            self.handles.append(unit.register_forward_pre_hook(self._loader(names)))
            self.handles.append(unit.register_forward_hook(self._releaser(names)))

    def _loader(self, names):
        def hook(module, args):
            by_shard = {}
            for name in names:
                by_shard.setdefault(self.weight_map[name], []).append(name)
            size = 0
            for shard, group in by_shard.items():
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
                        self.loads[name] = self.loads.get(name, 0)+1
                        self.source_bytes += original.numel()*2
                        size += promoted.numel()*promoted.element_size()
                        del original
            self.resident = size
            self.largest_module_bytes = max(self.largest_module_bytes, size)
        return hook

    def _releaser(self, names):
        def hook(module, args, output):
            for name in names:
                parent, field = name.rsplit('.', 1)
                owner = self.model.get_submodule(parent)
                old = getattr(owner, field)
                setattr(owner, field, torch.nn.Parameter(
                    torch.empty(old.shape, dtype=self.dtype, device='meta'), requires_grad=False))
            self.resident = 0
        return hook


def build(directory, dtype):
    config = AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    if config.model_type != 'qwen2' or config.tie_word_embeddings:
        raise ValueError('this oracle is for untied Qwen2ForCausalLM checkpoints')
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, attn_implementation='eager',
                                                trust_remote_code=False, dtype=dtype)
    # Non-persistent rotary buffer: materialize with the upstream constructor (stock FP32).
    rotary = model.model.rotary_emb
    model.model.rotary_emb = type(rotary)(config=config, device='cpu')
    if any(b.is_meta for b in model.buffers()):
        raise ValueError('unmaterialized buffer')
    return model.eval(), config


def run_oracle(directory, contract, sequence, chunks):
    dtype, profile = CONTRACTS[contract]
    model, config = build(directory, dtype)
    streamer = Streamer(model, directory, dtype)
    residuals = []
    capture = [layer.register_forward_hook(
        lambda module, inputs, output: residuals.append(
            (output[0] if isinstance(output, tuple) else output).detach()[0].clone()))
        for layer in model.model.layers]
    # Releaser hooks (registered first) only swap parameters back to meta; the
    # layer output tensor captured here is already computed and independent of them.
    logits, hidden, calls = [], [], []
    width = config.num_key_value_heads*config.hidden_size//config.num_attention_heads
    cache = None
    began = time.monotonic()
    try:
        with torch.inference_mode(), torch.autocast('cpu', enabled=False), precision_profile(model, profile):
            start = 0
            for size in chunks:
                residuals.clear()
                ids = sequence[start:start+size]
                t = time.monotonic()
                out = model(torch.tensor([ids]), past_key_values=cache, use_cache=True, return_dict=True)
                cache = out.past_key_values
                logits.append(out.logits[0].double().numpy().copy())
                hidden.append(np.stack([r.double().numpy() for r in residuals]))
                calls.append(dict(tokens=ids, wall_s=time.monotonic()-t,
                                  argmax=out.logits[0].argmax(-1).tolist()))
                start += size
            kv = np.stack([np.stack([z[0].transpose(0, 1).contiguous().double().numpy().reshape(-1, width)
                                     for z in (layer.keys, layer.values)]) for layer in cache.layers])
    finally:
        for h in capture+streamer.handles:
            h.remove()
    if start != len(sequence):
        raise ValueError('chunks do not cover the sequence')
    counts = set(streamer.loads.values())
    if set(streamer.loads) != set(streamer.weight_map) or counts != {len(chunks)}:
        raise ValueError('not every original tensor was loaded once per forward call')
    return dict(logits=np.concatenate(logits), hidden=np.concatenate(hidden, axis=1), kv=kv), dict(
        contract=contract, dtype=str(dtype).removeprefix('torch.'), precision_profile=profile,
        attention_implementation=model.config._attn_implementation, chunks=chunks, calls=calls,
        oracle_wall_s=time.monotonic()-began, loader=dict(
            one_top_level_module_resident=True, largest_module_bytes=streamer.largest_module_bytes,
            source_bf16_bytes_read=streamer.source_bytes, tensors=len(streamer.loads),
            loads_per_tensor=len(chunks)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--contract', choices=sorted(CONTRACTS), required=True)
    p.add_argument('--sequence', required=True, help='comma-separated token IDs (prompt + continuation)')
    p.add_argument('--prompt-length', type=int, required=True)
    p.add_argument('--chunks', default=None, help='e.g. 5,3; default one full-sequence call')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.with_suffix('.json').exists():
        raise FileExistsError(a.output)
    sequence = [int(x) for x in a.sequence.split(',')]
    chunks = [int(x) for x in a.chunks.split(',')] if a.chunks else [len(sequence)]
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    began = time.monotonic()
    arrays, meta = run_oracle(a.checkpoint, a.contract, sequence, chunks)
    tokenizer = AutoTokenizer.from_pretrained(a.checkpoint, local_files_only=True, trust_remote_code=False)
    texts = ['The capital of France is', 'Ciao, mondo! é\n42', '<|endoftext|>Hello', '<|im_start|>user\nhi']
    argmax = arrays['logits'].argmax(-1).tolist()
    # Causal model: argmax at position i of the full sequence is the upstream greedy
    # choice after prefix[:i+1]. Equality below = independent greedy continuation.
    greedy = argmax[a.prompt_length-1:len(sequence)-1] == sequence[a.prompt_length:]
    with a.output.open('xb') as f:
        np.savez_compressed(f, **{f'full.{k}': v for k, v in arrays.items()})
    with a.output.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    meta |= dict(versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                               numpy=np.__version__),
                 revision=json.loads((a.checkpoint/'provenance.json').read_text())['revision'],
                 sequence=sequence, prompt_length=a.prompt_length, argmax=argmax,
                 upstream_greedy_matches_sequence=greedy,
                 tokenizer=[dict(text=t, ids=tokenizer.encode(t),
                                 decoded=tokenizer.decode(tokenizer.encode(t), skip_special_tokens=True))
                            for t in texts],
                 shapes={k: list(v.shape) for k, v in arrays.items()},
                 elapsed_s=time.monotonic()-began,
                 peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                 capture_sha256=digest, threads=torch.get_num_threads())
    with a.output.with_suffix('.json').open('x') as f:
        json.dump(meta, f, indent=2); f.write('\n')
    print(json.dumps({k: meta[k] for k in ('contract', 'oracle_wall_s', 'peak_rss_bytes',
                                           'upstream_greedy_matches_sequence', 'loader')}, indent=1))


if __name__ == '__main__':
    main()

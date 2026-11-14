"""Independent standard Transformers FP32 oracle with tensor-at-a-time promotion.

No InWiMe imports. No runtime-provided inputs/intermediates. Source BF16 mappings
are released after each FP32 tensor assignment; no resident BF16 model or FP32
disk copy. This loader is verification tooling, not a new inference backend.
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


def load_fp32(directory):
    config = AutoConfig.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, attn_implementation='eager',
                                                trust_remote_code=False, dtype=torch.float32)
    # Nonpersistent rotary frequencies are not in state_dict. Materialize them
    # with the installed upstream constructor, retaining its stock FP32 policy.
    rotary = model.model.rotary_emb
    model.model.rotary_emb = type(rotary)(config=config, device='cpu')
    index = directory/'model.safetensors.index.json'
    if index.exists():
        weight_map = json.loads(index.read_text())['weight_map']
    else:
        with safe_open(directory/'model.safetensors', framework='pt', device='cpu') as f:
            weight_map = {name: 'model.safetensors' for name in f.keys()}
    expected = set(model.state_dict())
    # This model ties the head to embedding; original stores only one copy.
    if not config.tie_word_embeddings or expected != set(weight_map) | {'lm_head.weight'}:
        raise ValueError('unexpected parameter set or unsupported tying in verification loader')
    original_bytes = largest = 0
    for name, shard in weight_map.items():
        with safe_open(directory/shard, framework='pt', device='cpu') as f:
            original = f.get_tensor(name)
            if original.dtype != torch.bfloat16:
                raise ValueError('original checkpoint is not entirely BF16')
            parent, field = name.rsplit('.', 1)
            module = model.get_submodule(parent)
            if tuple(original.shape) != tuple(getattr(module, field).shape):
                raise ValueError('parameter shape mismatch')
            size = original.numel()*2
            original_bytes += size
            largest = max(largest, size)
            promoted = original.to(dtype=torch.float32)
            setattr(module, field, torch.nn.Parameter(promoted, requires_grad=False))
            del original, promoted
        # No view of this mapping escapes the context. Only FP32 stays resident.
    model.tie_weights()
    if any(p.is_meta or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('unloaded or non-FP32 parameters')
    if any(b.is_meta for b in model.buffers()):
        raise ValueError('unmaterialized nonpersistent model buffer')
    if model.lm_head.weight is not model.model.embed_tokens.weight:
        raise ValueError('head is not physically shared')
    if sum(p.numel()*2 for p in model.parameters()) != original_bytes:
        raise ValueError('unique parameter count mismatch')
    return model.eval(), dict(original_bf16_bytes=original_bytes,
                               resident_fp32_parameter_bytes=original_bytes*2,
                               maximum_source_tensor_bytes=largest,
                               one_source_mapping_at_a_time=True,
                               no_resident_bf16_model=True, tied_storage=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or a.output.with_suffix('.json').exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    began = time.monotonic()
    model, loader = load_fp32(a.checkpoint)
    loader['load_s'] = time.monotonic()-began
    tokenizer = AutoTokenizer.from_pretrained(a.checkpoint, local_files_only=True, trust_remote_code=False)
    texts = ['The capital of France is', 'Ciao, mondo! é\n42', '<|endoftext|>Hello']
    tokens = tokenizer.encode(texts[0])
    payload, scenarios = {}, {}

    def capture(name, chunks):
        cache = None
        residuals = []
        handles = [layer.register_forward_hook(
            lambda module, inputs, output: residuals.append(
                (output[0] if isinstance(output, tuple) else output).detach().numpy().copy()))
            for layer in model.model.layers]
        records = []
        try:
            for i, chunk in enumerate(chunks):
                residuals.clear()
                result = model(torch.tensor([chunk]), past_key_values=cache,
                               use_cache=True, return_dict=True)
                cache = result.past_key_values
                prefix = f'{name}.{i}'
                payload[prefix+'.logits'] = result.logits[0].numpy().copy()
                payload[prefix+'.hidden'] = np.stack([x[0] for x in residuals])
                legacy = cache.to_legacy_cache()
                width = model.config.num_key_value_heads*model.config.hidden_size//model.config.num_attention_heads
                payload[prefix+'.kv'] = np.stack([np.stack([
                    z[0].transpose(0, 1).contiguous().numpy().reshape(-1, width)
                    for z in pair]) for pair in legacy])
                records.append(dict(tokens=chunk, greedy=result.logits[0].argmax(-1).tolist()))
        finally:
            for h in handles:
                h.remove()
        scenarios[name] = records

    with torch.inference_mode(), torch.autocast('cpu', enabled=False):
        # Independently obtain continuation with incremental upstream KV.
        sequence, continuation, cache = list(tokens), [], None
        chunk = tokens
        for _ in range(3):
            result = model(torch.tensor([chunk]), past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            chosen = int(result.logits[0, -1].argmax())
            continuation.append(chosen); sequence.append(chosen); chunk = [chosen]
        del result, cache
        capture('decode', [tokens]+[[t] for t in continuation])
        capture('extended', [sequence])
        capture('chunks', [tokens[:2], tokens[2:]])
    with a.output.open('xb') as f:
        np.savez_compressed(f, **payload)
    with a.output.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    meta = dict(contract='transformers_standard_float32', dtype='float32',
                precision_profile='stock', attention_implementation=model.config._attn_implementation,
                versions=dict(torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__),
                revision=json.loads((a.checkpoint/'provenance.json').read_text())['revision'],
                loader=loader, scenarios=scenarios, continuation=continuation,
                tokenizer=[dict(text=t, ids=tokenizer.encode(t),
                                decoded=tokenizer.decode(tokenizer.encode(t), skip_special_tokens=True)) for t in texts],
                elapsed_s=time.monotonic()-began,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                capture_sha256=digest,
                limitation='FP32 parity only for 1.7B; no full FP64 resident oracle under this WSL budget')
    with a.output.with_suffix('.json').open('x') as f:
        json.dump(meta, f, indent=2); f.write('\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()

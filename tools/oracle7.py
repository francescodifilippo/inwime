"""Independent upstream capture. No InWiMe imports. Missing inputs are errors."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from .oracle_precision7 import precision_profile, FloatingDtypeAudit, PROFILES


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--dtype', choices=['float64', 'float32'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--diagnostic-rms64', action='store_true', help='explicit non-stock RMSNorm intervention')
    p.add_argument('--precision-profile', choices=PROFILES, default='stock')
    args = p.parse_args()
    profile = 'rms64' if args.diagnostic_rms64 else args.precision_profile
    if args.diagnostic_rms64 and args.precision_profile != 'stock':
        p.error('legacy diagnostic flag cannot be combined with a precision profile')
    if profile != 'stock' and args.dtype != 'float64':
        p.error('precision interventions require --dtype float64')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    started = time.monotonic()
    # Load original BF16 first, then convert only the resident reference.
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, local_files_only=True,
        trust_remote_code=False, dtype=torch.bfloat16, attn_implementation='eager').eval()
    assert all(v.dtype == torch.bfloat16 for v in model.parameters())
    model = model.to(dtype=getattr(torch, args.dtype))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True,
                                             trust_remote_code=False)
    texts = ['The capital of France is', 'Once upon a time', 'Ciao, mondo! é\n42', '<|endoftext|>Hello']
    tokens = tokenizer.encode(texts[0])
    payload, scenarios = {}, {}

    def capture(name, chunks):
        cache = None
        residuals = []
        handles = [layer.register_forward_hook(
            lambda module, inputs, output: residuals.append((output[0] if isinstance(output, tuple) else output).detach().cpu().numpy().copy()))
            for layer in model.model.layers]
        used, steps = [], []
        try:
            with torch.inference_mode():
                for i, chunk in enumerate(chunks):
                    residuals.clear()
                    with audit:
                        result = model(torch.tensor([chunk]), past_key_values=cache, use_cache=True,
                                       output_hidden_states=True, return_dict=True)
                    cache = result.past_key_values
                    prefix = f'{name}.{i}'
                    payload[prefix+'.logits'] = result.logits[0].cpu().numpy()
                    payload[prefix+'.hidden'] = np.stack([x[0] for x in residuals])
                    # Cache layout [batch, KV heads, sequence, head dim] -> [sequence, KV width].
                    legacy = cache.to_legacy_cache() if hasattr(cache, 'to_legacy_cache') else cache
                    payload[prefix+'.kv'] = np.stack([np.stack([
                        z[0].transpose(0, 1).contiguous().cpu().numpy().reshape(-1, model.config.num_key_value_heads*model.config.hidden_size//model.config.num_attention_heads)
                        for z in pair]) for pair in legacy])
                    chosen = result.logits[0].argmax(-1).tolist()
                    steps.append(dict(tokens=chunk, greedy=chosen))
                    used.extend(chunk)
        finally:
            for handle in handles:
                handle.remove()
        scenarios[name] = steps

    with precision_profile(model, profile), torch.autocast('cpu', enabled=False):
        audit = FloatingDtypeAudit()
        with torch.inference_mode(), audit:
            model(torch.tensor([tokens]), use_cache=True)
        dtype_audit = audit.summary()
        if profile == 'full64' and dtype_audit['non_fp64_output_operations']:
            raise AssertionError('fully FP64 oracle still emits lower precision tensors')
        capture('single', [tokens[:1]])
        capture('full', [tokens])
        capture('chunks', [tokens[:2], tokens[2:]])
        # Greedy continuation computed by upstream independently, no EOS stop for parity.
        sequence = list(tokens)
        continuation = []
        with torch.inference_mode():
            for _ in range(3):
                token = int(model(torch.tensor([sequence])).logits[0, -1].argmax())
                continuation.append(token)
                sequence.append(token)
        capture('decode', [tokens]+[[t] for t in continuation])
        capture('extended', [sequence])
        capture('repeat', [tokens])
    dtype_audit = audit.summary()
    if profile == 'full64' and dtype_audit['non_fp64_output_operations']:
        raise AssertionError('fully FP64 capture emits lower precision tensors')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as f:
        np.savez_compressed(f, **payload)
    metadata = dict(precision_profile=profile, contract=('transformers_standard_'+args.dtype if profile == 'stock' else 'pytorch_pure_fp64' if profile == 'full64' else 'diagnostic_'+profile), dtype_audit=dtype_audit, attention_implementation=model.config._attn_implementation, diagnostic_rms64=args.diagnostic_rms64, dtype=args.dtype, versions=dict(torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__),
                    revision=json.loads((args.checkpoint/'provenance.json').read_text())['revision'],
                    tokenizer=[dict(text=t, ids=tokenizer.encode(t), decoded=tokenizer.decode(tokenizer.encode(t),skip_special_tokens=True)) for t in texts],
                    scenarios=scenarios, continuation=continuation,
                    elapsed_s=time.monotonic()-started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                    capture_sha256=hashlib.file_digest(args.output.open('rb'),'sha256').hexdigest())
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()

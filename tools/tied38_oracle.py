"""Oracle side of the #38 gate: stock Transformers on the same sequences, then the comparison.

Independence: no InWiMe import (`inwime`, `inwime_runtime`); the only inputs taken from the Motore
side are its JSON report and its captured arrays. For each case of `tools/tied38_motore.py`:

1. Rendering parity: Transformers' own `apply_chat_template` (add_generation_prompt) on the same
   messages must give the prompt ids the Motore used.
2. Numerical equivalence: `tools.check_oracle10.stock` (stock `from_pretrained`, eager attention,
   residual hooks and DynamicCache K/V; pinned and unchanged) recomputes the Motore's sequence
   with the same chunks, under two contracts:
     stock_fp32  FP32 weights and arithmetic: the Transformers oracle;
     full64      FP64 weights with the increment-7/9 FP64 precision profile: the FP64 reference.
   `tools.compare10.compare` checks every logit, residual and K/V entry (FULL coverage) at the
   thresholds preregistered in M10: fp32 atol 2e-3 rtol 2e-4, fp64 atol 2e-5 rtol 2e-6.
3. Greedy token parity: each contract generates up to the Motore's max_tokens from its own
   rendering, by argmax (the lowest id wins a tie, as torch.argmax and the M10 rule), stopping
   at an EOS of generation_config.json; the tokens must equal the Motore's.

The verdicts of the two contracts and of the parity are reported separately.

    .venv-oracle7/bin/python -m tools.tied38_oracle --motore DIR --output FILE
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import compare10
from .check_oracle10 import stock
from .oracle_precision9 import precision_profile

CONTRACTS = {'stock_fp32': (torch.float32, 'stock', 'fp32'), 'full64': (torch.float64, 'full64', 'fp64')}


def greedy(directory, dtype, profile, prompt, max_tokens, eos):
    model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True, dtype=dtype,
                                                 attn_implementation='eager').eval()
    generated, cache, step = [], None, prompt
    with torch.inference_mode(), precision_profile(model, profile):
        while len(generated) < max_tokens:
            out = model(torch.tensor([step]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            token = int(torch.argmax(out.logits[0, -1]))
            generated.append(token)
            if token in eos:
                break
            step = [token]
    return generated


def eos_ids(directory):
    value = json.loads((Path(directory)/'generation_config.json').read_text()).get('eos_token_id')
    return set(value if isinstance(value, list) else [value])


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--motore', type=Path, required=True, help='output directory of tools/tied38_motore.py')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    motore = json.loads((args.motore/'motore.json').read_text())
    directory = motore['snapshot_dir']
    config = json.loads((Path(directory)/'config.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    eos = eos_ids(directory)
    layers, width, vocab = config['num_hidden_layers'], config['hidden_size'], config['vocab_size']
    kv_width = config['num_key_value_heads']*width//config['num_attention_heads']
    cases = []
    for index, case in enumerate(motore['cases']):
        path = args.motore/case['arrays']
        if hashlib.sha256(path.read_bytes()).hexdigest() != case['arrays_sha256']:
            raise SystemExit('%s differs from the digest in motore.json' % path)
        with np.load(path) as saved:
            actual = {k: saved[k] for k in compare10.KINDS}
        prompt, completion = case['prompt_token_ids'], case['completion_token_ids']
        sequence = prompt+completion
        rendered = tokenizer.apply_chat_template(case['messages'], add_generation_prompt=True,
                                                 tokenize=True, return_dict=False)
        shapes = compare10.required_shapes(layers, len(sequence), width, vocab, kv_width)
        result = dict(rendering_parity=list(rendered) == prompt, contracts={})
        for contract, (dtype, profile, precision) in CONTRACTS.items():
            begun = time.monotonic()
            reference = stock(directory, dtype, profile, sequence, case['capture']['chunks'])
            numeric = compare10.compare(actual, reference, shapes, precision)
            tokens = greedy(directory, dtype, profile, list(rendered), motore['max_tokens'], eos)
            numeric['argmax_equal'] = bool((actual['logits'].argmax(-1) == reference['logits'].argmax(-1)).all())
            result['contracts'][contract] = dict(
                numerical=numeric, greedy_token_ids=tokens, token_parity=tokens == completion,
                seconds=time.monotonic()-begun)
            print('case %d %s: %s/%s, tokens %s' % (index, contract, numeric['coverage'], numeric['verdict'],
                                                    'equal' if tokens == completion else 'DIFFER'), flush=True)
            del reference
        cases.append(result)
    verdicts = {contract: dict(
        numerical='PASS' if all(c['contracts'][contract]['numerical']['verdict'] == 'PASS' for c in cases) else 'FAIL',
        token_parity='PASS' if all(c['contracts'][contract]['token_parity'] for c in cases) else 'FAIL')
        for contract in CONTRACTS}
    report = dict(schema='inwime-tied38-oracle-v1', model=motore['model'], cases=cases, verdicts=verdicts,
                  rendering_parity='PASS' if all(c['rendering_parity'] for c in cases) else 'FAIL',
                  thresholds={k: compare10.THRESHOLDS[v[2]] for k, v in CONTRACTS.items()},
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                                numpy=np.__version__, python=platform.python_version()),
                  threads=args.threads)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(verdicts=verdicts, rendering_parity=report['rendering_parity'])))


if __name__ == '__main__':
    main()

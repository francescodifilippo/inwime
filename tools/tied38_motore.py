"""Motore side of the #38 gate: Qwen2.5-0.5B-Instruct (tied head) served by InWiMe, captured.

For each short conversation of `PROMPTS` this tool:
1. asks the real Motore (`inwime.engine.Coordinatore` over the installed Checkpoint, as
   `inwime serve` builds it) for a greedy chat completion: the prompt is rendered with the
   Checkpoint's chat template, the tokens are chosen by the M10 greedy rule. The request sets
   temperature 0 and repetition_penalty 1: Qwen2.5-0.5B-Instruct's generation_config.json has
   repetition_penalty 1.1, which the Motore, as Transformers, would apply to greedy decoding too;
2. runs the same Percorso computazionale in a fresh session (the catalog the Motore builds for an
   installed tied Checkpoint, the same tiles and kernel) on the prompt followed by the
   completion, as one prefill of the prompt and one decode per generated token, and saves every
   logit (all positions x full vocabulary), every decoder residual and the whole KV in FP64.

The oracle side (`tools/tied38_oracle.py`, no InWiMe import) recomputes the same sequence with
Transformers and compares. Nothing here reads the oracle; nothing is downloaded.

    PYTHONPATH=. INWIME_CPU_KERNEL=build/libinwime_cpu.so .venv-product/bin/python \\
        -m tools.tied38_motore --output DIR
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np

from inwime import registry
from inwime.engine import CatalogQwen2Runtime, Coordinatore

REPO_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
MAX_TOKENS = 8
GREEDY = {'temperature': 0, 'repetition_penalty': 1.0}     # plain argmax, see step 1
# Short conversations: the template's default system prompt, an Italian one, an explicit system.
PROMPTS = (
    [{'role': 'user', 'content': 'What is the capital of Italy?'}],
    [{'role': 'user', 'content': 'Scrivi una frase sul mare.'}],
    [{'role': 'system', 'content': 'You are a concise assistant.'},
     {'role': 'user', 'content': 'Name three primary colors.'}],
)


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


def capture(catalog, math, prompt, completion):
    """FP64 logits, residuals and KV of prompt+completion: one prefill, then one decode per token."""
    sequence = prompt+completion
    started = time.monotonic()
    with CatalogQwen2Runtime(catalog, len(sequence), integrity='full', math_backend=math) as r:
        logits, hidden = [], []
        calls = [(r.prefill, prompt, len(prompt))] + [(r.decode, t, 1) for t in completion]
        for call, argument, size in calls:
            with call(argument) as out:
                logits.append(np.frombuffer(out.logits, dtype='<f8').reshape(size, -1).copy())
                hidden.append(np.stack([np.frombuffer(h, dtype='<f8').reshape(size, -1) for h in out.hidden]))
                kv = np.stack([np.stack([np.frombuffer(v, dtype='<f8').reshape(r.length, -1) for v in pair])
                               for pair in out.kv]).copy()
        stats = r.control.stats()
    arrays = dict(logits=np.concatenate(logits), hidden=np.concatenate(hidden, axis=1), kv=kv)
    return arrays, dict(seconds=time.monotonic()-started, chunks=[len(prompt)]+[1]*len(completion),
                        inference_read_bytes=stats['inference_read_bytes'],
                        integrity_read_bytes=stats['integrity_read_bytes'],
                        reserved_peak_bytes=stats['reserved_peak_bytes'])


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model_id = '%s@%s' % (REPO_ID, REVISION)
    manifest = registry.read_json(registry.manifest_path(REPO_ID, REVISION))
    if not registry.intact(manifest):
        raise SystemExit('%s is not a verified, complete install' % model_id)
    snapshot_dir = manifest['snapshot_dir']
    home = str(Path.home())
    # The report is committed evidence: the home directory is written as '~'.
    shown = '~'+snapshot_dir[len(home):] if snapshot_dir.startswith(home+'/') else snapshot_dir
    started = time.monotonic()
    motore = Coordinatore(model_id, snapshot_dir, manifest)
    load = dict(seconds=time.monotonic()-started, integrity_read_bytes=motore.integrity_read_bytes,
                kernel=motore.math.name, tied=motore.tied)
    cases = []
    try:
        for index, messages in enumerate(PROMPTS):
            prepared = motore.prepare_chat(messages, None, MAX_TOKENS, dict(GREEDY))
            begun = time.monotonic()
            reply = motore.generate(prepared).future.result()
            served = time.monotonic()-begun
            prompt = reply['diagnostics']['prompt_token_ids']
            completion = reply['diagnostics']['completion_token_ids']
            arrays, run = capture(motore.catalog, motore.math, prompt, completion)
            path = args.output/('case%d.npz' % index)
            np.savez(path, **arrays)
            cases.append(dict(messages=messages, prompt_token_ids=prompt, completion_token_ids=completion,
                              text=reply['text'], finish_reason=reply['finish_reason'],
                              computation=reply['diagnostics']['computation'],
                              sampling=reply['diagnostics']['sampling'], motore_seconds=served,
                              capture=run, arrays=path.name,
                              arrays_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              shapes={k: list(v.shape) for k, v in arrays.items()}))
            print('case %d: %d prompt + %d completion tokens, %r (%.0f s served, %.0f s captured)'
                  % (index, len(prompt), len(completion), reply['text'], served, run['seconds']), flush=True)
    finally:
        motore.close()
    report = dict(schema='inwime-tied38-motore-v1', model=model_id, snapshot_dir=shown,
                  max_tokens=MAX_TOKENS, load=load, cases=cases, peak_rss_bytes=peak_rss_bytes(),
                  host=dict(python=platform.python_version(), machine=platform.machine(),
                            kernel_release=platform.release()))
    (args.output/'motore.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()

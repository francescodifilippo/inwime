"""Local evidence of #41 on Qwen/Qwen2.5-0.5B-Instruct (tied head), installed and verified.

Two parts, run as separate processes so that each peak RSS is its own:

- `identity`: a chat prompt rendered by the Motore, its greedy completion through the Motore
  (Percorso additivo), then the same sequence (one prefill of the prompt, one decode per
  generated token) in two fresh sessions over the catalog the Motore builds for the installed
  Checkpoint: the frozen computation (`CatalogQwen2Runtime`, i.e. `Qwen2Runtime` unchanged over
  the tied catalog of #38) and the additive runtime with the operators the Motore's gate chose,
  in blocks of `--block`. Every logit (all positions, full vocabulary), every residual and the
  whole KV are compared byte for byte; their SHA256 are reported. A third session runs the
  configuration the Motore serves (no trace, the default block, head and final norm on the last
  position only): its last logits and KV after every call are compared with the frozen ones.
- `long`: a prompt of `--tokens` token ids (the tokenized text of docs/VISION_AND_ROADMAP.md,
  repeated and cut) prefilled through a session built exactly as the Motore builds it, then one
  greedy decode: wall times, bytes read, the memory plan by category. For comparison the plan the
  Percorso congelato would reserve for the same capacity is computed, not run.

Nothing is downloaded; the home directory is written as '~'.

    PYTHONPATH=. INWIME_CPU_KERNEL=build/libinwime_cpu.so INWIME_OPS_KERNEL=build/libinwime_ops.so \\
        /usr/bin/time -v .venv-product/bin/python -m tools.additive41_motore identity --output FILE
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time

from inwime import registry
from inwime.engine import CatalogQwen2Runtime, Coordinatore, Prepared
from inwime.generation import Sampling
from inwime_runtime.llama import llama_plan
from inwime.additive import AdditiveQwen2Runtime

REPO_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
GREEDY = {'temperature': 0, 'repetition_penalty': 1.0}
MESSAGES = [{'role': 'user', 'content': 'Descrivi in due frasi il ciclo dell\'acqua.'}]
ROOT = Path(__file__).resolve().parents[1]


def peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024


def host():
    return dict(python=platform.python_version(), machine=platform.machine(),
                kernel_release=platform.release(), cpus=os.cpu_count(),
                load_average_before=os.getloadavg(), threads='1 (single-threaded kernels)',
                temperature_and_power='NOT MEASURED')


def motore():
    model_id = '%s@%s' % (REPO_ID, REVISION)
    manifest = registry.read_json(registry.manifest_path(REPO_ID, REVISION))
    if not registry.intact(manifest):
        raise SystemExit('%s is not a verified, complete install' % model_id)
    started = time.monotonic()
    m = Coordinatore(model_id, manifest['snapshot_dir'], manifest)
    return m, dict(seconds=time.monotonic()-started, kernel=m.math.name, tied=m.tied,
                   operators=m.operators)


def frozen_steps(catalog, math, prompt, completion):
    sequence = prompt+completion
    steps = []
    with CatalogQwen2Runtime(catalog, len(sequence), integrity='metadata', math_backend=math) as r:
        width, vocab = r.spec.width, r.spec.vocab
        calls = [(r.prefill, prompt, 0)] + [(r.decode, t, len(prompt)+i) for i, t in enumerate(completion)]
        for call, argument, first in calls:
            with call(argument) as out:
                logits = bytes(out.logits)
                size = len(logits)//(vocab*8)
                steps.append(dict(
                    logits={first+i: logits[i*vocab*8:(i+1)*vocab*8] for i in range(size)},
                    residuals={(l, first+i): bytes(h)[i*width*8:(i+1)*width*8]
                               for l, h in enumerate(out.hidden) for i in range(size)},
                    kv=b''.join(bytes(v) for pair in out.kv for v in pair)))
        stats = r.stats()
    return steps, stats


class Trace:
    def __init__(self):
        self.logits, self.residuals = {}, {}

    def residual(self, layer, position, values):
        self.residuals[layer, position] = bytes(values)

    def logit(self, position, values):
        self.logits[position] = bytes(values)


def additive_steps(catalog, math, ops, prompt, completion, block):
    sequence = prompt+completion
    steps = []
    with AdditiveQwen2Runtime(catalog, len(sequence), integrity='metadata', math_backend=math, ops=ops,
                              block=block, all_logits=True) as r:
        for call, argument in [(r.prefill, prompt)] + [(r.decode, t) for t in completion]:
            trace = Trace()
            with call(argument, trace=trace) as out:
                steps.append(dict(logits=trace.logits, residuals=trace.residuals,
                                  kv=b''.join(bytes(v) for pair in out.kv for v in pair)))
        stats = r.stats()
    return steps, stats


def served_steps(catalog, math, ops, prompt, completion):
    sequence = prompt+completion
    steps = []
    with AdditiveQwen2Runtime(catalog, len(sequence), integrity='metadata', math_backend=math, ops=ops) as r:
        for call, argument in [(r.prefill, prompt)] + [(r.decode, t) for t in completion]:
            with call(argument) as out:
                steps.append(dict(logits=bytes(out.logits), kv=b''.join(bytes(v) for pair in out.kv for v in pair)))
        block = r.block
    return steps, block


def digest(steps, kind):
    h = hashlib.sha256()
    for step in steps:
        value = step[kind]
        if isinstance(value, dict):
            for key in sorted(value):
                h.update(value[key])
        else:
            h.update(value)
    return h.hexdigest()


def identity(args):
    m, load = motore()
    try:
        prepared = m.prepare_chat(MESSAGES, None, args.max_tokens, dict(GREEDY))
        begun = time.monotonic()
        reply = m.generate(prepared).future.result()
        served = time.monotonic()-begun
        prompt = reply['diagnostics']['prompt_token_ids']
        completion = reply['diagnostics']['completion_token_ids']
        begun = time.monotonic()
        frozen, frozen_stats = frozen_steps(m.catalog, m.math, prompt, completion)
        frozen_seconds = time.monotonic()-begun
        begun = time.monotonic()
        additive, additive_stats = additive_steps(m.catalog, m.math, m.ops, prompt, completion, args.block)
        additive_seconds = time.monotonic()-begun
        served_runs, served_block = served_steps(m.catalog, m.math, m.ops, prompt, completion)
    finally:
        m.close()
    kinds = {kind: dict(identical=all(a[kind] == f[kind] for a, f in zip(additive, frozen, strict=True)),
                        sha256_additive=digest(additive, kind), sha256_frozen=digest(frozen, kind))
             for kind in ('logits', 'residuals', 'kv')}
    ends = [len(prompt)+i for i in range(len(completion)+1)]     # last position of each call
    kinds['served_last_logits'] = dict(block=served_block, identical=all(
        s["logits"] == f["logits"][end-1] for s, f, end in zip(served_runs, frozen, ends, strict=True)))
    kinds['served_kv'] = dict(block=served_block, identical=all(
        s["kv"] == f["kv"] for s, f in zip(served_runs, frozen, strict=True)))
    return dict(part='identity', load=load, messages=MESSAGES, prompt_token_ids=prompt,
                completion_token_ids=completion, text=reply['text'], served_seconds=served,
                computation=reply['diagnostics']['computation'], block=args.block,
                positions=len(prompt)+len(completion), calls=1+len(completion),
                comparison=kinds, verdict='PASS' if all(k['identical'] for k in kinds.values()) else 'FAIL',
                frozen=dict(seconds=frozen_seconds, inference_read_bytes=frozen_stats['inference_read_bytes'],
                            reserved_bytes=frozen_stats['reserved_peak_bytes']),
                additive=dict(seconds=additive_seconds, inference_read_bytes=additive_stats['inference_read_bytes'],
                              reserved_bytes=additive_stats['reserved_peak_bytes'],
                              note='all_logits: head and final norm on every position, for the comparison'))


def long_prompt(args):
    m, load = motore()
    m.close()          # only its catalog, kernel and gate are reused, on this thread
    text = (ROOT/'docs/VISION_AND_ROADMAP.md').read_text()
    ids = []
    while len(ids) < args.tokens:
        ids += m.tokenizer.encode(text)
    ids = ids[:args.tokens]
    capacity = len(ids)+1
    frozen_plan = llama_plan(m.catalog, capacity, 262144, 262144, 'sync').required
    with m._session(capacity, 'metadata') as r:
        plan = dict(r.plan.required)
        begun = time.monotonic()
        with r.prefill(ids) as out:
            vocab = r.spec.vocab
            logits = list(out.logits)
        prefill_seconds = time.monotonic()-begun
        after_prefill = r.stats()
        token = max(range(vocab), key=logits.__getitem__)      # greedy, lowest id on ties
        begun = time.monotonic()
        with r.decode(token) as out:
            decoded = max(range(vocab), key=list(out.logits).__getitem__)
        decode_seconds = time.monotonic()-begun
        stats = r.stats()
    return dict(part='long', load=load, prompt_tokens=len(ids),
                prompt_source='docs/VISION_AND_ROADMAP.md tokenized, repeated, cut',
                prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                block=r.block, operators=stats['operators'], kernel=stats['math_backend'],
                prefill_seconds=prefill_seconds, prefill_tokens_per_second=len(ids)/prefill_seconds,
                decode_seconds=decode_seconds, greedy_tokens=[token, decoded],
                prefill_inference_read_bytes=after_prefill['inference_read_bytes'],
                total_inference_read_bytes=stats['inference_read_bytes'],
                additive_plan_bytes=plan, additive_plan_total=sum(plan.values()),
                frozen_plan_bytes_not_run=dict(frozen_plan), frozen_plan_total_not_run=sum(frozen_plan.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('part', choices=('identity', 'long'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--block', type=int, default=8)
    p.add_argument('--max-tokens', type=int, default=6)
    p.add_argument('--tokens', type=int, default=5317)
    args = p.parse_args()
    before = host()
    report = identity(args) if args.part == 'identity' else long_prompt(args)
    report.update(model='%s@%s' % (REPO_ID, REVISION), host=before, peak_rss_bytes=peak_rss_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k in ('verdict', 'prefill_seconds', 'peak_rss_bytes')}))


if __name__ == '__main__':
    main()

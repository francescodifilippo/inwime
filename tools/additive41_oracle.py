"""Offline oracle values for the CI regression of the Percorso additivo (#41).

Runs stock Transformers (`tools.check_oracle10.stock`: `from_pretrained`, eager attention,
residual hooks and DynamicCache K/V; pinned and unchanged) on the tiny tied and untied Qwen2
fixtures of `tests_inwime/tinyqwen.generator`, under the two contracts of the M10/#38 gates:

    stock_fp32  FP32 weights and arithmetic: the Transformers oracle;
    full64      FP64 weights with the increment-7/9 FP64 precision profile: the FP64 reference.

It writes every logit, residual and K/V entry of one sequence (prefill of 5, then 3 decodes) to a
versioned JSON file with its provenance: the producing tool and Git commit, library versions,
SHA256 of every fixture file, the M10 thresholds, and the SHA256 of the canonical values. The
product test `tests_inwime/test_oracle_reference.py` checks those digests and compares the
additive runtime with the values, in CI, without Transformers or PyTorch.

Independence: no InWiMe import (`inwime`, `inwime_runtime`); only the fixture generator is shared.

    .venv-oracle7/bin/python -m tools.additive41_oracle --output tests_inwime/reference/additive41_oracle.json
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import numpy as np
import torch
import transformers

from . import compare10
from .check_oracle10 import stock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tests_inwime'))
import tinyqwen  # noqa: E402  (fixture generator only)

SCHEMA = 'inwime-oracle-reference/1'
SEQUENCE_PREFIX = [3, None, 7, 30, 5, None, 12, 2]      # None: the fixture's TINY_TOKEN
CHUNKS = [5, 1, 1, 1]
CONTRACTS = {'stock_fp32': (torch.float32, 'stock', 'fp32'), 'full64': (torch.float64, 'full64', 'fp64')}


def encode(array):
    array = np.ascontiguousarray(array, dtype='<f8')
    return dict(shape=list(array.shape), float64_le_base64=base64.b64encode(array.tobytes()).decode())


def canonical_digest(values):
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    tiny = tinyqwen.generator().TINY_TOKEN
    sequence = [tiny if t is None else t for t in SEQUENCE_PREFIX]
    fixtures, values = {}, {}
    with tempfile.TemporaryDirectory() as scratch:
        for head in ('untied', 'tied'):
            directory = tinyqwen.generator(head).write(Path(scratch)/head)
            fixtures[head] = dict(generator='tests_inwime/tinyqwen.py generator(%r).write' % head, files={
                f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(directory.iterdir())})
            values[head] = {}
            for contract, (dtype, profile, _) in CONTRACTS.items():
                reference = stock(str(directory), dtype, profile, sequence, CHUNKS)
                values[head][contract] = {kind: encode(reference[kind]) for kind in compare10.KINDS}
                print(head, contract, {k: v['shape'] for k, v in values[head][contract].items()}, flush=True)
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    # Tracked files only, except the output itself (the product of this run): the tool, the fixture
    # generator and the pinned oracle helpers are then those of `git_commit`.
    output = args.output.resolve().relative_to(ROOT) if args.output.resolve().is_relative_to(ROOT) else None
    exclude = [':(exclude)%s' % output] if output else []
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no', '--', '.', *exclude],
                                capture_output=True, text=True, cwd=ROOT).stdout)
    report = dict(
        schema=SCHEMA, produced_by='tools/additive41_oracle.py',
        produced_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        git_commit=commit, git_worktree_dirty=dirty,
        versions=dict(torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
                      python=platform.python_version()),
        threads=args.threads, sequence=sequence, chunks=CHUNKS, fixtures=fixtures,
        contracts={k: dict(precision=v[2], atol=compare10.THRESHOLDS[v[2]][0], rtol=compare10.THRESHOLDS[v[2]][1])
                   for k, v in CONTRACTS.items()},
        layout=dict(logits='positions x vocab', hidden='layers x positions x width (residual after each layer)',
                    kv='layers x (K, V) x positions x kv_width, K after RoPE'),
        values=values, values_sha256=canonical_digest(values))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True)+'\n')
    print('wrote', args.output, report['values_sha256'])


if __name__ == '__main__':
    main()

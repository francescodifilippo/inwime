"""End-to-end equivalence of oracle10 against the increment-9 oracle9 captures (Qwen2.5-7B).

Preregistered prerequisite of every 14B oracle (protocol.numerical_contract.oracle10
.equivalence.7b_end_to_end_prerequisite):

  stock_fp32           logits, residuals and K/V bit-identical
  full64, stock_fp64   residuals and K/V bit-identical; logits max |delta| <= 1e-9 and
                       identical argmax (blocked FP64 head)

Both sides must describe the same experiment: contract, dtype, precision profile,
checkpoint revision, token sequence, prompt length and chunks are compared first;
any difference is FAIL (never "not comparable"). NumPy only; no torch, no InWiMe.
A failure is recorded as such; thresholds come from the protocol, never from here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import compare10

RULE = dict(stock_fp32=dict(require_bitwise_logits=True), full64=dict(require_bitwise_logits=False),
            stock_fp64=dict(require_bitwise_logits=False))
IDENTITY = ('contract', 'dtype', 'precision_profile', 'revision', 'sequence', 'prompt_length', 'chunks')


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load(npz):
    with np.load(npz) as z:
        return {k.removeprefix('full.'): z[k] for k in z.files}


def evaluate(new_npz, new_meta, old_npz, old_meta, protocol):
    contract = new_meta.get('contract')
    problems = []
    if contract not in RULE:
        problems.append(f'unknown contract {contract!r}')
    for key in IDENTITY:
        if new_meta.get(key) != old_meta.get(key):
            problems.append(f'{key} differs: {new_meta.get(key)!r} != {old_meta.get(key)!r}')
    if old_meta.get('capture_sha256') != sha256(old_npz):
        problems.append('oracle9 capture SHA256 differs from its metadata')
    if new_meta.get('capture_sha256') != sha256(new_npz):
        problems.append('oracle10 capture SHA256 differs from its metadata')
    if not new_meta.get('head_mode_is_preregistered'):
        problems.append('oracle10 head mode is not the preregistered one')
    limit = protocol['numerical_contract']['oracle10']['equivalence']
    if '1e-9' not in limit['7b_end_to_end_prerequisite']:
        problems.append('protocol rule text changed: 1e-9 not found')
    new, old = load(new_npz), load(old_npz)
    result = compare10.equivalence(new, old, logits_max_abs=1e-9, **RULE.get(contract, {}))
    greedy = dict(oracle10=new_meta.get('upstream_greedy_matches_sequence'),
                  oracle9=old_meta.get('upstream_greedy_matches_sequence'),
                  argmax_equal=new_meta.get('argmax') == old_meta.get('argmax'))
    if not (greedy['oracle10'] is True and greedy['argmax_equal']):
        problems.append('greedy continuation or argmax list differs')
    passed = result['passed'] and not problems
    return dict(schema='inwime-equivalence10-e2e/1', contract=contract, verdict='PASS' if passed else 'FAIL',
                coverage='FULL' if all('bit_identical' in result['kinds'].get(k, {}) for k in compare10.KINDS)
                else 'PARTIAL', rule=RULE.get(contract), problems=problems, comparison=result, greedy=greedy,
                shapes={k: list(v.shape) for k, v in new.items()},
                elements={k: int(v.size) for k, v in new.items()},
                oracle10=dict(capture_sha256=new_meta.get('capture_sha256'), head_mode=new_meta.get('head_mode'),
                              oracle_wall_s=new_meta.get('oracle_wall_s'), peak_rss_bytes=new_meta.get('peak_rss_bytes'),
                              versions=new_meta.get('versions'), threads=new_meta.get('threads'),
                              source_bf16_bytes_read=new_meta.get('loader', {}).get('source_bf16_bytes_read')),
                oracle9=dict(capture_sha256=old_meta.get('capture_sha256'), versions=old_meta.get('versions'),
                             threads=old_meta.get('threads')))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--new', type=Path, required=True, help='oracle10 capture (.npz, metadata next to it)')
    p.add_argument('--old', type=Path, required=True, help='oracle9 capture (artifacts/increment9/oracle-*.npz)')
    p.add_argument('--protocol', type=Path, default=Path('results/increment10/protocol.json'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    out = evaluate(a.new, json.loads(a.new.with_suffix('.json').read_text()), a.old,
                   json.loads(a.old.with_suffix('.json').read_text()), json.loads(a.protocol.read_text()))
    out['inputs'] = dict(new=str(a.new), new_sha256=sha256(a.new), old=str(a.old), old_sha256=sha256(a.old))
    with a.output.open('x') as f:
        json.dump(out, f, indent=2); f.write('\n')
    print(json.dumps({k: out[k] for k in ('contract', 'verdict', 'coverage', 'problems')}
                     | {k: v for k, v in out['comparison']['kinds'].items()}, indent=1))
    return 0 if out['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

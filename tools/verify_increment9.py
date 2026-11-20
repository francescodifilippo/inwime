"""Offline full-array comparison of an increment-9 runtime capture with a bounded oracle.

Runtime phases (prefill, then decodes) are compared against the oracle's
full-sequence arrays: logits rows, residuals of every layer and the K/V prefix
published after each phase. Thresholds come from the preregistered protocol and
are refused if altered. The upstream greedy continuation is recomputed from the
oracle arrays and each decode input must be the runtime's previous greedy choice.
Never executes an InWiMe forward.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from inwime_runtime.tokenizer import LlamaTokenizer
from .verify_increment7 import metrics
from .verify_increment8 import read_capture

CONTRACTS = {'stock_fp32': ('float32', 'stock', 'fp32_comparison', (2e-3, 2e-4)),
             'full64': ('float64', 'full64', 'fp64_comparison', (2e-5, 2e-6)),
             'stock_fp64': ('float64', 'stock', 'fp64_comparison', (2e-5, 2e-6))}


def verify(runtime_path, capture_dir, oracle_path, protocol_path, revision_key='phase_b'):
    runtime = json.loads(runtime_path.read_text())
    meta = json.loads(oracle_path.with_suffix('.json').read_text())
    protocol = json.loads(protocol_path.read_text())
    contract = meta['contract']
    dtype, profile, key, declared = CONTRACTS[contract]
    tolerance = protocol[key]
    if (tolerance['atol'], tolerance['rtol']) != declared:
        raise ValueError('preregistered thresholds changed')
    if (meta['dtype'] != dtype or meta['precision_profile'] != profile
            or meta['attention_implementation'] != 'eager'):
        raise ValueError('oracle contract/dtype/profile inconsistent')
    if meta['revision'] != protocol[revision_key]['revision']:
        raise ValueError('oracle revision differs from protocol')
    with oracle_path.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != meta['capture_sha256']:
            raise ValueError('oracle digest mismatch')
    sequence = [t for p in runtime['phases'] for t in p['ids']]
    if meta['sequence'] != sequence or meta['prompt_length'] != len(runtime['input_ids']):
        raise ValueError('oracle sequence differs from the runtime sequence')
    if len(runtime['captures']) != len(runtime['phases']) or len(runtime['phases']) < 2:
        raise ValueError('incomplete runtime scenario: need prefill and at least one decode')
    tokenizer = LlamaTokenizer(runtime['checkpoint'])
    for item in meta['tokenizer']:
        if tokenizer.encode(item['text']) != item['ids'] or tokenizer.decode(item['ids']) != item['decoded']:
            raise ValueError('official tokenizer mismatch: '+repr(item['text']))
    atol, rtol = tolerance['atol'], tolerance['rtol']
    report = dict(contract=contract, oracle_sha256=meta['capture_sha256'], oracle_metadata=meta,
                  runtime_report=str(runtime_path), atol=atol, rtol=rtol, comparisons={},
                  compared_elements=dict(logits=0, hidden=0, kv=0))
    with np.load(oracle_path, allow_pickle=False) as oracle:
        full = {k: oracle['full.'+k] for k in ('logits', 'hidden', 'kv')}
    # Upstream greedy is recomputed from the oracle arrays, never read from the oracle's
    # own claim: by causality the argmax at position i is the upstream choice after
    # prefix[:i+1]. The runtime must also have fed back exactly its own previous choice.
    prompt = len(runtime['input_ids'])
    upstream = full['logits'].argmax(-1).tolist()
    recomputed = len(upstream) == len(sequence) and upstream[prompt-1:len(sequence)-1] == sequence[prompt:]
    follows = runtime['phases'][0]['ids'] == runtime['input_ids'] and all(
        phase['ids'] == previous['greedy'][-1:] for previous, phase in zip(runtime['phases'], runtime['phases'][1:]))
    declared = bool(meta['upstream_greedy_matches_sequence'])
    report['upstream_greedy'] = dict(recomputed_from_oracle_argmax=bool(recomputed), declared_by_oracle=declared,
                                     decode_inputs_follow_runtime_greedy=bool(follows))
    report['upstream_greedy_matches_sequence'] = bool(recomputed and declared)
    passed = bool(recomputed and declared and follows)
    start = 0
    for index, phase in enumerate(runtime['phases']):
        stop = start+len(phase['ids'])
        prefix = f'decode.{index}'
        actual = {k: read_capture(capture_dir, runtime['captures'][prefix][k]) for k in ('logits', 'hidden', 'kv')}
        expected = dict(logits=full['logits'][start:stop], hidden=full['hidden'][:, start:stop],
                        kv=full['kv'][:, :, :stop])
        record = {}
        for kind in ('logits', 'hidden', 'kv'):
            record[kind] = metrics(actual[kind], expected[kind], atol, rtol)
            report['compared_elements'][kind] += actual[kind].size
            if kind != 'logits':
                record[kind]['layers'] = [metrics(x, y, atol, rtol) for x, y in zip(actual[kind], expected[kind])]
            passed &= record[kind]['failures'] == 0
        runtime_argmax = actual['logits'].argmax(-1).tolist()
        record['argmax_equal'] = runtime_argmax == expected['logits'].argmax(-1).tolist()
        record['runtime_greedy_recorded'] = runtime_argmax == phase['greedy']
        passed &= record['argmax_equal'] and record['runtime_greedy_recorded']
        record['top10'] = [dict(actual=np.argsort(-x, kind='stable')[:10].tolist(),
                                upstream=np.argsort(-y, kind='stable')[:10].tolist())
                           for x, y in zip(actual['logits'], expected['logits'])]
        report['comparisons'][prefix] = record
        start = stop
    report['coverage'] = dict(positions=len(sequence), logits_per_position=int(full['logits'].shape[1]),
                              layers=int(full['hidden'].shape[0]), kv_layers=int(full['kv'].shape[0]),
                              full_arrays=True)
    report['passed'] = bool(passed)
    report['role'] = 'mandatory diagnostic, raw verdict kept' if contract == 'stock_fp64' else 'gate contract'
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=Path('results/increment9/protocol.json'))
    p.add_argument('--revision-key', default='phase_b')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = verify(a.runtime, a.capture, a.oracle, a.protocol, a.revision_key)
    with a.output.open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print('PASS' if report['passed'] else 'FAIL', report['contract'], a.output)
    raise SystemExit(0 if report['passed'] else 1)

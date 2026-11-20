"""Full-coverage comparison of an increment-10 runtime capture against an oracle10 capture.

NumPy only (no torch, no InWiMe import). Inputs:
  --forward   the runtime report written by tools/benchmark_increment9.py (--capture)
  --capture   its capture directory (decode.<i>.{logits,hidden,kv}.f64, raw little-endian FP64)
  --oracle    the oracle10 capture (.npz, metadata .json next to it)

Checks, each recorded (never a fallback to a smaller requirement):
  * binding: SHA256 and size of every capture file equal the values recorded by the
    runtime; oracle capture SHA256 equals its metadata; the oracle token sequence equals
    the runtime input sequence and the oracle chunks equal the runtime phases;
  * K/V: the runtime K/V of every phase (lengths 5, 6, 7, 8) is a byte-identical prefix
    of the final K/V (protocol coverage_full.kv_rule);
  * elementwise comparison by tools/compare10.compare with the preregistered
    thresholds of the contract (FP32 for stock_fp32, FP64 for full64/stock_fp64):
    logits of every position over the full vocabulary, residuals of every layer and
    position, final K/V of every layer, K and V;
  * argmax and top-10 of every position (reported), and the upstream greedy
    continuation against the runtime continuation (required).

Verdict: compare10's verdict (PASS only with FULL coverage and zero violations), turned
into FAIL by any binding, K/V-prefix or greedy failure. stock_fp64 is diagnostic: its
verdict is reported raw and never rewritten.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import compare10

PRECISION = {'stock_fp32': 'fp32', 'full64': 'fp64', 'stock_fp64': 'fp64'}
GATING = ('stock_fp32', 'full64')


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def runtime_arrays(forward, capture):
    """(arrays, phase_kv, binding problems) from the runtime capture files."""
    problems, logits, hidden, phase_kv = [], [], [], []
    names = [p['name'] for p in forward['phases']]
    if set(forward.get('captures', {})) != set(names):
        problems.append('capture set differs from the phases')
        return None, None, problems
    for name in names:
        entry = forward['captures'][name]
        arrays = {}
        for kind in ('logits', 'hidden', 'kv'):
            meta = entry[kind]
            path = Path(capture)/meta['file']
            if not path.is_file():
                problems.append(f'missing capture file {meta["file"]}')
                return None, None, problems
            if path.stat().st_size != meta['nbytes'] or sha256(path) != meta['sha256']:
                problems.append(f'capture file differs from the runtime record: {meta["file"]}')
            values = np.fromfile(path, dtype='<f8')
            if values.size != int(np.prod(meta['shape'])):
                problems.append(f'capture size contradicts its shape: {meta["file"]}')
                return None, None, problems
            arrays[kind] = values.reshape(meta['shape'])
        logits.append(arrays['logits']); hidden.append(arrays['hidden']); phase_kv.append(arrays['kv'])
    final = phase_kv[-1]
    return dict(logits=np.concatenate(logits), hidden=np.concatenate(hidden, axis=1), kv=final), phase_kv, problems


def kv_prefixes(phase_kv):
    final = phase_kv[-1]
    rows = []
    for kv in phase_kv:
        length = kv.shape[2]
        same = kv.shape[:2] == final.shape[:2] and kv.shape[3] == final.shape[3] and \
            np.ascontiguousarray(kv).tobytes() == np.ascontiguousarray(final[:, :, :length]).tobytes()
        rows.append(dict(length=int(length), byte_identical_prefix=bool(same)))
    return rows


def topk(logits, k=10):
    # stable ordering: value descending, then index ascending
    order = np.lexsort((np.broadcast_to(np.arange(logits.shape[1]), logits.shape), -logits), axis=1)
    return order[:, :k]


def evaluate(forward, capture, oracle_npz, oracle_meta):
    contract = oracle_meta.get('contract')
    problems = []
    if contract not in PRECISION:
        problems.append(f'unknown contract {contract!r}')
    sequence = [t for p in forward['phases'] for t in p['ids']]
    chunks = [len(p['ids']) for p in forward['phases']]
    if oracle_meta.get('sequence') != sequence:
        problems.append('oracle sequence differs from the runtime input sequence')
    if oracle_meta.get('chunks') != chunks:
        problems.append('oracle chunks differ from the runtime phases')
    if oracle_meta.get('capture_sha256') != sha256(oracle_npz):
        problems.append('oracle capture SHA256 differs from its metadata')
    if not oracle_meta.get('head_mode_is_preregistered'):
        problems.append('oracle head mode is not the preregistered one')
    actual, phase_kv, binding = runtime_arrays(forward, capture)
    problems += binding
    with np.load(oracle_npz) as z:
        reference = {k.removeprefix('full.'): z[k] for k in z.files}
    cfg = forward['config']
    layers, width, vocab = cfg['num_hidden_layers'], cfg['hidden_size'], cfg['vocab_size']
    kv_width = cfg['num_key_value_heads']*width//cfg['num_attention_heads']
    shapes = compare10.required_shapes(layers, len(sequence), width, vocab, kv_width)
    result = compare10.compare(actual, reference, shapes, PRECISION.get(contract, 'fp64'))
    prefixes = kv_prefixes(phase_kv) if phase_kv else []
    if not prefixes or not all(r['byte_identical_prefix'] for r in prefixes) or \
            [r['length'] for r in prefixes] != list(np.cumsum(chunks)):
        problems.append('runtime per-phase K/V is not a byte-identical prefix of the final K/V (lengths '
                        + ','.join(str(r['length']) for r in prefixes) + ')')
    argmax = top = None
    greedy_ok = False
    if actual is not None and tuple(actual['logits'].shape) == tuple(reference['logits'].shape):
        a, r = actual['logits'], reference['logits']
        argmax = dict(runtime=a.argmax(-1).tolist(), oracle=r.argmax(-1).tolist())
        argmax['equal_every_position'] = argmax['runtime'] == argmax['oracle']
        ta, tr = topk(a), topk(r)
        top = dict(ordered_equal_positions=int((ta == tr).all(axis=1).sum()),
                   set_equal_positions=int(sum(set(x) == set(y) for x, y in zip(ta.tolist(), tr.tolist()))),
                   positions=int(a.shape[0]))
        prompt = len(forward['phases'][0]['ids'])
        continuation = forward['continuation']
        upstream = argmax['oracle'][prompt-1:]
        greedy_ok = upstream == continuation
        argmax['upstream_greedy'] = upstream
        argmax['runtime_continuation'] = continuation
    if not greedy_ok:
        problems.append('upstream greedy continuation differs from the runtime continuation')
    verdict = 'FAIL' if problems or result['verdict'] == 'FAIL' else result['verdict']
    return dict(schema='inwime-parity10/1', contract=contract, gating=contract in GATING,
                verdict=verdict, coverage=result['coverage'], problems=problems, comparison=result,
                kv_prefixes=prefixes, argmax=argmax, top10=top, greedy_equal=greedy_ok,
                sequence=sequence, chunks=chunks, required_elements={k: int(np.prod(v)) for k, v in shapes.items()},
                oracle=dict(capture_sha256=oracle_meta.get('capture_sha256'), head_mode=oracle_meta.get('head_mode'),
                            dtype=oracle_meta.get('dtype'), precision_profile=oracle_meta.get('precision_profile'),
                            versions=oracle_meta.get('versions'), revision=oracle_meta.get('revision'),
                            peak_rss_bytes=oracle_meta.get('peak_rss_bytes'), oracle_wall_s=oracle_meta.get('oracle_wall_s')))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--forward', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    forward = json.loads(a.forward.read_text())
    meta = json.loads(a.oracle.with_suffix('.json').read_text())
    out = evaluate(forward, a.capture, a.oracle, meta)
    out['inputs'] = dict(forward=str(a.forward), forward_sha256=sha256(a.forward), capture=str(a.capture),
                         oracle=str(a.oracle), oracle_sha256=sha256(a.oracle),
                         oracle_meta_sha256=sha256(a.oracle.with_suffix('.json')))
    with a.output.open('x') as f:
        json.dump(out, f, indent=2); f.write('\n')
    k = out['comparison']['kinds']
    print(json.dumps(dict(contract=out['contract'], verdict=out['verdict'], coverage=out['coverage'],
                          problems=out['problems'],
                          kinds={n: {x: k[n].get(x) for x in ('elements', 'failures', 'max_abs_error',
                                                              'max_threshold_utilization')} for n in k}), indent=1))
    return 0 if out['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())

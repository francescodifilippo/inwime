"""Required parity runner: compares every captured value; no skips/fallback oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from inwime_runtime.llama import LlamaRuntime
from inwime_runtime.math_cpu import NativeMath
from inwime_runtime.tokenizer import LlamaTokenizer

TOLERANCES = {'float64': (2e-5, 2e-6), 'float32': (2e-3, 2e-4)}


def metrics(actual, expected, atol, rtol):
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    delta = np.abs(actual-expected)
    return dict(count=actual.size, max_abs=float(delta.max()),
                rms=float(np.sqrt(np.mean(delta**2))),
                max_rel_floor_1e_8=float((delta/np.maximum(np.abs(expected), 1e-8)).max()),
                failures=int(np.count_nonzero(delta > atol+rtol*np.abs(expected))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', type=Path)
    p.add_argument('oracle', type=Path)
    p.add_argument('--mode', choices=['sync','pipeline'], default='sync')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    meta = json.loads(args.oracle.with_suffix('.json').read_text())
    with args.oracle.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == meta['capture_sha256']
    oracle = np.load(args.oracle, allow_pickle=False)
    atol, rtol = TOLERANCES[meta['dtype']]
    tokenizer = LlamaTokenizer(args.checkpoint)
    for sample in meta['tokenizer']:
        assert tokenizer.encode(sample['text']) == sample['ids']
        assert tokenizer.decode(sample['ids']) == sample['decoded']
    report = dict(oracle_contract=meta.get('contract', 'historical_unspecified'), oracle_precision_profile=meta.get('precision_profile', 'historical_unspecified'), dtype_audit=meta.get('dtype_audit'), oracle_executed=True, oracle_diagnostic_rms64=meta.get('diagnostic_rms64', False), oracle_sha256=meta['capture_sha256'], dtype=meta['dtype'],
                  mode=args.mode, atol=atol, rtol=rtol, comparisons={}, tokenizer=meta['tokenizer'])
    failure = False
    captures = {}
    for name, steps in meta['scenarios'].items():
        began = time.monotonic()
        with LlamaRuntime(args.checkpoint, capacity=16, io_mode=args.mode,
                          math_backend=NativeMath('build/libinwime_cpu.so')) as runtime:
            startup = time.monotonic()-began
            records = []
            for i, step in enumerate(steps):
                started = time.monotonic()
                with runtime.prefill(step['tokens']) as result:
                    forward = time.monotonic()-started
                    prefix = f'{name}.{i}'
                    actual = dict(logits=np.array(result.logits).reshape(-1,runtime.spec.vocab),
                                  hidden=np.array(result.hidden).reshape(runtime.spec.layers,-1,runtime.spec.width),
                                  kv=np.array(result.kv).reshape(runtime.spec.layers,2,runtime.length,runtime.spec.kv_width))
                    captures[prefix] = actual
                    comparisons = {}
                    for kind, values in actual.items():
                        expected = oracle[prefix+'.'+kind]
                        comparisons[kind] = metrics(values, expected, atol, rtol)
                        if kind in ('hidden', 'kv'):
                            comparisons[kind]['layers'] = [metrics(values[l], expected[l], atol, rtol)
                                                          for l in range(runtime.spec.layers)]
                        failure |= comparisons[kind]['failures'] > 0
                    chosen = actual['logits'].argmax(-1).tolist()
                    comparisons['greedy_equal'] = chosen == step['greedy']
                    failure |= not comparisons['greedy_equal']
                    expected = oracle[prefix+'.logits']
                    comparisons['top10'] = [dict(actual=np.argsort(-row,kind='stable')[:10].tolist(),
                         upstream=np.argsort(-ref,kind='stable')[:10].tolist()) for row, ref in zip(actual['logits'],expected)]
                    records.append(dict(digests={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in actual.items()}, tokens=step['tokens'], forward_s=forward, comparisons=comparisons))
            report['comparisons'][name] = dict(startup_s=startup,steps=records,stats=runtime.stats())
    invariants = {}
    for kind in ('logits', 'hidden'):
        axis = 1 if kind == 'hidden' else 0
        full = captures['full.0'][kind]
        chunked = np.concatenate([captures['chunks.0'][kind],captures['chunks.1'][kind]],axis=axis)
        incremental = np.concatenate([captures[f'decode.{i}'][kind] for i in range(4)],axis=axis)
        invariants[kind] = dict(chunks_exact=bool(np.array_equal(full,chunked)),
            repeated_exact=bool(np.array_equal(full,captures['repeat.0'][kind])),
            extended_prefix_exact=bool(np.array_equal(full,np.take(captures['extended.0'][kind],range(len(meta['scenarios']['full'][0]['tokens'])),axis=axis))),
            incremental_exact=bool(np.array_equal(incremental,captures['extended.0'][kind])))
    invariants['kv'] = dict(chunks_exact=bool(np.array_equal(captures['chunks.1']['kv'],captures['full.0']['kv'])),
        incremental_exact=bool(np.array_equal(captures['decode.3']['kv'],captures['extended.0']['kv'])))
    report['engine_invariants'] = invariants
    failure |= not all(v for item in invariants.values() for v in item.values())
    report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report['passed'] = not failure
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f: json.dump(report,f,indent=2); f.write('\n')
    print('PASS' if not failure else 'FAIL', args.output)
    if failure: raise SystemExit(1)


if __name__ == '__main__': main()

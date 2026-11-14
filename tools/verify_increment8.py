"""Offline full-array comparison; never execute an InWiMe forward in the oracle."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from inwime_runtime.tokenizer import LlamaTokenizer
from .verify_increment7 import metrics


def read_capture(directory, descriptor):
    path = directory/descriptor['file']
    if path.name != descriptor['file'] or path.stat().st_size != descriptor['nbytes']:
        raise ValueError('capture path/size mismatch')
    with path.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != descriptor['sha256']:
            raise ValueError('runtime capture digest mismatch')
    data = np.fromfile(path, dtype=descriptor['dtype'])
    if descriptor['dtype'] != '<f8':
        raise ValueError('expected FP64 runtime capture')
    return data.reshape(descriptor['shape'])


def verify(runtime_path, directory, oracle_path, protocol_path):
    runtime = json.loads(runtime_path.read_text())
    meta = json.loads(oracle_path.with_suffix('.json').read_text())
    protocol = json.loads(protocol_path.read_text())
    tolerance = protocol['fp32_comparison']
    if (tolerance['atol'], tolerance['rtol']) != (2e-3, 2e-4):
        raise ValueError('predeclared FP32 thresholds changed')
    if (meta['contract'] != 'transformers_standard_float32' or meta['precision_profile'] != 'stock'
            or meta['attention_implementation'] != 'eager' or meta['dtype'] != 'float32'
            or meta['revision'] != protocol['revision']):
        raise ValueError('wrong upstream contract/revision')
    with oracle_path.open('rb') as f:
        if hashlib.file_digest(f, 'sha256').hexdigest() != meta['capture_sha256']:
            raise ValueError('oracle digest mismatch')
    scenario = runtime['scenario']
    steps = meta['scenarios'][scenario]
    if len(runtime['phases']) != len(steps) or len(runtime['captures']) != len(steps):
        raise ValueError('incomplete scenario')
    tokenizer = LlamaTokenizer(runtime['checkpoint'])
    for item in meta['tokenizer']:
        if tokenizer.encode(item['text']) != item['ids'] or tokenizer.decode(item['ids']) != item['decoded']:
            raise ValueError('official tokenizer mismatch')
    report = dict(oracle_contract=meta['contract'], oracle_sha256=meta['capture_sha256'],
                  oracle_metadata=meta, runtime_report=str(runtime_path), scenario=scenario,
                  atol=tolerance['atol'], rtol=tolerance['rtol'], comparisons={})
    passed = True
    with np.load(oracle_path, allow_pickle=False) as oracle:
        for index, (phase, step) in enumerate(zip(runtime['phases'], steps)):
            prefix = f'{scenario}.{index}'
            if phase['ids'] != step['tokens']:
                raise ValueError('independently generated input sequence differs')
            record = dict(greedy_equal=phase['greedy'] == step['greedy'])
            passed &= record['greedy_equal']
            for kind in ('logits', 'hidden', 'kv'):
                actual = read_capture(directory, runtime['captures'][prefix][kind])
                expected = oracle[prefix+'.'+kind]
                record[kind] = metrics(actual, expected, tolerance['atol'], tolerance['rtol'])
                if kind != 'logits':
                    record[kind]['layers'] = [metrics(x, y, tolerance['atol'], tolerance['rtol'])
                                              for x, y in zip(actual, expected)]
                else:
                    argmax = actual.argmax(-1).tolist()
                    record['argmax_equal'] = argmax == expected.argmax(-1).tolist() == step['greedy']
                    passed &= record['argmax_equal']
                    record['top10'] = [dict(actual=np.argsort(-x, kind='stable')[:10].tolist(),
                                           upstream=np.argsort(-y, kind='stable')[:10].tolist())
                                       for x, y in zip(actual, expected)]
                passed &= record[kind]['failures'] == 0
            report['comparisons'][prefix] = record
    report['passed'] = bool(passed)
    return report


def invariants(runs):
    """Exact mode/format and full/chunk/incremental invariants, independent of upstream tolerances."""
    reports = [json.loads(p.read_text()) for p, _ in runs]
    arrays = [{prefix: {kind: read_capture(directory, desc) for kind, desc in kinds.items()}
               for prefix, kinds in report['captures'].items()}
              for report, (_, directory) in zip(reports, runs)]
    by_scenario = {}
    for index, report in enumerate(reports):
        by_scenario.setdefault(report['scenario'], []).append(index)
    if set(by_scenario) != {'decode', 'chunks', 'extended'}:
        raise ValueError('need all three runtime scenarios')
    result = {}
    for scenario, indices in by_scenario.items():
        base = indices[0]
        for index in indices[1:]:
            # np.array_equal alone considers +0/-0 equal; compare raw bytes here.
            same = reports[index]['input_ids'] == reports[base]['input_ids']
            same &= [p['ids'] for p in reports[index]['phases']] == [p['ids'] for p in reports[base]['phases']]
            for prefix, kinds in arrays[base].items():
                for kind, values in kinds.items():
                    other = arrays[index][prefix][kind]
                    same &= values.shape == other.shape and values.tobytes() == other.tobytes()
            result[str(runs[index][0])+'_exact'] = bool(same)
    decode = arrays[by_scenario['decode'][0]]
    extended = arrays[by_scenario['extended'][0]]['extended.0']
    chunks = arrays[by_scenario['chunks'][0]]
    for kind in ('logits', 'hidden'):
        axis = 0 if kind == 'logits' else 1
        incremental = np.concatenate([decode[f'decode.{i}'][kind] for i in range(4)], axis=axis)
        chunked = np.concatenate([chunks[f'chunks.{i}'][kind] for i in range(len(chunks))], axis=axis)
        prefix = np.take(extended[kind], range(chunked.shape[axis]), axis=axis)
        result[kind+'_incremental_exact'] = incremental.tobytes() == extended[kind].tobytes()
        result[kind+'_chunks_exact'] = chunked.tobytes() == prefix.tobytes()
    result['kv_incremental_exact'] = decode['decode.3']['kv'].tobytes() == extended['kv'].tobytes()
    full_prefill = decode['decode.0']['kv']
    result['kv_chunks_exact'] = chunks[f'chunks.{len(chunks)-1}']['kv'].tobytes() == full_prefill.tobytes()
    return dict(checks=result, passed=all(result.values()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path)
    p.add_argument('--capture', type=Path)
    p.add_argument('--oracle', type=Path)
    p.add_argument('--protocol', type=Path, default=Path('results/increment8/protocol.json'))
    p.add_argument('--exact-run', nargs=2, action='append', metavar=('REPORT', 'CAPTURE'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.exact_run:
        report = invariants([(Path(r), Path(c)) for r, c in a.exact_run])
    else:
        if any(v is None for v in (a.runtime, a.capture, a.oracle)):
            p.error('runtime, capture, oracle required for parity')
        report = verify(a.runtime, a.capture, a.oracle, a.protocol)
    with a.output.open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print('PASS' if report['passed'] else 'FAIL', a.output)
    raise SystemExit(0 if report['passed'] else 1)

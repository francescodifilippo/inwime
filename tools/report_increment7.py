"""Historical gate plus explicit, separately recorded numerical-contract resolution."""
import hashlib
import json
from pathlib import Path
import statistics


def report(directory):
    d=Path(directory)
    names={'stock_fp32':'final-smollm2-135m-93efa2f-sync-float32.json',
           'stock_fp64':'final-smollm2-135m-93efa2f-sync-float64.json',
           'diagnostic_rms64':'diagnostic-rms64-final-parity.json'}
    result={'status':'partial','parity':{},'benchmark':{}}
    for key,name in names.items():
        r=json.loads((d/name).read_text())
        result['parity'][key]={'passed':r['passed'],'argmax_all_equal':all(s['comparisons']['greedy_equal'] for c in r['comparisons'].values() for s in c['steps']),
            'max_abs':{kind:max(s['comparisons'][kind]['max_abs'] for c in r['comparisons'].values() for s in c['steps']) for kind in ['logits','hidden','kv']},
            'failed_elements':{kind:sum(s['comparisons'][kind]['failures'] for c in r['comparisons'].values() for s in c['steps']) for kind in ['logits','hidden','kv']}}
    for mode in ['sync','pipeline']:
        rows=[json.loads((d/f'measured-{mode}-{i}.json').read_text()) for i in range(1,4)]
        result['benchmark'][mode]=dict(repetitions=3,arena_plan_bytes=rows[0]['arena_plan_bytes'],
            peak_rss_bytes=max(r['peak_rss_bytes'] for r in rows),
            median_startup_s=statistics.median(r['startup_s'] for r in rows),
            median_prefill_s=statistics.median(r['phases'][0]['wall_s'] for r in rows),
            median_decode_s=statistics.median(p['wall_s'] for r in rows for p in r['phases'][1:]),
            median_host_residual_s=statistics.median(p['host_residual_s'] for r in rows for p in r['phases']),
            inference_bytes=rows[0]['stats']['inference_read_bytes'],
            integrity_bytes=rows[0]['stats']['integrity_read_bytes'],
            physical_proc_read_bytes=[r['proc_io_delta']['read_bytes'] for r in rows])
    if result['parity']['stock_fp32']['passed'] and result['parity']['stock_fp64']['passed']:
        result['status']='passed'
    return result


RESOLVED_INPUTS = {
    'stock_fp32': 'parity-stock-float32-sync.json',
    'pure_fp64': 'parity-full64-float64-sync.json',
    'standard_fp64_diagnostic': 'parity-stock-float64-sync.json',
    'pure_fp64_pipeline': 'parity-full64-float64-pipeline.json',
    'pure_fp64_sharded': 'parity-full64-float64-sharded.json',
}


def numerical_gate(reports):
    """No test omitted: standard FP64 remains mandatory as a diagnostic contract.

    Only pure FP64/standard FP32 claim numerical parity. Keep the old thresholds
    for all three and preserve standard FP64's raw failing verdict unchanged.
    """
    from .verify_increment7 import TOLERANCES
    if set(reports) != set(RESOLVED_INPUTS):
        raise ValueError('all numerical contracts/modes must actually be compared')
    scenarios = {'single':1, 'full':1, 'chunks':2, 'decode':4, 'extended':1, 'repeat':1}
    output = {}
    for label, r in reports.items():
        pure = label.startswith('pure_fp64')
        dtype = 'float32' if label == 'stock_fp32' else 'float64'
        contract = 'pytorch_pure_fp64' if pure else 'transformers_standard_'+dtype
        if (r.get('oracle_executed') is not True or r.get('oracle_contract') != contract
                or r.get('oracle_precision_profile') != ('full64' if pure else 'stock')
                or r.get('dtype') != dtype
                or (r.get('atol'),r.get('rtol')) != TOLERANCES[dtype]
                or len(r.get('oracle_sha256','')) != 64):
            raise ValueError('wrong/missing oracle contract, execution or original tolerances')
        audit = r.get('dtype_audit')
        if (not isinstance(audit, dict) or 'non_fp64_output_operations' not in audit
                or (pure and audit['non_fp64_output_operations'])):
            raise ValueError('FP64 reference must have an actual clean floating dtype audit')
        if set(r.get('comparisons',{})) != set(scenarios):
            raise ValueError('missing upstream scenarios')
        comparisons = []
        for name, count in scenarios.items():
            steps = r['comparisons'][name]['steps']
            if len(steps) != count:
                raise ValueError('missing prefill/decode comparison steps')
            for step in steps:
                c = step['comparisons']
                for kind in ('logits','hidden','kv'):
                    if c[kind]['count'] <= 0 or c[kind]['failures'] < 0:
                        raise ValueError('empty or invalid full-array comparison')
                comparisons.append(c)
        invariants = r.get('engine_invariants')
        if not invariants or any(not values for values in invariants.values()):
            raise ValueError('missing causal/KV equivalence verification')
        passed = (all(c[k]['failures'] == 0 for c in comparisons for k in ('logits','hidden','kv'))
                  and all(c['greedy_equal'] for c in comparisons)
                  and all(v for values in invariants.values() for v in values.values()))
        if r.get('passed') is not passed:
            raise ValueError('recorded raw verdict inconsistent with complete comparisons')
        output[label] = dict(required_for_parity=label != 'standard_fp64_diagnostic',
            executed=True, raw_parity_passed=passed,
            max_abs={k:max(c[k]['max_abs'] for c in comparisons) for k in ('logits','hidden','kv')},
            failed_elements={k:sum(c[k]['failures'] for c in comparisons) for k in ('logits','hidden','kv')},
            argmax_all_equal=all(c['greedy_equal'] for c in comparisons))
    return dict(criterion='standard FP32 parity AND genuinely FP64 parity; standard FP64 mixed precision diagnostic required',
                passed=all(v['raw_parity_passed'] for v in output.values() if v['required_for_parity']),
                contracts=output)


def resolved_report(directory):
    d = Path(directory)
    reports = {k:json.loads((d/name).read_text()) for k,name in RESOLVED_INPUTS.items()}
    gate = numerical_gate(reports)
    preservation = json.loads((d/'preservation-after.json').read_text())
    proof = json.loads((d/'attribution.json').read_text())
    if not preservation['all_unchanged'] or not proof['historical_stock_captures_exact']:
        raise ValueError('historical evidence/runtime/libraries/checkpoint changed or stock behavior differs')
    if proof['first_nonexact_operator'] != 'layers.0.norm1':
        raise ValueError('first divergent operator must remain explicitly demonstrated')
    gate['criterion_change'] = dict(previous='stock FP32 AND stock FP64',
        previous_status=json.loads((d.parent/'summary.json').read_text())['status'],
        previous_evidence='../summary.json',
        rationale='stock double parameters still execute RMSNorm, RoPE and attention softmax in FP32; InWiMe executes FP64')
    gate['status'] = 'passed' if gate['passed'] else 'partial'
    gate['preservation_verified'] = True
    return gate


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory')
    p.add_argument('--resolve-numerical-contracts', action='store_true')
    p.add_argument('--output', type=Path)
    a=p.parse_args()
    r=resolved_report(a.directory) if a.resolve_numerical_contracts else report(a.directory)
    destination=a.output or Path(a.directory)/'summary.json'
    with destination.open('x') as f:
        json.dump(r,f,indent=2);f.write('\n')
    print(json.dumps(r,indent=2))
    raise SystemExit(0 if r['status']=='passed' else 1)

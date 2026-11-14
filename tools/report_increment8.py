"""Rebuild the increment8 evidence gate without changing historical evidence."""
import argparse
import json
from pathlib import Path
import re
import statistics


def full_fp32_parity(parity, runtime):
    c = runtime['config']
    expected_keys = {p['name'] for p in runtime['phases']}
    if (not parity['passed'] or (parity['atol'], parity['rtol']) != (2e-3, 2e-4)
            or set(parity['comparisons']) != expected_keys):
        return False
    for phase in runtime['phases']:
        comparison = parity['comparisons'][phase['name']]
        count = len(phase['ids'])
        expected = dict(logits=count*c['vocab_size'],
            hidden=count*c['num_hidden_layers']*c['hidden_size'],
            kv=phase['length']*2*c['num_hidden_layers']*c['num_key_value_heads']*
                (c['hidden_size']//c['num_attention_heads']))
        if not comparison['greedy_equal'] or not comparison['argmax_equal']:
            return False
        if any(comparison[k]['failures'] != 0 or comparison[k]['count'] != n for k, n in expected.items()):
            return False
    return bool(expected_keys)


def report(directory):
    d = Path(directory)
    def read(name):
        return json.loads((d/name).read_text())
    acquisition = read('acquisition.json')
    original = read('smoke-sync-1024.json')
    payload = original['unique_active_bf16_bytes']
    checks = {}
    official = acquisition['official']
    expected_sha = '1193528982f4ac0c0b707ce36fd7dc03a0ef6f3e1a432deb886dce2e90c300c0'
    checks['A_original_public_bf16'] = (
        official['sha'] == 'd4c18ccb8a25e6e637e0609aac08a5d1a0436d1b'
        and official['safetensors']['parameters'] == {'BF16': payload//2}
        and read('integrity-after.json')['original_sha256'] == expected_sha
        and payload > original['rlimit_as_bytes'][0])
    rows = {mode: [read(f'measured-{mode}-{i}.json') for i in range(1, 4)]
            for mode in ('sync', 'pipeline')}
    all_rows = [original, read('strict-sync-128.json'), read('strict-pipeline-128.json'), read('sharded-sync.json')]
    all_rows += [r for values in rows.values() for r in values]
    checks['B_full_prefill_and_three_decodes'] = all(
        [len(p['ids']) for p in r['phases']] == [5, 1, 1, 1]
        and r['coverage_exact'] and r['all_tensors_used']
        and r['stats']['tensor_read_bytes'] == r['expected_tensor_read_bytes']
        and r['stats']['inference_read_bytes'] == sum(r['expected_tensor_read_bytes'].values())
        and all(p['kv_prefix_preserved'] and p['length'] == 5+i and p['generation'] == i+1
                for i, p in enumerate(r['phases']))
        and r['released_fds'] and r['released_arenas'] for r in all_rows)
    checks['B_repeated_outputs_bitwise_exact'] = all(
        [p['digests'] for p in r['phases']] == [p['digests'] for p in original['phases']]
        and r['continuation'] == original['continuation'] for r in all_rows)
    checks['C_bounded_process_below_payload'] = all(
        r['rlimit_as_bytes'][0] < payload
        and r['peak_rss_bytes'] < r['rlimit_as_bytes'][0]
        and r['process_after_close']['VmPeak_bytes'] <= r['rlimit_as_bytes'][0]
        and not r['heavy_modules_loaded'] for r in all_rows)
    parities = {name: read(f'parity-{name}.json') for name in
                ('original-sync', 'original-pipeline', 'extended', 'chunks', 'sharded')}
    runtime_reports = {'original-sync': original, 'original-pipeline': rows['pipeline'][0],
                       'extended': read('extended.json'), 'chunks': read('chunks.json'),
                       'sharded': read('sharded-sync.json')}
    checks['D_full_standard_fp32_parity'] = all(full_fp32_parity(p, runtime_reports[name])
                                               for name, p in parities.items())
    exact = read('exact-invariants.json')
    checks['D_exact_modes_formats_causality'] = exact['passed'] and all(exact['checks'].values())
    faults = read('faults.json')
    checks['E_atomic_KV_and_failures'] = faults['passed'] and len(faults['cases']) == 6 and all(
        all(c[k] for k in ('budget_rejected_before_payload', 'old_kv_all_bytes_preserved',
                          'descriptor_preserved', 'candidate_was_modified', 'drained', 'released'))
        for c in faults['cases'])
    gate7 = read('regression7-gate.json')
    checks['F_preserved_regressions'] = (read('preservation-after.json')['all_consolidated_unchanged']
        and gate7['passed'] and not gate7['contracts']['standard_fp64_diagnostic']['raw_parity_passed']
        and read('regression-tests.json')['exit_code'] == 0
        and int(re.search(r'(\d+) passed', (d/'pytest-final.txt').read_text())[1]) >= 382
        and 'OK' in (d/'oracle-loader-green.txt').read_text()
        and 'OK' in (d/'oracle-precision7-tests.txt').read_text())
    performance = {}
    for mode, values in rows.items():
        supervisors = [read(f'supervisor-{mode}-{i}.json') for i in range(1, 4)]
        phases = [p for r in values for p in r['phases']]
        decodes = [p['wall_s'] for r in values for p in r['phases'][1:]]
        performance[mode] = dict(samples=3,
            prefill_s_median=statistics.median(r['phases'][0]['wall_s'] for r in values),
            decode_s_median=statistics.median(decodes), decode_s_range=[min(decodes), max(decodes)],
            decode_tokens_per_s=1/statistics.median(decodes),
            workload_s_median=statistics.median(sum(p['wall_s'] for p in r['phases']) for r in values),
            peak_rss_bytes=max(r['peak_rss_bytes'] for r in values),
            peak_vmhwm_bytes=max(r['process_after_close']['VmHWM_bytes'] for r in values),
            peak_virtual_bytes=max(r['process_after_close']['VmPeak_bytes'] for r in values),
            sampled_peak_rss_bytes=max(s['sampled_peaks_bytes']['VmRSS'] for s in supervisors),
            sampled_peak_vmhwm_bytes=max(s['sampled_peaks_bytes']['VmHWM'] for s in supervisors),
            sampled_peak_virtual_bytes=max(s['sampled_peaks_bytes']['VmPeak'] for s in supervisors),
            phase_vmsize_bytes_range=[min(p['process']['VmSize_bytes'] for p in phases),
                                     max(p['process']['VmSize_bytes'] for p in phases)],
            arena_plan_bytes=values[0]['arena_plan_bytes'],
            median_open_index_s=statistics.median(r['open_index_s'] for r in values),
            median_sha256_s=statistics.median(r['sha256_s'] for r in values),
            median_startup_full_integrity_s=statistics.median(r['startup_full_integrity_s'] for r in values),
            median_gemv_s=statistics.median(r['stats']['gemv_ns']/1e9 for r in values),
            median_bf16_expand_s=statistics.median(r['stats']['bf16_expand_ns']/1e9 for r in values),
            median_read_service_s=statistics.median(r['stats']['read_service_ns']/1e9 for r in values),
            median_data_wait_s=statistics.median(r['stats']['data_wait_ns']/1e9 for r in values),
            median_overlap_s=statistics.median(r['stats']['overlap_ns']/1e9 for r in values),
            median_host_residual_s=statistics.median(p['host_residual_s'] for p in phases),
            inference_logical_bytes=[r['stats']['inference_read_bytes'] for r in values],
            inference_physical_proc_bytes=[r['inference_proc_io_delta']['read_bytes'] for r in values],
            max_observed_phase_swap_bytes=max(p['process']['VmSwap_bytes'] for p in phases))
    return dict(passed=all(checks.values()), checks=checks,
        declared_parity='full standard Transformers FP32, complete logits/residuals/KV; 1.7B FP64 oracle not executed',
        unique_active_bf16_bytes=payload, performance=performance,
        max_abs={kind: max(c[kind]['max_abs'] for p in parities.values()
                          for c in p['comparisons'].values()) for kind in ('logits', 'hidden', 'kv')},
        limitations=['weights below total WSL and physical host RAM',
                     'buffered OS cache, no controlled cold-NVMe inference',
                     'short context: maximum eight positions, capacity sixteen',
                     '1.7B independent oracle is FP32, not pure FP64',
                     'temperature and CPU power not controlled'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = report(a.directory)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)

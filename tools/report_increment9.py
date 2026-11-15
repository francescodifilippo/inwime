"""Increment-9 gate A-H with PASS / FAIL / NOT PROVEN, recomputed from raw evidence.

Missing evidence is NOT PROVEN, never PASS. Every verdict carries the facts it
used. Exit code 0 only when every condition is PASS.
"""
import argparse
import hashlib
import json
from pathlib import Path

PASS, FAIL, NOT_PROVEN = 'PASS', 'FAIL', 'NOT PROVEN'
GIB2 = 2147483648


def load(directory, name):
    path = directory/name
    return json.loads(path.read_text()) if path.exists() else None


def verdict(ok, facts, missing=False):
    return dict(verdict=NOT_PROVEN if missing else (PASS if ok else FAIL), facts=facts)


def scope_facts(supervisor, runtime):
    fin = (supervisor or {}).get('finalizer') or {}
    events = fin.get('memory.events', {})
    return dict(
        exit_code=supervisor.get('exit_code'), watchdog=supervisor.get('watchdog_reason'),
        memory_max=fin.get('memory.max'), swap_max=fin.get('memory.swap.max'),
        memory_peak=fin.get('memory.peak'), swap_peak=fin.get('memory.swap.peak'),
        oom=events.get('oom'), oom_kill=events.get('oom_kill'), max_events=events.get('max'),
        scope_verified_inside=bool(runtime and runtime['scope'].get('verified_before_opening_weights')),
        global_oom_kill_delta=(supervisor.get('global_oom_kill_after', 0) or 0)
                              - (supervisor.get('global_oom_kill_before', 0) or 0),
        worker_read_bytes=runtime['inference_proc_io_delta']['read_bytes'] if runtime else None,
        startup_read_bytes=runtime['startup_proc_io_delta']['read_bytes'] if runtime else None,
        scope_file_peak=supervisor.get('sampled_peaks', {}).get('stat.file'),
        scope_anon_peak=supervisor.get('sampled_peaks', {}).get('stat.anon'),
        host_disk_delta=supervisor.get('host_disk_delta'))


def scope_ok(f):
    return (f['exit_code'] == 0 and f['watchdog'] is None and f['memory_max'] == GIB2
            and f['swap_max'] == 0 and f['memory_peak'] is not None and f['memory_peak'] <= GIB2
            and f['oom'] == 0 and f['oom_kill'] == 0 and (f['swap_peak'] or 0) == 0
            and f['scope_verified_inside'] and f['global_oom_kill_delta'] == 0)


def residency(supervisor, key):
    values = list((supervisor.get(key) or {}).values())
    return sum(v['resident_bytes'] for v in values), sum(v['size_bytes'] for v in values)


def gate(d):
    inc8 = json.loads(Path('results/increment8/strict-sync-128.json').read_text())
    protocol = load(d, 'protocol.json')
    out = {}
    # A — integrity of originals
    acq, integ = load(d, 'qwen-acquisition.json'), load(d, 'integrity-after.json')
    ra, rb = load(d, 'phaseA-P-runtime.json'), load(d, 'phaseB-sync-runtime.json')
    missing = not all((acq, integ, ra, rb))
    facts = {}
    ok = False
    if not missing:
        shards = {n: v for n, v in acq['downloaded'].items() if n.endswith('.safetensors')}
        facts['qwen_shards_streaming_sha_equals_lfs'] = all(
            v['sha256'] == acq['files'][n]['lfs_sha256'] for n, v in shards.items()) and len(shards) == 4
        facts['final_rehash_all_match'] = integ['all_match']
        facts['runtime_full_integrity_bytes'] = dict(
            smollm=ra['integrity_read_bytes'], qwen=rb['integrity_read_bytes'])
        facts['runtime_integrity_covers_files'] = (
            ra['integrity_read_bytes'] == sum(v['nbytes'] for v in ra['source_files'].values())
            and rb['integrity_read_bytes'] == sum(v['nbytes'] for v in rb['source_files'].values()))
        ok = all(v is True for k, v in facts.items() if k != 'runtime_full_integrity_bytes')
    out['A'] = verdict(ok, facts, missing)

    # B — SmolLM2 under a real 2 GiB scope
    sp, parity_a = load(d, 'phaseA-P-supervisor.json'), load(d, 'phaseA-P-parity-inc8.json')
    missing = not all((sp, ra, parity_a))
    facts, ok = {}, False
    if not missing:
        facts = scope_facts(sp, ra)
        before, size = residency(sp, 'residency_after_eviction')
        facts['resident_bytes_after_eviction'], facts['file_bytes'] = before, size
        facts['digests_equal_increment8'] = [p['digests'] for p in ra['phases']] == [p['digests'] for p in inc8['phases']]
        facts['prefill_tokens'], facts['decodes'] = len(ra['phases'][0]['ids']), len(ra['phases'])-1
        facts['parity_increment8_fp32_oracle'] = parity_a['passed']
        ok = (scope_ok(facts) and before <= size//100 and facts['worker_read_bytes'] > 0
              and facts['digests_equal_increment8'] and parity_a['passed']
              and facts['prefill_tokens'] >= 2 and facts['decodes'] >= 3)
    out['B'] = verdict(ok, facts, missing)

    # C — I/O characterised with correct attribution (warm vs evicted, both present)
    sw, rw = load(d, 'phaseA-W-supervisor.json'), load(d, 'phaseA-W-runtime.json')
    sbs = load(d, 'phaseB-sync-supervisor.json')
    missing = not all((sw, rw, sp, ra, sbs, rb))
    facts, ok = {}, False
    if not missing:
        w, p, b = scope_facts(sw, rw), scope_facts(sp, ra), scope_facts(sbs, rb)
        warm_resident, warm_size = residency(sw, 'residency_after_warm')
        facts = dict(
            warm=dict(resident_before_launch=warm_resident, file_bytes=warm_size,
                      worker_block_read_bytes=w['worker_read_bytes'], startup_block_read_bytes=w['startup_read_bytes'],
                      scope_memory_peak=w['memory_peak'], scope_file_peak=w['scope_file_peak'],
                      logical_inference_bytes=rw['stats']['inference_read_bytes'], host_disk_delta=w['host_disk_delta']),
            evicted=dict(worker_block_read_bytes=p['worker_read_bytes'], startup_block_read_bytes=p['startup_read_bytes'],
                         scope_memory_peak=p['memory_peak'], scope_file_peak=p['scope_file_peak'],
                         logical_inference_bytes=ra['stats']['inference_read_bytes'], host_disk_delta=p['host_disk_delta']),
            qwen=dict(worker_block_read_bytes=b['worker_read_bytes'], startup_block_read_bytes=b['startup_read_bytes'],
                      logical_inference_bytes=rb['stats']['inference_read_bytes'], scope_memory_peak=b['memory_peak'],
                      host_disk_delta=b['host_disk_delta']))
        # Correct attribution: warm pages charged elsewhere are not charged to the scope
        # and are not block reads; evicted pages are re-read and charged under the limit.
        ok = (warm_resident >= warm_size*0.99 and w['worker_read_bytes'] < warm_size//100
              and p['worker_read_bytes'] >= ra['payload_bytes']
              and b['worker_read_bytes'] > 0 and all(x['host_disk_delta'] is not None for x in (w, p, b)))
        facts['note'] = 'host disk counters are global to the Windows physical disk'
    out['C'] = verdict(ok, facts, missing)

    # D — unique active BF16 bytes > MemTotal at run start (independent recomputation)
    pre = load(d, 'qwen-preflight.json')
    missing = not all((rb, pre))
    facts, ok = {}, False
    if not missing:
        width = pre['config']['hidden_size']
        sequence = [t for p in rb['phases'] for t in p['ids']]
        independent = (pre['payload_bytes'] - pre['embedding_bytes'] + len(set(sequence))*width*2)
        facts = dict(memtotal_at_start=rb['meminfo_start']['MemTotal'],
                     runtime_unique_active_bf16_bytes=rb['unique_active_bf16_bytes'],
                     independent_from_official_headers=independent,
                     margin_bytes=independent-rb['meminfo_start']['MemTotal'])
        ok = independent == rb['unique_active_bf16_bytes'] and independent > facts['memtotal_at_start']
    out['D'] = verdict(ok, facts, missing)

    # E — complete forward
    missing = rb is None
    facts, ok = {}, False
    if not missing:
        facts = dict(prefill_tokens=len(rb['phases'][0]['ids']), decodes=len(rb['phases'])-1,
                     all_tensors_used=rb['all_tensors_used'], coverage_exact=rb['coverage_exact'],
                     layers=rb['config']['num_hidden_layers'], continuation=rb['continuation'],
                     generated_text=rb['generated_text'], lengths=[p['length'] for p in rb['phases']],
                     generations=[p['generation'] for p in rb['phases']],
                     released=rb['released_fds'] and rb['released_arenas'])
        ok = (facts['prefill_tokens'] >= 2 and facts['decodes'] >= 1 and facts['all_tensors_used']
              and facts['coverage_exact'] and facts['released']
              and facts['lengths'] == list(range(facts['prefill_tokens'], facts['prefill_tokens']+len(rb['phases']))))
    out['E'] = verdict(ok, facts, missing)

    # F — independent numerical verification, coverage declared
    contracts = {c: load(d, f'parity-{c}.json') for c in ('stock_fp32', 'full64', 'stock_fp64')}
    facts = {c: (None if r is None else dict(passed=r['passed'], role=r['role'], coverage=r['coverage'],
                                             compared=r['compared_elements'],
                                             max_abs={k: max(v[k]['max_abs'] for v in r['comparisons'].values())
                                                      for k in ('logits', 'hidden', 'kv')},
                                             failures={k: sum(v[k]['failures'] for v in r['comparisons'].values())
                                                       for k in ('logits', 'hidden', 'kv')},
                                             upstream_greedy=r['upstream_greedy_matches_sequence']))
             for c, r in contracts.items()}
    checks = (d/'oracle9-checks.txt')
    facts['small_case_checks_ok'] = checks.exists() and checks.read_text().rstrip().endswith('OK')
    facts['phase_a_parity'] = bool(parity_a and parity_a['passed'])
    gating = [contracts['stock_fp32'], contracts['full64']]
    if any(r is None for r in gating):
        out['F'] = verdict(False, facts, missing=True)
    else:
        out['F'] = verdict(all(r['passed'] for r in gating) and facts['small_case_checks_ok']
                           and facts['phase_a_parity'], facts)
    facts['stock_fp64_diagnostic_raw_verdict'] = (None if contracts['stock_fp64'] is None
                                                  else ('PASS' if contracts['stock_fp64']['passed'] else 'FAIL'))

    # G — resources within declared limits, no OOM, no swap attributed
    missing = not all((sbs, rb, sp, sw))
    facts, ok = {}, False
    if not missing:
        b = scope_facts(sbs, rb)
        facts = dict(phase_b=b, phase_b_peak_rss=rb['peak_rss_bytes'], phase_b_vm_peak=rb['process_after_close'].get('VmPeak_bytes'),
                     phase_b_rlimit_as=rb['rlimit_as_bytes'], phase_b_arena_plan=rb['arena_plan_bytes'],
                     phase_b_vmswap=rb['process_after_close'].get('VmSwap_bytes'),
                     phase_a_oom_kill=[scope_facts(s, r)['oom_kill'] for s, r in ((sp, ra), (sw, rw))])
        others = {}
        for label in ('phaseA-W', 'phaseA-P', 'phaseB-sync', 'phaseB-pipeline', 'faults'):
            name = 'supervisor-faults.json' if label == 'faults' else f'{label}-supervisor.json'
            sup = load(d, name)
            fin = (sup or {}).get('finalizer') or {}
            events = fin.get('memory.events', {})
            others[label] = None if sup is None else dict(
                exit_code=sup['exit_code'], memory_peak=fin.get('memory.peak'), memory_max=fin.get('memory.max'),
                oom=events.get('oom'), oom_kill=events.get('oom_kill'), swap_peak=fin.get('memory.swap.peak'),
                global_oom_kill_delta=sup['global_oom_kill_after']-sup['global_oom_kill_before'])
        facts['all_supervised_runs'] = others
        ok = (scope_ok(b) and facts['phase_b_vmswap'] == 0 and facts['phase_a_oom_kill'] == [0, 0]
              and all(r is not None and r['exit_code'] == 0 and r['memory_max'] == GIB2
                      and r['memory_peak'] <= GIB2 and r['oom'] == 0 and r['oom_kill'] == 0
                      and (r['swap_peak'] or 0) == 0 and r['global_oom_kill_delta'] == 0
                      for r in others.values()))
    out['G'] = verdict(ok, facts, missing)

    # H — regressions preserved
    pres, gate8, gate7 = load(d, 'preservation.json'), load(d, 'gate8-rerun.json'), load(d, 'gate7-rerun.json')
    pytest = d/'pytest-final.txt'
    missing = not all((pres, gate8, gate7)) or not pytest.exists()
    facts, ok = {}, False
    if not missing:
        tail = pytest.read_text().strip().splitlines()[-1]
        facts = dict(pytest=tail, preservation=pres['passed'], modified_tracked=pres['modified_tracked'],
                     gate8_all_pass=all(v == 'PASS' for v in gate8['verdicts'].values()) if 'verdicts' in gate8 else gate8.get('passed'),
                     gate7_passed=gate7.get('passed'))
        ok = (' passed' in tail and 'failed' not in tail and 'skipped' not in tail and facts['preservation']
              and facts['gate8_all_pass'] and facts['gate7_passed'])
    out['H'] = verdict(ok, facts, missing)
    return dict(protocol_sha256=hashlib.sha256((d/'protocol.json').read_bytes()).hexdigest() if protocol else None,
                verdicts={k: v['verdict'] for k, v in out.items()}, conditions=out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = gate(a.directory)
    with a.output.open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(report['verdicts'], indent=1))
    raise SystemExit(0 if all(v == PASS for v in report['verdicts'].values()) else 1)

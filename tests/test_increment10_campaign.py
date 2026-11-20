"""Increment 10, campaign and gate tooling (amendment 004): synthetic, no weights, no host access.

tools/campaign10.py: block-device counters, the BLOCKED path (a non-GO host start check
writes blocked-<run>.json and never launches the supervisor), the in-run observer
killing only the run scope on a preregistered kill action.
tools/equivalence10_e2e.py and tools/parity10.py: bit identity / thresholds / binding /
K/V prefix / greedy rules on synthetic arrays, including mutations that must FAIL.
tools/report_increment10.py: verdict algebra (FAIL > BLOCKED > NOT PROVEN > PASS, a
missing and a contradicted field in the same check give FAIL, a blocked file without
cause is NOT PROVEN) and the N8 mutations of criterion D and of the I/O attribution
(W declared as C, 7B bytes, missing RAM, MemTotal as RAM, sustained host paging).
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import campaign10, equivalence10_e2e as e2e, parity10, report_increment10 as g10
from tests.support_increment10_monitor import GiB, MiB, guest, monitor, win

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT/'results'/'increment10'/'protocol.json').read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- campaign10

def test_block_stat_and_delta(tmp_path):
    stat = tmp_path/'stat'
    stat.write_text(' 10 0 2048 5 1 0 8 1 0 7 9\n')
    entry = dict(name='sdx', stat_path=str(stat))
    before = campaign10.block_stat(entry)
    assert before['read_bytes'] == 2048*512 and before['reads_completed'] == 10
    stat.write_text(' 15 0 4096 6 1 0 8 1 0 9 9\n')
    d = campaign10.block_delta(before, campaign10.block_stat(entry))
    assert d['read_bytes'] == 2048*512 and d['reads_completed'] == 5 and 'in_flight' not in d
    assert campaign10.block_stat(dict(name='x', stat_path=str(tmp_path/'absent')))['error']
    assert campaign10.block_delta(dict(error='x'), before) is None


def test_non_go_host_writes_blocked_and_never_launches(tmp_path, monkeypatch):
    baseline = tmp_path/'baseline.json'
    baseline.write_text(json.dumps(dict(after=dict(utc='2026-11-19T21:00:00Z'))))

    def host_check(profile, base, output, checkpoint, backing):
        output.write_text(json.dumps(dict(verdict='DEFER', violated=['host_pagefile_growth_since_baseline'],
                                          missing=[])))
        return 3, json.loads(output.read_text())
    monkeypatch.setattr(campaign10, 'host_check', host_check)
    monkeypatch.setattr(campaign10.subprocess, 'run', lambda *a, **k: pytest.fail('supervisor launched'))
    code = campaign10.main(['--run', 'RX', '--label', 'x', '--kind', 'runtime', '--profile', 'runtime',
                            '--cache', 'C', '--purpose', 'p', '--condition', 'evicted', '--memory-max', '1',
                            '--output-dir', str(tmp_path/'c'), '--baseline', str(baseline),
                            '--checkpoint-volume', str(tmp_path), '--', 'true'])
    assert code == 3
    blocked = json.loads((tmp_path/'c'/'blocked-RX.json').read_text())
    assert blocked['cause'] == 'host start check not GO' and blocked['verdict'] == 'DEFER' and blocked['utc']
    assert not (tmp_path/'c'/'RX-run.json').exists()
    with pytest.raises(FileExistsError):              # never repeated under the same run id
        campaign10.main(['--run', 'RX', '--label', 'x', '--kind', 'runtime', '--profile', 'runtime',
                         '--cache', 'C', '--purpose', 'p', '--condition', 'evicted', '--memory-max', '1',
                         '--output-dir', str(tmp_path/'c'), '--baseline', str(baseline),
                         '--checkpoint-volume', str(tmp_path), '--', 'true'])


def test_observer_kills_only_the_run_scope(tmp_path, monkeypatch):
    app = tmp_path/'app.slice'
    mine, other = app/'inwime9-lab-1.scope', app/'inwime9-other-1.scope'
    for d in (mine, other):
        d.mkdir(parents=True)
        (d/'cgroup.kill').write_text('')
    monkeypatch.setattr(campaign10, 'USER_APP', app)
    policy = campaign10.RUN_POLICY
    m = monitor(lambda i, t: win(t, committed=50*GiB - MiB), tmp_path, tmp_path, policy=policy)
    obs = campaign10.Observer(m, 'lab')
    obs.monitor.start()
    obs.start_decision = {'stop': False}
    # one poll cycle as in Observer.run
    obs.monitor.poll()
    kill = [a for a in obs.monitor.annotations if a.get('applied') == 'kill-scope']
    assert kill and kill[0]['quantity'] == 'host_commit_headroom_bytes'
    obs.stop_event.set()
    scope = obs.scope()
    assert scope == mine
    (scope/'cgroup.kill').write_text('1')
    assert (mine/'cgroup.kill').read_text() == '1' and (other/'cgroup.kill').read_text() == ''


def test_run_policy_maps_every_frozen_action():
    actions = {c['action'] for c in PROTOCOL['host_memory_preflight']['abort_criteria']}
    assert set(campaign10.RUN_POLICY['action_mapping']) == actions
    assert campaign10.RUN_POLICY['action_mapping']['kill-scope-invalidate-run'] == 'kill-scope'


# ---------------------------------------------------------------- equivalence10_e2e

def arrays(seed=0, positions=3, layers=2, width=4, vocab=7, kvw=2):
    r = np.random.default_rng(seed)
    return dict(logits=r.standard_normal((positions, vocab)), hidden=r.standard_normal((layers, positions, width)),
                kv=r.standard_normal((layers, 2, positions, kvw)))


def save(path, a, meta):
    with path.open('wb') as f:
        np.savez_compressed(f, **{f'full.{k}': v for k, v in a.items()})
    meta = dict(meta, capture_sha256=sha(path))
    path.with_suffix('.json').write_text(json.dumps(meta))
    return meta


META = dict(dtype='float64', precision_profile='full64', revision='r', sequence=[1, 2, 3], prompt_length=2,
            chunks=[2, 1], head_mode_is_preregistered=True, upstream_greedy_matches_sequence=True)


def e2e_case(tmp_path, contract, mutate=None, meta_change=None):
    old = arrays()
    new = copy.deepcopy(old)
    if mutate:
        mutate(new)
    meta_old = dict(META, contract=contract, argmax=old['logits'].argmax(-1).tolist())
    meta_new = dict(meta_old, argmax=new['logits'].argmax(-1).tolist(), **(meta_change or {}))
    mo = save(tmp_path/'old.npz', old, meta_old)
    mn = save(tmp_path/'new.npz', new, meta_new)
    return e2e.evaluate(tmp_path/'new.npz', mn, tmp_path/'old.npz', mo, PROTOCOL)


@pytest.mark.parametrize('contract', ['stock_fp32', 'full64'])
def test_e2e_identical_passes(tmp_path, contract):
    r = e2e_case(tmp_path, contract)
    assert r['verdict'] == 'PASS' and r['coverage'] == 'FULL'


@pytest.mark.parametrize('contract,mutate,verdict', [
    ('stock_fp32', lambda a: a['logits'].__setitem__((0, 0), np.nextafter(a['logits'][0, 0], 9)), 'FAIL'),
    ('full64', lambda a: a['logits'].__setitem__((0, 0), a['logits'][0, 0] + 1e-10), 'PASS'),
    ('full64', lambda a: a['logits'].__setitem__((0, 0), a['logits'][0, 0] + 1e-8), 'FAIL'),
    ('full64', lambda a: a['hidden'].__setitem__((0, 0, 0), np.nextafter(a['hidden'][0, 0, 0], 9)), 'FAIL'),
    ('stock_fp32', lambda a: a['kv'].__setitem__((1, 1, 2, 1), np.nextafter(a['kv'][1, 1, 2, 1], -9)), 'FAIL'),
])
def test_e2e_rules(tmp_path, contract, mutate, verdict):
    assert e2e_case(tmp_path, contract, mutate)['verdict'] == verdict


@pytest.mark.parametrize('change', [dict(sequence=[1, 2, 4]), dict(chunks=[3]), dict(revision='x'),
                                    dict(precision_profile='stock'), dict(head_mode_is_preregistered=False),
                                    dict(upstream_greedy_matches_sequence=False)])
def test_e2e_identity_mismatch_fails(tmp_path, change):
    r = e2e_case(tmp_path, 'full64', meta_change=change)
    assert r['verdict'] == 'FAIL' and r['problems']


def test_e2e_capture_sha_mismatch_fails(tmp_path):
    old = arrays()
    mo = save(tmp_path/'old.npz', old, dict(META, contract='full64', argmax=[0]))
    mn = dict(mo, capture_sha256='0'*64)
    save(tmp_path/'new.npz', old, mn)
    r = e2e.evaluate(tmp_path/'new.npz', mn, tmp_path/'old.npz', mo, PROTOCOL)
    assert r['verdict'] == 'FAIL' and any('SHA256' in p for p in r['problems'])


# ---------------------------------------------------------------- parity10

CFG = dict(num_hidden_layers=2, hidden_size=4, vocab_size=7, num_attention_heads=2, num_key_value_heads=1)


def parity_case(tmp_path, contract='full64', mutate_runtime=None, break_prefix=False, continuation=None,
                drop=None, tamper=None):
    ref = arrays(1, positions=4, kvw=2)
    sequence, chunks = [3, 1, 4, 5], [2, 1, 1]
    greedy = ref['logits'].argmax(-1).tolist()
    act = copy.deepcopy(ref)
    if mutate_runtime:
        mutate_runtime(act)
    cap = tmp_path/'cap'
    cap.mkdir()
    phases, captures, start = [], {}, 0
    for i, size in enumerate(chunks):
        name = f'decode.{i}'
        end = start + size
        kv = act['kv'][:, :, :end].copy()
        if break_prefix and i == 1:
            kv[0, 0, 0, 0] += 1.0
        parts = dict(logits=act['logits'][start:end], hidden=act['hidden'][:, start:end], kv=kv)
        captures[name] = {}
        for kind, values in parts.items():
            path = cap/f'{name}.{kind}.f64'
            np.ascontiguousarray(values, dtype='<f8').tofile(path)
            captures[name][kind] = dict(file=path.name, shape=list(values.shape), dtype='<f8',
                                        nbytes=path.stat().st_size, sha256=sha(path))
        phases.append(dict(name=name, ids=sequence[start:end]))
        start = end
    if drop:
        (cap/drop).unlink()
    if tamper:
        with (cap/tamper).open('r+b') as f:
            f.write(b'\x00'*8)
    forward = dict(phases=phases, captures=captures, config=CFG,
                   continuation=continuation if continuation is not None else greedy[1:])
    meta = save(tmp_path/'oracle.npz', ref, dict(contract=contract, sequence=sequence, chunks=chunks,
                                                 head_mode_is_preregistered=True))
    return parity10.evaluate(forward, cap, tmp_path/'oracle.npz', meta)


@pytest.mark.parametrize('contract', ['stock_fp32', 'full64'])
def test_parity_identical_passes_full(tmp_path, contract):
    r = parity_case(tmp_path, contract)
    assert r['verdict'] == 'PASS' and r['coverage'] == 'FULL' and r['greedy_equal']
    assert [x['length'] for x in r['kv_prefixes']] == [2, 3, 4]
    assert r['comparison']['kinds']['logits']['elements'] == 28


@pytest.mark.parametrize('kw,needle', [
    (dict(mutate_runtime=lambda a: a['hidden'].__setitem__((1, 3, 2), a['hidden'][1, 3, 2] + 1e-3)), None),
    (dict(break_prefix=True), 'prefix'),
    (dict(continuation=[0, 0, 0]), 'greedy'),
    (dict(drop='decode.2.kv.f64'), 'missing capture'),
    (dict(tamper='decode.0.logits.f64'), 'differs from the runtime record'),
])
def test_parity_mutations_fail(tmp_path, kw, needle):
    r = parity_case(tmp_path, 'full64', **kw)
    assert r['verdict'] == 'FAIL'
    if needle:
        assert any(needle in p for p in r['problems']), r['problems']


def test_parity_fp32_tolerance_is_the_preregistered_one(tmp_path):
    bump = lambda a: a['logits'].__setitem__((0, 1), a['logits'][0, 1] + 1e-3)   # inside FP32, outside FP64
    assert parity_case(tmp_path/'a', 'stock_fp32', mutate_runtime=bump)['verdict'] == 'PASS'
    assert parity_case(tmp_path/'b', 'full64', mutate_runtime=bump)['verdict'] == 'FAIL'


@pytest.fixture(autouse=True)
def _mk(tmp_path):
    for d in ('a', 'b'):
        (tmp_path/d).mkdir()


# ---------------------------------------------------------------- report_increment10: verdict algebra

def test_precedence_and_settle():
    assert g10.combine('PASS', 'NOT PROVEN', 'BLOCKED') == 'BLOCKED'
    assert g10.combine('BLOCKED', 'FAIL') == 'FAIL' and g10.combine('weird') == 'FAIL'

    def missing():
        raise g10.Missing('x')

    def blocked():
        raise g10.Blocked('y')
    with pytest.raises(g10.Blocked):
        g10.settle(dict(a=lambda: True, b=missing, c=blocked))
    out = g10.settle(dict(a=lambda: False, b=missing, c=blocked))      # mixed: contradicted + missing
    assert out == dict(a=False, b=None, c=None)


def blocked_evidence(tmp_path, with_cause=True):
    c = tmp_path/'campaign'
    c.mkdir()
    (c/'blocked-R1.json').write_text(json.dumps(dict(cause='host start check not GO', utc='2026-11-19T00:00:00Z')
                                                if with_cause else dict(note='x')))
    return g10.Evidence(campaign=c, artifacts=tmp_path/'a', root=ROOT)


def test_blocked_run_makes_condition_blocked(tmp_path):
    r = g10.gate(blocked_evidence(tmp_path), only=['D'])
    assert r['verdicts'] == dict(D='BLOCKED') and 'host start check not GO' in r['conditions']['D']['facts']['blocked']


def test_blocked_file_without_cause_is_not_proven(tmp_path):
    assert g10.gate(blocked_evidence(tmp_path, with_cause=False), only=['D'])['verdicts'] == dict(D='NOT PROVEN')


def test_no_evidence_is_never_pass(tmp_path):
    (tmp_path/'campaign').mkdir()
    e = g10.Evidence(campaign=tmp_path/'campaign', artifacts=tmp_path/'a', root=ROOT)
    r = g10.gate(e, only=list('ABCDEG'))
    assert all(v == 'NOT PROVEN' for v in r['verdicts'].values()), r['verdicts']


# ---------------------------------------------------------------- N8: criterion D and attribution mutations

def d_evidence(tmp_path, **mutations):
    """Minimal R1 evidence for criterion D, values as the real 14B run would record them."""
    c = tmp_path/'campaign'
    c.mkdir(exist_ok=True)
    seq_phases = [dict(ids=[785, 6722, 315, 9625, 374]), dict(ids=[12095]), dict(ids=[13]), dict(ids=[576])]
    aw = PROTOCOL['active_weights']
    forward = dict(phases=seq_phases, unique_active_bf16_bytes=aw['unique_bytes_8_distinct_rows'],
                   inference_proc_io_delta=dict(read_bytes=222_550_794_240),
                   stats=dict(inference_read_bytes=aw['logical_bytes_8_positions']),
                   meminfo_start=dict(MemTotal=11_483_848_704), all_tensors_used=True,
                   expected_tensor_read_bytes={str(i): 1 for i in range(579)})
    run = dict(cache='C', condition='evicted')
    host = dict(after=dict(windows=dict(TotalPhysicalMemory=16_900_739_072, DimmBytes=17_179_869_184)))
    for key, fn in mutations.items():
        fn(dict(forward=forward, run=run, host=host))
    (c/'R1-forward.json').write_text(json.dumps(forward))
    (c/'R1-run.json').write_text(json.dumps(run))
    (c/'R1-host-start.json').write_text(json.dumps(host))
    return g10.Evidence(campaign=c, artifacts=tmp_path/'a', root=ROOT)


def test_d_passes_on_consistent_evidence(tmp_path):
    r = g10.gate(d_evidence(tmp_path), only=['D'])
    assert r['verdicts'] == dict(D='PASS'), r['conditions']['D']
    assert r['conditions']['D']['facts']['ratio_host_ram'] > 1.62


@pytest.mark.parametrize('name,mutation,verdict', [
    ('warm_as_cold', lambda x: x['run'].update(cache='W', condition='as-is'), 'FAIL'),
    ('bytes_of_7b', lambda x: x['forward'].update(unique_active_bf16_bytes=14_141_295_616), 'FAIL'),
    ('missing_ram', lambda x: x['host']['after']['windows'].pop('TotalPhysicalMemory'), 'NOT PROVEN'),
    ('memtotal_as_ram', lambda x: x['host']['after']['windows'].update(TotalPhysicalMemory=11_483_848_704), 'FAIL'),
    ('not_really_read', lambda x: x['forward']['inference_proc_io_delta'].update(read_bytes=10**9), 'FAIL'),
    ('ram_as_float', lambda x: x['host']['after']['windows'].update(TotalPhysicalMemory=16_900_739_072.0), 'FAIL'),
    ('missing_and_contradicted', lambda x: (x['host']['after']['windows'].pop('DimmBytes'),
                                            x['run'].update(cache='W')), 'FAIL'),
])
def test_d_mutations(tmp_path, name, mutation, verdict):
    assert g10.gate(d_evidence(tmp_path, m=mutation), only=['D'])['verdicts'] == dict(D=verdict)


def test_sustained_paging_invalidates_attribution_only():
    rec = dict(observer=dict(summary=dict(annotations=[dict(preregistered_action='annotate-io-attribution-invalid',
                                                              quantity='host_pages_input_per_s')])))
    assert g10.io_invalidated(rec)
    assert not g10.io_invalidated(dict(observer=dict(summary=dict(annotations=[]))))


@pytest.mark.parametrize('summary,kills,ok', [
    (dict(read_only=True, counts=dict(valid=10), stop_reasons=[]), [], True),
    (dict(read_only=True, counts=dict(valid=0), stop_reasons=[]), [], False),
    (dict(read_only=True, counts=dict(valid=10), stop_reasons=['host not monitored: x']), [], False),
    (dict(read_only=True, counts=dict(valid=10), stop_reasons=[]), [dict(scope='s')], False),
])
def test_observer_validity(summary, kills, ok):
    assert g10.observer_ok(dict(observer=dict(summary=summary, kills=kills))) is ok


def test_windows_disk_selects_c_only():
    assert g10.windows_disk({'0 C:': dict(DiskReadBytesPersec=5), '1 D:': dict(DiskReadBytesPersec=9)}) == 5
    assert g10.windows_disk({'1 D:': dict(DiskReadBytesPersec=9)}) is None
    assert g10.windows_disk(None) is None
    assert g10.windows_disk({'0 C:': dict(DiskReadBytesPersec=5.0)}) is None



# ---------------------------------------------------------------- fault10 (N1-N4)

def test_fault10_rlimit_per_mode_is_the_preregistered_one():
    """The unchanged M9 probe uses the real 14B/7B token IDs (785, 6722), so it runs only on real
    checkpoints (F-sync / F-pipeline evidence); here the per-mode limits are bound to the protocol."""
    from tools import fault10
    assert fault10.RLIMIT == PROTOCOL['memory_limits']['rlimit_as_bytes'] == g10.RLIMIT
    assert fault10.FAULTS == ('io', 'short', 'cancel')


def test_digest_only_runs_replace_capture_checks_without_weakening():
    """R2-R4 carry digests only (protocol io_protocol); a capture-less run cannot pass as captured,
    and a non-SHA256 digest or a declared capture on a digest-only run is a FAIL."""
    real = ROOT/'results'/'increment10'/'campaign'/'R2-forward.json'
    if not real.is_file():
        pytest.skip('campaign evidence R2-forward.json not present')
    r = json.loads(real.read_text())
    assert g10.true(g10.runtime_forward(r, captured=False))
    with pytest.raises(KeyError):                       # treated as captured: the capture files are missing
        g10.runtime_forward(r, captured=True)
    bad = copy.deepcopy(r)
    bad['phases'][2]['digests']['kv'] = 'not-a-digest'
    assert g10.runtime_forward(bad, captured=False)['phase_digests_are_sha256'] is False
    declared = copy.deepcopy(r)
    declared['captures'] = {'decode.0': {}}
    assert g10.runtime_forward(declared, captured=False)['phase_names'] is False


def test_fault_position_is_midway_through_layer0_o_proj():
    """Synthetic catalogs with the 7B and 14B layer-0 shapes: the fault lands after the K/V writes."""
    from types import SimpleNamespace
    from tools import fault10

    def catalog(width, kv_width):
        el = {'layers.0.norm1': width, 'layers.0.q': width*width, 'layers.0.q.bias': width,
              'layers.0.k': kv_width*width, 'layers.0.k.bias': kv_width, 'layers.0.v': kv_width*width,
              'layers.0.v.bias': kv_width, 'layers.0.o': width*width}
        return SimpleNamespace(tensors={n: SimpleNamespace(elements=e) for n, e in el.items()})
    assert fault10.fault_read(catalog(3584, 512), 262144) == (92, 68)       # 7B: M9's read 100 is also in o_proj
    assert fault10.fault_read(catalog(5120, 1024), 262144) == (195, 145)    # 14B: read 100 would be inside q_proj
    at, before_o = fault10.fault_read(catalog(5120, 1024), 262144)
    assert 100 < before_o < at                                               # after q, k and v of layer 0


# ---------------------------------------------------------------- amendment 006: review findings

def samples(rates, step=5.0):
    return [dict(t=i*step, host_pages_input_per_s=r) for i, r in enumerate(rates)]


@pytest.mark.parametrize('rates,invalid', [
    ([100, 3000, 1300, 3000, 100], False),               # transient spikes (as in R1)
    ([100, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 100], True),   # > 2048 for >= 30 s
    ([None, 2100, 2100, 2100, 2100, 2100, 2100, 2100], True),
    ([2100, 2100, 2100, 2100, 2100, 2100], False),        # 25 s only
])
def test_paging_rule_recomputed_from_samples(rates, invalid):
    rec = dict(observer=dict(summary=dict(annotations=[], last_samples=samples(rates))))
    assert bool(g10.io_invalidated(rec)) is invalid


def test_attempts_are_ordered_numerically_and_unjustified_retries_fail(tmp_path):
    c = tmp_path/'campaign'
    c.mkdir()
    for name in ('O14-full64', 'O14-full64-a2', 'O14-full64-a10'):
        (c/f'{name}-run.json').write_text('{}')
    e = g10.Evidence(campaign=c, artifacts=tmp_path/'a', root=ROOT)
    assert e.attempts('O14-full64') == ['O14-full64', 'O14-full64-a2', 'O14-full64-a10']
    assert e.latest('O14-full64') == 'O14-full64-a10'
    f = {}
    assert g10.retries_justified(e, f) is False             # repeated until pass, no amendment names it
    assert f['attempts_with_records'] == {'O14-full64': ['O14-full64', 'O14-full64-a2', 'O14-full64-a10']}
    d = tmp_path/'campaign2'
    d.mkdir()
    for name in ('F-sync', 'F-sync-a2'):
        (d/f'{name}-run.json').write_text('{}')
    assert g10.retries_justified(g10.Evidence(campaign=d, artifacts=tmp_path/'a', root=ROOT), {}) is True


def test_produced_files_bound_directly_or_through_the_sanitization_map(tmp_path):
    c = tmp_path/'campaign'
    c.mkdir()
    (c/'X-scope.json').write_text('{"a": "<recovery>/py"}')
    (c/'X-forward.json').write_text('{"b": 1}')
    original = hashlib.sha256(b'{"a": "/somewhere/py"}').hexdigest()
    rec = dict(produced_sha256={'X-scope.json': original, 'X-forward.json': sha(c/'X-forward.json')})
    e = g10.Evidence(campaign=c, artifacts=tmp_path/'a', root=ROOT)
    assert g10.produced_bound(e, 'X', rec) == {'X-scope.json': False, 'X-forward.json': True}
    (c/'sanitization-post-write.json').write_text(json.dumps({'X-scope.json': dict(
        original_sha256=original, sanitized_sha256=sha(c/'X-scope.json'))}))
    e = g10.Evidence(campaign=c, artifacts=tmp_path/'a', root=ROOT)
    assert all(g10.produced_bound(e, 'X', rec).values())
    (c/'X-forward.json').write_text('{"b": 2}')            # edited after the run: detected
    assert g10.produced_bound(e, 'X', rec)['X-forward.json'] is False


# ---------------------------------------------------------------- amendment 007: diagnostic 7B equivalence in the gate

def e2e_evidence(tmp_path, contract, mutate=None, bind_new=True):
    """Synthetic 7B end-to-end evidence laid out as the gate reads it: oracle10 capture bound to an
    O7 run record, oracle9 capture bound to an increment-9 capture manifest, stored comparison."""
    root, c, a = tmp_path/'root', tmp_path/'campaign', tmp_path/'art'
    for d in (root/'artifacts'/'increment9', root/'results'/'increment9', root/'results'/'increment10', c, a):
        d.mkdir(parents=True, exist_ok=True)
    (root/'results'/'increment10'/'protocol.json').write_text(json.dumps(PROTOCOL))
    old = arrays()
    new = copy.deepcopy(old)
    if mutate:
        mutate(new)
    meta_old = dict(META, contract=contract, argmax=old['logits'].argmax(-1).tolist())
    oldp, newp = root/'artifacts'/'increment9'/f'oracle-{contract}.npz', a/f'oracle10-7b-{contract}.npz'
    mo = save(oldp, old, meta_old)
    mn = save(newp, new, dict(meta_old, argmax=new['logits'].argmax(-1).tolist()))
    (root/'results'/'increment9'/'capture-manifest.json').write_text(json.dumps(
        {'files': {f'artifacts/increment9/oracle-{contract}.npz': dict(sha256=sha(oldp))}}))
    (c/f'O7-{contract}-a3-run.json').write_text(json.dumps({'produced_sha256': {
        f'artifacts/increment10/oracle10-7b-{contract}.npz': sha(newp) if bind_new else '0'*64}}))
    stored = e2e.evaluate(newp, mn, oldp, mo, PROTOCOL)
    stored['inputs'] = dict(new_sha256=sha(newp), old_sha256=sha(oldp))
    stored = json.loads(json.dumps(stored))
    (c/f'equivalence-7b-e2e-{contract}.json').write_text(json.dumps(stored))
    return g10.Evidence(campaign=c, artifacts=a, root=root), stored


@pytest.mark.parametrize('mutate,verdict', [
    (None, 'PASS'),
    (lambda a: a['hidden'].__setitem__((1, 2, 3), a['hidden'][1, 2, 3] + 1e-6), 'FAIL'),
])
def test_diagnostic_7b_equivalence_is_reported_raw(tmp_path, mutate, verdict):
    e, stored = e2e_evidence(tmp_path, 'stock_fp64', mutate)
    assert stored['verdict'] == verdict
    assert g10.e2e_recomputed(e, 'stock_fp64', stored) is True     # raw verdict, recomputed == stored


@pytest.mark.parametrize('tamper', ['verdict', 'comparison', 'unbound_capture', 'input_digest'])
def test_diagnostic_7b_equivalence_contradicted_is_false(tmp_path, tamper):
    e, stored = e2e_evidence(tmp_path, 'stock_fp64', bind_new=tamper != 'unbound_capture')
    if tamper == 'verdict':
        stored['verdict'] = 'FAIL'
    elif tamper == 'comparison':
        stored['comparison']['kinds']['logits']['max_abs_difference'] = 1e-3
    elif tamper == 'input_digest':
        stored['inputs']['new_sha256'] = '0'*64
    assert g10.e2e_recomputed(e, 'stock_fp64', stored) is False


def test_gating_7b_equivalence_still_requires_pass(tmp_path):
    e, stored = e2e_evidence(tmp_path, 'full64', lambda a: a['logits'].__setitem__((0, 0), a['logits'][0, 0] + 1e-8))
    assert stored['verdict'] == 'FAIL'
    assert g10.e2e_recomputed(e, 'full64', stored) is False


def test_diagnostic_7b_check_exists_only_with_a_run_record(tmp_path):
    c = tmp_path/'campaign'
    c.mkdir()
    e = g10.Evidence(campaign=c, artifacts=tmp_path/'a', root=ROOT)
    assert '7b_e2e.stock_fp64.reported_raw' not in g10.gate(e, only=['F'])['conditions']['F']['facts']['checks']
    (c/'blocked-O7-stock_fp64.json').write_text(json.dumps(dict(cause='host not GO', utc='2026-11-20T00:00:00Z')))
    e = g10.Evidence(campaign=c, artifacts=tmp_path/'a', root=ROOT)     # blocked only: still no check
    assert '7b_e2e.stock_fp64.reported_raw' not in g10.gate(e, only=['F'])['conditions']['F']['facts']['checks']
    (c/'O7-stock_fp64-a3-run.json').write_text('{}')
    e = g10.Evidence(campaign=c, artifacts=tmp_path/'a', root=ROOT)
    r = g10.gate(e, only=['F'])
    assert '7b_e2e.stock_fp64.reported_raw' in r['conditions']['F']['facts']['checks']
    assert r['verdicts']['F'] != 'PASS'

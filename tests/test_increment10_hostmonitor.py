"""Increment 10, amendment 002: read-only in-run host monitor during the acquisition.

Review of PR #24 at e8c588f: the start GO can go stale while 29.55 GB flow. The
monitor applies the frozen abort criteria of protocol.json during the transfer and
stops it cooperatively, keeping the owned .part for a later resume. Covered here:
GO -> critical host pressure during the first and during a later file, each frozen
criterion, sustain windows, annotation-only criterion, sampling errors (error lines,
unreadable values, dead sampler, no first sample, sampler that cannot start), bounds,
resume after a stop, and the real subprocess stream with a fake sampler.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import pytest

from tools import acquire10 as acq
from tools import hostmem10
from tools import hostmonitor10 as hmon
from tools import protocol10
from tests.support_increment10_monitor import (ABORT, BASELINE, POLICY, GiB, MiB, ROOMY, FakeClock, Usage,
                                               guest, healthy, monitor, win)
from tests.test_increment10_acquire import FakeHF, GoGate, authorize, build_repo, small_reserves

ROOT = Path(__file__).resolve().parents[1]
SHARD1, SHARD2 = 'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors'


def drive(m, n):
    """start() then n polls; returns the list of decisions."""
    out = [m.start()]
    for _ in range(n):
        out.append(m.poll())
    return out


def stop_index(decisions):
    return next((i for i, d in enumerate(decisions) if d['stop']), None)


# ---------------------------------------------------------------- policy

def test_policy_is_preregistered_and_maps_no_stop_to_annotation():
    assert POLICY['criteria'] == 'host_memory_preflight.abort_criteria'
    assert POLICY['action_mapping'] == {'pause-and-annotate': 'stop', 'kill-scope-invalidate-run': 'stop',
                                        'stop-acquisition-retain-part': 'stop',
                                        'annotate-io-attribution-invalid': 'annotate'}
    assert (POLICY['sample_interval_s'], POLICY['max_unmonitored_s'], POLICY['startup_timeout_s']) == (5, 20, 60)
    assert protocol10.check()['passed']


def test_frozen_abort_criteria_are_the_ones_applied():
    frozen = json.loads((ROOT/'results'/'increment10'/'protocol.json').read_text())
    assert ABORT == frozen['host_memory_preflight']['abort_criteria']
    assert {(c['quantity'], c['op'], c['threshold'], c.get('sustain_s')) for c in ABORT} == {
        ('host_available_bytes', '>=', 300_000_000, 10), ('guest_mem_available', '>=', 768*MiB, None),
        ('host_pagefile_growth_since_baseline', '<=', 4*GiB, None), ('host_commit_headroom_bytes', '>=', GiB, None),
        ('host_pages_input_per_s', '<=', 2048, 30), ('backing_volume_free', '>=', 40*GiB, None)}


def test_missing_action_mapping_is_rejected():
    bad = dict(POLICY, action_mapping={k: v for k, v in POLICY['action_mapping'].items()
                                       if k != 'kill-scope-invalidate-run'})
    with pytest.raises(ValueError, match='kill-scope'):
        monitor(lambda i, t: win(t), 'd', 'b', policy=bad)


# ---------------------------------------------------------------- GO -> critical (unit)

def test_healthy_host_never_stops(tmp_path):
    m = healthy(tmp_path, tmp_path)
    decisions = drive(m, 40)
    assert stop_index(decisions) is None
    s = m.summary()
    assert s['counts']['valid'] == 41 and s['counts']['errors'] == 0 and not s['stopped']


def test_available_below_300mb_stops_only_when_sustained_10s(tmp_path):
    # GO for 4 samples, then 200 MB: violations at t=25, 30, 35 -> sustained 10 s at t=35
    m = monitor(lambda i, t: win(t, available=4*GiB if i < 5 else 200*10**6), tmp_path, tmp_path)
    decisions = drive(m, 20)
    assert stop_index(decisions) == 7
    assert 'host_available_bytes' in decisions[7]['reasons'][0] and 'pause-and-annotate' in decisions[7]['reasons'][0]
    record = m.summary()['annotations'][-1]
    assert record['applied'] == 'stop' and record['preregistered_action'] == 'pause-and-annotate'


def test_transient_dip_below_300mb_does_not_stop(tmp_path):
    m = monitor(lambda i, t: win(t, available=200*10**6 if i in (5, 6) else 4*GiB), tmp_path, tmp_path)
    decisions = drive(m, 20)
    assert stop_index(decisions) is None
    assert m.summary()['counts']['transient_violations'] == 2


@pytest.mark.parametrize('name,sample,guest_mem', [
    ('guest_mem_available', dict(), 700*MiB),
    ('host_pagefile_growth_since_baseline', dict(pf_mb=7000 + 4097), 7*GiB),
    ('host_commit_headroom_bytes', dict(committed=50*GiB - GiB + 1), 7*GiB),
])
def test_immediate_criteria_stop_on_first_violating_sample(tmp_path, name, sample, guest_mem):
    m = monitor(lambda i, t: win(t, **(sample if i >= 3 else {})), tmp_path, tmp_path,
                guest_reader=lambda: guest(guest_mem if m.source.index > 3 else 7*GiB))
    decisions = drive(m, 10)
    assert stop_index(decisions) == 3
    assert decisions[3]['reasons'][0].startswith(name)


def test_backing_volume_below_40gib_stops(tmp_path):
    calls = []

    def usage(path):
        calls.append(path)
        return Usage(10**15, 0, 40*GiB - 1 if len(calls) > 8 else 10**12)
    m = monitor(lambda i, t: win(t), tmp_path, tmp_path, usage=usage)
    decisions = drive(m, 10)
    i = stop_index(decisions)
    assert i is not None and 'backing_volume_free' in decisions[i]['reasons'][0]
    assert 'stop-acquisition-retain-part' in decisions[i]['reasons'][0]


def test_paging_criterion_is_annotation_only(tmp_path):
    total = [0]

    def script(i, t):                     # 5000 pages/s for samples 2..13, idle otherwise (counter monotonic)
        total[0] += 5000*5 if 2 <= i < 14 else 0
        return win(t, pages_total=total[0])
    m = monitor(script, tmp_path, tmp_path)
    decisions = drive(m, 30)
    assert stop_index(decisions) is None
    notes = [a for a in m.summary()['annotations'] if a['quantity'] == 'host_pages_input_per_s']
    assert len(notes) == 1 and notes[0]['applied'] == 'annotate'      # once per episode


# ---------------------------------------------------------------- sampling errors

def test_error_lines_lead_to_stop_after_max_unmonitored(tmp_path):
    m = monitor(lambda i, t: win(t) if i < 3 else 'error', tmp_path, tmp_path)
    decisions = drive(m, 10)
    i = stop_index(decisions)
    # last valid sample at t=15; stop when the clock passes 15 + 20
    assert i == 7 and 'not monitored' in decisions[i]['reasons'][0]
    assert m.summary()['counts']['errors'] >= 4


def test_unreadable_values_make_samples_invalid(tmp_path):
    bad = lambda t: {k: v for k, v in win(t).items() if k != 'AvailableBytes'}
    m = monitor(lambda i, t: win(t) if i < 2 else bad(t), tmp_path, tmp_path)
    decisions = drive(m, 10)
    assert stop_index(decisions) is not None
    s = m.summary()
    assert s['counts']['invalid'] >= 4 and 'host_available_bytes' in s['errors'][0]['error']


def test_unreadable_guest_makes_samples_invalid(tmp_path):
    m = monitor(lambda i, t: win(t), tmp_path, tmp_path,
                guest_reader=lambda: dict(meminfo=None, psi=None) if m.source.index > 2 else guest())
    decisions = drive(m, 10)
    i = stop_index(decisions)
    assert i is not None and 'not monitored' in decisions[i]['reasons'][0]


def test_silent_sampler_stops(tmp_path):
    m = monitor(lambda i, t: win(t) if i < 2 else None, tmp_path, tmp_path)
    decisions = drive(m, 10)
    assert 'not monitored' in decisions[stop_index(decisions)]['reasons'][0]


def test_dead_sampler_stops_immediately(tmp_path):
    m = monitor(lambda i, t: win(t) if i < 3 else 'eof', tmp_path, tmp_path)
    decisions = drive(m, 5)
    assert stop_index(decisions) == 3 and 'sampler stopped' in decisions[3]['reasons'][0]


def test_no_first_sample_within_startup_timeout(tmp_path):
    m = monitor(lambda i, t: 'error', tmp_path, tmp_path)
    d = m.start()
    assert d['stop'] and 'within 60 s' in d['reasons'][0]


def test_first_sample_already_critical_stops_at_start(tmp_path):
    m = monitor(lambda i, t: win(t, committed=50*GiB), tmp_path, tmp_path)
    assert m.start()['stop']


def test_sampler_that_cannot_start(tmp_path):
    m = monitor(lambda i, t: win(t), tmp_path, tmp_path)

    def broken():
        raise FileNotFoundError('powershell.exe')
    m.source.start = broken
    d = m.start()
    assert d['stop'] and 'could not start' in d['reasons'][0]


def test_bounded_history_and_sticky_stop(tmp_path):
    policy = dict(POLICY, max_samples_kept=8, max_errors_kept=3)
    m = monitor(lambda i, t: win(t) if i % 2 == 0 else 'error', tmp_path, tmp_path, policy=policy)
    drive(m, 60)
    s = m.summary()
    assert len(s['last_samples']) == 8 and len(s['errors']) == 3 and s['counts']['errors'] == 30
    m2 = monitor(lambda i, t: win(t) if i < 1 else 'eof', tmp_path, tmp_path)
    drive(m2, 2)
    consumed = m2.source.index
    assert m2.poll()['stop'] and m2.source.index == consumed          # no work after a stop


# ---------------------------------------------------------------- integration with execute

@pytest.fixture
def setup(tmp_path):
    manifest, files = build_repo()
    fake = FakeHF(manifest, files)
    dest = tmp_path/'ckpt'
    yield manifest, files, fake, dest
    fake.close()


def part_size(dest, name):
    p = dest/(name+'.part')
    return p.stat().st_size if p.exists() else 0


def run(fake, manifest, dest, auth, mon):
    return acq.execute(dest, dest.parent, auth, GoGate(), mon, manifest=manifest, endpoint=fake.endpoint,
                       transport=acq.Transport(allow_loopback_http=True), reserves=small_reserves(),
                       usage=lambda p: ROOMY, chunk=512)


def critical_when(condition, **pressure):
    return lambda i, t: win(t, **pressure) if condition() else win(t)


def assert_stopped_with_part(fake, manifest, dest, auth, mon, name):
    with pytest.raises(acq.HostPressureStop) as info:
        run(fake, manifest, dest, auth, mon)
    report = info.value.report
    assert report['host_monitor']['stopped'] and report['parts_retained'][name] == part_size(dest, name)
    assert 0 < part_size(dest, name) < manifest.size(name)
    marker = json.loads((dest/(name+'.part.acquire10.json')).read_text())
    assert marker['state'] == 'partial' and marker['bytes'] == part_size(dest, name)
    assert not (dest/name).exists() and not (dest/'.acquire10.lock').exists()
    assert mon.source.closed
    return report


def test_go_to_critical_during_first_shard_then_resume(tmp_path, setup):
    manifest, files, fake, dest = setup
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    mon = monitor(critical_when(lambda: part_size(dest, SHARD1) > 0, available=200*10**6), dest, dest.parent)
    report = assert_stopped_with_part(fake, manifest, dest, auth, mon, SHARD1)
    assert 'host_available_bytes' in report['stopped']
    assert report['payload_bytes_transferred'] < manifest.total_bytes()
    kept = part_size(dest, SHARD1)
    # resume: new GO and a healthy host; the .part is re-hashed and only the remainder is requested
    again = run(fake, manifest, dest, auth, healthy(dest, dest.parent))
    v = again['verification'][SHARD1]
    assert v['method'] == 'resumed .part re-hashed then streamed' and v['resumed_from_bytes'] == kept
    for name, data in files.items():
        assert (dest/name).read_bytes() == data
    assert again['committed'][SHARD1]['computed_sha256'] == hashlib.sha256(files[SHARD1]).hexdigest()


def test_go_to_critical_during_a_later_shard(tmp_path, setup):
    manifest, files, fake, dest = setup
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    mon = monitor(critical_when(lambda: part_size(dest, SHARD2) > 0, committed=50*GiB - 1), dest, dest.parent)
    report = assert_stopped_with_part(fake, manifest, dest, auth, mon, SHARD2)
    assert (dest/SHARD1).read_bytes() == files[SHARD1]                  # earlier file committed and verified
    assert 'host_commit_headroom_bytes' in report['stopped'] and 'kill-scope-invalidate-run' in report['stopped']


def test_guest_pressure_during_transfer_stops(tmp_path, setup):
    manifest, files, fake, dest = setup
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    mon = monitor(lambda i, t: win(t), dest, dest.parent,
                  guest_reader=lambda: guest(500*MiB if part_size(dest, SHARD1) > 0 else 7*GiB))
    report = assert_stopped_with_part(fake, manifest, dest, auth, mon, SHARD1)
    assert 'guest_mem_available' in report['stopped']


def test_critical_before_first_payload_request(tmp_path, setup):
    manifest, files, fake, dest = setup
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    mon = monitor(lambda i, t: win(t, pf_mb=7000 + 5000), dest, dest.parent)
    with pytest.raises(acq.HostPressureStop, match='before the first payload'):
        run(fake, manifest, dest, auth, mon)
    assert fake.payload_gets() == [] and mon.source.closed
    assert not any(p.name.endswith('.part') for p in dest.iterdir())


def test_sampling_errors_during_transfer_stop_safely(tmp_path, setup):
    manifest, files, fake, dest = setup
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    mon = monitor(lambda i, t: 'error' if part_size(dest, SHARD1) > 0 else win(t), dest, dest.parent)
    report = assert_stopped_with_part(fake, manifest, dest, auth, mon, SHARD1)
    assert 'not monitored' in report['stopped'] and report['host_monitor']['counts']['errors'] >= 4


def test_dead_sampler_during_transfer_stops_safely(tmp_path, setup):
    manifest, files, fake, dest = setup
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    mon = monitor(lambda i, t: 'eof' if part_size(dest, SHARD2) > 0 else win(t), dest, dest.parent)
    report = assert_stopped_with_part(fake, manifest, dest, auth, mon, SHARD2)
    assert 'sampler stopped' in report['stopped']


def test_annotation_only_paging_does_not_stop_transfer(tmp_path, setup):
    manifest, files, fake, dest = setup
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    mon = monitor(lambda i, t: win(t, pages_per_s=9000), dest, dest.parent)
    report = run(fake, manifest, dest, auth, mon)
    assert report['payload_bytes_transferred'] == manifest.total_bytes()
    notes = report['host_monitor']['annotations']
    assert [n['quantity'] for n in notes] == ['host_pages_input_per_s'] and notes[0]['applied'] == 'annotate'
    assert mon.source.closed


@pytest.mark.parametrize('bad', [None, object(), GoGate()])
def test_execute_requires_a_real_monitor(tmp_path, setup, bad):
    manifest, files, fake, dest = setup
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    with pytest.raises(acq.AcquireError, match='in-run host monitor missing'):
        run(fake, manifest, dest, auth, bad)
    assert fake.requests == [] and not dest.exists()


def test_cli_stop_writes_evidence(tmp_path, setup, monkeypatch):
    manifest, files, fake, dest = setup
    stop = acq.HostPressureStop('host stop condition: x')
    stop.report = dict(mode='execute', stopped='x')

    def fake_execute(*args, **kw):
        raise stop
    monkeypatch.setattr(acq, 'execute', fake_execute)
    monkeypatch.setattr(hmon.HostMonitor, 'for_acquisition', classmethod(lambda cls, *a, **k: None))
    evidence = tmp_path/'stop.json'
    with pytest.raises(acq.HostPressureStop):
        acq.main([str(dest), '--backing-volume', str(tmp_path), '--execute', '--authorization', 'a.json',
                  '--host-report', 'r.json', '--host-baseline', 'b.json', '--evidence', str(evidence)])
    assert json.loads(evidence.read_text())['stopped'] == 'x'


# ---------------------------------------------------------------- real subprocess stream (fake sampler)

SAMPLER = r'''
import json, sys, time
mode = sys.argv[1]
def sample(i):
    return dict(TotalPhysicalMemory=16900739072, DimmBytes=17179869184, AvailableBytes=4*2**30,
                CommittedBytes=40*2**30, CommitLimit=50*2**30, PagesInput=i, PagesOutput=0, PageReads=0,
                Timestamp100ns=i*10**6, Frequency100ns=10**7,
                PageFiles=[dict(Name='p', AllocatedBaseSize=1, CurrentUsage=7000, PeakUsage=1)], CFreeBytes=1)
if mode == 'garbage':
    print('x' * 200000); print('not json'); print('[1, 2]')
for i in range(3 if mode == 'exit' else 10**6):
    print(json.dumps(sample(i)), flush=True)
    time.sleep(0.05)
'''


def stream(tmp_path, mode):
    script = tmp_path/'sampler.py'
    script.write_text(SAMPLER)
    return hmon.StreamSource([sys.executable, '-I', str(script), mode], max_line_bytes=POLICY['max_line_bytes'])


def real_monitor(source, tmp_path, policy=None):
    return hmon.HostMonitor(source, ABORT, BASELINE, policy or POLICY, tmp_path, tmp_path,
                            usage=lambda p: ROOMY, guest=lambda: guest())


def wait_for(predicate, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_stream_delivers_samples_and_close_terminates_own_child(tmp_path):
    source = stream(tmp_path, 'steady')
    m = real_monitor(source, tmp_path)
    assert m.start()['stop'] is False
    assert wait_for(lambda: (m.poll(), m.summary()['counts']['valid'] >= 5)[1])
    m.close()
    assert source.proc.poll() is not None
    assert m.summary()['counts']['errors'] == 0


def test_stream_bounds_long_and_garbage_lines(tmp_path):
    source = stream(tmp_path, 'garbage')
    m = real_monitor(source, tmp_path)
    assert m.start()['stop'] is False
    assert wait_for(lambda: (m.poll(), m.summary()['counts']['valid'] >= 2)[1])
    m.close()
    errors = [e['error'] for e in m.summary()['errors']]
    assert any('longer than' in e for e in errors) and 'unparsable sample line' in errors
    assert 'sample is not a JSON object' in errors


def test_stream_exit_is_a_stop(tmp_path):
    source = stream(tmp_path, 'exit')
    m = real_monitor(source, tmp_path)
    m.start()
    assert wait_for(lambda: m.poll()['stop'])
    assert 'sampler stopped' in m.summary()['stop_reasons'][0]
    m.close()


def test_powershell_loop_is_read_only_and_bounded():
    command = hmon.StreamSource.powershell(5).command
    assert command[:4] == [hostmem10.POWERSHELL, '-NoProfile', '-NonInteractive', '-Command']
    script = command[4]
    assert 'Start-Sleep -Milliseconds 5000' in script and 'ConvertTo-Json -Compress' in script
    verbs = {w.split('-')[0] for w in script.replace('(', ' ').replace(';', ' ').split() if '-' in w and w[0].isupper()}
    assert verbs <= {'Get', 'Measure', 'Select', 'ConvertTo', 'Start'}, verbs
    assert 'Start-Sleep' in script and 'Start-Process' not in script


# ---------------------------------------------------------------- amendment rules for the monitor section

def monitor_chain(tmp_path, mutate=None):
    from tests.test_increment10_amendment import chain_root, write_amendment
    root, raw, protocol, amendment = chain_root(tmp_path)
    protocol['host_memory_preflight']['abort_criteria'] = ABORT
    section = json.loads(json.dumps(POLICY))
    if mutate:
        mutate(section)
    amendment['changes']['acquisition_host_monitor'] = section
    return protocol10.check_amendments(raw, protocol, write_amendment(root, amendment), root)


def test_monitor_section_valid(tmp_path):
    assert monitor_chain(tmp_path) == []


WEAKENINGS = {
    'kill_to_annotate': lambda s: s['action_mapping'].update({'kill-scope-invalidate-run': 'annotate'}),
    'pause_to_annotate': lambda s: s['action_mapping'].update({'pause-and-annotate': 'annotate'}),
    'stop_action_missing': lambda s: s['action_mapping'].pop('stop-acquisition-retain-part'),
    'unknown_value': lambda s: s['action_mapping'].update({'annotate-io-attribution-invalid': 'ignore'}),
    'interval_not_below_unmonitored': lambda s: s.update(sample_interval_s=20),
    'unmonitored_zero': lambda s: s.update(max_unmonitored_s=0),
    'startup_missing': lambda s: s.update(startup_timeout_s=None),
    'samples_zero': lambda s: s.update(max_samples_kept=0),
    'custom_criteria': lambda s: s.update(criteria='custom thresholds'),
}


@pytest.mark.parametrize('name', sorted(WEAKENINGS))
def test_monitor_section_weakening_rejected(tmp_path, name):
    assert any('acquisition_host_monitor' in p for p in monitor_chain(tmp_path, WEAKENINGS[name]))

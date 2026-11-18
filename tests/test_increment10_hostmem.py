"""Increment 10, C2: read-only host memory preflight (pure evaluation, no powershell needed)."""
import json
from pathlib import Path
import subprocess

import pytest

from tools import hostmem10 as hm

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'results'/'increment10'/'protocol.json'
GiB, MiB = 2**30, 2**20


def win(available=4*GiB, pf_mb=7000, pages_in=0, pages_out=0, ts=0):
    return dict(TotalPhysicalMemory=16_900_739_072, DimmBytes=17_179_869_184, AvailableBytes=available,
                CommittedBytes=40*GiB, CommitLimit=50*GiB, PagesInput=pages_in, PagesOutput=pages_out,
                PageReads=pages_in//4, Timestamp100ns=ts, Frequency100ns=10_000_000,
                PageFiles=[dict(Name='C:\\pagefile.sys', AllocatedBaseSize=34816, CurrentUsage=pf_mb,
                                PeakUsage=13799)], CFreeBytes=150*10**9)


def snap(available=4*GiB, pf_mb=7000, pages_in=0, ts=0, mem_available=9*GiB, swap_used=500*MiB,
         free=150*10**9, psi=0.0):
    return dict(windows=win(available, pf_mb, pages_in, 0, ts),
                guest=dict(meminfo=dict(MemTotal=11_483_848_704, MemAvailable=mem_available,
                                        SwapTotal=8*GiB, SwapFree=8*GiB-swap_used),
                           psi=dict(some=dict(avg10=psi))),
                checkpoint_volume=dict(free=10**12), backing_volume=dict(free=free))


def criteria(profile):
    return json.loads(PROTOCOL.read_text())['host_memory_preflight']['start_criteria'][profile]


def test_parsers():
    m = hm.meminfo('MemTotal:       11214696 kB\nMemAvailable:    8609676 kB\nHugePages_Total:       0\n')
    assert m == {'MemTotal': 11214696*1024, 'MemAvailable': 8609676*1024, 'HugePages_Total': 0}
    p = hm.psi('some avg10=0.50 avg60=0.10 avg300=0.00 total=12\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=3\n')
    assert p['some']['avg10'] == 0.5 and p['full']['total'] == 3.0
    assert hm.vmstat('pswpin 4\npswpout 9\n') == {'pswpin': 4, 'pswpout': 9}


def test_rates_from_raw_counters():
    before, after = win(pages_in=1000, ts=0), win(pages_in=4000, ts=30*10_000_000)
    assert hm.rate(before, after, 'PagesInput') == pytest.approx(100.0)
    assert hm.rate(after, before, 'PagesInput') is None        # negative interval or counter reset
    assert hm.rate(before, dict(after, Timestamp100ns=0), 'PagesInput') is None


def test_classification_and_baseline_growth():
    base = snap(pf_mb=7000, swap_used=500*MiB)
    q = hm.measurements(snap(ts=0), snap(pf_mb=7600, ts=10**8, swap_used=700*MiB), base)
    assert q['host_total_physical_memory']['kind'] == 'capacity'
    assert q['host_available_bytes']['kind'] == 'observation'
    assert q['host_pages_input_per_s']['kind'] == 'estimate'
    assert q['host_pagefile_growth_since_baseline']['value'] == 600*MiB
    assert q['guest_swap_growth_since_baseline']['value'] == 200*MiB


def test_available_uses_worst_of_two_snapshots():
    q = hm.measurements(snap(available=3*GiB), snap(available=1*GiB, ts=10**8))
    assert q['host_available_bytes']['value'] == 1*GiB


@pytest.mark.parametrize('profile', ['download', 'runtime', 'oracle'])
def test_go_on_healthy_host(profile):
    q = hm.measurements(snap(ts=0), snap(ts=3*10**8), snap())
    assert hm.evaluate(q, criteria(profile))['verdict'] == 'GO'


def test_defer_on_current_like_host_pressure():
    """2026-11-18 observation: ~0.65 GB available at rest. Runtime/oracle campaigns are deferred."""
    q = hm.measurements(snap(available=648_921_088, ts=0), snap(available=648_921_088, ts=3*10**8), snap())
    for profile in ('runtime', 'oracle'):
        r = hm.evaluate(q, criteria(profile))
        assert r['verdict'] == 'DEFER' and 'host_available_bytes' in r['violated']


def test_defer_on_paging_and_pagefile_growth():
    q = hm.measurements(snap(pages_in=0, ts=0), snap(pages_in=30*10_000, ts=30*10_000_000, pf_mb=9000),
                        snap(pf_mb=7000))
    r = hm.evaluate(q, criteria('runtime'))
    assert r['verdict'] == 'DEFER'
    assert {'host_pages_input_per_s', 'host_pagefile_growth_since_baseline'} <= set(r['violated'])


def test_not_measured_when_windows_unreachable():
    before, after = snap(), snap(ts=10**8)
    before['windows'] = after['windows'] = {'error': 'powershell.exe not reachable'}
    r = hm.evaluate(hm.measurements(before, after, snap()), criteria('runtime'))
    assert r['verdict'] == 'NOT MEASURED' and 'host_available_bytes' in r['missing']


def test_not_measured_without_baseline():
    r = hm.evaluate(hm.measurements(snap(), snap(ts=10**8)), criteria('runtime'))
    assert r['verdict'] == 'NOT MEASURED'


def test_violation_takes_precedence_over_missing():
    before, after = snap(mem_available=1*GiB), snap(mem_available=1*GiB, ts=10**8)
    before['windows'] = after['windows'] = {'error': 'x'}
    r = hm.evaluate(hm.measurements(before, after, snap()), criteria('oracle'))
    assert r['verdict'] == 'DEFER'


def test_bool_is_not_a_measurement():
    r = hm.evaluate({'host_available_bytes': dict(value=True)},
                    [dict(quantity='host_available_bytes', op='>=', threshold=1)])
    assert r['verdict'] == 'NOT MEASURED'


def test_abort_criteria_sustain():
    abort = json.loads(PROTOCOL.read_text())['host_memory_preflight']['abort_criteria']
    q = {'host_available_bytes': dict(value=200*10**6), 'guest_mem_available': dict(value=4*GiB)}
    assert hm.abort_reasons(q, abort, history={'host_available_bytes': 3}) == []
    reasons = hm.abort_reasons(q, abort, history={'host_available_bytes': 11})
    assert [r['quantity'] for r in reasons] == ['host_available_bytes']
    q = {'guest_mem_available': dict(value=700*MiB)}
    assert [r['action'] for r in hm.abort_reasons(q, abort)] == ['kill-scope-invalidate-run']


def test_pagefile_parse_rejects_bad_values():
    assert hm.pagefile_usage_bytes({'PageFiles': [{'CurrentUsage': -1}]}) is None
    assert hm.pagefile_usage_bytes({'PageFiles': []}) is None
    assert hm.pagefile_usage_bytes({'PageFiles': [{'CurrentUsage': 2}, {'CurrentUsage': 3}]}) == 5*MiB


def test_windows_probe_is_read_only_query(monkeypatch):
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(win()), stderr='')
    monkeypatch.setattr(hm.os.path, 'exists', lambda p: True)
    data = hm.windows(fake_run)
    assert data['AvailableBytes'] == 4*GiB
    script = seen[0][-1]
    for verb in ('Set-', 'Remove-', 'New-', 'Stop-', 'Invoke-WmiMethod', 'Invoke-CimMethod', 'wmic', 'bcdedit'):
        assert verb not in script
    assert script.count('Get-CimInstance') == 6


def test_pair_uses_one_process_and_sleeps_inside(monkeypatch):
    """Observer effect: powershell start-up must precede the first sample (one process, two samples)."""
    seen = []

    def fake_run(cmd, **kw):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(dict(before=win(ts=0), after=win(ts=10**8))),
                                           stderr='')
    monkeypatch.setattr(hm.os.path, 'exists', lambda p: True)
    before, after = hm.windows_pair(10, fake_run)
    assert len(seen) == 1 and before['Timestamp100ns'] == 0 and after['Timestamp100ns'] == 10**8
    script = seen[0][-1]
    assert script.count('Start-Sleep -Milliseconds 10000') == 1 and script.index('$a=& $f') < script.index('Start-Sleep')
    for verb in ('Set-', 'Remove-', 'New-', 'Stop-', 'Invoke-WmiMethod', 'Invoke-CimMethod'):
        assert verb not in script


def test_pair_failure_is_not_measured(monkeypatch):
    monkeypatch.setattr(hm.os.path, 'exists', lambda p: True)
    bad = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout='', stderr='x')
    before, after = hm.windows_pair(1, bad)
    q = hm.measurements(dict(snap(), windows=before), dict(snap(), windows=after), snap())
    assert hm.evaluate(q, criteria('runtime'))['verdict'] == 'NOT MEASURED'

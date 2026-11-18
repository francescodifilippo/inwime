"""Read-only host memory-pressure preflight for increment 10 (Windows host + WSL2 guest).

Reads, never changes: Windows CIM classes through powershell.exe (physical RAM,
available bytes, pagefile usage, raw paging counters), the WSL guest's
/proc/meminfo, /proc/vmstat, PSI and cgroup files, and statvfs of the checkpoint
volume and of its physical backing volume. No virtual-memory, swap, pagefile,
process or Windows setting is touched.

Two snapshots separated by --interval seconds give paging *rates* from the raw
cumulative counters. With --baseline (an earlier output of this tool) the
pagefile and guest-swap growth since the baseline is evaluated.

Every value is classified:
  capacity     installed or configured size (RAM, volume size, MemTotal)
  observation  measured state at the snapshot (available, pagefile in use, rates)
  estimate     derived quantity (rates over the interval, growth vs baseline)

Verdicts (thresholds are preregistered in results/increment10/protocol.json):
  GO            every start criterion measured and satisfied
  DEFER         a measured value violates a start criterion: postpone the campaign
  NOT MEASURED  a required value could not be read: the campaign must not start
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

POWERSHELL = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
SCHEMA = 'inwime-hostmem10/1'
MiB, GiB = 2**20, 2**30

# Read-only CIM query; emits one compact JSON object.
PS_SAMPLE = (
    "$cs=Get-CimInstance Win32_ComputerSystem;"
    "$dimm=(Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum).Sum;"
    "$os=Get-CimInstance Win32_OperatingSystem;"
    "$m=Get-CimInstance Win32_PerfRawData_PerfOS_Memory;"
    "$pf=@(Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage);"
    "$c=Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='C:'\";"
    "[pscustomobject]@{TotalPhysicalMemory=[int64]$cs.TotalPhysicalMemory;DimmBytes=[int64]$dimm;"
    "FreePhysicalMemoryKB=[int64]$os.FreePhysicalMemory;AvailableBytes=[int64]$m.AvailableBytes;"
    "CommittedBytes=[int64]$m.CommittedBytes;CommitLimit=[int64]$m.CommitLimit;"
    "PagesInput=[int64]$m.PagesInputPersec;PagesOutput=[int64]$m.PagesOutputPersec;"
    "PageReads=[int64]$m.PageReadsPersec;PageWrites=[int64]$m.PageWritesPersec;"
    "Timestamp100ns=[int64]$m.Timestamp_Sys100NS;Frequency100ns=[int64]$m.Frequency_Sys100NS;"
    "PageFiles=$pf;CFreeBytes=[int64]$c.FreeSpace;CSizeBytes=[int64]$c.Size}")
PS_QUERY = PS_SAMPLE + " | ConvertTo-Json -Compress -Depth 4"
# Both samples come from ONE powershell process: its start-up (loading .NET under memory
# pressure pages in ~100 MB) happens before the first sample, so it is not counted in the
# paging rate of the interval (observer effect seen on 2026-11-18 with one process per sample).
PS_PAIR = ("$f={" + PS_SAMPLE + "}; $a=& $f; Start-Sleep -Milliseconds %d; $b=& $f;"
           " [pscustomobject]@{before=$a;after=$b} | ConvertTo-Json -Compress -Depth 5")


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def windows(run=subprocess.run):
    if not os.path.exists(POWERSHELL):
        return dict(error='powershell.exe not reachable')
    try:
        out = run([POWERSHELL, '-NoProfile', '-NonInteractive', '-Command', PS_QUERY],
                  capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(error=type(error).__name__)
    if out.returncode != 0:
        return dict(error='powershell exit %d' % out.returncode)
    try:
        data = json.loads(out.stdout.strip())
    except ValueError:
        return dict(error='unparsable powershell output')
    pf = data.get('PageFiles') or []
    data['PageFiles'] = pf if isinstance(pf, list) else [pf]
    return data


def windows_pair(interval, run=subprocess.run):
    """Two samples interval seconds apart from a single read-only powershell process."""
    if not os.path.exists(POWERSHELL):
        error = dict(error='powershell.exe not reachable')
        return error, error
    try:
        out = run([POWERSHELL, '-NoProfile', '-NonInteractive', '-Command', PS_PAIR % int(interval*1000)],
                  capture_output=True, text=True, timeout=120+interval)
    except (OSError, subprocess.TimeoutExpired) as error:
        e = dict(error=type(error).__name__)
        return e, e
    try:
        data = json.loads(out.stdout.strip()) if out.returncode == 0 else None
    except ValueError:
        data = None
    if not isinstance(data, dict) or not isinstance(data.get('before'), dict) or not isinstance(data.get('after'), dict):
        e = dict(error='powershell failed or unparsable output')
        return e, e
    for w in (data['before'], data['after']):
        pf = w.get('PageFiles') or []
        w['PageFiles'] = pf if isinstance(pf, list) else [pf]
    return data['before'], data['after']


def meminfo(text):
    out = {}
    for line in text.splitlines():
        key, _, rest = line.partition(':')
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[key] = int(parts[0]) * (1024 if parts[1:] == ['kB'] else 1)
    return out


def psi(text):
    out = {}
    for line in text.splitlines():
        kind, *fields = line.split()
        out[kind] = {k: float(v) for k, v in (f.split('=') for f in fields)}
    return out


def vmstat(text):
    return {k: int(v) for k, v in (line.split() for line in text.splitlines() if line.strip())}


def read(path):
    try:
        return Path(path).read_text()
    except OSError:
        return None


def guest():
    info = read('/proc/meminfo')
    pressure = read('/proc/pressure/memory')
    vm = read('/proc/vmstat')
    cgroup = {}
    for name in ('memory.max', 'memory.swap.max', 'memory.current'):
        raw = read('/sys/fs/cgroup/user.slice/user-%d.slice/user@%d.service/%s' % (os.getuid(), os.getuid(), name))
        cgroup[name] = raw.strip() if raw else None
    return dict(meminfo=meminfo(info) if info else None, psi=psi(pressure) if pressure else None,
                vmstat={k: v for k, v in vmstat(vm).items() if k in ('pswpin', 'pswpout', 'pgmajfault')}
                if vm else None, user_cgroup=cgroup)


def volume(path, usage=shutil.disk_usage):
    p = Path(path).absolute()
    while not p.exists():
        p = p.parent
    u = usage(p)
    return dict(path_checked=str(path), total=u.total, free=u.free)


def snapshot(checkpoint_volume, backing_volume, run=subprocess.run, usage=shutil.disk_usage):
    return dict(utc=now(), monotonic=time.monotonic(), windows=windows(run), guest=guest(),
                checkpoint_volume=volume(checkpoint_volume, usage), backing_volume=volume(backing_volume, usage))


# ---------------------------------------------------------------- evaluation (pure)

def pagefile_usage_bytes(w):
    files = w.get('PageFiles')
    if not isinstance(files, list) or not files:
        return None
    total = 0
    for f in files:
        v = f.get('CurrentUsage') if isinstance(f, dict) else None
        if type(v) is not int or v < 0:
            return None
        total += v * MiB            # Win32_PageFileUsage reports MB (MiB)
    return total


def rate(before, after, key):
    """Pages per second from raw cumulative counters and the 100 ns system timestamp."""
    try:
        dc = after[key] - before[key]
        dt = (after['Timestamp100ns'] - before['Timestamp100ns']) / after['Frequency100ns']
    except (KeyError, TypeError, ZeroDivisionError):
        return None
    if dt <= 0 or dc < 0:
        return None
    return dc / dt


def measurements(before, after, baseline=None):
    """Flatten the two snapshots into classified quantities; missing values stay None."""
    wb, wa = before.get('windows', {}), after.get('windows', {})
    ga = (after.get('guest') or {}).get('meminfo') or {}
    gb = (before.get('guest') or {}).get('meminfo') or {}
    q = {}

    def put(name, value, kind, unit):
        q[name] = dict(value=value, kind=kind, unit=unit)

    put('host_total_physical_memory', wa.get('TotalPhysicalMemory'), 'capacity', 'B')
    put('host_installed_dimm_bytes', wa.get('DimmBytes'), 'capacity', 'B')
    put('host_available_bytes', min(v for v in (wb.get('AvailableBytes'), wa.get('AvailableBytes')) if type(v) is int)
        if any(type(v) is int for v in (wb.get('AvailableBytes'), wa.get('AvailableBytes'))) else None,
        'observation', 'B')
    put('host_committed_bytes', wa.get('CommittedBytes'), 'observation', 'B')
    put('host_commit_limit', wa.get('CommitLimit'), 'capacity', 'B')
    committed, limit = wa.get('CommittedBytes'), wa.get('CommitLimit')
    put('host_commit_headroom_bytes', limit-committed if type(committed) is int and type(limit) is int else None,
        'estimate', 'B')
    put('host_pagefile_in_use', pagefile_usage_bytes(wa), 'observation', 'B')
    for key, name in (('PagesInput', 'host_pages_input_per_s'), ('PagesOutput', 'host_pages_output_per_s'),
                      ('PageReads', 'host_page_reads_per_s')):
        put(name, rate(wb, wa, key), 'estimate', 'pages/s')
    put('guest_mem_total', ga.get('MemTotal'), 'capacity', 'B')
    put('guest_mem_available', min(gb.get('MemAvailable', 1 << 62), ga['MemAvailable'])
        if 'MemAvailable' in ga else None, 'observation', 'B')
    swap_used = (ga['SwapTotal'] - ga['SwapFree']) if 'SwapTotal' in ga and 'SwapFree' in ga else None
    put('guest_swap_used', swap_used, 'observation', 'B')
    pa = (after.get('guest') or {}).get('psi') or {}
    put('guest_psi_memory_some_avg10', (pa.get('some') or {}).get('avg10'), 'observation', '%')
    put('checkpoint_volume_free', (after.get('checkpoint_volume') or {}).get('free'), 'observation', 'B')
    put('backing_volume_free', (after.get('backing_volume') or {}).get('free'), 'observation', 'B')
    put('host_c_free_cim', wa.get('CFreeBytes'), 'observation', 'B')
    if baseline is not None:
        bw = baseline.get('windows', {})
        b_pf, a_pf = pagefile_usage_bytes(bw), q['host_pagefile_in_use']['value']
        put('host_pagefile_growth_since_baseline', a_pf-b_pf if None not in (a_pf, b_pf) else None, 'estimate', 'B')
        bg = (baseline.get('guest') or {}).get('meminfo') or {}
        b_sw = (bg['SwapTotal'] - bg['SwapFree']) if 'SwapTotal' in bg and 'SwapFree' in bg else None
        put('guest_swap_growth_since_baseline', swap_used-b_sw if None not in (swap_used, b_sw) else None,
            'estimate', 'B')
    else:
        put('host_pagefile_growth_since_baseline', None, 'estimate', 'B')
        put('guest_swap_growth_since_baseline', None, 'estimate', 'B')
    return q


OPS = {'>=': lambda v, t: v >= t, '<=': lambda v, t: v <= t}


def evaluate(quantities, criteria):
    """criteria: list of {quantity, op, threshold, required}. Returns checks and verdict."""
    checks, missing, violated = [], [], []
    for c in criteria:
        value = quantities.get(c['quantity'], {}).get('value')
        ok = None if not isinstance(value, (int, float)) or isinstance(value, bool) else OPS[c['op']](value, c['threshold'])
        checks.append(dict(c, value=value, kind=quantities.get(c['quantity'], {}).get('kind'), ok=ok))
        if ok is None and c.get('required', True):
            missing.append(c['quantity'])
        elif ok is False:
            violated.append(c['quantity'])
    verdict = 'DEFER' if violated else 'NOT MEASURED' if missing else 'GO'
    return dict(verdict=verdict, violated=violated, missing=missing, checks=checks)


def abort_reasons(sample_quantities, criteria, history=None):
    """In-run monitor criteria; 'sustain_s' requires the violation to persist."""
    reasons = []
    for c in criteria:
        value = sample_quantities.get(c['quantity'], {}).get('value')
        if not isinstance(value, (int, float)):
            continue
        if not OPS[c['op']](value, c['threshold']):
            sustained = c.get('sustain_s', 0)
            if not sustained or (history and history.get(c['quantity'], 0) >= sustained):
                reasons.append(dict(quantity=c['quantity'], value=value, op=c['op'], threshold=c['threshold'],
                                    action=c['action']))
    return reasons


def make_report(before, after, baseline_raw, criteria, profile, interval, protocol_sha256):
    """Assemble the report. Amendment 001: the baseline is identified by its SHA256 and UTC time so
    that tools/hostgate10.py can bind a report to the exact baseline file and check its freshness."""
    baseline = json.loads(baseline_raw)['after'] if baseline_raw is not None else None
    q = measurements(before, after, baseline)
    result = evaluate(q, criteria)
    report = dict(schema=SCHEMA, profile=profile, interval_s=interval, protocol_sha256=protocol_sha256,
                  before=before, after=after, baseline_used=baseline_raw is not None,
                  baseline_sha256=hashlib.sha256(baseline_raw).hexdigest() if baseline_raw is not None else None,
                  baseline_utc=baseline.get('utc') if baseline is not None else None,
                  quantities=q, **result,
                  read_only=True, note='No system setting was changed; values are point observations.')
    for snap in (before, after):
        for key in ('checkpoint_volume', 'backing_volume'):
            snap[key]['path_checked'] = os.path.relpath(Path(snap[key]['path_checked']).absolute())
    return report


def load_criteria(protocol, profile):
    section = json.loads(Path(protocol).read_text())['host_memory_preflight']
    if profile not in section['start_criteria']:
        raise SystemExit('unknown profile '+profile)
    return section['start_criteria'][profile], section


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint-volume', type=Path, required=True)
    p.add_argument('--backing-volume', type=Path, required=True)
    p.add_argument('--protocol', type=Path, default=Path('results/increment10/protocol.json'))
    p.add_argument('--profile', default='runtime')
    p.add_argument('--interval', type=float, default=30.0)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args(argv)
    criteria, section = load_criteria(a.protocol, a.profile)
    began = dict(utc=now(), guest=guest())
    w_before, w_after = windows_pair(a.interval)
    before = dict(began, windows=w_before, checkpoint_volume=volume(a.checkpoint_volume),
                  backing_volume=volume(a.backing_volume))
    after = dict(utc=now(), guest=guest(), windows=w_after, checkpoint_volume=volume(a.checkpoint_volume),
                 backing_volume=volume(a.backing_volume))
    baseline_raw = a.baseline.read_bytes() if a.baseline else None
    report = make_report(before, after, baseline_raw, criteria, a.profile, a.interval,
                         hashlib.sha256(a.protocol.read_bytes()).hexdigest())
    q, result = report['quantities'], report
    if a.output:
        with a.output.open('x') as f:
            json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(dict(verdict=result['verdict'], violated=result['violated'], missing=result['missing'],
                          values={k: v['value'] for k, v in q.items()}), indent=1))
    return 0 if result['verdict'] == 'GO' else 3


if __name__ == '__main__':
    sys.exit(main())

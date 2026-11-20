"""Run one experiment in a transient cgroup v2 scope and supervise it from outside.

Uses only the memory controller already delegated to the user manager
(`systemd-run --user --scope`): no sudo, no persistent unit, no global setting,
no other cgroup modified. The supervisor stays in its own cgroup; a tiny /bin/sh
finalizer inside the scope records the scope counters after the worker exits,
including after an OOM kill. Page-cache eviction is targeted (POSIX_FADV_DONTNEED
on the experiment files only) and verified with mincore via fincore.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from .audit_increment9 import POWERSHELL, meminfo
from .sanitize9 import portable

USER_APP = Path('/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/app.slice')
FINALIZER = Path(__file__).with_name('cgroup9_finalize.sh')
STAT_KEYS = ('anon', 'file', 'kernel', 'kernel_stack', 'pagetables', 'sock', 'shmem', 'file_mapped',
             'file_dirty', 'file_writeback', 'active_file', 'inactive_file', 'active_anon',
             'inactive_anon', 'slab', 'workingset_refault_file', 'workingset_activate_file',
             'pgscan', 'pgsteal', 'pgmajfault', 'pgfault')


def keyed(text):
    result = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and re.fullmatch(r'-?\d+', parts[1]):
            result[parts[0]] = int(parts[1])
    return result


def global_oom_kills():
    """System-wide OOM-kill counter (/proc/vmstat), to detect any system OOM during a run."""
    for line in Path('/proc/vmstat').read_text().splitlines():
        if line.startswith('oom_kill '):
            return int(line.split()[1])
    return None


def read_int(path):
    value = Path(path).read_text().strip()
    return None if value == 'max' else int(value)


def fincore(paths):
    """Resident page-cache bytes per file (mincore). Not attribution to a cgroup."""
    out = {}
    for path in paths:
        result = subprocess.run(['fincore', '--bytes', '--noheadings', '--output', 'RES,SIZE', str(path)],
                                capture_output=True, text=True, check=True)
        resident, size = result.stdout.split()
        out[str(path)] = dict(resident_bytes=int(resident), size_bytes=int(size))
    return out


def evict(paths):
    """Advisory, file-scoped eviction of clean unmapped pages; verified separately."""
    for path in paths:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def warm(paths):
    """Populate the global page cache from the supervisor's own cgroup."""
    total = 0
    for path in paths:
        with open(path, 'rb', buffering=0) as f:
            while data := f.read(8*2**20):
                total += len(data)
    return total


def host_disk():
    """Global Windows physical-disk raw counters (host-wide; not attributable)."""
    if not Path(POWERSHELL).exists():
        return {'error': 'powershell.exe not reachable'}
    command = ("Get-CimInstance Win32_PerfRawData_PerfDisk_PhysicalDisk | Where-Object {$_.Name -ne '_Total'} |"
               " Select Name,DiskReadBytesPersec,DiskReadsPersec,DiskWriteBytesPersec | ConvertTo-Json")
    began = time.time()
    try:
        out = subprocess.run([POWERSHELL, '-NoProfile', '-NonInteractive', '-Command', command],
                             capture_output=True, text=True, timeout=120)
        data = json.loads(out.stdout)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return {'error': repr(error)}
    return {'captured_unix_s': began, 'disks': data if isinstance(data, list) else [data]}


def disk_delta(before, after):
    if 'disks' not in before or 'disks' not in after:
        return None
    old = {d['Name']: d for d in before['disks']}
    return {d['Name']: {k: int(d[k])-int(old[d['Name']][k]) for k in
                        ('DiskReadBytesPersec', 'DiskReadsPersec', 'DiskWriteBytesPersec')}
            for d in after['disks'] if d['Name'] in old}


def proc(pid):
    data = {}
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            key, _, value = line.partition(':')
            if key.startswith('Vm'):
                data[key] = int(value.split()[0])*1024
        data['io'] = keyed(Path(f'/proc/{pid}/io').read_text().replace(':', ' '))
        data['comm'] = Path(f'/proc/{pid}/comm').read_text().strip()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    return data


def sample_scope(directory):
    snap = {'memory.current': read_int(directory/'memory.current'),
            'memory.swap.current': read_int(directory/'memory.swap.current'),
            'memory.events': keyed((directory/'memory.events').read_text()),
            'memory.stat': {k: v for k, v in keyed((directory/'memory.stat').read_text()).items()
                            if k in STAT_KEYS}}
    if (directory/'memory.peak').exists():
        snap['memory.peak'] = read_int(directory/'memory.peak')
    snap['pids'] = [int(p) for p in (directory/'cgroup.procs').read_text().split()]
    return snap


def run(a):
    uid = os.getuid()
    unit = f'inwime9-{a.label}-{int(time.time())}'
    scope = Path(str(USER_APP).format(uid=uid))/f'{unit}.scope'
    final = a.output.with_name(a.output.stem+'.final.txt')
    if a.output.exists() or final.exists() or a.output.with_suffix('.log').exists():
        raise FileExistsError(a.output)
    files = [Path(p) for p in a.files]
    report = dict(label=a.label, unit=unit, scope=str(scope), command=a.command,
                  memory_max=a.memory_max, swap_max=a.swap_max, condition=a.condition,
                  available_stop_bytes=a.available_stop_mib*2**20,
                  global_before=meminfo(), files=[str(p) for p in files],
                  global_oom_kill_before=global_oom_kills())
    report['residency_initial'] = fincore(files) if files else {}
    if a.condition == 'evicted':
        evict(files)
        report['residency_after_eviction'] = fincore(files)
    elif a.condition == 'warm':
        began = time.monotonic()
        report['warm_read_bytes'] = warm(files)
        report['warm_read_s'] = time.monotonic()-began
        report['residency_after_warm'] = fincore(files)
    report['host_disk_before'] = host_disk() if a.host_disk else None
    report['global_at_launch'] = meminfo()
    env = dict(os.environ, TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2', INWIME9_SCOPE_UNIT=unit,
               INWIME9_MEMORY_MAX=str(a.memory_max), INWIME9_SWAP_MAX=str(a.swap_max))
    launcher = ['systemd-run', '--user', '--scope', '--quiet', '--collect', '--expand-environment=no',
                f'--unit={unit}', '-p', f'MemoryMax={a.memory_max}', '-p', f'MemorySwapMax={a.swap_max}',
                # Keep the scope (and its finalizer) alive after a kernel OOM kill of the
                # worker so the kill is recorded; the default 'stop' would discard it.
                '-p', 'OOMPolicy=continue',
                '--', '/bin/sh', str(FINALIZER), str(final.resolve()), *a.command]
    report['launcher'] = launcher
    peaks, minimum_available, samples, series = {}, None, 0, []
    reason = None
    began = time.monotonic()
    last_series = 0.
    with a.output.with_suffix('.log').open('x') as log:
        child = subprocess.Popen(launcher, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                if not scope.exists():
                    time.sleep(a.interval)
                    continue
                try:
                    snap = sample_scope(scope)
                except (FileNotFoundError, ProcessLookupError, ValueError):
                    time.sleep(a.interval)
                    continue
                samples += 1
                for key in ('memory.current', 'memory.swap.current', 'memory.peak'):
                    if snap.get(key) is not None:
                        peaks[key] = max(peaks.get(key, 0), snap[key])
                for key, value in snap['memory.stat'].items():
                    peaks['stat.'+key] = max(peaks.get('stat.'+key, 0), value)
                for pid in snap['pids']:
                    p = proc(pid)
                    if p:
                        name = f'pid{pid}.{p["comm"]}'
                        for key in ('VmRSS', 'VmHWM', 'VmSize', 'VmPeak', 'VmSwap'):
                            if key in p:
                                peaks[name+'.'+key] = max(peaks.get(name+'.'+key, 0), p[key])
                        for key in ('read_bytes', 'rchar'):
                            peaks[name+'.io.'+key] = max(peaks.get(name+'.io.'+key, 0), p['io'].get(key, 0))
                g = meminfo()
                minimum_available = min(minimum_available or g['MemAvailable'], g['MemAvailable'])
                now = time.monotonic()-began
                if now-last_series >= a.series_interval and len(series) < 20000:
                    last_series = now
                    series.append(dict(t=round(now, 3), current=snap['memory.current'],
                                       file=snap['memory.stat'].get('file'), anon=snap['memory.stat'].get('anon'),
                                       kernel=snap['memory.stat'].get('kernel'),
                                       swap=snap['memory.swap.current'], events=snap['memory.events'],
                                       mem_available=g['MemAvailable'], cached=g['Cached'],
                                       swap_free=g['SwapFree']))
                peaks['memory.events'] = snap['memory.events']
                if g['MemAvailable'] < a.available_stop_mib*2**20:
                    reason = 'global MemAvailable below stop threshold: cgroup.kill'
                    (scope/'cgroup.kill').write_text('1')
                    break
                time.sleep(a.interval)
        finally:
            code = child.wait()
    report.update(exit_code=code, watchdog_reason=reason, wall_s=time.monotonic()-began,
                  samples=samples, sample_interval_s=a.interval, sampled_peaks=peaks,
                  minimum_global_memavailable=minimum_available, series=series,
                  scope_removed_after_exit=not scope.exists(), global_after=meminfo(),
                  global_oom_kill_after=global_oom_kills())
    report['finalizer'] = parse_final(final) if final.exists() else None
    report['residency_after'] = fincore(files) if files else {}
    report['host_disk_after'] = host_disk() if a.host_disk else None
    report['host_disk_delta'] = (disk_delta(report['host_disk_before'], report['host_disk_after'])
                                 if a.host_disk else None)
    report['limitations'] = [
        'sampling can miss transients; memory.peak from the finalizer is the kernel high-water mark',
        'io.stat unavailable: io controller not enabled for child cgroups',
        'host disk counters are global for the Windows physical disk (all host activity)',
        'fincore reports residency, not which cgroup is charged']
    # Evidence is Git-bound: the checkout root is written as <repo>, values are unchanged.
    with a.output.open('x') as f:
        json.dump(portable(report), f, indent=2); f.write('\n')
    fin = report['finalizer'] or {}
    print(json.dumps(dict(exit_code=code, reason=reason, unit=unit, wall_s=report['wall_s'],
                          peak=fin.get('memory.peak'), events=fin.get('memory.events'),
                          swap_peak=fin.get('memory.swap.peak')), indent=1))
    return code


def parse_final(path):
    sections, current = {}, None
    for line in path.read_text().splitlines():
        if line.startswith('== '):
            current = line[3:].strip()
            sections[current] = []
        elif current:
            sections[current].append(line)
    out = {}
    for name, lines in sections.items():
        text = '\n'.join(lines)
        if name in ('memory.peak', 'memory.current', 'memory.swap.current', 'memory.swap.peak',
                    'memory.max', 'memory.swap.max', 'worker.exit'):
            out[name] = None if text.strip() in ('max', '') else int(text.strip())
        elif name == 'cgroup.path':
            out[name] = text.strip()
        else:
            out[name] = keyed(text)
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--memory-max', type=int, default=2147483648)
    p.add_argument('--swap-max', type=int, default=0)
    p.add_argument('--condition', choices=['as-is', 'evicted', 'warm'], default='as-is')
    p.add_argument('--file', dest='files', action='append', default=[])
    p.add_argument('--available-stop-mib', type=int, default=768)
    p.add_argument('--interval', type=float, default=0.05)
    p.add_argument('--series-interval', type=float, default=1.0)
    p.add_argument('--host-disk', action='store_true')
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.command[:1] == ['--']:
        a.command = a.command[1:]
    if not a.command or not shutil.which('systemd-run'):
        p.error('child command and systemd-run required')
    raise SystemExit(run(a))

"""Supervised increment-10 campaign runs (runtime R1-R4 and oracles), additive wrapper.

One invocation = one run, never repeated automatically. Steps, each recorded:

  1. host start check: tools/hostmem10.py with the preregistered profile (runtime or
     oracle) against the campaign baseline. Anything but GO writes
     blocked-<run>.json (cause, UTC, report SHA256) and the run is NOT started;
  2. L3: /sys/block/<dev>/stat of the checkpoint block device and of the swap device;
  3. L4b: in-run host observer (tools/hostmonitor10 sampler + HostMonitor) applying
     the frozen abort criteria (protocol host_memory_preflight.abort_criteria) with
     the run mapping below; kill actions write cgroup.kill of the run scope only;
  4. the run itself through the unchanged tools/cgroup9.py supervisor (transient
     scope, MemoryMax, MemorySwapMax 0, OOMPolicy=continue, targeted eviction or
     as-is, fincore residency, global Windows PhysicalDisk counters L4);
  5. L3 after, observer summary, SHA256 of every produced file.

Nothing on the host or in WSL is changed: no drop_caches, sysctl, swap, .wslconfig,
pagefile or mount. Eviction is POSIX_FADV_DONTNEED on the experiment files only
(done by cgroup9). The wrapper does not interpret results: tools/report_increment10.py
derives verdicts from the raw files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

from . import hostmem10
from .hostmonitor10 import HostMonitor, StreamSource
from .sanitize9 import portable

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'results'/'increment10'/'protocol.json'
SCHEMA = 'inwime-campaign10/1'
USER_APP = Path('/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/app.slice')
# Run mapping of the frozen abort actions (the acquisition mapping of amendment 002 does
# not apply to runs; amendment 003 records this mapping).
RUN_POLICY = dict(
    action_mapping={'pause-and-annotate': 'annotate', 'kill-scope-invalidate-run': 'kill-scope',
                    'annotate-io-attribution-invalid': 'annotate',
                    'stop-acquisition-retain-part': 'annotate'},
    sample_interval_s=5.0, startup_timeout_s=90.0, max_unmonitored_s=60.0, max_line_bytes=65536,
    max_annotations=64, max_errors_kept=64, max_samples_kept=2000)
SAMPLE_KEYS = ('host_available_bytes', 'host_commit_headroom_bytes', 'host_pagefile_growth_since_baseline',
               'host_pages_input_per_s', 'guest_mem_available', 'backing_volume_free')


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def block_device(path):
    """major:minor of the filesystem holding path, and its /sys/block entry (whole disk)."""
    dev = os.stat(path).st_dev
    major, minor = os.major(dev), os.minor(dev)
    link = Path(f'/sys/dev/block/{major}:{minor}').resolve()
    disk = link if (link/'device').exists() or not (link.parent/'stat').exists() else link.parent
    return dict(major_minor=f'{major}:{minor}', name=disk.name, stat_path=str(disk/'stat'))


def swap_devices():
    out = []
    for line in Path('/proc/swaps').read_text().splitlines()[1:]:
        name = line.split()[0]
        if name.startswith('/dev/'):
            out.append(dict(name=Path(name).name, stat_path=f'/sys/class/block/{Path(name).name}/stat'))
    return out


BLOCK_FIELDS = ('reads_completed', 'reads_merged', 'sectors_read', 'read_ticks_ms', 'writes_completed',
                'writes_merged', 'sectors_written', 'write_ticks_ms', 'in_flight', 'io_ticks_ms',
                'time_in_queue_ms')


def block_stat(entry):
    try:
        values = [int(x) for x in Path(entry['stat_path']).read_text().split()]
    except (OSError, ValueError) as error:
        return dict(device=entry['name'], error=type(error).__name__)
    stat = dict(zip(BLOCK_FIELDS, values))
    stat['read_bytes'] = stat['sectors_read']*512   # /sys/block stat sectors are 512 B by definition
    return dict(device=entry['name'], monotonic_s=time.monotonic(), utc=utc(), **stat)


def block_delta(before, after):
    if 'error' in before or 'error' in after:
        return None
    return {k: after[k]-before[k] for k in (*BLOCK_FIELDS, 'read_bytes') if k != 'in_flight'}


class Observer(threading.Thread):
    """Polls a HostMonitor every second during the run; kill actions kill the run scope only."""

    def __init__(self, monitor, label):
        super().__init__(daemon=True)
        self.monitor, self.label, self.stop_event = monitor, label, threading.Event()
        self.kills, self.start_decision = [], None

    def scope(self):
        app = Path(str(USER_APP).format(uid=os.getuid()))
        found = sorted(app.glob(f'inwime9-{self.label}-*.scope'))
        return found[-1] if found else None

    def run(self):
        self.start_decision = self.monitor.start()
        while not self.stop_event.wait(1.0):
            decision = self.monitor.poll()
            kill = [a for a in self.monitor.annotations if a.get('applied') == 'kill-scope']
            if kill and not self.kills:
                scope = self.scope()
                if scope is not None and (scope/'cgroup.kill').exists():
                    (scope/'cgroup.kill').write_text('1')
                    self.kills.append(dict(utc=utc(), scope=scope.name, reason=kill[0]))
            if decision['stop'] and not self.monitor.annotations:
                # sampler failure (stale or dead): recorded, the run is not killed for it
                pass

    def finish(self):
        self.stop_event.set()
        self.join(timeout=10)
        self.monitor.close()
        return dict(start=self.start_decision, kills=self.kills, summary=self.monitor.summary())


def host_check(profile, baseline, output, checkpoint, backing):
    command = [sys.executable, '-m', 'tools.hostmem10', '--checkpoint-volume', str(checkpoint),
               '--backing-volume', str(backing), '--profile', profile, '--interval', '30',
               '--baseline', str(baseline), '--output', str(output)]
    code = subprocess.run(command, cwd=ROOT, capture_output=True, text=True).returncode
    report = json.loads(output.read_text()) if output.exists() else None
    return code, report


def run(a):
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    record_path = out/f'{a.run}-run.json'
    blocked_path = out/f'blocked-{a.run}.json'
    for p in (record_path, blocked_path, out/f'{a.run}-host-start.json', out/f'{a.run}-scope.json'):
        if p.exists():
            raise FileExistsError(p)
    protocol = json.loads(PROTOCOL.read_text())
    record = dict(schema=SCHEMA, run=a.run, kind=a.kind, profile=a.profile, cache=a.cache, purpose=a.purpose,
                  utc_start=utc(), command=a.command, memory_max=a.memory_max, condition=a.condition,
                  files=[os.path.relpath(f, ROOT) for f in a.files], protocol_sha256=sha256(PROTOCOL),
                  baseline=os.path.relpath(a.baseline, ROOT), baseline_sha256=sha256(a.baseline),
                  python=sys.version.split()[0], read_only_host=True)
    host_path = out/f'{a.run}-host-start.json'
    code, report = host_check(a.profile, a.baseline, host_path, a.checkpoint_volume, a.backing_volume)
    record['host_start'] = dict(path=host_path.name, exit_code=code,
                                verdict=report.get('verdict') if report else None,
                                violated=report.get('violated') if report else None,
                                missing=report.get('missing') if report else None,
                                sha256=sha256(host_path) if host_path.exists() else None)
    if not report or report.get('verdict') != 'GO':
        blocked = dict(schema=SCHEMA, run=a.run, utc=utc(), cause='host start check not GO',
                       verdict=record['host_start']['verdict'], violated=record['host_start']['violated'],
                       missing=record['host_start']['missing'], host_report=host_path.name,
                       host_report_sha256=record['host_start']['sha256'])
        with blocked_path.open('x') as f:
            json.dump(blocked, f, indent=2); f.write('\n')
        print(json.dumps(blocked, indent=1))
        return 3
    devices = dict(checkpoint=block_device(a.files[0] if a.files else a.checkpoint_volume),
                   swap=swap_devices())
    record['block_devices'] = devices
    record['l3_before'] = dict(checkpoint=block_stat(devices['checkpoint']),
                               swap=[block_stat(d) for d in devices['swap']])
    baseline_after = json.loads(Path(a.baseline).read_text())['after']
    monitor = HostMonitor(StreamSource.powershell(RUN_POLICY['sample_interval_s'],
                                                  max_line_bytes=RUN_POLICY['max_line_bytes']),
                          protocol['host_memory_preflight']['abort_criteria'], baseline_after, RUN_POLICY,
                          a.checkpoint_volume, a.backing_volume)
    observer = Observer(monitor, a.label)
    observer.start()
    while observer.start_decision is None and observer.is_alive():
        time.sleep(0.25)
    record['observer_started'] = observer.start_decision
    scope_json = out/f'{a.run}-scope.json'
    supervisor = [sys.executable, '-m', 'tools.cgroup9', '--label', a.label, '--output', str(scope_json),
                  '--memory-max', str(a.memory_max), '--swap-max', '0', '--condition', a.condition,
                  '--host-disk']
    for f in a.files:
        supervisor += ['--file', str(f)]
    supervisor += ['--', *a.command]
    record['supervisor'] = supervisor
    began = time.monotonic()
    code = subprocess.run(supervisor, cwd=ROOT).returncode
    record['wall_s'] = time.monotonic()-began
    record['supervisor_exit'] = code
    record['l3_after'] = dict(checkpoint=block_stat(devices['checkpoint']),
                              swap=[block_stat(d) for d in devices['swap']])
    record['l3_delta'] = dict(checkpoint=block_delta(record['l3_before']['checkpoint'],
                                                     record['l3_after']['checkpoint']),
                              swap=[block_delta(b, c) for b, c in zip(record['l3_before']['swap'],
                                                                      record['l3_after']['swap'])])
    record['observer'] = observer.finish()
    record['utc_end'] = utc()
    produced = {}
    for p in sorted(out.glob(f'{a.run}-*')):
        if p != record_path and p.is_file():
            produced[p.name] = sha256(p)
    for p in a.outputs:
        if Path(p).exists():
            produced[os.path.relpath(p, ROOT)] = sha256(p)
    record['produced_sha256'] = produced
    with record_path.open('x') as f:
        json.dump(portable(record), f, indent=2); f.write('\n')     # no absolute home path in evidence
    print(json.dumps(dict(run=a.run, supervisor_exit=code, wall_s=record['wall_s'],
                          l3_read_bytes=(record['l3_delta']['checkpoint'] or {}).get('read_bytes'),
                          observer_stopped=record['observer']['summary']['stopped'],
                          annotations=len(record['observer']['summary']['annotations']),
                          kills=record['observer']['kills']), indent=1))
    return code


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run', required=True, help='run identifier, e.g. R1, O14-stock_fp32')
    p.add_argument('--label', required=True, help='cgroup9 label (unit inwime9-<label>-<time>)')
    p.add_argument('--kind', choices=['runtime', 'oracle', 'fault'], required=True)
    p.add_argument('--profile', choices=['runtime', 'oracle'], required=True)
    p.add_argument('--cache', choices=['C', 'W', 'as-is'], required=True)
    p.add_argument('--purpose', required=True)
    p.add_argument('--condition', choices=['as-is', 'evicted', 'warm'], required=True)
    p.add_argument('--memory-max', type=int, required=True)
    p.add_argument('--file', dest='files', action='append', default=[], type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--produces', dest='outputs', action='append', default=[], type=Path)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--checkpoint-volume', type=Path, required=True)
    p.add_argument('--backing-volume', type=Path, default=Path('/mnt/c'))
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    if a.command[:1] == ['--']:
        a.command = a.command[1:]
    if not a.command:
        p.error('run command required after --')
    return run(a)


if __name__ == '__main__':
    sys.exit(main())

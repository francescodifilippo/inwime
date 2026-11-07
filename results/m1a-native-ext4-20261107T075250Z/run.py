"""One scoped M1a acquisition. Existing tools/binary; no historical output changes."""
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PS = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
BIN = ROOT / 'build/bf16_stream'
SOURCE = ROOT / 'data/weights-5GiB.bf16'
EXPECTED = 'fbfd729651af3054f46600bd042c08f71d87f5a915479af9329bd6617a2c8c0b'
ENV = os.environ.copy()
ENV['PYTHONDONTWRITEBYTECODE'] = '1'
ENV['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib' + (':' + ENV['LD_LIBRARY_PATH'] if ENV.get('LD_LIBRARY_PATH') else '')


def save(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def windows():
    script = r'''& { $ErrorActionPreference='Stop'; @{Time=(Get-Date -Format o);Disks=@(Get-Disk | Select-Object Number,FriendlyName,Size,UniqueId,BusType,IsOffline);Counters=@(Get-CimInstance Win32_PerfRawData_PerfDisk_PhysicalDisk | Select-Object Name,DiskReadBytesPersec,DiskWriteBytesPersec);Memory=(Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory | Select-Object AvailableMBytes,CommittedBytes,CommitLimit,PagesInputPersec,PagesOutputPersec);Paging=@(Get-CimInstance Win32_PageFileUsage | Select-Object AllocatedBaseSize,CurrentUsage,PeakUsage);Battery=@(Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining);Power=(& powercfg /getactivescheme)} | ConvertTo-Json -Depth 5 }'''
    p = subprocess.run([PS, '-NoProfile', '-NonInteractive', '-Command', script], capture_output=True, timeout=30, check=True)
    return json.loads(p.stdout.decode('utf-8-sig'))


def linux():
    return {'time': time.time(), 'diskstats': Path('/proc/diskstats').read_text(),
            'meminfo': Path('/proc/meminfo').read_text(), 'vmstat': Path('/proc/vmstat').read_text()}


def run_logged(command, log):
    with (OUT / log).open('x') as f:
        p = subprocess.run(list(map(str, command)), cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f'command exit {p.returncode}; see {log}')


def main():
    save('started.json', {'time': time.time(), 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                          'binary_sha256': digest(BIN), 'bytes_per_pass': 5 * 1024**3,
                          'repeats': 10, 'buffer_mib': 64, 'n': 1, 'cache_mib': 0,
                          'threads': 8, 'cache_policy': 'O_DIRECT, no global cache flush; host/controller cache uncontrolled'})
    save('windows-before.json', windows())
    run_logged([sys.executable, '-B', ROOT / 'tests/smoke.py', '--gpu'], 'smoke.log')
    print('PASS: CPU/GPU smoke, complete FP64 reference, corruption and tail checks', flush=True)
    mount = Path('/mnt/wsl/Sabrent')
    fm = json.loads(subprocess.check_output(['findmnt', '--json', '--mountpoint', str(mount)], text=True))['filesystems'][0]
    if fm['fstype'] != 'ext4' or fm['source'] != '/dev/sdd1':
        raise RuntimeError('external mount changed')
    uuid = subprocess.check_output(['lsblk', '-n', '-o', 'UUID', fm['source']], text=True).strip()
    if uuid != '5ee418f3-419e-4434-9337-ef5267c3a403':
        raise RuntimeError('external filesystem UUID changed')
    if SOURCE.stat().st_size != 5 * 1024**3 or shutil.disk_usage(mount).free < 6 * 1024**3:
        raise RuntimeError('dataset size or free-space guard failed')
    if digest(SOURCE) != EXPECTED:
        raise RuntimeError('source checksum mismatch')
    external = mount / OUT.name
    external.mkdir()
    fixture = external / 'small-reference.bf16'
    base = [BIN, '--file', fixture, '--m', 32, '--k', 128, '--n', 7, '--blocks', 3,
            '--threads', 2, '--repeats', 1, '--full-check']
    run_logged([*base, '--prepare'], 'external-small-prepare.log')
    for mode in ['cpu', 'sync', 'async']:
        run_logged([*base, '--mode', mode], f'external-small-{mode}.log')
    print('PASS: external ext4 O_DIRECT + CPU/sync/async full reference', flush=True)
    target = external / SOURCE.name
    save('dataset-planned.json', {'source': str(SOURCE), 'external': str(target), 'external_uuid': uuid, 'sha256': EXPECTED})
    with SOURCE.open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst, 8 * 1024**2)
        dst.flush()
        os.fsync(dst.fileno())
    if target.stat().st_size != SOURCE.stat().st_size or digest(target) != EXPECTED:
        raise RuntimeError('external dataset checksum mismatch')
    save('dataset-verified.json', {'source': str(SOURCE), 'external': str(target), 'bytes': target.stat().st_size,
                                  'sha256': EXPECTED, 'external_fixture_retained': str(fixture), 'external_dataset_retained': True})
    print('PASS: exclusive external 5 GiB copy and SHA256', flush=True)
    cases = [(route, mode) for route in ['external-ext4', 'internal-ext4'] for mode in ['io', 'cpu', 'async']]
    random.Random(20261107).shuffle(cases)
    save('order.json', {'seed': 20261107, 'order': cases, 'note': '10 consecutive repetitions per case; shuffled case order, no thermal control'})
    for index, (route, mode) in enumerate(cases):
        path = target if route == 'external-ext4' else SOURCE
        before = {'linux': linux(), 'windows': windows()}
        print(f'START {index+1}/6 {route} {mode}', flush=True)
        command = [sys.executable, '-B', ROOT / 'tools/benchmark_v2.py', '--file', path,
                   '--output', OUT / route / 'sweep', '--stage', 'buffers', '--buffers', 64,
                   '--ns', 1, '--modes', mode, '--repeats', 10, '--threads', 8]
        save(f'{route}-{mode}-before.json', {'command': list(map(str, command)), **before})
        run_logged(command, f'{route}-{mode}-progress.log')
        save(f'{route}-{mode}-after.json', {'linux': linux(), 'windows': windows()})
        print(f'DONE {route} {mode}', flush=True)
    for route in ['external-ext4', 'internal-ext4']:
        run_logged([sys.executable, '-B', ROOT / 'tools/summarize_v2.py', '--input', OUT / route / 'sweep',
                    '--output', OUT / route], f'{route}-summarize.log')
    save('windows-after.json', windows())
    preserved = json.loads((OUT / 'preserved-files.json').read_text())
    changed = [name for name, sha in preserved.items() if not (ROOT / name).exists() or digest(ROOT / name) != sha]
    save('preservation-check.json', {'checked_files': len(preserved), 'changed': changed})
    if changed:
        raise RuntimeError('pre-existing file changed')
    save('completed.json', {'time': time.time(), 'cases': 6, 'passes': 60,
                            'logical_timed_read_bytes': 300 * 1024**3,
                            'warmup_read_bytes': 60 * 1024**2 * 64,
                            'note': 'warmup: 2 slots per async pass + 1 block per CPU pass, on both routes'})
    print('COMPLETE: 60 passes; original tracked/untracked files preserved', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        save('failed.json', {'time': time.time(), 'error': repr(exc)})
        raise

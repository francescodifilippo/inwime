"""Read-only memory/storage/cgroup audit for increment 9. Never changes the system.

Separates process RSS, Linux page cache, WSL RAM (MemTotal), swap and host RAM.
Windows facts come from read-only CIM queries; failures are recorded, not hidden.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

POWERSHELL = '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
CGROUP = Path('/sys/fs/cgroup')


def meminfo():
    result = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, _, value = line.partition(':')
        parts = value.split()
        result[key] = int(parts[0])*(1024 if parts[1:] == ['kB'] else 1)
    return result


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as error:
        return f'unavailable: {error.__class__.__name__}'


def cgroup_chain():
    own = Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1]
    chain, path = [], CGROUP
    for part in [''] + [p for p in own.split('/') if p]:
        path = path/part if part else path
        chain.append({'path': str(path), 'controllers': read(path/'cgroup.controllers'),
                      'subtree_control': read(path/'cgroup.subtree_control'),
                      **{f: read(path/f) for f in ('memory.max', 'memory.high', 'memory.swap.max')
                         if (path/f).exists()}})
    return own, chain


def powershell(command):
    if not Path(POWERSHELL).exists():
        return {'error': 'powershell.exe not reachable'}
    try:
        out = subprocess.run([POWERSHELL, '-NoProfile', '-NonInteractive', '-Command', command],
                             capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'error': repr(error)}
    if out.returncode:
        return {'error': out.stderr.strip()[-2000:]}
    try:
        return json.loads(out.stdout)
    except ValueError:
        return {'raw': out.stdout.strip()[-4000:]}


def windows_facts():
    facts = _windows_facts()
    # Keep only what the evidence needs: VHDX drive letter and wslconfig values,
    # not Windows user-profile paths or PowerShell provider objects.
    distributions = facts.get('WslDistributions')
    if isinstance(distributions, dict):
        distributions = [distributions]
    if isinstance(distributions, list):
        facts['WslDistributions'] = [{'Name': d.get('Name'),
                                      'BasePathDrive': str(d.get('BasePath', ''))[:2]}
                                     for d in distributions]
    config = facts.get('WslConfig')
    if isinstance(config, dict):
        config = config.get('value')
    if isinstance(config, str):
        facts['WslConfig'] = [line.strip() for line in config.splitlines() if line.strip()]
    return facts


def _windows_facts():
    return powershell(
        "$cs=Get-CimInstance Win32_ComputerSystem; $os=Get-CimInstance Win32_OperatingSystem;"
        "$d=Get-CimInstance Win32_LogicalDisk | Select DeviceID,FileSystem,Size,FreeSpace;"
        "$p=Get-CimInstance Win32_DiskDrive | Select Index,Model,Size,InterfaceType;"
        "$l=Get-ChildItem HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss |"
        " ForEach-Object { $i=Get-ItemProperty $_.PSPath; [pscustomobject]@{Name=$i.DistributionName;BasePath=$i.BasePath} };"
        "$w=Get-Content $env:USERPROFILE\\.wslconfig -Raw -ErrorAction SilentlyContinue;"
        "[pscustomobject]@{TotalPhysicalMemory=$cs.TotalPhysicalMemory;"
        "FreePhysicalMemoryKiB=$os.FreePhysicalMemory;LogicalDisks=$d;PhysicalDisks=$p;"
        "WslDistributions=$l;WslConfig=$w} | ConvertTo-Json -Depth 4")


def checkpoint_inventory(root):
    entries = []
    for directory in sorted(Path(root).iterdir()) if Path(root).exists() else ():
        if directory.is_dir():
            files = [f for f in directory.iterdir() if f.is_file()]
            entries.append({'directory': directory.name, 'files': len(files),
                            'bytes': sum(f.stat().st_size for f in files),
                            'safetensors': sorted(f.name for f in files if f.suffix == '.safetensors')})
    return entries


def audit(repository):
    own, chain = cgroup_chain()
    memory = meminfo()
    usage = shutil.disk_usage(repository)
    try:
        backing = shutil.disk_usage('/mnt/c')._asdict()
    except OSError as error:
        backing = {'error': repr(error)}
    swaps = Path('/proc/swaps').read_text().splitlines()
    try:
        user_manager = subprocess.run(['systemctl', '--user', 'is-system-running'],
                                      capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        user_manager = repr(error)
    hf = Path.home()/'.cache/huggingface/hub'
    return {
        'captured_unix_s': time.time(),
        'kernel': os.uname().release, 'logical_cpus': os.cpu_count(),
        'wsl_meminfo_bytes': {k: memory[k] for k in (
            'MemTotal', 'MemFree', 'MemAvailable', 'Cached', 'Buffers', 'Active(file)',
            'Inactive(file)', 'AnonPages', 'Shmem', 'SwapTotal', 'SwapFree', 'Dirty')},
        'swap_devices': swaps, 'swappiness': read('/proc/sys/vm/swappiness'),
        'cgroup_filesystem': read('/proc/filesystems').count('cgroup2') > 0
                             and os.statvfs(CGROUP) is not None,
        'cgroup_v2_mounted': (CGROUP/'cgroup.controllers').exists(),
        'own_cgroup': own, 'cgroup_chain': chain,
        'root_subtree_control': read(CGROUP/'cgroup.subtree_control'),
        'io_controller_enabled_for_children': 'io' in read(CGROUP/'cgroup.subtree_control').split(),
        'user_manager_state': user_manager,
        'systemd_run': shutil.which('systemd-run'), 'fincore': shutil.which('fincore'),
        'transparent_hugepage': read('/sys/kernel/mm/transparent_hugepage/enabled'),
        'repository_filesystem_bytes': usage._asdict(),
        'mnt_c_statfs_bytes': backing,
        'windows': windows_facts(),
        'checkpoints': checkpoint_inventory(Path(repository)/'checkpoints'),
        'huggingface_hub_cache_entries': sorted(p.name for p in hf.iterdir()) if hf.exists() else [],
        'semantics': {
            'MemTotal': 'RAM assigned to the WSL2 VM, not host RAM',
            'Cached': 'global Linux page cache, not attributable to one process by itself',
            'mnt_c_statfs_bytes': 'Windows volume holding the WSL VHDX (verify BasePath)',
            'repository_filesystem_bytes': 'virtual ext4 size; may exceed real Windows free space'}}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = audit(Path.cwd())
    with a.output.open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps({k: report[k] for k in ('wsl_meminfo_bytes', 'mnt_c_statfs_bytes',
                                             'io_controller_enabled_for_children')}, indent=1))

"""Read-only queries of the Windows host behind WSL2, through PowerShell interop.

Needed because WSL's MemTotal is the VM size, not host RAM, and the VHDX location
(hence its physical volume) is only recorded in the host registry. Interop may be
disabled or the Windows PATH not appended, so the executable is searched on the
drvfs mounts. Every failure returns None: callers must treat that as unknown.
"""
import subprocess
from pathlib import Path

from .storage import drvfs_drive

POWERSHELL = 'Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
SCRIPT = (
    # Windows PowerShell 5.1 writes in the OEM code page by default; force UTF-8 for non-ASCII names.
    "[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
    "$ErrorActionPreference='Stop';"
    "Get-ChildItem 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss' | ForEach-Object {"
    " $p = Get-ItemProperty $_.PSPath; 'D|{0}|{1}' -f $p.DistributionName, $p.BasePath };"
    "'M|{0}' -f (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"
)
TIMEOUT_SECONDS = 30


def find_powershell(table):
    for m in table:
        if drvfs_drive(m):
            candidate = Path(m.point)/POWERSHELL
            if candidate.is_file():
                return candidate
    return None


def run_powershell(executable, script=SCRIPT, timeout=TIMEOUT_SECONDS):
    try:
        done = subprocess.run([str(executable), '-NoProfile', '-NonInteractive', '-Command', script],
                              stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.decode('utf-8', errors='replace')


def parse(text, distro):
    if text is None:
        return None
    distros, memory = {}, None
    for line in text.splitlines():
        kind, _, rest = line.strip().partition('|')
        if kind == 'D':
            name, _, base = rest.partition('|')
            if name:
                distros[name] = base
        elif kind == 'M' and rest.strip().isdigit():
            memory = int(rest.strip())
    return dict(distro=distro, distros=distros, total_physical_memory_bytes=memory)


def query(table, distro, runner=run_powershell):
    executable = find_powershell(table)
    if executable is None:
        return None
    return parse(runner(executable), distro)

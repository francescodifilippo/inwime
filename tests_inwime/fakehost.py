"""A fake host: /proc, /sys and /etc files under a temporary root, plus fake statvfs and PowerShell.

Mount points are real directories inside the temporary tree so path resolution works,
while their filesystem identity comes only from the fake mountinfo.
"""
import os
from pathlib import Path
from types import SimpleNamespace

GIB = 1024**3


class FakeHost:
    def __init__(self, base, *, wsl=True):
        self.base = Path(base)
        self.root = self.base/'fsroot'
        self.linux = self.base/'linux'      # stands for the distribution's '/'
        self.cdrive = self.base/'c'         # stands for /mnt/c
        for d in (self.root, self.linux, self.cdrive):
            d.mkdir(parents=True, exist_ok=True)
        self.space = {}                     # mount point -> (free, total)
        self.powershell_output = None
        self.powershell_calls = 0
        self.write('/proc/sys/kernel/osrelease',
                   '6.18.58.1-microsoft-standard-WSL2\n' if wsl else '6.8.0-45-generic\n')
        self.write('/etc/os-release', 'PRETTY_NAME="Ubuntu 26.04 LTS"\nID=ubuntu\n')
        self.write('/proc/cpuinfo', ''.join(
            f'processor\t: {i}\nmodel name\t: Test CPU 9000\nflags\t\t: fpu sse4_2 avx avx2 fma f16c\n\n'
            for i in range(4)))
        self.write('/proc/meminfo', 'MemTotal:       10485760 kB\nMemAvailable:    8388608 kB\n'
                                    'SwapTotal:       2097152 kB\nSwapFree:        2097152 kB\n')
        self.write('/proc/self/cgroup', '0::/user.slice/app.scope\n')
        self.mounts = []
        self.add_mount('/', 'ext4', '/dev/sdd', device='8:48', options='rw,discard')
        self.set_space('/', free=800*GIB, total=1000*GIB)
        self.add_mount(self.linux, 'ext4', '/dev/sdd', device='8:48', options='rw,discard')
        self.set_space(self.linux, free=800*GIB, total=1000*GIB)

    def write(self, path, text):
        target = self.root/path.lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def add_mount(self, point, fstype, source, *, device='0:99', options='rw'):
        n = len(self.mounts)+20
        point = str(point).replace(' ', '\\040')
        self.mounts.append(f'{n} 1 {device} / {point} rw,relatime - {fstype} {source} {options}')
        self.write('/proc/self/mountinfo', '\n'.join(self.mounts)+'\n')

    def set_space(self, point, *, free, total):
        self.space[str(point)] = (free, total)

    def add_c_drive(self, *, free, total, powershell=True):
        self.add_mount(self.cdrive, '9p', 'C:\\134', device='0:69',
                       options='rw,aname=drvfs;path=C:\\;uid=1000;gid=1000')
        self.set_space(self.cdrive, free=free, total=total)
        if powershell:
            exe = self.cdrive/'Windows/System32/WindowsPowerShell/v1.0/powershell.exe'
            exe.parent.mkdir(parents=True, exist_ok=True)
            exe.write_bytes(b'')

    def statvfs(self, path):
        best = max((p for p in self.space if str(path) == p or str(path).startswith(p.rstrip('/')+'/')),
                   key=len, default=None)
        if best is None:
            raise FileNotFoundError(path)
        free, total = self.space[best]
        return SimpleNamespace(f_bavail=free, f_frsize=1, f_blocks=total)

    def runner(self, executable):
        self.powershell_calls += 1
        return self.powershell_output

    def kwargs(self, **extra):
        out = dict(root=self.root, env={'WSL_DISTRO_NAME': 'Ubuntu', 'HOME': str(self.linux/'home'),
                                        'INWIME_CPU_KERNEL': 'off', 'INWIME_OPS_KERNEL': 'off'},
                   statvfs=self.statvfs, runner=self.runner, access=lambda p, mode: True)
        out.update(extra)
        return out

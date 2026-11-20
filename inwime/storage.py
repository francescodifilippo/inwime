"""Where bytes really land: mount, filesystem, backing volume and usable space.

Under WSL2 the distribution's ext4 filesystem lives in a dynamically growing VHDX
file on a Windows volume, so Linux ``statvfs`` overstates what can be written: the
physical limit is the free space of that host volume. ``describe`` reports both and
derives ``usable_bytes`` = physical free space minus a safety margin (default 30 GiB).
Read-only: nothing is created, written, compacted or deleted.
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

from .probe import read_text

GIB = 1024**3
DEFAULT_MARGIN_BYTES = 30*GIB
DRVFS_TYPES = ('9p', 'drvfs', 'v9fs')
# `wsl --mount` of a physical disk: /mnt/wsl/PHYSICALDRIVE<n>[p<partition>]
PHYSICAL_DRIVE = re.compile(r'/mnt/wsl/PHYSICALDRIVE\d+(p\d+)?$')


@dataclass(frozen=True)
class Mount:
    device: str
    point: str
    fstype: str
    source: str
    options: str


def unescape(field):
    return re.sub(r'\\([0-7]{3})', lambda m: chr(int(m.group(1), 8)), field)


def parse_mountinfo(text):
    table = []
    for line in (text or '').splitlines():
        left, sep, right = line.partition(' - ')
        a, b = left.split(), right.split()
        if not sep or len(a) < 5 or len(b) < 2:
            continue
        table.append(Mount(device=a[2], point=unescape(a[4]), fstype=b[0],
                           source=unescape(b[1]), options=' '.join(b[2:])))
    return table


def mounts(root='/'):
    return parse_mountinfo(read_text(root, '/proc/self/mountinfo'))


def mount_for(path, table):
    """Mount whose point is the longest component-wise prefix of ``path``; last one wins ties."""
    parts = Path(path).parts
    best = None
    for m in table:
        point = Path(m.point).parts
        if parts[:len(point)] == point and (best is None or len(point) >= len(Path(best.point).parts)):
            best = m
    return best


def drvfs_drive(m):
    """Windows drive letter behind a WSL drvfs mount, else None."""
    if m.fstype not in DRVFS_TYPES:
        return None
    match = re.search(r'(?:^|[;,])path=([A-Za-z]):', m.options)
    if match is None and m.fstype in ('drvfs', '9p'):
        match = re.match(r'([A-Za-z]):', m.source)
    if m.fstype == '9p' and 'aname=drvfs' not in m.options:
        return None
    return match.group(1).upper() if match else None


def drive_mount(drive, table):
    for m in table:
        if drvfs_drive(m) == drive.upper():
            return m
    return None


def existing_ancestor(path):
    path = Path(os.path.abspath(os.path.expanduser(str(path))))
    while not path.exists() and path != path.parent:
        path = path.parent
    return Path(os.path.realpath(path))


def space(point, statvfs):
    try:
        st = statvfs(point)
    except OSError:
        return None, None
    return st.f_bavail*st.f_frsize, st.f_blocks*st.f_frsize


def describe(path, *, table=None, wsl=False, root_device=None, host=None,
             margin_bytes=DEFAULT_MARGIN_BYTES, statvfs=os.statvfs, access=os.access, root='/'):
    """Report where ``path`` is stored and how many bytes can be written there safely.

    ``host`` is the result of ``wsl_host.query`` (or None if unavailable/disabled).
    Fail-closed: an unknown backing gives ``usable_bytes=None`` and status 'unknown'.
    """
    table = mounts(root) if table is None else table
    anchor = existing_ancestor(path)
    m = mount_for(str(anchor), table)
    free, total = space(str(anchor), statvfs)
    out = dict(path=str(path), exists=Path(os.path.expanduser(str(path))).exists(),
               existing_ancestor=str(anchor), writable=bool(access(anchor, os.W_OK)),
               mount_point=m.point if m else None, fstype=m.fstype if m else None,
               source=m.source if m else None, linux_free_bytes=free, linux_total_bytes=total,
               margin_bytes=margin_bytes, backing='unknown', host_volume=None, vhdx=None,
               physical_free_bytes=None, usable_bytes=None, notes=[])
    if m is None:
        out['notes'].append('mount point not found in mountinfo')
    elif not wsl:
        out.update(backing='local', physical_free_bytes=free)
    elif drvfs_drive(m):
        out.update(backing='wsl-host-volume', physical_free_bytes=free,
                   host_volume=dict(drive=drvfs_drive(m), mount_point=m.point,
                                    free_bytes=free, total_bytes=total))
        out['notes'].append('Windows volume through drvfs: slow for streaming reads')
    elif PHYSICAL_DRIVE.search(m.point):
        out.update(backing='wsl-physical-disk', physical_free_bytes=free)
    elif m.fstype == 'ext4' and root_device is not None and m.device == root_device:
        out['backing'] = 'wsl-vhdx'
        _vhdx_host(out, host, table, statvfs, free)
    else:
        out['notes'].append('WSL filesystem with unrecognized backing (e.g. an attached VHD)')
    if out['physical_free_bytes'] is not None:
        out['usable_bytes'] = out['physical_free_bytes']-margin_bytes
    out['status'] = ('unknown' if out['usable_bytes'] is None else
                     'warn' if not out['writable'] or out['usable_bytes'] <= 0 else 'ok')
    return out


def _vhdx_host(out, host, table, statvfs, linux_free):
    distro = (host or {}).get('distro')
    base = ((host or {}).get('distros') or {}).get(distro) if distro else None
    if not base:
        out['notes'].append('VHDX location unknown: Windows host query unavailable or distro not found')
        return
    base = PureWindowsPath(base[4:] if base.startswith('\\\\?\\') else base)
    drive = base.drive.rstrip(':').upper()
    hm = drive_mount(drive, table) if len(drive) == 1 else None
    if hm is None:
        out['notes'].append(f'host drive {drive or "?"} of the VHDX is not mounted in WSL')
        return
    host_free, host_total = space(hm.point, statvfs)
    vhdx = Path(hm.point, *base.parts[1:], 'ext4.vhdx')
    try:
        vhdx_bytes = vhdx.stat().st_size
    except OSError:
        vhdx_bytes = None
    out.update(host_volume=dict(drive=drive, mount_point=hm.point, free_bytes=host_free,
                                total_bytes=host_total),
               vhdx=dict(path=str(base/'ext4.vhdx'), bytes=vhdx_bytes))
    if host_free is not None and linux_free is not None:
        out['physical_free_bytes'] = min(host_free, linux_free)
    else:
        out['notes'].append('host volume free space unreadable')

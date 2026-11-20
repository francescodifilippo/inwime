"""Read-only probes of the host: OS, CPU, memory, cgroup v2 and resource limits.

Every reader takes ``root`` so tests can point it at a fake /proc and /sys tree.
Nothing here writes files, allocates large buffers or starts processes.
"""
import os
import platform
import resource
from pathlib import Path, PurePosixPath

# Flags that matter for current or planned CPU kernels; reported, never required.
CPU_FLAGS = ('sse4_2', 'avx', 'avx2', 'fma', 'f16c', 'avx512f', 'avx512_bf16', 'amx_bf16')
MEMINFO_KEYS = ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree')
CGROUP_ROOT = '/sys/fs/cgroup'


def read_text(root, path):
    try:
        return (Path(root)/str(path).lstrip('/')).read_text()
    except OSError:
        return None


def os_release(root='/'):
    fields = {}
    for line in (read_text(root, '/etc/os-release') or '').splitlines():
        key, sep, value = line.partition('=')
        if sep:
            fields[key.strip()] = value.strip().strip('"')
    return fields.get('PRETTY_NAME')


def is_wsl(root='/'):
    release = (read_text(root, '/proc/sys/kernel/osrelease') or '').lower()
    return 'microsoft' in release or 'wsl' in release


def system(root='/', env=None):
    env = os.environ if env is None else env
    wsl = is_wsl(root)
    release = (read_text(root, '/proc/sys/kernel/osrelease') or platform.release()).strip()
    return dict(os=platform.system(), kernel_release=release, distribution=os_release(root),
                wsl=wsl, wsl_distro=env.get('WSL_DISTRO_NAME') if wsl else None,
                machine=platform.machine(), python=platform.python_version(),
                implementation=platform.python_implementation())


def parse_cpuinfo(text):
    model, flags, logical = None, set(), 0
    for line in (text or '').splitlines():
        key, sep, value = line.partition(':')
        if not sep:
            continue
        key = key.strip()
        if key == 'processor':
            logical += 1
        elif key == 'model name' and model is None:
            model = value.strip()
        elif key == 'flags' and not flags:
            flags = set(value.split())
    return dict(model=model, logical_cpus=logical or None,
                flags={flag: flag in flags for flag in CPU_FLAGS} if flags else None)


def parse_meminfo(text):
    out = {}
    for line in (text or '').splitlines():
        key, sep, value = line.partition(':')
        parts = value.split()
        if sep and key in MEMINFO_KEYS and parts and parts[0].isdigit():
            unit = parts[1] if len(parts) > 1 else ''
            out[key] = int(parts[0])*(1024 if unit == 'kB' else 1)
    return {key: out.get(key) for key in MEMINFO_KEYS}


def cgroup_path(root='/'):
    """cgroup v2 path of this process, or None on a v1/hybrid or unreadable host."""
    for line in (read_text(root, '/proc/self/cgroup') or '').splitlines():
        hierarchy, _, rest = line.partition(':')
        controllers, _, path = rest.partition(':')
        if hierarchy == '0' and controllers == '':
            return path or '/'
    return None


def limit_value(text):
    text = (text or '').strip()
    if not text:
        return None, False
    if text == 'max':
        return None, True
    return (int(text), True) if text.isdigit() else (None, False)


def cgroup(root='/'):
    """Own cgroup v2 files, the effective memory limit and the headroom left under it.

    Every ancestor with a finite ``memory.max`` bounds this process; its headroom is its own
    limit minus its own ``memory.current``. Headroom is None if any limited node's usage is unreadable.
    """
    path = cgroup_path(root)
    if path is None:
        return dict(version=None, path=None)
    leaf = PurePosixPath(CGROUP_ROOT)/path.lstrip('/')
    files = {}
    for name in ('memory.max', 'memory.high', 'memory.swap.max', 'memory.current'):
        text = read_text(root, leaf/name)
        files[name] = limit_value(text)[0] if text is not None else None
    files['cpu.max'] = (read_text(root, leaf/'cpu.max') or '').strip() or None
    effective, headroom, headroom_known, any_readable, node = None, None, True, False, leaf
    while True:
        limit, readable = limit_value(read_text(root, node/'memory.max'))
        any_readable |= readable
        if readable and limit is not None:
            effective = limit if effective is None else min(effective, limit)
            used, used_readable = limit_value(read_text(root, node/'memory.current'))
            if used_readable and used is not None:
                left = max(limit-used, 0)
                headroom = left if headroom is None else min(headroom, left)
            else:
                headroom_known = False
        if str(node) == CGROUP_ROOT:
            break
        node = node.parent
    # 'unknown' when no memory.max on the path could be read: never report an unread limit as absent.
    status = 'limited' if effective is not None else 'unlimited' if any_readable else 'unknown'
    return dict(version=2, path=path, files=files, effective_memory_max=effective, memory_limit=status,
                headroom_bytes=headroom if headroom_known else None)


def rlimits():
    out = {}
    for name in ('RLIMIT_AS', 'RLIMIT_DATA', 'RLIMIT_NOFILE'):
        soft, _ = resource.getrlimit(getattr(resource, name))
        out[name] = None if soft == resource.RLIM_INFINITY else soft
    return out


def cpu(root='/'):
    info = parse_cpuinfo(read_text(root, '/proc/cpuinfo'))
    try:
        info['affinity_cpus'] = len(os.sched_getaffinity(0)) if str(root) == '/' else None
    except (AttributeError, OSError):
        info['affinity_cpus'] = None
    return info


def memory(root='/', group=None):
    """Memory the application can use: MemAvailable bounded by the cgroup headroom.

    If a cgroup limit applies but its usage is unreadable, the exact value is unknown (None) and
    only an upper bound, min(MemAvailable, limit), is reported.
    """
    info = parse_meminfo(read_text(root, '/proc/meminfo'))
    group = cgroup(root) if group is None else group
    available, limit = info.get('MemAvailable'), group.get('effective_memory_max')
    bound = available if limit is None else limit if available is None else min(available, limit)
    if limit is not None:
        headroom = group.get('headroom_bytes')
        available = None if headroom is None or available is None else min(available, headroom)
    info.update(cgroup=group, rlimits=rlimits() if str(root) == '/' else None,
                application_available_bytes=available, application_available_upper_bound_bytes=bound)
    return info

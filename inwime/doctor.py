"""`inwime doctor`: a non-destructive audit of the host, its memory, storage and CPU kernel.

The report distinguishes measured facts from what could not be read: anything
unavailable is ``None`` with status 'unknown', never a guess. Nothing is written,
downloaded or created, and RAM never gates a download.
"""
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, kernel, probe, storage, wsl_host

SCHEMA = 'inwime-doctor/1'


def default_store(env):
    """Hugging Face hub cache, resolved as huggingface_hub does."""
    for name in ('HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE'):   # the second is the legacy name
        if env.get(name):
            return Path(env[name])
    if env.get('HF_HOME'):
        return Path(env['HF_HOME'])/'hub'
    cache = Path(env['XDG_CACHE_HOME']) if env.get('XDG_CACHE_HOME') else Path(env.get('HOME', '~'))/'.cache'
    return cache/'huggingface'/'hub'


def report(*, root='/', env=None, paths=None, host_query=True, margin_bytes=storage.DEFAULT_MARGIN_BYTES,
           runner=wsl_host.run_powershell, statvfs=os.statvfs, access=os.access):
    env = dict(os.environ if env is None else env)
    system = probe.system(root, env)
    table = storage.mounts(root)
    root_mount = storage.mount_for('/', table)
    host = None
    if system['wsl'] and host_query:
        host = wsl_host.query(table, system['wsl_distro'], runner)
    memory = probe.memory(root)
    memory['host_physical_bytes'] = (host or {}).get('total_physical_memory_bytes')
    targets = list(paths) if paths else [default_store(env)]
    entries = [storage.describe(p, table=table, wsl=system['wsl'],
                                root_device=root_mount.device if root_mount else None,
                                host=host, margin_bytes=margin_bytes, statvfs=statvfs,
                                access=access, root=root) for p in targets]
    out = dict(schema=SCHEMA, inwime_version=__version__,
               generated_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
               system=system, cpu=probe.cpu(root), memory=memory,
               host_query=dict(enabled=bool(system['wsl'] and host_query), available=host is not None),
               storage=entries, cpu_kernel=kernel.describe(env))
    out['checks'] = checks(out)
    out['status'] = max((c['status'] for c in out['checks']), key=SEVERITY.index, default='ok')
    return out


SEVERITY = ('ok', 'unknown', 'warn', 'fail')


def checks(r):
    s, k, m = r['system'], r['cpu_kernel'], r['memory']
    out = []

    def add(name, status, message):
        out.append(dict(id=name, status=status, message=message))

    tested = s['os'] == 'Linux' and s['machine'] in ('x86_64', 'AMD64')
    add('platform', 'ok' if tested else 'warn',
        f"{s['os']} {s['machine']}" + ('' if tested else ': untested platform, support is experimental'))
    add('cpu-kernel', k['status'], k['detail'])
    if s['wsl']:
        q = r['host_query']
        add('wsl-host', 'ok' if q['available'] else 'unknown',
            'Windows host queried (registry, physical RAM)' if q['available'] else
            'Windows host query disabled' if not q['enabled'] else
            'Windows host query unavailable: host RAM and VHDX volume unknown')
    available = m['application_available_bytes']
    bound = m.get('application_available_upper_bound_bytes')
    add('memory', 'unknown' if available is None else 'ok',
        ('application memory unknown' + (f' (at most {bound} bytes)' if bound is not None else ''))
        if available is None else
        f'application memory available: {available} bytes (informative, never gates downloads)')
    for e in r['storage']:
        usable = e['usable_bytes']
        add('storage:'+e['path'], e['status'],
            f"{e['backing']}: usable bytes unknown" if usable is None else
            f"{e['backing']}: not writable" if not e['writable'] else
            f"{e['backing']}: {usable} bytes usable after a {e['margin_bytes']}-byte margin"
            + (' (downloads refused)' if usable <= 0 else ''))
    return out


# Windows profile directories: C:\Users\<name>, C:/Users/<name> and drvfs /mnt/c/Users/<name>.
WINDOWS_USER = re.compile(r'((?:\b[A-Za-z]:|/mnt/[A-Za-z])[\\/]+Users[\\/]+)[^\\/]+')


def redact(value, *, home):
    """Copy of ``value`` with the home directory shown as '~' and Windows user names hidden."""
    home = home.rstrip('/')
    if isinstance(value, dict):
        return {k: redact(v, home=home) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, home=home) for v in value]
    if isinstance(value, str):
        if home and (value == home or value.startswith(home+'/')):
            value = '~'+value[len(home):]
        elif home:
            value = value.replace(home+'/', '~/')
        return WINDOWS_USER.sub(r'\1<user>', value)
    return value


def size(n):
    if n is None:
        return 'unknown'
    sign, n = ('-' if n < 0 else ''), abs(n)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if n < 1024 or unit == 'TiB':
            return f'{sign}{n:.1f} {unit}' if unit != 'B' else f'{sign}{n} B'
        n /= 1024


def cgroup_limit(group):
    if not group['version']:
        return 'unknown (no cgroup v2)'
    if group['memory_limit'] == 'limited':
        return size(group['effective_memory_max'])
    return group['memory_limit']


def limits(values):
    if values is None:
        return 'rlimits unknown'
    return ' '.join(f"{name[7:]}={'unlimited' if v is None else v}" for name, v in values.items())


def render(r):
    s, c, m = r['system'], r['cpu'], r['memory']
    flags = c['flags'] or {}
    group = m['cgroup']
    lines = [f"InWiMe {r['inwime_version']} doctor ({r['generated_utc']})", '',
             f"System   {s['distribution'] or s['os']} | kernel {s['kernel_release']} | {s['machine']}"
             + (f" | WSL2 ({s['wsl_distro']})" if s['wsl'] else '')
             + f" | {s['implementation']} {s['python']}",
             f"CPU      {c['model'] or 'unknown'} | {c['logical_cpus'] or '?'} logical"
             + (f", {c['affinity_cpus']} usable" if c.get('affinity_cpus') else '')
             + ' | ' + (' '.join(f for f, on in flags.items() if on) or 'flags unknown'),
             f"Memory   MemTotal {size(m['MemTotal'])} | MemAvailable {size(m['MemAvailable'])}"
             f" | swap {size(m['SwapFree'])}/{size(m['SwapTotal'])}"
             + (f" | host RAM {size(m['host_physical_bytes'])}" if s['wsl'] else ''),
             f"         cgroup v2 limit {cgroup_limit(group)}"
             f" | application available {size(m['application_available_bytes'])}",
             f"         memory.current {size((group.get('files') or {}).get('memory.current'))}"
             + (f" | headroom {size(group.get('headroom_bytes'))}" if group.get('memory_limit') == 'limited' else '')
             + ' | ' + limits(m.get('rlimits')),
             f"CPU kernel {r['cpu_kernel']['implementation']}"
             + (f" ({r['cpu_kernel']['path']})" if r['cpu_kernel']['path'] else '')]
    for e in r['storage']:
        lines.append(f"Storage  {e['path']} -> {e['backing']} on {e['mount_point']} ({e['fstype']})"
                     f" | free {size(e['physical_free_bytes'])} physical, {size(e['linux_free_bytes'])} in Linux"
                     f" | usable {size(e['usable_bytes'])} after {size(e['margin_bytes'])} margin")
        if e.get('vhdx'):
            lines.append(f"         VHDX {e['vhdx']['path']} ({size(e['vhdx']['bytes'])})")
        lines += ['         note: '+n for n in e['notes']]
    lines += ['', 'Checks']
    lines += [f"  [{ch['status']}] {ch['id']}: {ch['message']}" for ch in r['checks']]
    lines += ['', 'Overall: '+r['status'], '']
    return '\n'.join(lines)

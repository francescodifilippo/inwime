"""Seam: the doctor report on a fake host (fake /proc and /sys root, injected statvfs and PowerShell)."""
import shutil
import subprocess
from pathlib import Path

import pytest

from inwime import doctor
from fakehost import FakeHost, GIB

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def host(tmp_path):
    return FakeHost(tmp_path)


def storage_entry(report, path):
    return next(s for s in report['storage'] if s['path'] == str(path))


def test_wsl_vhdx_usable_space_is_host_volume_free_minus_margin(host):
    host.add_c_drive(free=131*GIB, total=1700*GIB)
    host.powershell_output = ('D|Ubuntu|C:\\Users\\someone\\AppData\\Local\\wsl\\{guid}\n'
                              'M|16900739072\n')
    store = host.linux/'home/.cache/huggingface/hub'
    report = doctor.report(paths=[store], **host.kwargs())
    entry = storage_entry(report, store)
    assert entry['backing'] == 'wsl-vhdx'
    assert entry['host_volume']['free_bytes'] == 131*GIB
    assert entry['physical_free_bytes'] == 131*GIB      # the host volume, not ext4's 800 GiB
    assert entry['usable_bytes'] == 101*GIB             # 131 GiB - 30 GiB default margin
    assert entry['status'] == 'ok'
    assert report['memory']['host_physical_bytes'] == 16900739072


@pytest.mark.parametrize('case', ['powershell-missing', 'powershell-fails', 'distro-not-registered',
                                  'queries-disabled'])
def test_wsl_vhdx_without_host_facts_is_unknown_not_guessed(host, case):
    host.add_c_drive(free=131*GIB, total=1700*GIB, powershell=case != 'powershell-missing')
    host.powershell_output = {'powershell-fails': None,
                              'distro-not-registered': 'D|Debian|C:\\wsl\\debian\nM|1\n'}.get(
        case, 'D|Ubuntu|C:\\wsl\\ubuntu\nM|16900739072\n')
    store = host.linux/'store'
    report = doctor.report(paths=[store], host_query=case != 'queries-disabled', **host.kwargs())
    entry = storage_entry(report, store)
    assert entry['backing'] == 'wsl-vhdx'
    assert entry['usable_bytes'] is None and entry['physical_free_bytes'] is None
    assert entry['status'] == 'unknown'
    assert entry['linux_free_bytes'] == 800*GIB          # still reported, but never used as physical
    if case in ('powershell-missing', 'queries-disabled'):
        assert host.powershell_calls == 0
        assert report['memory']['host_physical_bytes'] is None


def test_storage_backings_and_margin(tmp_path):
    host = FakeHost(tmp_path)
    host.add_c_drive(free=40*GIB, total=1700*GIB)        # 40 GiB free: below 30 GiB margin + nothing
    host.powershell_output = 'D|Ubuntu|C:\\wsl\\ubuntu\nM|1\n'
    usb = host.base/'mnt/wsl/PHYSICALDRIVE1p1'
    usb.mkdir(parents=True)
    host.add_mount(usb, 'ext4', '/dev/sde1', device='8:65')
    host.set_space(usb, free=400*GIB, total=500*GIB)
    attached = host.base/'attached vhd'                  # space in the name: mountinfo escape
    attached.mkdir()
    host.add_mount(attached, 'ext4', '/dev/sdf', device='8:80')
    host.set_space(attached, free=50*GIB, total=60*GIB)
    paths = [host.cdrive/'models', host.linux/'store', attached/'models']
    report = doctor.report(paths=paths, **host.kwargs())
    on_c, on_vhdx, on_attached = (storage_entry(report, p) for p in paths)
    assert on_c['backing'] == 'wsl-host-volume' and on_c['usable_bytes'] == 10*GIB
    assert on_vhdx['usable_bytes'] == 10*GIB and on_vhdx['status'] == 'ok'
    assert on_attached['mount_point'] == str(attached)
    assert on_attached['backing'] == 'unknown' and on_attached['usable_bytes'] is None


def test_physical_disk_and_full_margin_and_unwritable(tmp_path):
    host = FakeHost(tmp_path)
    host.add_c_drive(free=20*GIB, total=1700*GIB)
    host.powershell_output = 'D|Ubuntu|C:\\wsl\\ubuntu\nM|1\n'
    usb = tmp_path/'mnt/wsl/PHYSICALDRIVE1p1'
    usb.mkdir(parents=True)
    host.add_mount(usb, 'ext4', '/dev/sde1', device='8:65')
    host.set_space(usb, free=400*GIB, total=500*GIB)
    report = doctor.report(paths=[host.linux/'store', usb/'models'],
                           **host.kwargs(access=lambda p, mode: 'PHYSICALDRIVE' not in str(p)))
    vhdx, disk = report['storage']
    assert vhdx['usable_bytes'] == -10*GIB and vhdx['status'] == 'warn'     # margin would be violated
    assert disk['backing'] == 'wsl-physical-disk' and disk['usable_bytes'] == 370*GIB
    assert disk['writable'] is False and disk['status'] == 'warn'


def test_native_linux_uses_filesystem_free_space_and_never_queries_windows(tmp_path):
    host = FakeHost(tmp_path, wsl=False)
    host.add_c_drive(free=1*GIB, total=2*GIB)
    report = doctor.report(paths=[host.linux/'store'], **host.kwargs())
    entry = report['storage'][0]
    assert report['system']['wsl'] is False and host.powershell_calls == 0
    assert entry['backing'] == 'local' and entry['usable_bytes'] == 770*GIB


def test_cgroup_limit_is_the_most_restrictive_ancestor_and_bounds_application_memory(host):
    host.write('/sys/fs/cgroup/user.slice/app.scope/memory.max', 'max\n')
    host.write('/sys/fs/cgroup/user.slice/app.scope/memory.current', str(512*1024**2)+'\n')
    host.write('/sys/fs/cgroup/user.slice/app.scope/memory.high', 'max\n')
    host.write('/sys/fs/cgroup/user.slice/memory.max', str(2*GIB)+'\n')
    host.write('/sys/fs/cgroup/memory.max', 'max\n')     # never present on a real root, harmless
    memory = doctor.report(**host.kwargs())['memory']
    assert memory['MemTotal'] == 10*GIB and memory['MemAvailable'] == 8*GIB
    assert memory['cgroup']['version'] == 2
    assert memory['cgroup']['files']['memory.max'] is None            # 'max' = unlimited here
    assert memory['cgroup']['effective_memory_max'] == 2*GIB          # limited by the parent
    # headroom of the binding ancestor is its own limit minus its own usage, not the leaf's
    assert memory['application_available_bytes'] is None              # parent's usage unreadable
    assert memory['application_available_upper_bound_bytes'] == 2*GIB
    host.write('/sys/fs/cgroup/user.slice/memory.current', str(1536*1024**2)+'\n')
    memory = doctor.report(**host.kwargs())['memory']
    assert memory['application_available_bytes'] == 512*1024**2


def test_headroom_is_the_minimum_over_limited_ancestors(host):
    host.write('/sys/fs/cgroup/user.slice/app.scope/memory.max', str(4*GIB)+'\n')
    host.write('/sys/fs/cgroup/user.slice/app.scope/memory.current', str(3*GIB)+'\n')   # 1 GiB left
    host.write('/sys/fs/cgroup/user.slice/memory.max', str(8*GIB)+'\n')
    host.write('/sys/fs/cgroup/user.slice/memory.current', str(5*GIB)+'\n')             # 3 GiB left
    memory = doctor.report(**host.kwargs())['memory']
    assert memory['cgroup']['effective_memory_max'] == 4*GIB
    assert memory['application_available_bytes'] == 1*GIB


@pytest.mark.parametrize('cgroup', ['1:memory:/user.slice\n', None])
def test_cgroup_v1_or_unreadable_is_unknown(host, cgroup):
    if cgroup is None:
        (host.root/'proc/self/cgroup').unlink()
    else:
        host.write('/proc/self/cgroup', cgroup)
    memory = doctor.report(**host.kwargs())['memory']
    assert memory['cgroup']['version'] is None
    assert memory['application_available_bytes'] == 8*GIB            # MemAvailable only


def test_absent_kernel_falls_back_to_python(host):
    from inwime import kernel
    if kernel.locate(env={}) is not None:
        pytest.skip('a packaged kernel is installed; the CI no-compiler install covers absence')
    env = {k: v for k, v in host.kwargs()['env'].items() if k != 'INWIME_CPU_KERNEL'}
    report = doctor.report(**host.kwargs(env=env))
    assert report['cpu_kernel']['implementation'] == 'python-fp64'
    assert check(report, 'cpu-kernel')['status'] == 'warn'


def test_unreadable_meminfo_is_null_not_zero(host):
    (host.root/'proc/meminfo').unlink()
    memory = doctor.report(**host.kwargs())['memory']
    assert memory['MemTotal'] is None and memory['application_available_bytes'] is None




def compile_kernel(tmp_path, source):
    if shutil.which('cc') is None:
        pytest.skip('no C compiler')
    src, lib = tmp_path/'k.c', tmp_path/'libk.so'
    src.write_text(source)
    subprocess.run(['cc', '-O2', '-std=c11', '-fPIC', '-shared', '-ffp-contract=off', str(src), '-o', str(lib)],
                   check=True)
    return lib


def check(report, name):
    return next(c for c in report['checks'] if c['id'] == name)


def test_kernel_disabled_falls_back_to_python_with_warning(host):
    report = doctor.report(**host.kwargs())
    assert report['cpu_kernel']['implementation'] == 'python-fp64'
    assert report['cpu_kernel']['native'] is False
    assert check(report, 'cpu-kernel')['status'] == 'warn'


def test_pinned_kernel_passes_the_bitwise_self_test(host, tmp_path):
    source = (ROOT/'src/cpu_kernel.c').read_text()
    lib = compile_kernel(tmp_path, source)
    env = host.kwargs()['env'] | {'INWIME_CPU_KERNEL': str(lib)}
    report = doctor.report(**host.kwargs(env=env))
    assert report['cpu_kernel']['native'] is True
    assert report['cpu_kernel']['implementation'] == 'c-fp64-ordered'
    assert report['cpu_kernel']['self_test']['passed'] is True
    assert check(report, 'cpu-kernel')['status'] == 'ok'


def test_kernel_that_reorders_the_sum_fails_the_self_test(host, tmp_path):
    # Same API, reversed accumulation order: mathematically equal, not bit-identical.
    source = (ROOT/'src/cpu_kernel.c').read_text().replace(
        'for (size_t j = 0; j < number; ++j) sum += weights[i+j]*x[column+j];',
        'for (size_t j = number; j-- > 0;) sum += weights[i+j]*x[column+j];')
    assert 'j-- > 0' in source
    lib = compile_kernel(tmp_path, source)
    env = host.kwargs()['env'] | {'INWIME_CPU_KERNEL': str(lib)}
    report = doctor.report(**host.kwargs(env=env))
    assert report['cpu_kernel']['native'] is False
    assert report['cpu_kernel']['self_test']['passed'] is False
    assert check(report, 'cpu-kernel')['status'] == 'fail' and report['status'] == 'fail'


def test_unloadable_kernel_is_a_failure(host, tmp_path):
    bogus = tmp_path/'not-a-library.so'
    bogus.write_bytes(b'not ELF')
    env = host.kwargs()['env'] | {'INWIME_CPU_KERNEL': str(bogus)}
    report = doctor.report(**host.kwargs(env=env))
    assert check(report, 'cpu-kernel')['status'] == 'fail'


def test_foreign_library_without_kernel_symbols_is_a_failure_not_a_crash(host, tmp_path):
    lib = compile_kernel(tmp_path, 'int unrelated(void) { return 0; }\n')
    env = host.kwargs()['env'] | {'INWIME_CPU_KERNEL': str(lib)}
    report = doctor.report(**host.kwargs(env=env))
    assert check(report, 'cpu-kernel')['status'] == 'fail'


def test_cgroup_without_any_limit_is_unlimited_not_unknown(host):
    host.write('/sys/fs/cgroup/user.slice/app.scope/memory.max', 'max\n')
    host.write('/sys/fs/cgroup/user.slice/memory.max', 'max\n')
    group = doctor.report(**host.kwargs())['memory']['cgroup']
    assert group['effective_memory_max'] is None and group['memory_limit'] == 'unlimited'
    assert 'unlimited' in doctor.render(doctor.report(**host.kwargs()))


def test_cgroup_with_unreadable_limits_is_unknown(host):
    group = doctor.report(**host.kwargs())['memory']['cgroup']     # v2 path, but no files
    assert group['memory_limit'] == 'unknown'


def compile_operators(tmp_path, source):
    if shutil.which('cc') is None:
        pytest.skip('no C compiler')
    src, lib = tmp_path/'ops.c', tmp_path/'libops.so'
    src.write_text(source)
    subprocess.run(['cc', '-O2', '-std=c11', '-fPIC', '-shared', '-ffp-contract=off', '-fno-builtin',
                    '-I', str(ROOT/'src'), str(src), '-o', str(lib), '-lm'], check=True)
    return lib


def test_operators_disabled_fall_back_to_python_with_warning(host):
    report = doctor.report(**host.kwargs())
    assert report['operators']['native'] is False
    assert {o['implementation'] for o in report['operators']['operators'].values()} == {'python-fp64'}
    assert check(report, 'operator-kernel')['status'] == 'warn'


def test_operator_kernel_passes_the_gate(host, tmp_path):
    lib = compile_operators(tmp_path, (ROOT/'src/inwime_ops.c').read_text())
    report = doctor.report(**host.kwargs(env=host.kwargs()['env'] | {'INWIME_OPS_KERNEL': str(lib)}))
    assert report['operators']['native'] is True
    assert check(report, 'operator-kernel')['status'] == 'ok'
    assert 'operators c-fp64-ordered-ops' in doctor.render(report)


def test_an_operator_that_reorders_its_sum_is_replaced_and_fails_the_check(host, tmp_path):
    # Same API, reversed accumulation order in the linear operator only.
    source = (ROOT/'src/inwime_ops.c').read_text().replace(
        'for (size_t j = 0; j < number; ++j) total += w[i+j]*xp[j];',
        'for (size_t j = number; j-- > 0;) total += w[i+j]*xp[j];')
    assert 'j-- > 0' in source
    lib = compile_operators(tmp_path, source)
    report = doctor.report(**host.kwargs(env=host.kwargs()['env'] | {'INWIME_OPS_KERNEL': str(lib)}))
    operators = report['operators']['operators']
    assert operators['linear']['implementation'] == 'python-fp64' and operators['linear']['passed'] is False
    assert operators['attention']['implementation'] == 'c-fp64-ordered-ops'
    assert check(report, 'operator-kernel')['status'] == 'fail' and report['status'] == 'fail'

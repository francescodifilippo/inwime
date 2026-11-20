"""Seam: the `inwime` command as a user runs it (real host, non-destructive)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import inwime
from inwime import doctor
from fakehost import FakeHost, GIB


# Run the command from a neutral directory against the same `inwime` this test imported:
# the installed package in CI, the source tree in a development checkout.
PACKAGE_PARENT = str(Path(inwime.__file__).resolve().parents[1])


def run(*args, env=None, cwd=None):
    full = dict(os.environ, **(env or {}))
    full['PYTHONPATH'] = PACKAGE_PARENT
    return subprocess.run([sys.executable, '-m', 'inwime', *args], capture_output=True, text=True,
                          env=full, timeout=120, cwd=cwd or Path(full.get('HOME', '/')))


@pytest.fixture
def home(tmp_path):
    h = tmp_path/'home-of-alice'
    h.mkdir()
    return h


def isolated(home, **extra):
    env = {'HOME': str(home), 'INWIME_CPU_KERNEL': 'off'}
    for key in ('HF_HOME', 'HF_HUB_CACHE', 'XDG_CACHE_HOME'):
        env[key] = ''
    env.update(extra)
    return env


INT, STR, BOOL, NUL = int, str, bool, type(None)

# Contract of schema inwime-doctor/1: required fields and their types (None = not detected).
SCHEMA = {
    '': dict(schema=STR, inwime_version=STR, generated_utc=STR, system=dict, cpu=dict, memory=dict,
             host_query=dict, storage=list, cpu_kernel=dict, checks=list, status=STR),
    'system': dict(os=STR, kernel_release=STR, distribution=(STR, NUL), wsl=BOOL, wsl_distro=(STR, NUL),
                   machine=STR, python=STR, implementation=STR),
    'cpu': dict(model=(STR, NUL), logical_cpus=(INT, NUL), flags=(dict, NUL), affinity_cpus=(INT, NUL)),
    'memory': dict(MemTotal=(INT, NUL), MemAvailable=(INT, NUL), SwapTotal=(INT, NUL), SwapFree=(INT, NUL),
                   cgroup=dict, rlimits=(dict, NUL), application_available_bytes=(INT, NUL),
                   application_available_upper_bound_bytes=(INT, NUL), host_physical_bytes=(INT, NUL)),
    'cpu_kernel': dict(path=(STR, NUL), implementation=STR, native=BOOL, self_test=(dict, NUL),
                       status=STR, detail=STR),
    'storage[]': dict(path=STR, exists=BOOL, existing_ancestor=STR, writable=BOOL, mount_point=(STR, NUL),
                      fstype=(STR, NUL), source=(STR, NUL), linux_free_bytes=(INT, NUL),
                      linux_total_bytes=(INT, NUL), margin_bytes=INT, backing=STR, host_volume=(dict, NUL),
                      vhdx=(dict, NUL), physical_free_bytes=(INT, NUL), usable_bytes=(INT, NUL),
                      notes=list, status=STR),
    'checks[]': dict(id=STR, status=STR, message=STR),
}
STATUSES = {'ok', 'unknown', 'warn', 'fail'}
BACKINGS = {'local', 'wsl-vhdx', 'wsl-host-volume', 'wsl-physical-disk', 'unknown'}


def assert_schema(report):
    def conforms(obj, fields, where):
        assert isinstance(obj, dict), where
        for name, kind in fields.items():
            assert name in obj, f'{where}.{name} missing'
            assert isinstance(obj[name], kind) and not (kind is INT and isinstance(obj[name], bool)), \
                f'{where}.{name}: {type(obj[name]).__name__}'
    conforms(report, SCHEMA[''], 'report')
    assert report['schema'] == 'inwime-doctor/1' and report['status'] in STATUSES
    for section in ('system', 'cpu', 'memory', 'cpu_kernel'):
        conforms(report[section], SCHEMA[section], section)
    for entry in report['storage']:
        conforms(entry, SCHEMA['storage[]'], 'storage[]')
        assert entry['backing'] in BACKINGS and entry['status'] in STATUSES
    for item in report['checks']:
        conforms(item, SCHEMA['checks[]'], 'checks[]')
        assert item['status'] in STATUSES


def test_version():
    done = run('--version')
    assert done.returncode == 0 and done.stdout.strip() == 'inwime '+inwime.__version__


def test_doctor_json_on_the_real_host_is_valid_and_writes_nothing(home, tmp_path):
    before = sorted(p for p in tmp_path.rglob('*'))
    done = run('doctor', '--json', '--no-host-query', env=isolated(home))
    assert done.returncode == 0, done.stderr
    report = json.loads(done.stdout)
    assert report['schema'] == 'inwime-doctor/1'
    assert_schema(report)
    assert sorted(p for p in tmp_path.rglob('*')) == before          # no cache dir created
    store = report['storage'][0]
    assert store['path'] == '~/.cache/huggingface/hub'               # home redacted by default
    assert str(home) not in done.stdout


def test_show_paths_disables_redaction(home):
    done = run('doctor', '--json', '--no-host-query', '--show-paths', env=isolated(home))
    assert json.loads(done.stdout)['storage'][0]['path'] == str(home/'.cache/huggingface/hub')


def test_text_output_lists_checks_and_explicit_paths(home, tmp_path):
    done = run('doctor', '--no-host-query', '--path', str(tmp_path), '--host-margin-gib', '0',
               env=isolated(home))
    assert done.returncode == 0, done.stderr
    assert '[warn] cpu-kernel' in done.stdout and 'Python FP64 fallback' in done.stdout
    assert 'storage' in done.stdout


def test_failed_check_gives_nonzero_exit(home, tmp_path):
    bogus = tmp_path/'bogus.so'
    bogus.write_bytes(b'x')
    done = run('doctor', '--json', '--no-host-query', env=isolated(home, INWIME_CPU_KERNEL=str(bogus)))
    assert done.returncode == 1 and json.loads(done.stdout)['status'] == 'fail'


def test_redaction_hides_windows_and_linux_user_names(tmp_path):
    host = FakeHost(tmp_path)
    host.add_c_drive(free=100*GIB, total=1700*GIB)
    host.powershell_output = 'D|Ubuntu|C:\\Users\\Alice Smith\\AppData\\Local\\wsl\\{g}\nM|1\n'
    home = host.linux/'home'
    report = doctor.report(paths=[home/'store'], **host.kwargs())
    text = json.dumps(doctor.redact(report, home=str(home)))
    assert 'Alice Smith' not in text and str(home) not in text
    assert 'C:\\\\Users\\\\<user>\\\\AppData' in text and '~/store' in text


@pytest.mark.parametrize('path, shown', [
    ('/mnt/c/Users/alice/models', '/mnt/c/Users/<user>/models'),
    ('C:/Users/alice/x', 'C:/Users/<user>/x'),
    ('D:\\Users\\alice\\x', 'D:\\Users\\<user>\\x'),
    ('/mnt/c/Program Files/x', '/mnt/c/Program Files/x'),
])
def test_redaction_of_windows_profile_paths_in_every_form(path, shown):
    assert doctor.redact({'p': path}, home='/home/bob') == {'p': shown}


def test_fake_host_report_conforms_to_the_schema(tmp_path):
    host = FakeHost(tmp_path)
    host.add_c_drive(free=100*GIB, total=1700*GIB)
    host.powershell_output = 'D|Ubuntu|C:\\wsl\\u\nM|17\n'
    assert_schema(doctor.report(paths=[host.linux/'a', host.cdrive/'b'], **host.kwargs()))


@pytest.mark.parametrize('env, expected', [
    ({'HF_HUB_CACHE': '/x/hub', 'HUGGINGFACE_HUB_CACHE': '/old', 'HF_HOME': '/h'}, '/x/hub'),
    ({'HUGGINGFACE_HUB_CACHE': '/old', 'HF_HOME': '/h'}, '/old'),
    ({'HF_HOME': '/h', 'XDG_CACHE_HOME': '/c'}, '/h/hub'),
    ({'XDG_CACHE_HOME': '/c', 'HOME': '/home/bob'}, '/c/huggingface/hub'),
    ({'HOME': '/home/bob'}, '/home/bob/.cache/huggingface/hub'),
])
def test_default_store_follows_huggingface_hub_resolution(env, expected):
    assert str(doctor.default_store(env)) == expected

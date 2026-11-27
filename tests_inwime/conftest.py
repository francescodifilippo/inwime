"""Shared harness for the product tests: the `inwime` command in a guarded subprocess.

Every command runs in a subprocess whose network is guarded (netguard/sitecustomize.py): a
name resolution or connection outside loopback fails and is logged, and each test asserts
that the log stays empty and that the fake Hub saw no request it does not implement. The
subprocess gets a private HOME, so the Hugging Face cache and the InWiMe registry are fresh.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import inwime
from fakehub import FakeHub


PACKAGE_PARENT = str(Path(inwime.__file__).resolve().parents[1])
GUARD = str(Path(__file__).resolve().parent/'netguard')
SCRUBBED = ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE',
            'HF_TOKEN_PATH', 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'XDG_CACHE_HOME', 'HTTP_PROXY',
            'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'INWIME_HOME',
            'XDG_DATA_HOME', 'INWIME_FAKE_FREE_BYTES', 'INWIME_OPS_KERNEL', 'INWIME_COMPUTE_PATH')


@pytest.fixture
def hub():
    with FakeHub() as h:
        yield h
        assert h.unexpected == []


@pytest.fixture
def home(tmp_path):
    h = tmp_path/'home-of-alice'
    h.mkdir()
    return h


class Run:
    def __init__(self, hub, home, tmp_path):
        self.hub, self.home, self.guard_log = hub, home, tmp_path/'netguard.log'

    def environment(self, token=None, endpoint=None, env=None, free=None):
        """`free`: physical free bytes the fake storage reports, one value per measure
        (the last repeats); 'unknown' for an undetectable volume. None: the real host."""
        full = {k: v for k, v in os.environ.items() if k not in SCRUBBED}
        full.update(HOME=str(self.home), HF_HOME=str(self.home/'.cache/huggingface'),
                    HF_ENDPOINT=endpoint or self.hub.endpoint, HF_HUB_DISABLE_TELEMETRY='1',
                    HF_HUB_DISABLE_PROGRESS_BARS='1', NO_PROXY='*', INWIME_CPU_KERNEL='off', INWIME_OPS_KERNEL='off',
                    INWIME_NETGUARD_LOG=str(self.guard_log),
                    PYTHONPATH=os.pathsep.join((GUARD, PACKAGE_PARENT)))
        if token:
            full['HF_TOKEN'] = token
        if free is not None:
            values = free if isinstance(free, (list, tuple)) else [free]
            full['INWIME_FAKE_FREE_BYTES'] = ','.join(str(v) for v in values)
        full.update(env or {})
        return full

    def command(self, args):
        return [sys.executable, '-m', 'inwime', 'models', *args]

    def check_guard(self, expect_guard=False):
        violations = self.guard_log.read_text() if self.guard_log.exists() else ''
        if not expect_guard:
            assert violations == '', violations
        return violations

    def __call__(self, *args, input=None, expect_guard=False, **env):
        result = subprocess.run(self.command(args), capture_output=True, text=True, input=input,
                                env=self.environment(**env), timeout=120, cwd=self.home)
        result.violations = self.check_guard(expect_guard)
        return result

    def start(self, *args, **env):
        """The command in the background, for concurrency tests."""
        return subprocess.Popen(self.command(args), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, env=self.environment(**env),
                                cwd=self.home)

    def json(self, *args, **kwargs):
        result = self(*args, '--json', **kwargs)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)


@pytest.fixture
def run(hub, home, tmp_path):
    return Run(hub, home, tmp_path)

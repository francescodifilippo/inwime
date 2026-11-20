"""Increment 10, C3: test inventory identity checks (Issue #22) and adversarial mutations.

A tiny throwaway project (git repository + tests/) is generated, inventoried and run
with --junitxml. Every mutation must give FAIL or NOT PROVEN, never PASS, including
the case where the number of passed tests stays the same.
"""
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from tools import inventory10 as inv

ROOT = Path(__file__).resolve().parents[1]

BASE = '''
import pytest

def test_alpha():
    assert 1 + 1 == 2

@pytest.mark.parametrize('x', [1, 2])
def test_beta(x):
    assert x > 0

class TestGroup:
    def test_gamma(self):
        assert True
'''
NEW = '''
def test_delta():
    assert 2 * 2 == 4
'''


def git(cwd, *args):
    subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True,
                   env={'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@example.invalid',
                        'GIT_COMMITTER_NAME': 't', 'GIT_COMMITTER_EMAIL': 't@example.invalid',
                        'PATH': '/usr/bin:/bin', 'HOME': str(cwd)})


def pytest_junit(project, name='junit.xml'):
    out = project/name
    # -B: same-size rewrites within one second must not reuse a stale .pyc
    subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests',
                    f'--junitxml={out}'], cwd=project, capture_output=True)
    return out


@pytest.fixture
def project(tmp_path):
    p = tmp_path/'proj'
    (p/'tests').mkdir(parents=True)
    (p/'tests'/'test_base.py').write_text(BASE)
    git(p, 'init', '-q')
    git(p, 'add', '.')
    git(p, 'commit', '-q', '-m', 'base')
    base = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=p, capture_output=True, text=True).stdout.strip()
    (p/'tests'/'test_increment10_new.py').write_text(NEW)
    inventory = inv.generate(p, base, venv=None)
    return p, inventory


def verdict(project, inventory, junit, **kw):
    return inv.verify(inventory, junit, project, **kw)


def test_generate_groups_counts_and_digests(project):
    p, inventory = project
    assert inventory['expected_counts'] == {'total': 5, 'baseline-m1-m9': 4, 'm10-new': 1}
    assert 'tests/test_base.py::TestGroup::test_gamma' in inventory['groups']['baseline-m1-m9']
    assert set(inventory['test_files_sha256']) == {'tests/test_base.py', 'tests/test_increment10_new.py'}
    assert all(len(d) == 64 for d in inventory['test_function_sha256'].values())
    assert inventory['policy']['skipped'] == 'FAIL' and inventory['tool_versions']['pytest']


def test_clean_run_passes(project):
    p, inventory = project
    r = verdict(p, inventory, pytest_junit(p))
    assert r['verdict'] == 'PASS' and r['required_passed'] == 5


def test_junit_key_matches_pytest():
    assert inv.junit_key('tests/test_x.py::TestA::test_b[p-1]') == ('tests.test_x.TestA', 'test_b[p-1]')
    assert inv.junit_key('tests/test_x.py::test_c') == ('tests.test_x', 'test_c')


def test_removed_test_fails(project):
    p, inventory = project
    f = p/'tests'/'test_base.py'
    f.write_text(BASE.replace('def test_alpha():\n    assert 1 + 1 == 2\n', ''))
    r = verdict(p, inventory, pytest_junit(p))
    assert r['verdict'] == 'FAIL' and r['missing_required'] == ['tests/test_base.py::test_alpha']


def test_substitution_with_same_pass_count_fails(project):
    """One test removed and another added: passed count unchanged, still FAIL."""
    p, inventory = project
    f = p/'tests'/'test_base.py'
    f.write_text(BASE.replace('def test_alpha():', 'def test_alpha_replacement():'))
    junit = pytest_junit(p)
    cases, _, _ = inv.parse_junit(junit)
    assert sum(o == 'passed' for _, o in cases) == inventory['expected_counts']['total']
    r = verdict(p, inventory, junit)
    assert r['verdict'] == 'FAIL' and 'tests/test_base.py::test_alpha' in r['missing_required']


def test_same_name_different_body_fails(project):
    p, inventory = project
    f = p/'tests'/'test_base.py'
    f.write_text(BASE.replace('assert 1 + 1 == 2', 'assert True'))
    r = verdict(p, inventory, pytest_junit(p))
    assert r['verdict'] == 'FAIL' and r['test_files_changed_or_missing'] == ['tests/test_base.py']


def test_skipped_and_xfailed_fail(project):
    p, inventory = project
    f = p/'tests'/'test_base.py'
    f.write_text(BASE.replace('def test_alpha():\n', "@pytest.mark.skip('x')\ndef test_alpha():\n")
                 .replace('    def test_gamma(self):\n', "    @pytest.mark.xfail(reason='x')\n    def test_gamma(self):\n"
                          "        assert False\n    def _unused(self):\n"))
    r = verdict(p, inventory, pytest_junit(p))
    assert r['verdict'] == 'FAIL'
    assert r['not_passed'] == {'tests/test_base.py::test_alpha': 'skipped',
                               'tests/test_base.py::TestGroup::test_gamma': 'xfailed'}


def test_failure_and_error_fail(project):
    p, inventory = project
    (p/'tests'/'test_increment10_new.py').write_text(NEW.replace('== 4', '== 5'))
    r = verdict(p, inventory, pytest_junit(p))
    assert r['verdict'] == 'FAIL' and r['not_passed'] == {'tests/test_increment10_new.py::test_delta': 'failure'}


def test_duplicated_testcase_fails(project):
    p, inventory = project
    junit = pytest_junit(p)
    raw = junit.read_text()
    case = re.search(r'<testcase [^>]*name="test_alpha"[^>]*/>', raw).group(0)
    raw = raw.replace(case, case+case).replace('tests="5"', 'tests="6"')
    junit.write_text(raw)
    r = verdict(p, inventory, junit)
    assert r['verdict'] == 'FAIL' and r['duplicates']


def test_falsified_junit_detected(project):
    """A failing run edited into a passing one: caught by the recorded digest or by counts."""
    p, inventory = project
    (p/'tests'/'test_increment10_new.py').write_text(NEW.replace('== 4', '== 5'))
    junit = pytest_junit(p)
    recorded = inv.sha256(junit)
    raw = junit.read_text()
    forged = re.sub(r'<failure[^>]*>.*?</failure>', '', raw, flags=re.S)
    junit.write_text(forged)
    assert verdict(p, inventory, junit, junit_sha256=recorded)['verdict'] == 'FAIL'
    r = verdict(p, inventory, junit)          # without the digest: declared failures="1" != observed 0
    assert r['verdict'] == 'FAIL' and 'declared counts' in ' '.join(r['fail_reasons'])


def test_extra_test_is_not_proven(project):
    p, inventory = project
    inventory = json.loads(json.dumps(inventory))
    (p/'tests'/'test_increment10_new.py').write_text(NEW + '\n\ndef test_epsilon():\n    pass\n')
    inventory['test_files_sha256']['tests/test_increment10_new.py'] = inv.sha256(p/'tests'/'test_increment10_new.py')
    r = verdict(p, inventory, pytest_junit(p))
    assert r['verdict'] == 'NOT PROVEN' and r['extra_not_in_inventory'] == ['tests.test_increment10_new::test_epsilon']


def test_new_test_file_is_not_proven(project):
    p, inventory = project
    junit = pytest_junit(p)
    (p/'tests'/'test_increment10_unlisted.py').write_text('def test_z():\n    pass\n')
    assert verdict(p, inventory, junit)['verdict'] == 'NOT PROVEN'


def test_missing_malformed_empty_and_dtd_junit(project, tmp_path):
    p, inventory = project
    assert verdict(p, inventory, tmp_path/'absent.xml')['verdict'] == 'NOT PROVEN'
    bad = tmp_path/'bad.xml'
    bad.write_text('<testsuites><testsuite tests="1">')
    assert verdict(p, inventory, bad)['verdict'] == 'FAIL'
    empty = tmp_path/'empty.xml'
    empty.write_text('<testsuites><testsuite tests="0" failures="0" errors="0" skipped="0"/></testsuites>')
    assert verdict(p, inventory, empty)['verdict'] == 'FAIL'
    dtd = tmp_path/'dtd.xml'
    dtd.write_text('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><testsuites/>')
    assert verdict(p, inventory, dtd)['verdict'] == 'FAIL'


def test_cli_pinned_inventory_digest(project, tmp_path):
    p, inventory = project
    path = tmp_path/'inventory.json'
    path.write_text(json.dumps(inventory))
    junit = pytest_junit(p)
    ok = inv.main(['verify', '--inventory', str(path), '--junit', str(junit), '--root', str(p),
                   '--inventory-sha256', inv.sha256(path)])
    assert ok == 0
    assert inv.main(['verify', '--inventory', str(path), '--junit', str(junit), '--root', str(p),
                     '--inventory-sha256', '0'*64]) == 5


def test_repository_inventory_is_pinned_and_consistent():
    """The frozen M10 inventory: pinned in protocol.json, baseline group equals the 639de22 node set."""
    protocol = json.loads((ROOT/'results'/'increment10'/'protocol.json').read_text())
    entry = protocol['test_inventory']
    path = ROOT/entry['path']
    assert inv.sha256(path) == entry['sha256']
    inventory = json.loads(path.read_text())
    assert inventory['base_commit'] == protocol['base_commit']
    assert inventory['expected_counts']['baseline-m1-m9'] == entry['baseline_node_ids_at_base'] == 910
    assert inventory['expected_counts']['total'] == sum(len(v) for v in inventory['groups'].values())
    assert all(v == 'FAIL' for k, v in inventory['policy'].items() if k in ('skipped', 'xfailed', 'error'))

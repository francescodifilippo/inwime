"""Increment 10, amendment 001: traceable amendment chain and runner-aware inventory check.

Review of PR #24, observation 2: the GitHub runner cannot run the 11 increment-9
tests that need local unsanitized originals. Those tests are declared in the
amendment with their exact skip message; on the runner they are NOT VERIFIABLE,
never PASS, and the full-inventory verdict of the runner stays NOT PROVEN. The
local attestation (full JUnit of a local run) is verified separately.

Also: the amendment chain cannot move frozen sections, must start every changed
digest from the effective one, never drops a required test, and the frozen M10
evidence (protocol, inventory, archived reports) is byte-identical.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools import inventory10 as inv
from tools import protocol10

ROOT = Path(__file__).resolve().parents[1]
R10 = ROOT/'results'/'increment10'
AMENDMENT = R10/'amendments'/'001-acquisition-host-gate-and-ci-inventory.json'
SKIP = 'local originals are not on this machine'

SOURCE = f'''
import os
import pytest

def test_portable():
    assert 1 + 1 == 2

def test_needs_originals():
    value = os.environ.get('HAVE_ORIGINALS')
    if not value:
        pytest.skip({SKIP!r})
    assert value != 'broken'

def test_other():
    assert True
'''


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(cwd, *args):
    subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True,
                   env={'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@example.invalid',
                        'GIT_COMMITTER_NAME': 't', 'GIT_COMMITTER_EMAIL': 't@example.invalid',
                        'PATH': '/usr/bin:/bin', 'HOME': str(cwd)})


def junit(project, name, originals):
    out = project/name
    env = {k: v for k, v in os.environ.items() if k != 'HAVE_ORIGINALS'}
    if originals:
        env['HAVE_ORIGINALS'] = '1' if originals is True else originals
    subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider', 'tests',
                    f'--junitxml={out}'], cwd=project, capture_output=True, env=env)
    return out


@pytest.fixture
def project(tmp_path):
    p = tmp_path/'proj'
    (p/'tests').mkdir(parents=True)
    (p/'tests'/'test_m9.py').write_text(SOURCE)
    git(p, 'init', '-q')
    git(p, 'add', '.')
    git(p, 'commit', '-q', '-m', 'base')
    base = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=p, capture_output=True, text=True).stdout.strip()
    return p, inv.generate(p, base, venv=None)


NODE = 'tests/test_m9.py::test_needs_originals'


# ---------------------------------------------------------------- runner classification

def test_declared_skip_is_not_verifiable_never_pass(project):
    p, inventory = project
    r = inv.verify(inventory, junit(p, 'runner.xml', False), p, not_verifiable={NODE: SKIP})
    assert r['not_verifiable_on_runner'] == [NODE]
    assert r['runner_verdict'] == 'PASS'
    assert r['verdict'] == 'NOT PROVEN'                       # the skip is never counted as PASS
    assert r['required_passed'] == 2 and r['required_total'] == 3


def test_same_skip_without_declaration_fails(project):
    p, inventory = project
    r = inv.verify(inventory, junit(p, 'runner.xml', False), p)
    assert r['verdict'] == 'FAIL' and r['not_passed'] == {NODE: 'skipped'}


@pytest.mark.parametrize('message', ['another reason', SKIP + ' ', SKIP.upper()])
def test_declared_test_skipped_for_another_reason_fails(project, message):
    p, inventory = project
    r = inv.verify(inventory, junit(p, 'runner.xml', False), p, not_verifiable={NODE: message})
    assert r['runner_verdict'] == 'FAIL' and r['verdict'] == 'FAIL'


def test_other_test_skipped_on_runner_fails(project):
    p, inventory = project
    r = inv.verify(inventory, junit(p, 'runner.xml', False), p,
                   not_verifiable={'tests/test_m9.py::test_portable': SKIP})
    # test_portable is declared but passed (fine); test_needs_originals skipped undeclared: FAIL
    assert r['runner_verdict'] == 'FAIL' and NODE in r['not_passed']


def test_declared_test_that_fails_is_fail(project):
    p, inventory = project
    r = inv.verify(inventory, junit(p, 'broken.xml', 'broken'), p, not_verifiable={NODE: SKIP})
    assert r['not_passed'] == {NODE: 'failure'}
    assert r['verdict'] == 'FAIL' and r['runner_verdict'] == 'FAIL'


def test_declared_test_passing_locally_is_full_pass(project):
    p, inventory = project
    r = inv.verify(inventory, junit(p, 'local.xml', True), p, not_verifiable={NODE: SKIP})
    assert r['verdict'] == 'PASS' and r['runner_verdict'] == 'PASS' and r['not_verifiable_on_runner'] == []
    assert inv.verify(inventory, junit(p, 'local2.xml', True), p)['verdict'] == 'PASS'


def test_declaration_outside_inventory_fails(project):
    p, inventory = project
    r = inv.verify(inventory, junit(p, 'runner.xml', False), p,
                   not_verifiable={NODE: SKIP, 'tests/test_m9.py::test_ghost': SKIP})
    assert r['runner_verdict'] == 'FAIL'


# ---------------------------------------------------------------- ci command

def ci_setup(monkeypatch, p, inventory, attestation=None, passed=True):
    path = p/'inventory.json'
    path.write_text(json.dumps(inventory))
    rules = dict(not_verifiable_on_runner=dict(skip_message=SKIP, node_ids=[NODE]), local_attestation='att.json')
    monkeypatch.setattr(protocol10, 'check', lambda *a: dict(passed=passed, problems=[] if passed else ['x']))
    monkeypatch.setattr(protocol10, 'effective', lambda *a: dict(
        test_inventory=dict(path='inventory.json', sha256=sha256(path)), ci_runner_verification=rules))
    if attestation is not None:
        (p/'att.json').write_text(json.dumps(attestation))


def attestation_for(p, xml, inventory_path):
    return dict(junit=str(xml.relative_to(p)), junit_sha256=sha256(xml), inventory_sha256=sha256(inventory_path))


def test_ci_passes_runner_subset_with_local_attestation(monkeypatch, project, capsys):
    p, inventory = project
    ci_setup(monkeypatch, p, inventory)
    local = junit(p, 'local.xml', True)
    (p/'att.json').write_text(json.dumps(attestation_for(p, local, p/'inventory.json')))
    out = p/'ci.json'
    assert inv.main(['ci', '--junit', str(junit(p, 'runner.xml', False)), '--root', str(p),
                     '--output', str(out)]) == 0
    result = json.loads(out.read_text())
    assert result['runner_verdict'] == 'PASS'
    assert result['full_inventory_verdict_on_runner'] == 'NOT PROVEN'
    assert result['runner']['not_verifiable_on_runner'] == [NODE]
    assert result['local_attestation']['verdict'] == 'PASS'
    assert 'not a runner result' in result['local_attestation']['source']


def test_ci_fails_without_or_with_bad_attestation(monkeypatch, project):
    p, inventory = project
    ci_setup(monkeypatch, p, inventory)
    runner = junit(p, 'runner.xml', False)
    assert inv.main(['ci', '--junit', str(runner), '--root', str(p)]) == 5            # attestation missing
    # an "attestation" that is itself a runner-like JUnit (skip) is not a full local PASS
    (p/'att.json').write_text(json.dumps(attestation_for(p, runner, p/'inventory.json')))
    assert inv.main(['ci', '--junit', str(runner), '--root', str(p)]) == 5
    local = junit(p, 'local.xml', True)
    bad = attestation_for(p, local, p/'inventory.json') | dict(junit_sha256='0'*64)
    (p/'att.json').write_text(json.dumps(bad))
    assert inv.main(['ci', '--junit', str(runner), '--root', str(p)]) == 5
    other = attestation_for(p, local, p/'inventory.json') | dict(inventory_sha256='1'*64)
    (p/'att.json').write_text(json.dumps(other))
    assert inv.main(['ci', '--junit', str(runner), '--root', str(p)]) == 5


def test_ci_fails_when_protocol_chain_broken(monkeypatch, project):
    p, inventory = project
    ci_setup(monkeypatch, p, inventory, passed=False)
    local = junit(p, 'local.xml', True)
    (p/'att.json').write_text(json.dumps(attestation_for(p, local, p/'inventory.json')))
    assert inv.main(['ci', '--junit', str(local), '--root', str(p)]) == 5


# ---------------------------------------------------------------- amendment chain (pure, synthetic root)

def chain_root(tmp_path):
    root = tmp_path/'root'
    (root/'tools').mkdir(parents=True)
    (root/'tools'/'a.py').write_text('a = 1\n')
    old = dict(groups=dict(g=['tests/t.py::test_a', 'tests/t.py::test_b']))
    new = dict(groups=dict(g=['tests/t.py::test_a', 'tests/t.py::test_b', 'tests/t.py::test_c']))
    (root/'inv0.json').write_text(json.dumps(old))
    (root/'inv1.json').write_text(json.dumps(new))
    protocol = dict(pinned_tools_sha256={'tools/a.py': sha256(root/'tools'/'a.py')},
                    test_inventory=dict(path='inv0.json', sha256=sha256(root/'inv0.json')),
                    host_memory_preflight=dict(start_criteria=dict(download=[])))
    raw = json.dumps(protocol).encode()
    (root/'tools'/'a.py').write_text('a = 2\n')
    amendment = dict(
        schema=protocol10.AMENDMENT_SCHEMA, number=1, author='t', utc='2026-11-18T00:00:00Z', reason=['r'],
        protocol_sha256=hashlib.sha256(raw).hexdigest(), previous_sha256=hashlib.sha256(raw).hexdigest(),
        changes=dict(
            pinned_tools_sha256={'tools/a.py': dict(**{'from': protocol['pinned_tools_sha256']['tools/a.py']},
                                                    to=sha256(root/'tools'/'a.py'))},
            test_inventory={'from': protocol['test_inventory'],
                            'to': dict(path='inv1.json', sha256=sha256(root/'inv1.json'), total_node_ids=3)},
            acquisition_host_gate=dict(profile='download', max_report_age_s=900, max_baseline_age_s=43200,
                                       max_future_skew_s=60, interval_tolerance=[0.95, 1.5]),
            ci_runner_verification=dict(not_verifiable_on_runner=dict(skip_message='s',
                                                                      node_ids=['tests/t.py::test_c']))))
    return root, raw, protocol, amendment


def write_amendment(root, amendment, name='001-x.json'):
    d = root/'amendments'
    d.mkdir(exist_ok=True)
    (d/name).write_text(json.dumps(amendment))
    return d


def test_valid_chain_passes(tmp_path):
    root, raw, protocol, amendment = chain_root(tmp_path)
    d = write_amendment(root, amendment)
    assert protocol10.check_amendments(raw, protocol, d, root) == []


def test_no_amendment_means_tool_change_detected(tmp_path):
    root, raw, protocol, amendment = chain_root(tmp_path)
    problems = protocol10.check_amendments(raw, protocol, root/'amendments', root)
    assert problems == ['pinned tool differs from its effective digest: tools/a.py']


MUTATIONS = {
    'wrong_from': lambda a: a['changes']['pinned_tools_sha256']['tools/a.py'].update({'from': '0'*64}),
    'wrong_to': lambda a: a['changes']['pinned_tools_sha256']['tools/a.py'].update(to='0'*64),
    'broken_previous': lambda a: a.update(previous_sha256='0'*64),
    'other_protocol': lambda a: a.update(protocol_sha256='0'*64),
    'numbering': lambda a: a.update(number=2),
    'schema': lambda a: a.update(schema='x'),
    'no_reason': lambda a: a.update(reason=[]),
    'threshold_section': lambda a: a['changes'].update(host_memory_preflight={}),
    'numerical_section': lambda a: a['changes'].update(numerical_contract={}),
    'inventory_from': lambda a: a['changes']['test_inventory'].update({'from': dict(path='x', sha256='y')}),
    'inventory_sha': lambda a: a['changes']['test_inventory']['to'].update(sha256='0'*64),
    'inventory_total': lambda a: a['changes']['test_inventory']['to'].update(total_node_ids=4),
    'gate_profile': lambda a: a['changes']['acquisition_host_gate'].update(profile='nope'),
    'gate_age': lambda a: a['changes']['acquisition_host_gate'].update(max_report_age_s=0),
    'gate_tolerance': lambda a: a['changes']['acquisition_host_gate'].update(interval_tolerance=[1.2, 1.5]),
    'ci_unknown_test': lambda a: a['changes']['ci_runner_verification']['not_verifiable_on_runner'].update(
        node_ids=['tests/t.py::test_ghost']),
    'ci_no_message': lambda a: a['changes']['ci_runner_verification']['not_verifiable_on_runner'].update(
        skip_message=''),
}


@pytest.mark.parametrize('mutation', sorted(MUTATIONS))
def test_chain_mutations_detected(tmp_path, mutation):
    root, raw, protocol, amendment = chain_root(tmp_path)
    MUTATIONS[mutation](amendment)
    d = write_amendment(root, amendment)
    assert protocol10.check_amendments(raw, protocol, d, root) != []


def test_amended_inventory_dropping_a_test_detected(tmp_path):
    root, raw, protocol, amendment = chain_root(tmp_path)
    (root/'inv1.json').write_text(json.dumps(dict(groups=dict(g=['tests/t.py::test_a', 'tests/t.py::test_c']))))
    amendment['changes']['test_inventory']['to'].update(sha256=sha256(root/'inv1.json'), total_node_ids=2)
    amendment['changes']['ci_runner_verification']['not_verifiable_on_runner']['node_ids'] = ['tests/t.py::test_a']
    d = write_amendment(root, amendment)
    problems = protocol10.check_amendments(raw, protocol, d, root)
    assert any('removed or renamed' in x for x in problems)


def test_misnamed_amendment_file_detected(tmp_path):
    root, raw, protocol, amendment = chain_root(tmp_path)
    d = write_amendment(root, amendment, name='007-x.json')
    assert any('numbering' in x for x in protocol10.check_amendments(raw, protocol, d, root))


# ---------------------------------------------------------------- repository amendment 001

def amendment():
    return json.loads(AMENDMENT.read_text())


def test_repository_chain_verifies():
    result = protocol10.check()
    assert result['passed'], result['problems']
    assert result['amendments'][0] == AMENDMENT.name     # later amendments extend the chain


def test_amendment_moves_no_threshold_and_keeps_every_required_test():
    a = amendment()
    assert set(a['changes']) <= protocol10.AMENDABLE
    frozen = json.loads((R10/'protocol.json').read_text())
    old = protocol10.node_ids(ROOT/frozen['test_inventory']['path'])
    new = protocol10.node_ids(ROOT/a['changes']['test_inventory']['to']['path'])
    assert old < new and len(old) == 1053


def test_frozen_evidence_is_byte_identical():
    """protocol.json, protocol.sha256, the frozen inventory and the archived evidence index are untouched."""
    frozen = amendment()['frozen_evidence_preserved']
    for rel, digest in frozen.items():
        assert sha256(ROOT/rel) == digest, rel
    sums = (R10/'SHA256SUMS').read_text().split('\n')
    for line in filter(None, sums):
        digest, name = line.split(maxsplit=1)
        assert sha256(R10/name.lstrip('*')) == digest, name


def test_runner_unverifiable_set_is_exactly_the_m9_local_originals_cases():
    rules = amendment()['changes']['ci_runner_verification']['not_verifiable_on_runner']
    assert rules['skip_message'] == 'local unsanitized originals are not on this machine'
    nodes = rules['node_ids']
    assert len(nodes) == 11 == len(set(nodes))
    assert all(n.startswith('tests/test_increment9_hardening.py::') and n.endswith('real]') for n in nodes)
    source = (ROOT/'tests'/'test_increment9_hardening.py').read_text()
    assert source.count("pytest.skip('local unsanitized originals are not on this machine')") >= 1
    # nothing from increment 10 may be excused on the runner
    assert not any('increment10' in n for n in nodes)


def test_ci_workflow_runs_the_inventory_check():
    workflow = (ROOT/'.github'/'workflows'/'offline-cpu.yml').read_text()
    assert 'python -m tools.inventory10 ci --junit ci-results/pytest.xml' in workflow
    assert 'python -m tools.protocol10 check' in workflow

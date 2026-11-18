"""Versioned pytest inventory for the increment-10 gate (Issue #22, D-5).

generate  Collect the exact node IDs, group them (baseline M1-M9 = test files that
          existed at the base commit; one group per new increment-10 test file),
          record SHA256 of every file under tests/, a per-test source digest, tool
          versions, expected counts and the outcome policy. Written once with 'x'
          (never overwritten); its SHA256 is pinned in protocol.json.
verify    Rebuild the identity and outcome of every node ID from a JUnit XML file
          and check the test files on disk against the pinned digests. A removed,
          renamed, substituted, duplicated, skipped, xfailed or errored required test
          is a FAIL even if the number of passed tests stays high enough. A test not
          in the inventory is NOT PROVEN (the inventory drifted: amend it explicitly).

Precedence of verdicts: FAIL > NOT PROVEN > PASS. A missing JUnit file is NOT PROVEN;
an unreadable, malformed or internally inconsistent one is FAIL.

ci        (protocol amendment 001, review of PR #24) Verify the GitHub runner's JUnit
          against the effective (amended) inventory. Tests that cannot run on the
          runner are declared in the amendment with their exact skip message: such a
          test skipped with exactly that message is NOT VERIFIABLE ON RUNNER, never
          PASS, so the full-inventory verdict of a runner JUnit is NOT PROVEN. Any
          other skip, a different message, a failure or error is FAIL. The runner
          verdict (every runner-verifiable test passed, nothing missing or extra) is
          reported separately, and so is the archived local attestation (full JUnit
          of a local run, verified against the same inventory and its pinned SHA256).
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

SCHEMA = 'inwime-test-inventory/1'
POLICY = dict(
    required_outcome='passed',
    skipped='FAIL', xfailed='FAIL', xpassed='FAIL', error='FAIL', failure='FAIL',
    missing_required='FAIL', duplicate_testcase='FAIL', test_file_digest_mismatch='FAIL',
    junit_missing='NOT PROVEN', junit_malformed='FAIL', junit_counts_inconsistent='FAIL',
    junit_sha256_mismatch='FAIL', extra_testcase_not_in_inventory='NOT PROVEN',
    precedence=['FAIL', 'NOT PROVEN', 'PASS'],
    minimum_count_only='never sufficient: identity of every node ID is required',
    amendments='new file results/increment10/amendments/NNN-*.json with its own SHA256; '
               'the frozen inventory is never edited retroactively')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def collect(root, tests='tests'):
    out = run([sys.executable, '-B', '-m', 'pytest', '--collect-only', '-q', '-p', 'no:cacheprovider', tests], root)
    if out.returncode != 0:
        raise RuntimeError('pytest collection failed:\n'+out.stdout[-2000:]+out.stderr[-2000:])
    nodes = [line.strip() for line in out.stdout.splitlines() if '::' in line and not line.startswith(' ')]
    if len(nodes) != len(set(nodes)):
        raise RuntimeError('duplicate node IDs in collection')
    return nodes


def junit_key(node):
    """pytest junitxml identity: classname = dotted module path (+ classes), name = last part."""
    path, *parts = node.split('::')
    module = path.removesuffix('.py').replace('/', '.')
    return '.'.join([module, *parts[:-1]]), parts[-1]


def function_digests(root, nodes):
    """SHA256 of the AST dump of each test function/method (diagnostic: locates substitutions)."""
    trees, out = {}, {}
    for node in nodes:
        path, *parts = node.split('::')
        if path not in trees:
            trees[path] = ast.parse((Path(root)/path).read_text())
        scope = trees[path].body
        target = None
        for part in parts:
            name = part.split('[')[0]
            target = next((n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                           and n.name == name), None)
            if target is None:
                break
            scope = target.body
        out[node] = hashlib.sha256(ast.dump(target).encode()).hexdigest() if target is not None else None
    return out


def git_files_at(root, commit, directory='tests'):
    out = run(['git', 'ls-tree', '-r', '--name-only', commit, directory+'/'], root)
    if out.returncode != 0:
        raise RuntimeError('cannot list files at the base commit '+commit)
    return {line for line in out.stdout.splitlines() if line}


def tool_versions(root, venv=None):
    versions = dict(python=platform.python_version(), implementation=platform.python_implementation(),
                    platform=platform.platform(terse=True))
    for module in ('pytest', 'numpy'):
        try:
            versions[module] = __import__(module).__version__
        except ImportError:
            versions[module] = None
    cc = run(['cc', '--version'], root)
    versions['cc'] = cc.stdout.splitlines()[0] if cc.returncode == 0 and cc.stdout else None
    kernel = Path(root)/'src'/'cpu_kernel.c'
    versions['cpu_kernel_c_sha256'] = sha256(kernel) if kernel.exists() else None
    if venv and (Path(root)/venv/'bin'/'python').exists():
        probe = run([str(Path(root)/venv/'bin'/'python'), '-c',
                     'import json,torch,transformers,safetensors,numpy;print(json.dumps(dict('
                     'torch=torch.__version__,transformers=transformers.__version__,'
                     'safetensors=safetensors.__version__,numpy=numpy.__version__)))'], root)
        versions['verification_venv'] = json.loads(probe.stdout) if probe.returncode == 0 else None
    return versions


def group_of(path, baseline_files):
    if path in baseline_files:
        return 'baseline-m1-m9'
    stem = Path(path).stem
    return 'm10-'+stem.removeprefix('test_increment10_') if stem.startswith('test_increment10_') else 'other-'+stem


def generate(root, base_commit, tests='tests', venv='.venv-oracle7'):
    root = Path(root)
    nodes = collect(root, tests)
    baseline_files = git_files_at(root, base_commit, tests)
    groups = {}
    for node in nodes:
        groups.setdefault(group_of(node.split('::')[0], baseline_files), []).append(node)
    files = {str(p.relative_to(root)): sha256(p) for p in sorted((root/tests).rglob('*.py'))
             if '__pycache__' not in p.parts}
    digests = function_digests(root, nodes)
    if any(v is None for v in digests.values()):
        raise RuntimeError('a collected node has no resolvable source definition')
    return dict(
        schema=SCHEMA, generated_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        base_commit=base_commit, tests_directory=tests, policy=POLICY, tool_versions=tool_versions(root, venv),
        expected_counts=dict(total=len(nodes), **{g: len(v) for g, v in sorted(groups.items())}),
        groups={g: sorted(v) for g, v in sorted(groups.items())},
        test_files_sha256=files, test_function_sha256=dict(sorted(digests.items())),
        baseline_files_at_base_commit=sorted(baseline_files),
        junit_identity='classname = dotted module path plus classes, name = final node part (pytest junitxml)')


# ---------------------------------------------------------------- verification

def parse_junit(path):
    raw = Path(path).read_bytes()
    # pytest never emits a DTD; refusing one rules out entity expansion and external entities
    # without third-party parsers (defusedxml is not a dependency of the offline suite).
    if b'<!DOCTYPE' in raw or b'<!ENTITY' in raw:
        raise ValueError('DTD/entity declarations are not accepted in JUnit XML')
    root = ET.fromstring(raw)
    suites = [root] if root.tag == 'testsuite' else list(root.iter('testsuite'))
    if not suites:
        raise ValueError('no testsuite element')
    cases, declared = [], dict(tests=0, failures=0, errors=0, skipped=0)
    for suite in suites:
        for k in declared:
            declared[k] += int(suite.get(k, 0))
        for case in suite.iter('testcase'):
            kinds = [child.tag for child in case if child.tag in ('failure', 'error', 'skipped')]
            skipped = case.find('skipped')
            if 'error' in kinds:
                outcome = 'error'
            elif 'failure' in kinds:
                outcome = 'failure'
            elif skipped is not None:
                outcome = 'xfailed' if (skipped.get('type') == 'pytest.xfail') else 'skipped'
            else:
                outcome = 'passed'
            cases.append(((case.get('classname'), case.get('name')), outcome))
    observed = dict(tests=len(cases), failures=sum(o == 'failure' for _, o in cases),
                    errors=sum(o == 'error' for _, o in cases),
                    skipped=sum(o in ('skipped', 'xfailed') for _, o in cases))
    return cases, declared, observed


def junit_skip_messages(path):
    """{(classname, name): message} of every <skipped> element (outcome detail for the runner check)."""
    root = ET.fromstring(Path(path).read_bytes())
    return {(c.get('classname'), c.get('name')): c.find('skipped').get('message')
            for c in root.iter('testcase') if c.find('skipped') is not None}


def verify(inventory, junit, root='.', junit_sha256=None, not_verifiable=None):
    """not_verifiable: {node_id: exact skip message} of tests that cannot run on this runner."""
    findings = dict(fail=[], not_proven=[])
    not_verifiable = dict(not_verifiable or {})
    required = {junit_key(n): n for nodes in inventory['groups'].values() for n in nodes}
    result = dict(schema=SCHEMA, inventory_counts=inventory['expected_counts'])
    # test files on disk against the pinned digests (substitution with identical names)
    disk = {str(p.relative_to(root)): sha256(p) for p in sorted((Path(root)/inventory['tests_directory']).rglob('*.py'))
            if '__pycache__' not in p.parts}
    changed = sorted(f for f, d in inventory['test_files_sha256'].items() if disk.get(f) != d)
    added_files = sorted(set(disk) - set(inventory['test_files_sha256']))
    result.update(test_files_changed_or_missing=changed, test_files_added=added_files)
    if changed:
        findings['fail'].append('test file digest mismatch: '+', '.join(changed))
    if added_files:
        findings['not_proven'].append('test files not in the inventory: '+', '.join(added_files))
    junit = Path(junit)
    if not junit.exists():
        findings['not_proven'].append('JUnit XML missing')
        return finish(result, findings)
    if junit_sha256 is not None and sha256(junit) != junit_sha256:
        findings['fail'].append('JUnit XML differs from its recorded SHA256')
    try:
        cases, declared, observed = parse_junit(junit)
    except (ET.ParseError, ValueError, TypeError) as error:
        findings['fail'].append('JUnit XML malformed: '+str(error))
        return finish(result, findings)
    result.update(junit_sha256=sha256(junit), junit_declared=declared, junit_observed=observed)
    if declared != observed:
        findings['fail'].append('JUnit declared counts differ from the testcase elements')
    if not cases:
        findings['fail'].append('JUnit XML has no testcase')
    seen, outcomes, duplicates = set(), {}, []
    for key, outcome in cases:
        if key in seen:
            duplicates.append('::'.join(key))
        seen.add(key)
        outcomes[key] = outcome
    missing = sorted(n for k, n in required.items() if k not in outcomes)
    messages = junit_skip_messages(junit) if not_verifiable else {}
    unverified = sorted(n for k, n in required.items() if n in not_verifiable and outcomes.get(k) == 'skipped'
                        and messages.get(k) == not_verifiable[n])
    unknown = sorted(set(not_verifiable) - set(required.values()))
    if unknown:
        findings['fail'].append('not-verifiable list names tests outside the inventory: '+', '.join(unknown[:5]))
    not_passed = {n: outcomes[k] for k, n in sorted(required.items(), key=lambda x: x[1])
                  if k in outcomes and outcomes[k] != 'passed' and n not in unverified}
    extra = sorted('::'.join(k) for k in outcomes if k not in required)
    groups = {g: dict(expected=len(v), passed=sum(outcomes.get(junit_key(n)) == 'passed' for n in v))
              for g, v in inventory['groups'].items()}
    result.update(missing_required=missing, not_passed=not_passed, duplicates=sorted(duplicates),
                  extra_not_in_inventory=extra, groups=groups, not_verifiable_on_runner=unverified,
                  required_passed=sum(1 for k in required if outcomes.get(k) == 'passed'),
                  required_total=len(required))
    if missing:
        findings['fail'].append(f'{len(missing)} required node IDs missing')
    if not_passed:
        findings['fail'].append(f'{len(not_passed)} required node IDs not passed')
    if duplicates:
        findings['fail'].append('duplicate testcases')
    if extra:
        findings['not_proven'].append(f'{len(extra)} testcases not in the inventory')
    if not_verifiable:
        # runner verdict: everything the runner can run passed; the full-inventory verdict stays
        # NOT PROVEN while any required test is only "not verifiable on runner" (never PASS).
        result['runner_verdict'] = 'FAIL' if findings['fail'] else 'NOT PROVEN' if findings['not_proven'] else 'PASS'
    if unverified:
        findings['not_proven'].append(f'{len(unverified)} required node IDs not verifiable on this runner '
                                      '(local attestation required)')
    return finish(result, findings)


def ci(junit, root='.', output=None):
    """Runner verification under amendment 001. Exit 0 only if the protocol chain verifies, the runner
    verdict is PASS and the archived local attestation is a full PASS of the same inventory."""
    from . import protocol10
    root = Path(root)
    check = protocol10.check()
    eff = protocol10.effective()
    rules = eff.get('ci_runner_verification')
    out = dict(schema=SCHEMA, mode='ci', protocol_check=check, inventory=eff['test_inventory'])
    if not check['passed'] or rules is None:
        out.update(runner_verdict='FAIL', reasons=['protocol chain not verified or no ci_runner_verification rule'])
        return emit_ci(out, output, 5)
    inventory_path = root/eff['test_inventory']['path']
    if sha256(inventory_path) != eff['test_inventory']['sha256']:
        out.update(runner_verdict='FAIL', reasons=['effective inventory differs from its pinned SHA256'])
        return emit_ci(out, output, 5)
    inventory = json.loads(inventory_path.read_text())
    unverifiable = rules['not_verifiable_on_runner']
    runner = verify(inventory, junit, root, not_verifiable={n: unverifiable['skip_message'] for n in unverifiable['node_ids']})
    out['runner'] = runner
    attestation_path = root/rules['local_attestation']
    try:
        attestation = json.loads(attestation_path.read_text())
        local = verify(inventory, root/attestation['junit'], root, junit_sha256=attestation['junit_sha256'])
        if attestation.get('inventory_sha256') != eff['test_inventory']['sha256']:
            local = finish(local, dict(fail=local['fail_reasons']+['attestation names another inventory'],
                                       not_proven=local['not_proven_reasons']))
    except (OSError, ValueError, KeyError, TypeError) as error:
        local = finish(dict(schema=SCHEMA), dict(fail=[], not_proven=['local attestation missing or unreadable: '
                                                                       + type(error).__name__]))
    out['local_attestation'] = dict(source='archived JUnit of a local run (not a runner result)',
                                    path=rules['local_attestation'], verdict=local['verdict'],
                                    required_passed=local.get('required_passed'),
                                    required_total=local.get('required_total'),
                                    fail_reasons=local['fail_reasons'], not_proven_reasons=local['not_proven_reasons'])
    out.update(runner_verdict=runner.get('runner_verdict', runner['verdict']),
               full_inventory_verdict_on_runner=runner['verdict'])
    ok = out['runner_verdict'] == 'PASS' and local['verdict'] == 'PASS'
    return emit_ci(out, output, 0 if ok else 5)


def emit_ci(out, output, code):
    if output:
        with Path(output).open('x') as f:
            json.dump(out, f, indent=1); f.write('\n')
    runner = out.get('runner') or {}
    print(json.dumps(dict(
        runner_verdict=out.get('runner_verdict'),
        full_inventory_verdict_on_runner=out.get('full_inventory_verdict_on_runner'),
        runner_passed=runner.get('required_passed'), required_total=runner.get('required_total'),
        not_verifiable_on_runner=runner.get('not_verifiable_on_runner'),
        runner_fail_reasons=runner.get('fail_reasons'), runner_not_proven_reasons=runner.get('not_proven_reasons'),
        local_attestation=out.get('local_attestation'), reasons=out.get('reasons'),
        protocol_check=(out.get('protocol_check') or {}).get('problems')), indent=1))
    return code


def finish(result, findings):
    result['fail_reasons'] = findings['fail']
    result['not_proven_reasons'] = findings['not_proven']
    result['verdict'] = 'FAIL' if findings['fail'] else 'NOT PROVEN' if findings['not_proven'] else 'PASS'
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    g = sub.add_parser('generate')
    g.add_argument('--base-commit', required=True)
    g.add_argument('--root', type=Path, default=Path('.'))
    g.add_argument('--venv', default='.venv-oracle7')
    g.add_argument('--output', type=Path, required=True)
    v = sub.add_parser('verify')
    v.add_argument('--inventory', type=Path, required=True)
    v.add_argument('--inventory-sha256', help='pinned digest (protocol.json); mismatch is FAIL')
    v.add_argument('--junit', type=Path, required=True)
    v.add_argument('--junit-sha256')
    v.add_argument('--root', type=Path, default=Path('.'))
    v.add_argument('--output', type=Path)
    c = sub.add_parser('ci', help='runner JUnit vs effective inventory, plus the archived local attestation')
    c.add_argument('--junit', type=Path, required=True)
    c.add_argument('--root', type=Path, default=Path('.'))
    c.add_argument('--output', type=Path)
    a = p.parse_args(argv)
    if a.command == 'ci':
        return ci(a.junit, a.root, a.output)
    if a.command == 'generate':
        inventory = generate(a.root, a.base_commit, venv=a.venv)
        with a.output.open('x') as f:
            json.dump(inventory, f, indent=1, sort_keys=True); f.write('\n')
        print(json.dumps(dict(sha256=sha256(a.output), counts=inventory['expected_counts']), indent=1))
        return 0
    if a.inventory_sha256 is not None and sha256(a.inventory) != a.inventory_sha256:
        result = finish(dict(schema=SCHEMA), dict(fail=['inventory differs from the pinned SHA256'], not_proven=[]))
    else:
        result = verify(json.loads(a.inventory.read_text()), a.junit, a.root, a.junit_sha256)
    if a.output:
        with a.output.open('x') as f:
            json.dump(result, f, indent=1); f.write('\n')
    print(json.dumps({k: result.get(k) for k in ('verdict', 'fail_reasons', 'not_proven_reasons',
                                                  'required_passed', 'required_total')}, indent=1))
    return {'PASS': 0, 'NOT PROVEN': 4, 'FAIL': 5}[result['verdict']]


if __name__ == '__main__':
    sys.exit(main())

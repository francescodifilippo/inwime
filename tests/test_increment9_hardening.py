"""M9 post-closure hardening (Issues #17 and #18): regressions for D-2, D-3, D-4 and D-6.

Every case mutates a temporary copy of the real increment-9 evidence, as in
test_increment9_review.py, and was first reproduced against the gate of commit
5eb25da (results/increment9/post-closure/hardening-red.txt). No forward, oracle,
benchmark or download is run.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from test_increment9_review import (CLEAN_PRESERVATION, EVERY, EVIDENCE, copy, edit, own_verdicts,  # noqa: F401
                                    personal_evidence, pins, reseal, verdicts)
from tools import sanitize9
from tools import report_increment9
from tools.report_increment9 import SANITIZED, gate

MISSING = object()
SYNC_RUN, SYNC_SUP = 'phaseB-sync-runtime.json', 'phaseB-sync-supervisor.json'
# Invalid representations of a non-negative integer counter: none may satisfy an acceptance check.
INVALID = {'negative': -1, 'true': True, 'false': False, 'float_zero': 0.0, 'float': 1.5, 'none': None,
           'numeric_string': '0'}


def put(path, value):
    """A mutation setting (or, with MISSING, deleting) the field at `path` inside one JSON file."""
    def change(data):
        for key in path[:-1]:
            data = data[key]
        if value is MISSING:
            del data[path[-1]]
        else:
            data[path[-1]] = value
    return change


def own_and_final(d):
    own, report = own_verdicts(d)
    return own, report['verdicts'], report


# ---- D-3: malformed runtime counters --------------------------------------------------------

# name -> (file, path, valid non-zero value or None, zero allowed, conditions that read the field)
FIELDS = {
    'close_VmSwap': (SYNC_RUN, ('process_after_close', 'VmSwap_bytes'), None, True, 'CDG'),
    'phase_VmSwap': (SYNC_RUN, ('phases', 0, 'process', 'VmSwap_bytes'), None, True, 'CDG'),
    'close_VmPeak': (SYNC_RUN, ('process_after_close', 'VmPeak_bytes'), 142110720, True, 'CDG'),
    'close_VmHWM': (SYNC_RUN, ('process_after_close', 'VmHWM_bytes'), 124768256, True, 'CDG'),
    'peak_rss': (SYNC_RUN, ('peak_rss_bytes',), 124768256, True, 'CDG'),
    'stats_failed': (SYNC_RUN, ('stats', 'failed'), None, True, 'E'),
    'gpu_allocations': (SYNC_RUN, ('gpu_allocations_bytes',), None, True, 'E'),
    'swap_peak': (SYNC_SUP, ('finalizer', 'memory.swap.peak'), None, True, 'CDG'),
    'swap_current': (SYNC_SUP, ('finalizer', 'memory.swap.current'), None, True, 'CDG'),
    'sampled_swap': (SYNC_SUP, ('sampled_peaks', 'memory.swap.current'), None, True, 'CDG'),
    'exit_code': (SYNC_SUP, ('exit_code',), None, True, 'CDG'),
    'memory_peak': (SYNC_SUP, ('finalizer', 'memory.peak'), 2147483648, False, 'CDG'),
    'samples': (SYNC_SUP, ('samples',), 2358, False, 'CDG'),
    'stat_anon': (SYNC_SUP, ('sampled_peaks', 'stat.anon'), 105037824, True, 'CDG'),
    'global_oom_kill_after': (SYNC_SUP, ('global_oom_kill_after',), None, True, 'CDG'),
    'oracle_VmSwap': ('supervisor-oracle-full64.json', ('sampled_peaks_bytes', 'VmSwap'), None, True, 'G'),
    'oracle_exit_code': ('supervisor-oracle-full64.json', ('exit_code',), None, True, 'G'),
    'parity_failures': ('parity-full64.json', ('comparisons', 'decode.0', 'hidden', 'failures'), None, True, 'F'),
    'parity_layer_failures': ('parity-full64.json', ('comparisons', 'decode.0', 'hidden', 'layers', 0, 'failures'),
                              None, True, 'F'),
}
CASES = [(field, kind) for field in FIELDS for kind in ('valid', 'zero', 'missing', *INVALID)]


@pytest.mark.parametrize('field,kind', CASES, ids=[f'{f}-{k}' for f, k in CASES])
def test_runtime_counters_must_be_non_negative_json_integers(copy, field, kind):
    name, path, valid, zero_ok, conditions = FIELDS[field]
    if kind == 'valid' and valid is None:
        kind, value = 'positive_contradicts_zero', 4096  # a well-formed counter that must be 0: a contradicted fact
    elif kind == 'zero' and not zero_ok:
        kind, value = 'zero_contradicts_positive', 0
    else:
        value = dict(valid=valid, zero=0, missing=MISSING, **INVALID)[kind]
    if field == 'global_oom_kill_after' and kind == 'positive_contradicts_zero':
        value = 1
    edit(copy, name, put(path, value))
    reseal(copy)
    own, final, _ = own_and_final(copy)
    expected = {'valid': 'PASS', 'zero': 'PASS', 'missing': 'NOT PROVEN'}.get(kind, 'FAIL')
    assert all(own[c] == expected for c in conditions), (field, kind, own)
    changed = name in SANITIZED and hashlib.sha256(
        (copy/name).read_bytes()).hexdigest() != SANITIZED[name]['sanitized_sha256']
    if changed:  # the pinned digest of a sanitized file catches the edit too, but never as the only check
        assert set(final.values()) == {'FAIL'}, (field, kind, final)
    else:
        assert all(final[c] == expected for c in conditions), (field, kind, final)


@pytest.mark.parametrize('kind', ['negative', 'true', 'float_zero', 'none', 'numeric_string'])
def test_resident_bytes_after_eviction_must_be_a_counter(copy, kind):
    edit(copy, SYNC_SUP, lambda x: [v.update(resident_bytes=INVALID[kind]) for v in x['residency_after_eviction'].values()])
    reseal(copy)
    own, _, _ = own_and_final(copy)
    assert all(own[c] == 'FAIL' for c in 'CDG'), own


@pytest.mark.parametrize('kind', ['negative', 'true', 'float', 'none', 'numeric_string'])
def test_block_read_bytes_must_be_a_counter(copy, kind):
    value = dict(INVALID, float=1.2e11)[kind]  # a float above 90% of the logical bytes
    edit(copy, SYNC_RUN, lambda x: x['inference_proc_io_delta'].update(read_bytes=value))
    own, final, _ = own_and_final(copy)
    assert all(own[c] == final[c] == 'FAIL' for c in 'CDG'), own


def test_float_limit_is_not_the_approved_integer_limit(copy):
    edit(copy, SYNC_SUP, lambda x: x['finalizer'].update({'memory.max': 2147483648.0}))
    reseal(copy)
    own, _, _ = own_and_final(copy)
    assert all(own[c] == 'FAIL' for c in 'CDG'), own


# ---- D-2: verdict precedence ----------------------------------------------------------------

@pytest.mark.parametrize('own,preconditions,expected', [
    ('PASS', ['PASS', 'PASS'], 'PASS'),
    ('PASS', ['NOT PROVEN', 'PASS'], 'NOT PROVEN'),
    ('PASS', ['FAIL', 'NOT PROVEN'], 'FAIL'),
    ('NOT PROVEN', ['PASS', 'PASS'], 'NOT PROVEN'),
    ('NOT PROVEN', ['FAIL', 'PASS'], 'FAIL'),
    ('NOT PROVEN', ['NOT PROVEN', 'PASS'], 'NOT PROVEN'),
    ('FAIL', ['NOT PROVEN', 'PASS'], 'FAIL'),
    ('FAIL', ['PASS', 'PASS'], 'FAIL'),
    ('PASS', ['UNKNOWN', 'PASS'], 'FAIL'),  # an unknown verdict is never read as a PASS
])
def test_combine_fail_over_not_proven_over_pass(own, preconditions, expected):
    assert report_increment9.combine(own, *preconditions) == expected


def test_precondition_fail_turns_a_not_proven_condition_into_fail(copy):
    edit(copy, 'phaseA-W-supervisor.json', lambda x: x.update(wall_s=1.0))  # sanitization precondition FAIL
    edit(copy, 'faults.json', lambda x: x.pop('cases'))                     # G NOT PROVEN on its own evidence
    own, final, report = own_and_final(copy)
    assert report['preconditions']['sanitization']['verdict'] == 'FAIL'
    assert own['G'] == 'NOT PROVEN'
    assert set(final.values()) == {'FAIL'}, final  # before the fix G was NOT PROVEN


def test_altered_protocol_with_missing_amendment_is_fail(copy):
    edit(copy, 'protocol.json', lambda x: x.update(note='edited'))
    (copy/'protocol-amendment-remediation.json').unlink()
    _, final, report = own_and_final(copy)
    assert report['preconditions']['protocol']['verdict'] == 'FAIL'  # before the fix: NOT PROVEN
    assert report['preconditions']['protocol']['facts']['checks']['protocol_file_is_the_approved_one'] is False
    assert set(final.values()) == {'FAIL'}


def test_missing_amendment_alone_stays_not_proven(copy):
    (copy/'protocol-amendment-remediation.json').unlink()
    _, final, report = own_and_final(copy)
    assert report['preconditions']['protocol']['verdict'] == 'NOT PROVEN'
    assert set(final.values()) == {'NOT PROVEN'}


def test_precondition_not_proven_keeps_own_fail_and_own_not_proven(copy):
    edit(copy, 'phaseB-pipeline-supervisor.json', lambda x: x['finalizer']['memory.events'].update(oom_kill=1))
    edit(copy, 'faults.json', lambda x: x.pop('cases'))
    edit(copy, 'phaseB-sync-runtime.json', lambda x: x.pop('generated_text'))
    (copy/'sanitization-manifest.json').unlink()
    own, final, _ = own_and_final(copy)
    assert own['G'] == final['G'] == 'FAIL'
    assert own['E'] == final['E'] == 'NOT PROVEN'
    assert final['A'] == 'NOT PROVEN' and own['A'] == 'PASS'


# ---- D-4: structurally invalid sanitization manifest ----------------------------------------

STRUCTURAL = {
    'files_list': lambda d: edit(d, 'sanitization-manifest.json', lambda x: x.update(files=[])),
    'files_null': lambda d: edit(d, 'sanitization-manifest.json', lambda x: x.update(files=None)),
    'files_absent': lambda d: edit(d, 'sanitization-manifest.json', lambda x: x.pop('files')),
    'files_string': lambda d: edit(d, 'sanitization-manifest.json', lambda x: x.update(files='all')),
    'json_null': lambda d: (d/'sanitization-manifest.json').write_text('null\n'),
    'json_list': lambda d: (d/'sanitization-manifest.json').write_text('[]\n'),
    'entry_not_object': lambda d: edit(d, 'sanitization-manifest.json', lambda x: x['files'].update(
        {'faults.json': None})),
}
CONTRADICTORY = {
    'rules_changed': lambda d: edit(d, 'sanitization-manifest.json', lambda x: x['rules'].pop()),
    'original_digest_changed': lambda d: edit(d, 'sanitization-manifest.json', lambda x: x['files'][
        'environment-after.json'].update(original_sha256='0'*64)),
    'sanitized_digest_changed': lambda d: edit(d, 'sanitization-manifest.json', lambda x: x['files'][
        'environment-after.json'].update(sanitized_sha256='0'*64)),
}
REAL_ORIGINALS = Path('artifacts/increment9/original-evidence')


@pytest.mark.parametrize('originals', ['absent', 'real'])
@pytest.mark.parametrize('case', sorted(STRUCTURAL)+sorted(CONTRADICTORY))
def test_present_manifest_violating_the_schema_is_fail(copy, case, originals):
    if originals == 'real' and not REAL_ORIGINALS.is_dir():
        pytest.skip('local unsanitized originals are not on this machine')
    {**STRUCTURAL, **CONTRADICTORY}[case](copy)
    options = dict(originals=REAL_ORIGINALS) if originals == 'real' else {}
    own, report = own_verdicts(copy, **options)
    sanitization = report['preconditions']['sanitization']
    # Before the fix the first six STRUCTURAL cases were NOT PROVEN, as if the manifest were absent.
    assert sanitization['verdict'] == 'FAIL', sanitization
    assert set(own.values()) == {'PASS'}  # the evidence itself is untouched
    assert set(report['verdicts'].values()) == {'FAIL'}


@pytest.mark.parametrize('originals', ['absent', 'real'])
def test_absent_manifest_is_not_proven(copy, originals):
    if originals == 'real' and not REAL_ORIGINALS.is_dir():
        pytest.skip('local unsanitized originals are not on this machine')
    (copy/'sanitization-manifest.json').unlink()
    options = dict(originals=REAL_ORIGINALS) if originals == 'real' else {}
    _, report = own_verdicts(copy, **options)
    assert report['preconditions']['sanitization']['verdict'] == 'NOT PROVEN'
    assert set(report['verdicts'].values()) == {'NOT PROVEN'}


def test_valid_manifest_passes_with_and_without_originals(copy):
    _, report = own_verdicts(copy)
    assert report['preconditions']['sanitization']['facts']['originals'].startswith('absent')
    assert set(report['verdicts'].values()) == {'PASS'}
    if REAL_ORIGINALS.is_dir():
        _, report = own_verdicts(copy, originals=REAL_ORIGINALS)
        assert report['preconditions']['sanitization']['facts']['originals'].startswith('present')
        assert set(report['verdicts'].values()) == {'PASS'}


@pytest.mark.parametrize('manifest', [[], None, dict(files=[]), dict(files=None), dict(rules=[])])
def test_sanitize9_verify_present_malformed_versus_absent(tmp_path, manifest):
    d = tmp_path/'e'
    personal_evidence(d)
    result = sanitize9.verify(d, tmp_path/'absent', manifest, present=True)
    assert result['verdict'] == 'FAIL' and result['files'] == 0, result
    assert result['originals'].startswith('not compared')
    absent = sanitize9.verify(d, tmp_path/'absent', None, present=False)
    assert absent['verdict'] == 'NOT PROVEN'
    assert sanitize9.verify(d, tmp_path/'absent', None)['verdict'] == 'NOT PROVEN'  # historical call form


def test_sanitize9_cli_rejects_a_present_null_manifest(tmp_path):
    d = tmp_path/'e'
    personal_evidence(d)
    (d/'sanitization-manifest.json').write_text('null\n')
    run = subprocess.run(['python3', '-m', 'tools.sanitize9', str(d), '--verify', '--originals', str(tmp_path/'o')],
                         capture_output=True, text=True)
    assert run.returncode == 1 and json.loads(run.stdout)['verdict'] == 'FAIL', run.stdout+run.stderr


# ---- D-6: originals attested from the historical commit, without local copies ---------------

HISTORICAL = '6031b52'


def commit_available():
    return subprocess.run(['git', 'cat-file', '-e', HISTORICAL+'^{commit}'], capture_output=True).returncode == 0


@pytest.mark.skipif(not commit_available(), reason='historical commit 6031b52 not in this clone')
def test_sanitized_evidence_derives_from_the_historical_commit():
    result = sanitize9.verify_against_commit(EVIDENCE, HISTORICAL, SANITIZED)
    assert result['verdict'] == 'PASS' and result['files'] == 11, result
    assert result['commit'] == subprocess.run(['git', 'rev-parse', HISTORICAL], capture_output=True,
                                              text=True, check=True).stdout.strip()


@pytest.mark.skipif(not commit_available(), reason='historical commit 6031b52 not in this clone')
@pytest.mark.parametrize('case', ['number_resealed', 'pin_removed', 'wrong_commit'])
def test_historical_attestation_detects_changes(copy, case):
    expected, commit = dict(SANITIZED), HISTORICAL
    if case == 'number_resealed':
        edit(copy, 'phaseA-W-supervisor.json', lambda x: x.update(wall_s=1.0))
        digest = hashlib.sha256((copy/'phaseA-W-supervisor.json').read_bytes()).hexdigest()
        expected['phaseA-W-supervisor.json'] = dict(expected['phaseA-W-supervisor.json'], sanitized_sha256=digest)
    if case == 'pin_removed':
        expected = {}
    if case == 'wrong_commit':
        commit = '68ba9f3'  # the baseline: the increment-9 files do not exist there
    result = sanitize9.verify_against_commit(copy, commit, expected)
    assert result['verdict'] == 'FAIL', result


def test_historical_pins_match_the_manifest():
    assert pins(json.loads((EVIDENCE/'sanitization-manifest.json').read_text())) == SANITIZED


def test_gate_on_unmodified_evidence_is_still_pass(copy):
    report = gate(copy, originals=copy/'no-originals', preservation=lambda b, i: dict(CLEAN_PRESERVATION))
    assert report['verdicts'] == dict.fromkeys(EVERY, 'PASS')
    assert report['conditions']['F']['facts']['contracts']['stock_fp64']['raw_verdict'] == 'FAIL'
    assert verdicts(copy) == dict.fromkeys(EVERY, 'PASS')

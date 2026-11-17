"""Increment-9 review regressions: every false PASS found by the independent review.

Each gate case mutates a temporary copy of the real evidence and must turn the
affected conditions into FAIL or NOT PROVEN. Control tests prove the unmutated
copy is PASS, so a mutation cannot pass vacuously. Acquisition, greedy
recomputation, preservation and sanitization are tested without network,
checkpoints or forwards.
"""
import collections
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from tools.report_increment9 import gate

EVIDENCE = Path('results/increment9')
NOT_PASS = ('FAIL', 'NOT PROVEN')
EVERY = 'ABCDEFGH'
CLEAN_PYTEST = '.'*72+' [100%]\n9999 passed in 1.00s\n'
CLEAN_PRESERVATION = dict(modified_tracked=['README.md'], deleted=[], unexpected_modifications=[])
MIB = 2**20


def edit(d, name, change):
    path = d/name
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data, indent=2)+'\n')


def replace(d, name, text):
    (d/name).write_text(text)


def each(mapping, change):
    for value in mapping.values():
        change(value)


def phase_b(run, name, change):
    return lambda d: edit(d, f'phaseB-{run}-{name}.json', change)


def stock_as_full64(d):
    shutil.copyfile(d/'parity-stock_fp32.json', d/'parity-full64.json')


def reduce_coverage(x):
    x['coverage'].update(positions=1, logits_per_position=10)
    x['comparisons'] = {'decode.0': x['comparisons']['decode.0']}
    x['compared_elements'].update(logits=10)


def forged_preservation(x):
    x.update(modified_tracked=['README.md', 'inwime_runtime/core.py'], deleted=['src/kernel.c'],
             unexpected_modifications=[], passed=True)


# id -> (mutation of the evidence copy, conditions that must no longer be PASS)
MUTATIONS = {
    # P2-1: missing or empty evidence
    'diagnostic_report_missing': (lambda d: (d/'parity-stock_fp64.json').unlink(), 'F'),
    'phaseA_residency_missing': (lambda d: edit(d, 'phaseA-P-supervisor.json',
                                                lambda x: x.pop('residency_after_eviction')), 'B'),
    'phaseA_residency_empty': (lambda d: edit(d, 'phaseA-P-supervisor.json',
                                              lambda x: x.update(residency_after_eviction={})), 'B'),
    'phaseA_condition_not_evicted': (lambda d: edit(d, 'phaseA-P-supervisor.json',
                                                    lambda x: x.update(condition='as-is')), 'B'),
    'phaseA_residency_other_size': (lambda d: edit(d, 'phaseA-P-supervisor.json', lambda x: each(
        x['residency_after_eviction'], lambda v: v.update(size_bytes=4096))), 'B'),
    'integrity_files_empty': (lambda d: edit(d, 'integrity-after.json', lambda x: x.update(files={})), 'A'),
    'integrity_file_mismatch_all_match_true': (lambda d: edit(d, 'integrity-after.json', lambda x: x['files'][
        'qwen2.5-7b-d149729/model-00002-of-00004.safetensors'].update(match=False)), 'A'),
    'integrity_digest_altered_match_true': (lambda d: edit(d, 'integrity-after.json', lambda x: x['files'][
        'qwen2.5-7b-d149729/model-00003-of-00004.safetensors'].update(sha256='0'*64, expected='0'*64)), 'A'),
    'integrity_shard_missing': (lambda d: edit(d, 'integrity-after.json', lambda x: x['files'].pop(
        'qwen2.5-7b-d149729/model-00004-of-00004.safetensors')), 'A'),
    'integrity_smollm_missing': (lambda d: edit(d, 'integrity-after.json', lambda x: x['files'].pop(
        'smollm2-1.7b-d4c18cc/model.safetensors')), 'A'),
    'integrity_index_missing': (lambda d: edit(d, 'integrity-after.json', lambda x: x['files'].pop(
        'qwen2.5-7b-d149729/model.safetensors.index.json')), 'A'),
    # P2-3: derived booleans trusted, reports not bound together
    'full64_failures_and_argmax_passed_true': (lambda d: edit(d, 'parity-full64.json', lambda x: (
        x['comparisons']['decode.1']['logits'].update(failures=1000),
        x['comparisons']['decode.1'].update(argmax_equal=False))), 'F'),
    'full64_layer_failures_only': (lambda d: edit(d, 'parity-full64.json', lambda x: x['comparisons'][
        'decode.0']['hidden']['layers'][27].update(failures=31)), 'F'),
    'full64_coverage_reduced': (lambda d: edit(d, 'parity-full64.json', reduce_coverage), 'F'),
    'full64_element_count_reduced': (lambda d: edit(d, 'parity-full64.json', lambda x: x['comparisons'][
        'decode.0']['logits'].update(count=10)), 'F'),
    'full64_replaced_by_stock_fp32': (stock_as_full64, 'F'),
    'full64_other_runtime': (lambda d: edit(d, 'parity-full64.json', lambda x: x.update(
        runtime_report='results/increment9/other-runtime.json')), 'F'),
    'fp32_other_revision': (lambda d: edit(d, 'parity-stock_fp32.json', lambda x: x['oracle_metadata'].update(
        revision='0'*40)), 'F'),
    'fp32_other_sequence': (lambda d: edit(d, 'parity-stock_fp32.json', lambda x: x['oracle_metadata'].update(
        sequence=[785, 6722, 315, 9625, 374, 12095, 13, 1])), 'F'),
    'full64_other_oracle_digest': (lambda d: edit(d, 'parity-full64.json', lambda x: (
        x.update(oracle_sha256='0'*64), x['oracle_metadata'].update(capture_sha256='0'*64))), 'F'),
    'fp32_thresholds_in_report': (lambda d: edit(d, 'parity-stock_fp32.json', lambda x: x.update(atol=1.0)), 'F'),
    'full64_greedy_not_recorded': (lambda d: edit(d, 'parity-full64.json', lambda x: x['comparisons'][
        'decode.2'].update(runtime_greedy_recorded=False)), 'F'),
    'full64_upstream_greedy_differs': (lambda d: edit(d, 'parity-full64.json', lambda x: x[
        'oracle_metadata'].update(argmax=[2701, 315, 279, 374, 1, 13, 1084, 374])), 'F'),
    'diagnostic_rewritten_to_pass': (lambda d: edit(d, 'parity-stock_fp64.json', lambda x: x.update(passed=True)), 'F'),
    'diagnostic_other_runtime': (lambda d: edit(d, 'parity-stock_fp64.json', lambda x: x.update(
        runtime_report='results/increment9/other-runtime.json')), 'F'),
    'phaseA_parity_stub': (lambda d: replace(d, 'phaseA-P-parity-inc8.json', '{"passed":true}'), 'BF'),
    'phaseA_parity_other_oracle': (lambda d: edit(d, 'phaseA-P-parity-inc8.json', lambda x: (
        x.update(oracle_sha256='1'*64), x['oracle_metadata'].update(capture_sha256='1'*64))), 'BF'),
    'phaseA_parity_failures_passed_true': (lambda d: edit(d, 'phaseA-P-parity-inc8.json', lambda x: x[
        'comparisons']['decode.2']['kv'].update(failures=5)), 'BF'),
    'gate7_stub': (lambda d: replace(d, 'gate7-rerun.json', '{"passed":true}'), 'H'),
    'gate8_stub': (lambda d: replace(d, 'gate8-rerun.json', '{"passed":true}'), 'H'),
    'gate7_diagnostic_rewritten': (lambda d: edit(d, 'gate7-rerun.json', lambda x: x['contracts'][
        'standard_fp64_diagnostic'].update(raw_parity_passed=True)), 'H'),
    'gate8_check_removed': (lambda d: edit(d, 'gate8-rerun.json', lambda x: x['checks'].pop(
        'D_full_standard_fp32_parity')), 'H'),
    'phaseA_scope_unit_differs': (lambda d: edit(d, 'phaseA-P-runtime.json', lambda x: x['scope'].update(
        unit='inwime9-other-1')), 'BG'),
    'phaseB_scope_unit_differs': (phase_b('sync', 'runtime', lambda x: x['scope'].update(unit='inwime9-other-1')), 'CG'),
    'phaseB_runtime_output_differs': (phase_b('sync', 'supervisor', lambda x: x['command'].__setitem__(
        x['command'].index('--output')+1, 'results/increment9/other-runtime.json')), 'CG'),
    'phaseB_source_sha_zeroed': (phase_b('sync', 'runtime', lambda x: each(
        x['source_files'], lambda v: v.update(sha256='0'*64))), 'A'),
    'phaseB_pipeline_source_sha_zeroed': (phase_b('pipeline', 'runtime', lambda x: each(
        x['source_files'], lambda v: v.update(sha256='0'*64))), 'A'),
    'phaseA_source_sha_zeroed': (lambda d: edit(d, 'phaseA-P-runtime.json', lambda x: each(
        x['source_files'], lambda v: v.update(sha256='0'*64))), 'A'),
    'preservation_forged': (lambda d: edit(d, 'preservation.json', forged_preservation), 'H'),
    'preservation_stub': (lambda d: replace(d, 'preservation.json', '{"passed":true}'), 'H'),
    'tensor_counters_emptied': (phase_b('sync', 'runtime', lambda x: (
        x['stats'].update(tensor_read_bytes={}), x.update(expected_tensor_read_bytes={}))), 'E'),
    'tensor_counter_zeroed': (phase_b('sync', 'runtime', lambda x: (
        x['stats']['tensor_read_bytes'].update({'layers.3.q': 0}),
        x['expected_tensor_read_bytes'].update({'layers.3.q': 0}))), 'E'),
    'kv_prefix_not_preserved': (phase_b('sync', 'runtime', lambda x: x['phases'][2].update(
        kv_prefix_preserved=False)), 'E'),
    'generation_sequence_broken': (phase_b('sync', 'runtime', lambda x: x['phases'][2].update(generation=7)), 'E'),
    'decode_input_not_previous_greedy': (phase_b('sync', 'runtime', lambda x: x['phases'][1].update(ids=[1])), 'E'),
    'pipeline_digest_differs': (phase_b('pipeline', 'runtime', lambda x: x['phases'][3]['digests'].update(
        logits='0'*64)), 'E'),
    'protocol_threshold_altered': (lambda d: edit(d, 'protocol.json', lambda x: x['fp32_comparison'].update(
        atol=0.01)), EVERY),
    'protocol_limit_altered': (lambda d: edit(d, 'protocol.json', lambda x: x['phase_a']['cgroup'].update(
        {'memory.max': 2**33})), EVERY),
    # mutations the original gate already detected: they must stay detected
    'oom_kill_recorded': (phase_b('pipeline', 'supervisor', lambda x: x['finalizer']['memory.events'].update(
        oom_kill=1)), 'G'),
    'finalizer_missing': (phase_b('sync', 'supervisor', lambda x: x.update(finalizer=None)), 'G'),
    'increment8_digest_altered': (lambda d: edit(d, 'phaseA-P-runtime.json', lambda x: x['phases'][0][
        'digests'].update(logits='0'*64)), 'B'),
    'scope_not_required': (lambda d: edit(d, 'phaseA-P-runtime.json', lambda x: x.update(
        scope=dict(required=False))), 'B'),
    'acquisition_sha_altered': (lambda d: edit(d, 'qwen-acquisition.json', lambda x: x['downloaded'][
        'model-00001-of-00004.safetensors'].update(sha256='0'*64)), 'A'),
    # P3-2: fragile parsing of text evidence
    'pytest_passed_with_errors': (lambda d: replace(d, 'pytest-final.txt', '480 passed, 4 errors in 3.00s\n'), 'H'),
    'pytest_single_test': (lambda d: replace(d, 'pytest-final.txt', '.  [100%]\n1 passed in 1.00s\n'), 'H'),
    'pytest_skipped': (lambda d: replace(d, 'pytest-final.txt', '9999 passed, 1 skipped in 1.00s\n'), 'H'),
    'pytest_failure_in_progress_line': (lambda d: replace(
        d, 'pytest-final.txt', '..F.. [100%]\n9999 passed in 1.00s\n'), 'H'),
    'pytest_empty': (lambda d: replace(d, 'pytest-final.txt', ''), 'H'),
    'oracle9_checks_not_ok': (lambda d: replace(d, 'oracle9-checks.txt', 'Ran 3 tests in 0.2s\n\nNOT OK\n'), 'F'),
    'oracle9_checks_no_tests': (lambda d: replace(d, 'oracle9-checks.txt', 'Ran 0 tests in 0.000s\n\nOK\n'), 'F'),
    'oracle9_checks_skipped': (lambda d: replace(
        d, 'oracle9-checks.txt', 'Ran 3 tests in 0.2s\n\nOK (skipped=3)\n'), 'F'),
    'oracle8_check_not_ok': (lambda d: replace(d, 'check-oracle8.txt', 'Ran 1 test in 2.2s\n\nNOT OK\n'), 'H'),
    'oracle_precision7_check_missing': (lambda d: (d/'check-oracle-precision7.txt').unlink(), 'H'),
    # P3-3: the informative part of the cgroup amendment is now gated
    'phaseA_anon_peak_over_threshold': (lambda d: edit(d, 'phaseA-P-supervisor.json', lambda x: x[
        'sampled_peaks'].update({'stat.anon': 300*MIB})), 'BG'),
    'phaseB_anon_peak_over_threshold': (phase_b('sync', 'supervisor', lambda x: x['sampled_peaks'].update(
        {'stat.anon': 300*MIB})), 'G'),
    'phaseB_worker_vmhwm_over_threshold': (phase_b('pipeline', 'runtime', lambda x: x[
        'process_after_close'].update(VmHWM_bytes=300*MIB)), 'G'),
}
# P2-2: phase B must prove eviction and real block reads, in both modes
for run, extra in (('sync', 'D'), ('pipeline', '')):
    MUTATIONS.update({
        f'phaseB_{run}_condition_as_is': (phase_b(run, 'supervisor', lambda x: x.update(condition='as-is')), 'CG'+extra),
        f'phaseB_{run}_residency_missing': (phase_b(run, 'supervisor', lambda x: x.pop(
            'residency_after_eviction')), 'CG'+extra),
        f'phaseB_{run}_shards_still_resident': (phase_b(run, 'supervisor', lambda x: each(
            x['residency_after_eviction'], lambda v: v.update(resident_bytes=v['size_bytes']))), 'CG'+extra),
        f'phaseB_{run}_one_shard_resident': (phase_b(run, 'supervisor', lambda x: next(iter(
            x['residency_after_eviction'].values())).update(resident_bytes=2**30)), 'CG'+extra),
        f'phaseB_{run}_three_shards_only': (phase_b(run, 'supervisor', lambda x: x['residency_after_eviction'].pop(
            next(iter(x['residency_after_eviction'])))), 'CG'+extra),
        f'phaseB_{run}_no_block_reads': (phase_b(run, 'runtime', lambda x: x['inference_proc_io_delta'].update(
            read_bytes=4096)), 'CG'+extra),
        f'phaseB_{run}_half_block_reads': (phase_b(run, 'runtime', lambda x: x['inference_proc_io_delta'].update(
            read_bytes=x['stats']['inference_read_bytes']//2)), 'CG'+extra)})


def reseal(d):
    """Keep the sanitization manifest consistent, as a careful forger would."""
    path = d/'sanitization-manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        for name, entry in manifest['files'].items():
            if (d/name).exists():
                entry['sanitized_sha256'] = hashlib.sha256((d/name).read_bytes()).hexdigest()
        path.write_text(json.dumps(manifest, indent=2)+'\n')


@pytest.fixture
def copy(tmp_path):
    d = tmp_path/'evidence'
    shutil.copytree(EVIDENCE, d, ignore=shutil.ignore_patterns('review'))
    replace(d, 'pytest-final.txt', CLEAN_PYTEST)
    return d


def verdicts(d, **options):
    """Gate without local originals and with a fixed git state: only the evidence copy varies."""
    options.setdefault('originals', d/'no-originals')
    options.setdefault('preservation', lambda baseline, intentional: dict(CLEAN_PRESERVATION))
    return gate(d, **options)['verdicts']


def test_control_unmutated_copy_passes_every_condition(copy):
    report = gate(copy, originals=copy/'no-originals',
                  preservation=lambda baseline, intentional: dict(CLEAN_PRESERVATION))
    assert report['verdicts'] == dict.fromkeys(EVERY, 'PASS'), report['verdicts']
    assert report['preconditions']['protocol']['verdict'] == 'PASS'
    assert report['preconditions']['sanitization']['verdict'] == 'PASS'
    # The justified diagnostic FAIL is kept raw, reported, and never turned into a PASS.
    assert report['conditions']['F']['facts']['contracts']['stock_fp64']['raw_verdict'] == 'FAIL'
    assert report['conditions']['F']['facts']['contracts']['stock_fp64']['failures'] == dict(logits=0, hidden=31, kv=0)


@pytest.mark.parametrize('case', sorted(MUTATIONS))
def test_gate_rejects_review_mutation(copy, case):
    mutate, conditions = MUTATIONS[case]
    mutate(copy)
    reseal(copy)
    report = gate(copy, originals=copy/'no-originals',
                  preservation=lambda baseline, intentional: dict(CLEAN_PRESERVATION))
    result = report['verdicts']
    assert all(result[c] in NOT_PASS for c in conditions), (case, result)
    # Rejected on the condition's own evidence, not only through a failed precondition.
    # (Protocol mutations are caught by the protocol precondition alone, by design.)
    own = {c: report['conditions'][c]['verdict_before_preconditions'] for c in conditions}
    assert conditions == EVERY or all(v in NOT_PASS for v in own.values()), (case, own)


def test_missing_evidence_is_not_proven_not_fail(copy):
    (copy/'parity-stock_fp64.json').unlink()
    assert verdicts(copy)['F'] == 'NOT PROVEN'
    edit(copy, 'phaseA-P-supervisor.json', lambda x: x.pop('residency_after_eviction'))
    reseal(copy)
    assert verdicts(copy)['B'] == 'NOT PROVEN'


def test_contradicted_evidence_is_fail(copy):
    edit(copy, 'integrity-after.json', lambda x: x['files'][
        'qwen2.5-7b-d149729/model-00002-of-00004.safetensors'].update(match=False))
    edit(copy, 'phaseB-sync-supervisor.json', lambda x: x.update(condition='as-is'))
    reseal(copy)
    result = verdicts(copy)
    assert result['A'] == 'FAIL' and result['C'] == 'FAIL' and result['D'] == 'FAIL' and result['G'] == 'FAIL'


def test_live_git_state_is_recomputed_not_trusted(copy):
    dirty = dict(modified_tracked=['README.md', 'inwime_runtime/core.py'], deleted=[],
                 unexpected_modifications=['inwime_runtime/core.py'])
    assert verdicts(copy, preservation=lambda baseline, intentional: dirty)['H'] == 'FAIL'

    def unavailable(baseline, intentional):
        raise subprocess.CalledProcessError(128, 'git')
    assert verdicts(copy, preservation=unavailable)['H'] == 'NOT PROVEN'


def test_remediation_amendment_cannot_move_the_bar(copy):
    edit(copy, 'protocol-amendment-remediation.json',
         lambda x: x['gate_parameters'].update(anon_kernel_peak_max_bytes=2**31))
    assert set(verdicts(copy).values()) == {'FAIL'}
    (copy/'protocol-amendment-remediation.json').unlink()
    assert set(verdicts(copy).values()) == {'NOT PROVEN'}


def test_capture_manifest_binds_runtime_and_oracle_files(copy, tmp_path):
    from tools.report_increment9 import capture_files
    manifest = json.loads((copy/'capture-manifest.json').read_text())
    root = tmp_path/'root'
    assert capture_files(manifest, root)['state'].startswith('absent')
    name, entry = next(iter(manifest['files'].items()))
    target = root/name
    target.parent.mkdir(parents=True)
    target.write_bytes(b'x'*8)
    assert capture_files(manifest, root)['state'].startswith('mismatch')  # partial or altered set
    small = {'files': {name: dict(nbytes=8, sha256=hashlib.sha256(b'x'*8).hexdigest())}}
    assert capture_files(small, root)['state'].startswith('verified')
    edit(copy, 'capture-manifest.json', lambda x: x['files'][
        'artifacts/increment9/phaseB-sync/decode.0.logits.f64'].update(sha256='0'*64))
    assert verdicts(copy)['F'] in NOT_PASS


@pytest.mark.parametrize('text,ok', [
    (CLEAN_PYTEST, True), ('. [100%]\n510 passed in 41.27s\n', False),
    ('. [100%]\n9999 passed in 61.50s (0:01:01)\n', True), ('480 passed, 4 errors in 3.00s\n', False),
    ('. [100%]\n9999 passed, 2 warnings in 3.00s\n', False), ('s [100%]\n9999 passed in 3.00s\n', False),
    ('. [100%]\n9999 passed in 3.00s\nERROR collecting\n', False), ('', False)])
def test_pytest_summary_parsing_is_strict(text, ok):
    from tools.report_increment9 import pytest_ok
    assert pytest_ok(text, 600) is ok


@pytest.mark.parametrize('text,ok', [
    ('a ... ok\n\n---\nRan 3 tests in 0.198s\n\nOK\n', True), ('Ran 3 tests in 0.198s\n\nNOT OK\n', False),
    ('Ran 2 tests in 0.198s\n\nOK\n', False), ('Ran 3 tests in 0.198s\n\nOK (skipped=1)\n', False),
    ('a ... FAIL\nRan 3 tests in 0.198s\n\nFAILED (failures=1)\n', False),
    ('a ... ERROR\nRan 3 tests in 0.198s\n\nOK\n', False), ('OK\n', False), ('', False)])
def test_unittest_summary_parsing_is_strict(text, ok):
    from tools.report_increment9 import unittest_ok
    assert unittest_ok(text, 3) is ok


# ---- sanitization of Git-bound evidence -------------------------------------------------

PERSONAL = '/home/someone/GIT/account/ooc-inference'


def personal_evidence(directory):
    directory.mkdir()
    supervisor = dict(files=[PERSONAL+'/checkpoints/m/model.safetensors'], memory_max=2147483648,
                      residency={PERSONAL+'/checkpoints/m/model.safetensors': dict(resident_bytes=0, size_bytes=7)},
                      launcher=['/bin/sh', '/tmp/tool-1000/-home-someone/0c0ffee0-0000-4000-8000-000000000000'
                                '/scratchpad/out.final.txt'], wall_s=31.186610339034814)
    audit = dict(own_cgroup='/user.slice/app.slice/tmux-spawn-0c0ffee0-0000-4000-8000-000000000000.scope',
                 huggingface_hub_cache_entries=['models--Other--Model', 'models--Qwen--Qwen2.5-7B'], free=908396462080)
    for name, data in (('run-supervisor.json', supervisor), ('environment.json', audit),
                       ('clean.json', dict(value=1.5))):
        (directory/name).write_text(json.dumps(data, indent=2)+'\n')
    (directory/'run.log').write_text('phase decode.0 wall=1.00s\n')


def test_sanitization_replaces_only_declared_patterns(tmp_path):
    from tools import sanitize9
    d, originals = tmp_path/'e', tmp_path/'originals'
    personal_evidence(d)
    manifest = sanitize9.apply(d, originals)
    assert set(manifest['files']) == {'run-supervisor.json', 'environment.json'}
    supervisor = json.loads((d/'run-supervisor.json').read_text())
    assert supervisor['files'] == ['<repo>/checkpoints/m/model.safetensors']
    assert list(supervisor['residency']) == ['<repo>/checkpoints/m/model.safetensors']
    assert supervisor['launcher'][1] == '<scratch>/out.final.txt'
    assert supervisor['wall_s'] == 31.186610339034814 and supervisor['memory_max'] == 2147483648
    audit = json.loads((d/'environment.json').read_text())
    assert audit['huggingface_hub_cache_entries'] == ['models--Qwen--Qwen2.5-7B']
    assert audit['own_cgroup'].endswith('tmux-spawn-<session>.scope') and audit['free'] == 908396462080
    assert json.loads((originals/'environment.json').read_text())['huggingface_hub_cache_entries'][0] == 'models--Other--Model'
    assert sanitize9.verify(d, originals, manifest) == dict(
        verdict='PASS', files=2, originals='present: every sanitized file equals its original under the declared rules',
        problems=[])
    assert 'someone' not in json.dumps(manifest) and sanitize9.residual(d) == []
    # Idempotent and never silent: a second pass changes nothing and refuses to overwrite originals.
    assert sanitize9.apply(d, originals)['files'] == {}


@pytest.mark.parametrize('case', ['number', 'string', 'key_removed', 'original_swapped', 'residual_path',
                                  'unlisted_personal_file', 'manifest_digest', 'missing_file'])
def test_sanitization_check_detects_undeclared_changes(tmp_path, case):
    from tools import sanitize9
    d, originals = tmp_path/'e', tmp_path/'originals'
    personal_evidence(d)
    manifest = sanitize9.apply(d, originals)
    if case == 'number': edit(d, 'run-supervisor.json', lambda x: x.update(wall_s=30.0))
    if case == 'string': edit(d, 'run-supervisor.json', lambda x: x.update(files=['<repo>/checkpoints/other']))
    if case == 'key_removed': edit(d, 'environment.json', lambda x: x.pop('free'))
    if case == 'original_swapped': edit(originals, 'environment.json', lambda x: x.update(free=1))
    if case == 'residual_path': edit(d, 'clean.json', lambda x: x.update(path='/home/someone/x'))
    if case == 'unlisted_personal_file':
        (d/'new.json').write_text(json.dumps(dict(path=PERSONAL+'/x')))
    if case == 'manifest_digest': manifest['files']['environment.json']['sanitized_sha256'] = '0'*64
    if case == 'missing_file': (d/'environment.json').unlink()
    if case in ('number', 'string', 'key_removed'):  # a forger who also updates the manifest digest
        for name, entry in manifest['files'].items():
            entry['sanitized_sha256'] = hashlib.sha256((d/name).read_bytes()).hexdigest()
    assert sanitize9.verify(d, originals, manifest)['verdict'] == 'FAIL'


def test_sanitization_states_when_originals_are_absent(tmp_path):
    from tools import sanitize9
    d, originals = tmp_path/'e', tmp_path/'originals'
    personal_evidence(d)
    manifest = sanitize9.apply(d, originals)
    result = sanitize9.verify(d, tmp_path/'absent', manifest)
    assert result['verdict'] == 'PASS' and result['originals'].startswith('absent')
    assert sanitize9.verify(d, tmp_path/'absent', None)['verdict'] == 'NOT PROVEN'


def test_gate_sanitization_precondition(copy):
    assert set(verdicts(copy).values()) == {'PASS'}
    edit(copy, 'phaseA-W-supervisor.json', lambda x: x.update(wall_s=1.0))  # manifest not updated
    assert set(verdicts(copy).values()) == {'FAIL'}
    (copy/'sanitization-manifest.json').unlink()
    assert set(verdicts(copy).values()) == {'NOT PROVEN'}


# ---- second review: the manifest cannot shrink its own scope (P2-A) -------------------------

def own_verdicts(d, **options):
    options.setdefault('originals', d/'no-originals')
    options.setdefault('preservation', lambda baseline, intentional: dict(CLEAN_PRESERVATION))
    report = gate(d, **options)
    return {k: v['verdict_before_preconditions'] for k, v in report['conditions'].items()}, report


def pins(manifest):
    return {name: dict(original_sha256=e['original_sha256'], sanitized_sha256=e['sanitized_sha256'])
            for name, e in manifest['files'].items()}


@pytest.mark.parametrize('originals_present', [True, False])
def test_sanitization_entry_removed_with_altered_number_is_fail(tmp_path, originals_present):
    from tools import sanitize9
    d, originals = tmp_path/'e', tmp_path/'originals'
    personal_evidence(d)
    manifest = sanitize9.apply(d, originals)
    expected = pins(manifest)
    edit(d, 'run-supervisor.json', lambda x: x.update(wall_s=30.0))
    manifest['files'].pop('run-supervisor.json')
    where = originals if originals_present else tmp_path/'absent'
    result = sanitize9.verify(d, where, manifest, expected=expected)
    assert result['verdict'] == 'FAIL' and not result['originals'].startswith('present')
    assert any('run-supervisor.json' in problem for problem in result['problems'])
    if originals_present:  # detected even without the pinned set: the original is there and unlisted
        result = sanitize9.verify(d, originals, manifest)
        assert result['verdict'] == 'FAIL'
        assert 'local original is not listed in the manifest: run-supervisor.json' in result['problems']


@pytest.mark.parametrize('originals_present', [True, False])
def test_sanitization_empty_manifest_is_never_pass(tmp_path, originals_present):
    from tools import sanitize9
    d, originals = tmp_path/'e', tmp_path/'originals'
    personal_evidence(d)
    manifest = sanitize9.apply(d, originals)
    expected = pins(manifest)
    edit(d, 'run-supervisor.json', lambda x: x.update(wall_s=30.0))
    manifest['files'] = {}
    where = originals if originals_present else tmp_path/'absent'
    for options in (dict(), dict(expected=expected)):
        result = sanitize9.verify(d, where, manifest, **options)
        assert result['verdict'] == 'FAIL' and result['files'] == 0
        assert result['originals'].startswith('not compared')
        assert 'sanitization manifest lists no file: nothing was compared' in result['problems']


def test_sanitization_original_present_but_not_listed_is_fail(tmp_path):
    from tools import sanitize9
    d, originals = tmp_path/'e', tmp_path/'originals'
    personal_evidence(d)
    manifest = sanitize9.apply(d, originals)
    assert sanitize9.verify(d, originals, manifest)['verdict'] == 'PASS'
    (originals/'clean.json').write_text((d/'clean.json').read_text())  # an original the manifest does not list
    result = sanitize9.verify(d, originals, manifest)
    assert result['verdict'] == 'FAIL' and result['originals'].startswith('partial')
    assert 'local original is not listed in the manifest: clean.json' in result['problems']


def test_sanitization_pinned_set_and_digests(tmp_path):
    from tools import sanitize9
    d, originals = tmp_path/'e', tmp_path/'originals'
    personal_evidence(d)
    manifest = sanitize9.apply(d, originals)
    expected = pins(manifest)
    assert sanitize9.verify(d, tmp_path/'absent', manifest, expected=expected)['verdict'] == 'PASS'
    more = expected | {'other.json': dict(original_sha256='0'*64, sanitized_sha256='0'*64)}
    assert sanitize9.verify(d, tmp_path/'absent', manifest, expected=more)['verdict'] == 'FAIL'
    less = {k: v for k, v in expected.items() if k != 'environment.json'}
    assert sanitize9.verify(d, tmp_path/'absent', manifest, expected=less)['verdict'] == 'FAIL'
    # A forger who alters a number and re-seals the manifest digest, with no originals around.
    edit(d, 'run-supervisor.json', lambda x: x.update(wall_s=30.0))
    manifest['files']['run-supervisor.json']['sanitized_sha256'] = hashlib.sha256(
        (d/'run-supervisor.json').read_bytes()).hexdigest()
    assert sanitize9.verify(d, tmp_path/'absent', manifest)['verdict'] == 'PASS'  # what the pin is for
    assert sanitize9.verify(d, tmp_path/'absent', manifest, expected=expected)['verdict'] == 'FAIL'


def test_gate_pins_the_eleven_sanitized_files():
    from tools.report_increment9 import SANITIZED
    manifest = json.loads((EVIDENCE/'sanitization-manifest.json').read_text())
    assert len(SANITIZED) == 11 and SANITIZED == pins(manifest)


@pytest.mark.parametrize('case', ['entry_removed', 'entry_removed_resealed', 'manifest_empty', 'resealed_only'])
def test_gate_sanitization_cannot_be_bypassed_through_the_manifest(copy, case):
    edit(copy, 'phaseA-W-supervisor.json', lambda x: x.update(wall_s=1.0))
    if case.startswith('entry_removed'):
        edit(copy, 'sanitization-manifest.json', lambda x: x['files'].pop('phaseA-W-supervisor.json'))
    if case == 'manifest_empty':
        edit(copy, 'sanitization-manifest.json', lambda x: x.update(files={}))
    if case.endswith('resealed') or case == 'resealed_only':
        reseal(copy)
    own, report = own_verdicts(copy)
    assert set(report['verdicts'].values()) == {'FAIL'}, report['verdicts']
    assert set(own.values()) == {'PASS'}  # only the precondition catches an altered wall time
    facts = report['preconditions']['sanitization']['facts']
    assert facts['verdict'] == 'FAIL' and not facts['originals'].startswith('present')
    # With the real local originals, when this machine has them, the verdict is the same.
    real = Path('artifacts/increment9/original-evidence')
    if real.is_dir():
        assert set(verdicts(copy, originals=real).values()) == {'FAIL'}


def test_gate_original_present_but_not_listed(copy, tmp_path):
    originals = tmp_path/'originals'; originals.mkdir()
    (originals/'faults.json').write_text((copy/'faults.json').read_text())
    report = gate(copy, originals=originals, preservation=lambda baseline, intentional: dict(CLEAN_PRESERVATION))
    assert set(report['verdicts'].values()) == {'FAIL'}
    assert 'local original is not listed in the manifest: faults.json' in report[
        'preconditions']['sanitization']['facts']['problems']


# ---- second review: a contradicted fact stays FAIL (P3-2) ---------------------------------

def test_settle_keeps_a_contradicted_check_when_a_field_is_missing():
    from tools.report_increment9 import settle
    data = dict(oom_kill=1)
    assert settle(dict(a=lambda: data['oom_kill'] == 1)) == dict(a=True)
    with pytest.raises(KeyError):
        settle(dict(a=lambda: data['oom_kill'] == 1, b=lambda: data['samples'] > 0))
    assert settle(dict(a=lambda: data['oom_kill'] == 0, b=lambda: data['samples'] > 0)) == dict(a=False, b=None)


def test_precondition_not_proven_does_not_mask_a_fail(copy):
    edit(copy, 'integrity-after.json', lambda x: x['files'][
        'qwen2.5-7b-d149729/model-00002-of-00004.safetensors'].update(match=False))
    edit(copy, 'phaseB-pipeline-supervisor.json', lambda x: x['finalizer']['memory.events'].update(oom_kill=1))
    (copy/'sanitization-manifest.json').unlink()
    own, report = own_verdicts(copy)
    result = report['verdicts']
    assert report['preconditions']['sanitization']['verdict'] == 'NOT PROVEN'
    # Before the fix every condition, the OOM kill included, was reported NOT PROVEN.
    assert result['A'] == 'FAIL' and result['G'] == 'FAIL', result
    assert own['A'] == own['G'] == 'FAIL'
    assert all(result[c] == 'NOT PROVEN' for c in 'BEFH'), result  # their own PASS is not kept either
    assert all(report['conditions'][c]['facts']['precondition_not_met'] == ['sanitization'] for c in EVERY)


def test_missing_amendment_does_not_mask_a_fail(copy):
    edit(copy, 'integrity-after.json', lambda x: x['files'][
        'qwen2.5-7b-d149729/model-00002-of-00004.safetensors'].update(match=False))
    (copy/'protocol-amendment-remediation.json').unlink()
    result = verdicts(copy)
    assert result['A'] == 'FAIL' and all(result[c] == 'NOT PROVEN' for c in 'BCDEFGH'), result


@pytest.mark.parametrize('case', ['same_supervisor', 'same_supervisor_fact', 'other_file'])
def test_missing_unrelated_field_does_not_mask_a_fail(copy, case):
    edit(copy, 'phaseB-pipeline-supervisor.json', lambda x: x['finalizer']['memory.events'].update(oom_kill=1))
    if case == 'same_supervisor':
        edit(copy, 'phaseB-pipeline-supervisor.json', lambda x: x.pop('samples'))
    if case == 'same_supervisor_fact':
        edit(copy, 'phaseB-pipeline-supervisor.json', lambda x: x.pop('minimum_global_memavailable'))
    if case == 'other_file':
        edit(copy, 'faults.json', lambda x: x.pop('cases'))
    own, report = own_verdicts(copy)
    assert own['G'] == 'FAIL' and report['verdicts']['G'] == 'FAIL', (own, report['conditions']['G']['facts'])


def test_missing_field_alone_is_still_not_proven(copy):
    edit(copy, 'phaseB-pipeline-supervisor.json', lambda x: x.pop('samples'))
    own, _ = own_verdicts(copy)
    assert own['G'] == 'NOT PROVEN'
    edit(copy, 'phaseB-pipeline-runtime.json', lambda x: x.pop('process_after_close'))
    assert own_verdicts(copy)[0]['G'] == 'NOT PROVEN'


# ---- second review: unreadable evidence gives a verdict, not a crash (P3-3) ---------------

@pytest.mark.parametrize('name', ['protocol.json', 'protocol-amendment-remediation.json',
                                  'sanitization-manifest.json'])
def test_invalid_json_in_a_precondition_file_is_fail_not_a_crash(copy, name):
    replace(copy, name, '{"increment": 9,')
    report = gate(copy, originals=copy/'no-originals',
                  preservation=lambda baseline, intentional: dict(CLEAN_PRESERVATION))
    assert set(report['verdicts'].values()) == {'FAIL'}
    kind = 'sanitization' if name.startswith('sanitization') else 'protocol'
    assert report['preconditions'][kind]['verdict'] == 'FAIL'
    (copy/name).unlink()
    assert set(verdicts(copy).values()) == {'NOT PROVEN'}


def test_invalid_json_in_evidence_is_fail_not_a_crash(copy):
    replace(copy, 'integrity-final.json', '{"files": ')
    replace(copy, 'faults.json', 'not json')
    result = verdicts(copy)
    assert result['A'] == 'FAIL' and result['G'] == 'FAIL', result


def test_zero_logical_inference_bytes_is_fail_not_a_crash(copy):
    edit(copy, 'phaseB-sync-runtime.json', lambda x: x['stats'].update(inference_read_bytes=0))
    result = verdicts(copy)
    assert all(result[c] == 'FAIL' for c in 'CDEG'), result


# ---- second review: index entry and computed digests (P3-4) ---------------------------------

INDEX_ENTRY = 'qwen2.5-7b-d149729/model.safetensors.index.json'


@pytest.mark.parametrize('entry', [dict(match=True), dict(match=True, sha256='0'*64, expected='0'*64),
                                   dict(match=True, sha256='624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028'),
                                   dict(match=True, expected='624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028')])
def test_final_rehash_must_record_the_index_digest(copy, entry):
    assert json.loads((copy/'integrity-final.json').read_text())['files'][INDEX_ENTRY]['sha256'] == json.loads(
        (copy/'qwen-acquisition.json').read_text())['downloaded']['model.safetensors.index.json']['sha256']
    edit(copy, 'integrity-final.json', lambda x: x['files'].update({INDEX_ENTRY: entry}))
    report = gate(copy, originals=copy/'no-originals',
                  preservation=lambda baseline, intentional: dict(CLEAN_PRESERVATION))
    assert report['verdicts']['A'] == 'FAIL'
    assert report['conditions']['A']['facts']['checks']['rehash_after_remediation_equals_official'] is False


def test_historical_rehash_index_digest_cannot_be_wrong(copy):
    # integrity-after.json recorded only the outcome for the index (declared limit); a digest, if there, must agree.
    edit(copy, 'integrity-after.json', lambda x: x['files'][INDEX_ENTRY].update(sha256='0'*64))
    assert verdicts(copy)['A'] == 'FAIL'


@pytest.mark.parametrize('value,verdict', [('official', 'PASS'), ('zeros', 'FAIL'), ('short', 'FAIL'),
                                           ('upper', 'FAIL'), ('one_wrong', 'FAIL'), ('true', 'FAIL')])
def test_computed_digests_must_be_hex_and_equal_to_the_official_ones(copy, value, verdict):
    def change(x):
        for i, (name, entry) in enumerate(sorted(x['downloaded'].items())):
            if 'of-00004' in name:
                official = entry['sha256']
                entry['computed_sha256'] = dict(
                    official=official, zeros='0'*64, short=official[:10], upper=official.upper(), true=True,
                    one_wrong='0'*64 if name.startswith('model-00003') else official)[value]
    edit(copy, 'qwen-acquisition.json', change)
    assert verdicts(copy)['A'] == verdict


# ---- second review: one-line hardenings ------------------------------------------------------

def test_negative_residency_is_not_evidence_of_eviction(copy):
    edit(copy, 'phaseB-sync-supervisor.json', lambda x: each(
        x['residency_after_eviction'], lambda v: v.update(resident_bytes=-1)))
    own, _ = own_verdicts(copy)
    assert all(own[c] == 'FAIL' for c in 'CDG'), own


def test_json_false_is_not_a_zero_oom_counter(copy):
    edit(copy, 'phaseB-sync-supervisor.json', lambda x: x['finalizer']['memory.events'].update(oom_kill=False))
    own, _ = own_verdicts(copy)
    assert all(own[c] == 'FAIL' for c in 'CDG'), own


# ---- second review: honest amendment register (P3-1) -----------------------------------------

def test_amendment_register_labels_post_measurement_choices():
    amendment = json.loads((EVIDENCE/'protocol-amendment-remediation.json').read_text())
    changes = {c['id']: c for c in amendment['changes']}
    assert list(changes) == [f'R{i}' for i in range(1, 11)]
    assert amendment['status'].startswith('post-measurement') and 'every change tightens' not in amendment['status']
    assert changes['R10']['kind'].startswith('not a tightening')
    for name in ('R5', 'R6', 'R7'):
        assert 'chosen after measurement' in changes[name]['kind'], name
    assert hashlib.sha256((EVIDENCE/'protocol.json').read_bytes()).hexdigest() == amendment['amends_protocol_sha256']


def test_git_bound_evidence_has_no_personal_paths():
    from tools import sanitize9
    assert sanitize9.residual(EVIDENCE) == []
    manifest = json.loads((EVIDENCE/'sanitization-manifest.json').read_text())
    assert sanitize9.verify(EVIDENCE, Path('artifacts/increment9/original-evidence'), manifest)['verdict'] == 'PASS'


def test_future_reports_use_repository_relative_paths(tmp_path, monkeypatch):
    from tools import sanitize9
    monkeypatch.chdir(tmp_path)
    report = dict(files=[str(tmp_path/'checkpoints/m.safetensors')], n=3,
                  residency={str(tmp_path/'checkpoints/m.safetensors'): dict(size_bytes=7)},
                  launcher=['/bin/sh', str(tmp_path/'tools/cgroup9_finalize.sh')])
    assert sanitize9.portable(report) == dict(
        files=['<repo>/checkpoints/m.safetensors'], n=3,
        residency={'<repo>/checkpoints/m.safetensors': dict(size_bytes=7)},
        launcher=['/bin/sh', '<repo>/tools/cgroup9_finalize.sh'])
    assert sanitize9.portable(str(tmp_path)) == '<repo>'


def test_audit_keeps_only_project_cache_entries(tmp_path):
    from tools.audit_increment9 import relevant_cache_entries
    for name in ('models--Other--Model', 'models--Qwen--Qwen2.5-7B', 'models--HuggingFaceTB--SmolLM2-1.7B', 'x'):
        (tmp_path/name).mkdir()
    assert relevant_cache_entries(tmp_path) == dict(
        project_entries=['models--HuggingFaceTB--SmolLM2-1.7B', 'models--Qwen--Qwen2.5-7B'], unrelated_entries=2)
    assert relevant_cache_entries(tmp_path/'absent') == dict(project_entries=[], unrelated_entries=0)


def test_runtime_report_paths_are_repository_relative(tmp_path, monkeypatch):
    from tools import sanitize9
    from tools.benchmark_increment9 import repository_path
    monkeypatch.chdir(tmp_path)
    assert repository_path('checkpoints/m') == 'checkpoints/m'
    assert repository_path(tmp_path/'checkpoints/m') == 'checkpoints/m'
    assert repository_path('/mnt/other/m') == '/mnt/other/m'
    monkeypatch.undo()
    # Clean historical worker reports are already portable: the rules change nothing in them.
    for name in ('phaseA-P-runtime.json', 'phaseB-sync-runtime.json', 'faults.json'):
        data = json.loads((EVIDENCE/name).read_text())
        assert sanitize9.portable(data) == data


def test_rehash_records_every_digest_including_the_index(tmp_path):
    from tools.integrity9 import integrity
    directory = tmp_path/'model'; directory.mkdir()
    (directory/'a.safetensors').write_bytes(b'weights'); (directory/'index.json').write_bytes(b'{}')
    descriptor = dict(index='index.json', index_sha256=hashlib.sha256(b'{}').hexdigest(),
                      files={'a.safetensors': dict(nbytes=7, sha256=hashlib.sha256(b'weights').hexdigest())})
    (directory/'safetensors.json').write_text(json.dumps(descriptor))
    report = integrity([directory])
    assert report['all_match'] and report['files']['model/index.json'] == dict(
        sha256=descriptor['index_sha256'], expected=descriptor['index_sha256'], match=True)
    (directory/'index.json').write_bytes(b'{ }')
    report = integrity([directory])
    assert not report['all_match'] and report['files']['model/index.json']['match'] is False


# ---- preservation recomputed from git ----------------------------------------------------

def test_preservation_separates_modified_deleted_and_added(tmp_path):
    from tools.integrity9 import preservation
    git = lambda *a: subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@example.invalid', *a],
                                    cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()
    git('init', '-q')
    for name in ('README.md', 'core.py', 'old.py'):
        (tmp_path/name).write_text(name)
    git('add', '.'); git('commit', '-qm', 'baseline')
    baseline = git('rev-parse', 'HEAD')
    (tmp_path/'README.md').write_text('changed')
    (tmp_path/'tool9.py').write_text('new, committed after the baseline')
    git('add', 'tool9.py'); git('commit', '-qm', 'increment')
    (tmp_path/'notes.txt').write_text('untracked')
    report = preservation(baseline, ['README.md'], cwd=tmp_path)
    assert report['passed'] and report['modified_tracked'] == ['README.md'] and report['deleted'] == []
    assert report['added_since_baseline'] == ['tool9.py'] and report['new_untracked'] == ['notes.txt']
    (tmp_path/'core.py').write_text('changed'); (tmp_path/'old.py').unlink()
    report = preservation(baseline, ['README.md'], cwd=tmp_path)
    assert not report['passed'] and report['unexpected_modifications'] == ['core.py'] and report['deleted'] == ['old.py']


# ---- acquisition: computed digests, existing files, complete .part -------------------------

class FakeResponse(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def remote(tmp_path, monkeypatch):
    from tools import acquire_qwen9
    payload = bytes(range(256))*40
    calls = []

    def urlopen(request, timeout=None):
        calls.append(dict(request.header_items()))
        start = int(calls[-1].get('Range', 'bytes=0-')[6:-1])
        if start >= len(payload):
            raise OSError('HTTP Error 416: Requested Range Not Satisfiable')
        response = FakeResponse(payload[start:])
        response.status = 206 if start else 200
        return response
    monkeypatch.setattr(acquire_qwen9.urllib.request, 'urlopen', urlopen)
    usage = collections.namedtuple('usage', 'total used free')(2**40, 0, 2**40)
    monkeypatch.setattr(acquire_qwen9.shutil, 'disk_usage', lambda path: usage)
    destination = tmp_path/'checkpoint'
    destination.mkdir()
    return dict(module=acquire_qwen9, payload=payload, calls=calls, destination=destination,
                sha=hashlib.sha256(payload).hexdigest(), name='model-00001-of-00004.safetensors')


def fetch(remote, expected=None):
    return remote['module'].download(remote['name'], len(remote['payload']),
                                     remote['sha'] if expected is None else expected, remote['destination'], {})


def test_download_records_the_digest_it_computed(remote):
    result = fetch(remote)
    assert result['computed_sha256'] == remote['sha'] and result['method'] == 'streamed'
    assert (remote['destination']/remote['name']).read_bytes() == remote['payload'] and len(remote['calls']) == 1


def test_download_mismatch_keeps_part_and_raises(remote):
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        fetch(remote, expected='0'*64)
    assert not (remote['destination']/remote['name']).exists()
    assert (remote['destination']/(remote['name']+'.part')).exists()


def test_existing_file_is_hashed_before_being_accepted(remote):
    final = remote['destination']/remote['name']
    final.write_bytes(remote['payload'])
    result = fetch(remote)
    assert result == dict(method='existing file re-hashed', computed_sha256=remote['sha'], resumed_from_bytes=0)
    assert remote['calls'] == []
    corrupted = bytearray(remote['payload']); corrupted[5] ^= 1
    final.write_bytes(corrupted)  # same size, different content: the original code accepted it unverified
    with pytest.raises(ValueError, match='existing file'):
        fetch(remote)
    final.write_bytes(remote['payload'][:-1])
    with pytest.raises(ValueError, match='existing file'):
        fetch(remote)


def test_complete_part_is_verified_without_a_range_request(remote):
    part = remote['destination']/(remote['name']+'.part')
    part.write_bytes(remote['payload'])
    result = fetch(remote)
    assert result['method'] == 'complete .part re-hashed' and result['computed_sha256'] == remote['sha']
    assert remote['calls'] == [] and not part.exists()
    assert (remote['destination']/remote['name']).read_bytes() == remote['payload']


def test_complete_but_corrupt_part_is_refused(remote):
    part = remote['destination']/(remote['name']+'.part')
    corrupted = bytearray(remote['payload']); corrupted[-1] ^= 1
    part.write_bytes(corrupted)
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        fetch(remote)
    assert part.exists() and remote['calls'] == [] and not (remote['destination']/remote['name']).exists()


def test_partial_part_resumes_with_range_and_full_digest(remote):
    part = remote['destination']/(remote['name']+'.part')
    part.write_bytes(remote['payload'][:1000])
    result = fetch(remote)
    assert result == dict(method='resumed .part re-hashed then streamed', computed_sha256=remote['sha'],
                          resumed_from_bytes=1000)
    assert remote['calls'][0]['Range'] == 'bytes=1000-'
    assert (remote['destination']/remote['name']).read_bytes() == remote['payload']


def test_acquisition_evidence_separates_computed_and_official_digests(remote, tmp_path, monkeypatch):
    module, payload = remote['module'], remote['payload']
    monkeypatch.setattr(module, 'SMALL', (module.INDEX,))
    monkeypatch.setattr(module, 'SHARDS', {remote['name']: None})
    monkeypatch.setattr(module, 'preflight', lambda destination: dict(
        free_bytes=2**40, backing_volume_free_bytes=2**40, required_free_bytes=1, existing=[],
        files={module.INDEX: dict(nbytes=len(payload), blob_id=module.blob_id(payload), lfs_sha256=None),
               remote['name']: dict(nbytes=len(payload), blob_id='x', lfs_sha256=remote['sha'])}))
    report = module.acquire(remote['destination'], tmp_path/'evidence.json', True)
    shard = report['downloaded'][remote['name']]
    assert shard['sha256'] == shard['computed_sha256'] == shard['official_lfs_sha256'] == remote['sha']
    assert shard['verification'] == 'streamed' and 'verified' not in shard
    descriptor = json.loads((remote['destination']/'safetensors.json').read_text())
    assert descriptor['files'][remote['name']]['sha256'] == remote['sha']
    # A second run finds the files already present: it must hash them, not copy the expected digest.
    again = module.acquire(remote['destination'], tmp_path/'evidence-2.json', True)
    assert again['downloaded'][remote['name']]['verification'] == 'existing file re-hashed'
    corrupted = bytearray(payload); corrupted[0] ^= 1
    (remote['destination']/remote['name']).write_bytes(corrupted)
    with pytest.raises(ValueError, match='existing file'):
        module.acquire(remote['destination'], tmp_path/'evidence-3.json', True)
    assert not (tmp_path/'evidence-3.json').exists()


# ---- verifier: greedy recomputed from the oracle arrays --------------------------------------

@pytest.fixture
def capture(tmp_path, monkeypatch):
    from tools import verify_increment9
    directory = tmp_path/'cap'; directory.mkdir()
    rng = np.random.default_rng(0)
    vocab, layers, width, kv_width = 12, 2, 4, 2
    full = dict(logits=rng.normal(size=(3, vocab)), hidden=rng.normal(size=(layers, 3, width)),
                kv=rng.normal(size=(layers, 2, 3, kv_width)))
    full['logits'][0, 2] = 50; full['logits'][1, 9] = 50; full['logits'][2, 1] = 50  # greedy 2, 9, 1
    phases, captures, start = [], {}, 0
    for i, ids in enumerate([[5, 7], [9]]):
        stop = start+len(ids)
        arrays = dict(logits=full['logits'][start:stop], hidden=full['hidden'][:, start:stop], kv=full['kv'][:, :, :stop])
        captures[f'decode.{i}'] = {}
        for kind, values in arrays.items():
            raw = np.ascontiguousarray(values, dtype='<f8').tobytes()
            (directory/f'{i}.{kind}').write_bytes(raw)
            captures[f'decode.{i}'][kind] = dict(file=f'{i}.{kind}', shape=list(values.shape), dtype='<f8',
                                                 nbytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        phases.append(dict(ids=ids, greedy=arrays['logits'].argmax(-1).tolist()))
        start = stop
    runtime = tmp_path/'runtime.json'
    runtime.write_text(json.dumps(dict(phases=phases, captures=captures, input_ids=[5, 7], checkpoint='x')))
    protocol = tmp_path/'protocol.json'
    protocol.write_text(json.dumps(dict(fp32_comparison=dict(atol=2e-3, rtol=2e-4),
                                        fp64_comparison=dict(atol=2e-5, rtol=2e-6), phase_b=dict(revision='r'))))
    monkeypatch.setattr(verify_increment9, 'LlamaTokenizer', lambda checkpoint: None)

    def oracle(arrays, **claims):
        path = tmp_path/'oracle.npz'
        with path.open('wb') as f:
            np.savez(f, **{f'full.{k}': v for k, v in arrays.items()})
        meta = dict(contract='stock_fp32', dtype='float32', precision_profile='stock',
                    attention_implementation='eager', revision='r', sequence=[5, 7, 9], prompt_length=2,
                    tokenizer=[], upstream_greedy_matches_sequence=True,
                    capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest()) | claims
        path.with_suffix('.json').write_text(json.dumps(meta))
        return path
    return dict(runtime=runtime, directory=directory, protocol=protocol, full=full, oracle=oracle,
                verify=lambda path: verify_increment9.verify(runtime, directory, path, protocol))


def test_verifier_recomputes_upstream_greedy_from_arrays(capture):
    report = capture['verify'](capture['oracle'](capture['full']))
    assert report['passed'] and report['upstream_greedy_matches_sequence']
    assert report['upstream_greedy'] == dict(recomputed_from_oracle_argmax=True, declared_by_oracle=True,
                                             decode_inputs_follow_runtime_greedy=True)


def test_verifier_does_not_trust_the_declared_upstream_greedy(capture):
    # The runtime fed token 3, not its own greedy choice 9; upstream on that sequence also prefers 9.
    # Every array, argmax and recorded greedy still agrees: only the oracle's own claim said "greedy".
    data = json.loads(capture['runtime'].read_text())
    data['phases'][1]['ids'] = [3]
    capture['runtime'].write_text(json.dumps(data))
    report = capture['verify'](capture['oracle'](capture['full'], sequence=[5, 7, 3],
                                                 upstream_greedy_matches_sequence=True))
    assert not report['passed'] and not report['upstream_greedy_matches_sequence']
    assert report['upstream_greedy'] == dict(recomputed_from_oracle_argmax=False, declared_by_oracle=True,
                                             decode_inputs_follow_runtime_greedy=False)
    assert all(c[k]['failures'] == 0 for c in report['comparisons'].values() for k in ('logits', 'hidden', 'kv'))


def test_verifier_keeps_a_false_declared_greedy_as_failure(capture):
    report = capture['verify'](capture['oracle'](capture['full'], upstream_greedy_matches_sequence=False))
    assert not report['passed']

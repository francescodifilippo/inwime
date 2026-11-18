"""Increment 10, C2: the preregistered protocol is frozen, recomputable and binding.

Checks the protocol digest file, recomputes every derived section from the captured
official metadata and the increment-9 evidence, verifies the pinned tool and inventory
digests, and the increment-10 invariants: runtime and historical evidence unchanged.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tools import protocol10

ROOT = Path(__file__).resolve().parents[1]
R10 = ROOT/'results'/'increment10'


def protocol():
    return json.loads((R10/'protocol.json').read_text())


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def test_protocol_frozen_and_recomputable():
    result = protocol10.check()
    assert result['passed'], result['problems']


def test_protocol_digest_file():
    digest = hashlib.sha256((R10/'protocol.json').read_bytes()).hexdigest()
    assert (R10/'protocol.sha256').read_text() == f'{digest}  protocol.json\n'


def test_protocol_is_separate_and_m10_specific():
    PROTOCOL = protocol()
    assert PROTOCOL['schema'] == 'inwime-protocol-m10/1' and PROTOCOL['increment'] == 10
    assert PROTOCOL['checkpoint']['revision'] == '97e1e76335b7017d8f67c08a19d103c0504298c9'
    assert PROTOCOL['model_geometry']['gqa_ratio'] == 5 and PROTOCOL['model_geometry']['rms_norm_eps'] == 1e-5
    m9 = json.loads((ROOT/'results'/'increment9'/'protocol.json').read_text())
    assert m9['increment'] == 9                        # untouched and distinct


def test_thresholds_unchanged_from_m7_m9():
    PROTOCOL = protocol()
    m9 = json.loads((ROOT/'results'/'increment9'/'protocol.json').read_text())
    t = PROTOCOL['numerical_contract']['thresholds']
    assert (t['fp32']['atol'], t['fp32']['rtol']) == (m9['fp32_comparison']['atol'], m9['fp32_comparison']['rtol'])
    assert (t['fp64']['atol'], t['fp64']['rtol']) == (m9['fp64_comparison']['atol'], m9['fp64_comparison']['rtol'])


def test_workload_has_prefill_and_three_real_decodes():
    PROTOCOL = protocol()
    w = PROTOCOL['workload']
    assert w['prefill_tokens'] == 5 and w['decodes'] >= 3 and w['chunks'] == [5, 1, 1, 1]
    assert 'even if it is EOS' in w['decode_rule']


def test_active_weights_exceed_host_ram_lower_bound():
    PROTOCOL = protocol()
    a = PROTOCOL['active_weights']
    assert a['unique_bytes_5_distinct_rows'] > a['versus_host_ram']['installed_dimm']
    assert a['unique_bytes_5_distinct_rows'] > a['versus_host_ram']['total_physical_memory']
    assert a['class'].startswith('estimate')


def test_verdict_precedence_and_classes():
    PROTOCOL = protocol()
    v = PROTOCOL['verdicts']
    assert v['precedence'] == ['FAIL', 'BLOCKED', 'NOT PROVEN', 'PASS']
    assert set(PROTOCOL['host_memory_preflight']['verdicts']) == {'GO', 'DEFER', 'NOT MEASURED'}
    assert set(PROTOCOL['classification']) == {'capacity', 'observation', 'estimate'}
    assert set(PROTOCOL['gate_conditions']) == set('ABCDEFGH')


def test_every_criterion_quantity_is_produced_by_hostmem10():
    PROTOCOL = protocol()
    from tools import hostmem10
    snap = dict(windows={}, guest={}, checkpoint_volume={}, backing_volume={})
    produced = set(hostmem10.measurements(snap, snap, snap))
    h = PROTOCOL['host_memory_preflight']
    for profile in h['start_criteria'].values():
        assert {c['quantity'] for c in profile} <= produced
    assert {c['quantity'] for c in h['abort_criteria']} <= produced


def test_pinned_tools_exist_and_match():
    PROTOCOL = protocol()
    for path, digest in PROTOCOL['pinned_tools_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path


def test_runtime_and_kernel_unchanged_since_base():
    """M10 invariant (gate H): the inference engine is not modified by this increment."""
    PROTOCOL = protocol()
    base = PROTOCOL['base_commit']
    assert git('diff', '--name-only', base, '--', 'inwime_runtime', 'src') == ''
    assert git('status', '--porcelain', '--', 'inwime_runtime', 'src') == ''


@pytest.mark.parametrize('area', ['results/increment7', 'results/increment8', 'results/increment9', 'tests', 'tools'])
def test_no_preexisting_file_modified_or_deleted(area):
    """M10 is additive: historical evidence and earlier tests/tools are untouched (gate 9 H stays valid)."""
    PROTOCOL = protocol()
    base = PROTOCOL['base_commit']
    assert git('diff', '--name-only', '--no-renames', '--diff-filter=MDT', base, '--', area) == ''

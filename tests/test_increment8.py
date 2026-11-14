"""Evidence must fail closed on damaged captures and incomplete comparisons."""
from array import array
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.benchmark_increment8 import expected_reads, save_values
from tools.verify_increment8 import read_capture, invariants, verify
from tools.report_increment8 import full_fp32_parity, complete_exact_checks, report


def test_coverage_counts_tied_head_once_for_unique_but_twice_for_reads():
    # Every output coefficient uses the tied embedding, embedding lookup only
    # uses a row. Never count a second physical embedding in unique payload.
    tensor = SimpleNamespace(elements=24)
    catalog = SimpleNamespace(spec=SimpleNamespace(width=4),
                              tensors={'embedding': tensor, 'lm_head': tensor,
                                       'norm': SimpleNamespace(elements=4)})
    assert expected_reads(catalog, 5) == {'embedding': 40, 'lm_head': 240, 'norm': 40}


@pytest.mark.parametrize('fault', ['corrupt', 'truncate', 'path', 'dtype'])
def test_capture_integrity_and_representation(tmp_path, fault):
    values = array('d', [1., -0., 3., -4.])
    descriptor = save_values(tmp_path, 'decode.0', 'logits', [memoryview(values)], [2, 2])
    assert read_capture(tmp_path, descriptor).tobytes() == values.tobytes()
    path = tmp_path/descriptor['file']
    if fault == 'corrupt':
        raw = bytearray(path.read_bytes()); raw[0] ^= 1; path.write_bytes(raw)
    if fault == 'truncate':
        path.write_bytes(path.read_bytes()[:-1])
    if fault == 'path':
        descriptor['file'] = '../'+descriptor['file']
    if fault == 'dtype':
        descriptor['dtype'] = '<f4'
    with pytest.raises((ValueError, FileNotFoundError)):
        read_capture(tmp_path, descriptor)


def test_invariants_require_complete_scenarios():
    with pytest.raises(ValueError, match='all three'):
        invariants([])


def test_verification_requires_real_oracle(tmp_path):
    report = tmp_path/'runtime.json'; report.write_text('{}')
    with pytest.raises(FileNotFoundError):
        verify(report, tmp_path, tmp_path/'missing.npz', tmp_path/'protocol.json')


def test_bitwise_invariants_detect_signed_zero_and_missing_values(tmp_path):
    runs = []
    for scenario, count in [('decode', 4), ('chunks', 2), ('extended', 1), ('decode', 4)]:
        directory = tmp_path/str(len(runs)); directory.mkdir()
        captures = {}
        ids = [[1, 2], [3], [4], [5]] if scenario == 'decode' else [[1], [2]] if scenario == 'chunks' else [[1, 2, 3, 4, 5]]
        length = 0
        for i in range(count):
            length += len(ids[i])
            prefix = f'{scenario}.{i}'
            values = {'logits': ([len(ids[i]), 1], [0.]*len(ids[i])),
                      'hidden': ([1, len(ids[i]), 1], [0.]*len(ids[i])),
                      'kv': ([1, 2, length, 1], [0.]*(2*length))}
            captures[prefix] = {k: save_values(directory, prefix, k, [memoryview(array('d', data))], shape)
                                for k, (shape, data) in values.items()}
        report = directory/'report.json'
        report.write_text(json.dumps(dict(scenario=scenario, captures=captures,
            input_ids=[1, 2], phases=[dict(ids=chunk) for chunk in ids])))
        runs.append((report, directory))
    assert invariants(runs)['passed']
    report = json.loads(runs[-1][0].read_text())
    desc = report['captures']['decode.0']['logits']
    path = runs[-1][1]/desc['file']
    raw = array('d', [-0., 0.]).tobytes()
    path.write_bytes(raw); desc['sha256'] = hashlib.sha256(raw).hexdigest()
    runs[-1][0].write_text(json.dumps(report))
    assert not invariants(runs)['passed']


@pytest.mark.parametrize('fault', ['missing_phase', 'partial_logits', 'missing_kv', 'argmax', 'tolerance'])
def test_gate_rejects_partial_or_relaxed_parity(fault):
    runtime = dict(config=dict(vocab_size=7, hidden_size=4, num_hidden_layers=2,
        num_key_value_heads=2, num_attention_heads=2), phases=[dict(name='decode.0', ids=[1, 2], length=2)])
    comparison = dict(greedy_equal=True, argmax_equal=True,
        logits=dict(count=14, failures=0), hidden=dict(count=16, failures=0), kv=dict(count=32, failures=0))
    parity = dict(passed=True, atol=2e-3, rtol=2e-4, comparisons={'decode.0': comparison})
    parity.update(oracle_contract='transformers_standard_float32', oracle_sha256='a'*64,
        oracle_metadata=dict(contract='transformers_standard_float32', dtype='float32',
            precision_profile='stock', attention_implementation='eager',
            revision='d4c18ccb8a25e6e637e0609aac08a5d1a0436d1b', capture_sha256='a'*64))
    assert full_fp32_parity(parity, runtime)
    if fault == 'missing_phase':
        parity['comparisons'].clear()
    elif fault == 'partial_logits':
        comparison['logits']['count'] = 1
    elif fault == 'missing_kv':
        comparison['kv']['count'] = 0
    elif fault == 'argmax':
        comparison['argmax_equal'] = False
    else:
        parity['atol'] = 1.
    assert not full_fp32_parity(parity, runtime)


@pytest.mark.parametrize('field,value', [('contract', 'not_transformers'), ('dtype', 'float16'),
    ('precision_profile', 'full64'), ('attention_implementation', 'sdpa'),
    ('revision', '0'*40), ('capture_sha256', 'b'*64)])
def test_cached_parity_rejects_wrong_oracle_contract(field, value):
    directory = Path(__file__).resolve().parents[1]/'results/increment8'
    runtime = json.loads((directory/'smoke-sync-1024.json').read_text())
    parity = json.loads((directory/'parity-original-sync.json').read_text())
    assert full_fp32_parity(parity, runtime)
    parity['oracle_metadata'][field] = value
    assert not full_fp32_parity(parity, runtime)


@pytest.mark.parametrize('fault', ['missing_regression7', 'empty_exact', 'missing_exact_check'])
def test_aggregate_gate_rejects_incomplete_evidence(tmp_path, fault):
    source = Path(__file__).resolve().parents[1]/'results/increment8'
    directory = tmp_path/'evidence'
    shutil.copytree(source, directory, ignore=shutil.ignore_patterns('review'))
    if fault == 'missing_regression7':
        (directory/'regression7-stock_fp32.json').unlink()
    else:
        path = directory/'exact-invariants.json'
        exact = json.loads(path.read_text())
        if fault == 'empty_exact':
            exact['checks'] = {}
        else:
            exact['checks'].pop('kv_incremental_exact')
        path.write_text(json.dumps(exact))
    try:
        result = report(directory)
    except (ValueError, FileNotFoundError):
        return
    assert not result['passed']


@pytest.mark.parametrize('fault', ['empty', 'missing_causal', 'missing_pipeline', 'missing_shards'])
def test_cached_exact_gate_requires_each_comparison(fault):
    path = Path(__file__).resolve().parents[1]/'results/increment8/exact-invariants.json'
    exact = json.loads(path.read_text())
    assert complete_exact_checks(exact)
    if fault == 'empty':
        exact['checks'].clear()
    else:
        key = {'missing_causal': 'kv_incremental_exact',
               'missing_pipeline': 'results/increment8/measured-pipeline-1.json_exact',
               'missing_shards': 'results/increment8/sharded-sync.json_exact'}[fault]
        exact['checks'].pop(key)
    assert not complete_exact_checks(exact)

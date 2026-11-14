"""Numerical contracts: mixed FP64 diagnostics must not replace the pure64 oracle."""
from copy import deepcopy
import pytest
from tools.report_increment7 import numerical_gate


def fixture_reports():
    scenarios={'single':1,'full':1,'chunks':2,'decode':4,'extended':1,'repeat':1}
    base=dict(oracle_executed=True,oracle_contract='transformers_standard_float32',
              oracle_precision_profile='stock',dtype='float32',atol=2e-3,rtol=2e-4,passed=True,
              oracle_sha256='a'*64,engine_invariants={'logits':{'chunks_exact':True}},
              dtype_audit={'non_fp64_output_operations':{'softmax -> torch.float32':30}},
              comparisons={name:dict(steps=[dict(comparisons={
                'logits':dict(count=49152,max_abs=0.,failures=0),
                'hidden':dict(count=17280,max_abs=0.,failures=0),
                'kv':dict(count=11520,max_abs=0.,failures=0),
                'greedy_equal':True}) for _ in range(n)]) for name,n in scenarios.items()})
    pure=deepcopy(base);pure.update(oracle_contract='pytorch_pure_fp64',oracle_precision_profile='full64',dtype='float64',atol=2e-5,rtol=2e-6,dtype_audit={'non_fp64_output_operations':{}})
    diagnostic=deepcopy(base);diagnostic.update(oracle_contract='transformers_standard_float64',dtype='float64',atol=2e-5,rtol=2e-6,passed=False)
    diagnostic['comparisons']['single']['steps'][0]['comparisons']['logits']['failures']=1
    return {'stock_fp32':base,'pure_fp64':pure,'standard_fp64_diagnostic':diagnostic,
            'pure_fp64_pipeline':deepcopy(pure),'pure_fp64_sharded':deepcopy(pure)}


def test_gate_distinguishes_contracts():
    assert numerical_gate(fixture_reports())['passed']


def test_gate_requires_actual_pure64():
    r=fixture_reports();r['pure_fp64']['oracle_precision_profile']='rms64'
    with pytest.raises(ValueError):numerical_gate(r)


def test_gate_requires_diagnostic_run():
    r=fixture_reports();del r['standard_fp64_diagnostic']
    with pytest.raises(ValueError):numerical_gate(r)


@pytest.mark.parametrize('mutation',['threshold','float32','missing_scenario','not_executed'])
def test_gate_rejects_invalid_proof(mutation):
    r=fixture_reports();pure=r['pure_fp64']
    if mutation=='threshold':pure['atol']=1.
    if mutation=='float32':pure['dtype_audit']['non_fp64_output_operations']={'softmax -> torch.float32':1}
    if mutation=='missing_scenario':del pure['comparisons']['decode']
    if mutation=='not_executed':pure['oracle_executed']=False
    with pytest.raises(ValueError):numerical_gate(r)


def test_gate_keeps_real_divergence_failed():
    r=fixture_reports();r['pure_fp64']['passed']=False
    r['pure_fp64']['comparisons']['decode']['steps'][3]['comparisons']['kv']['failures']=1
    assert not numerical_gate(r)['passed']

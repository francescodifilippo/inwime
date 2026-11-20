"""Increment 10, C5: oracle10 independence, explicit memory plan and coverage classification.

Runs without torch (offline CI). The torch-side checks live in tools/check_oracle10.py
(verification venv) and the 7B component equivalence in tools/equivalence10.py.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from tools import compare10, fixture_qwen10, oracle10_plan

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT/'results'/'increment10'/'preflight-metadata'
GiB = 2**30


def imports(path):
    tree = ast.parse(Path(path).read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name.split('.')[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            out.add((node.module or '').split('.')[0] if node.level == 0 else '.'+(node.module or ''))
    return out


# ---------------------------------------------------------------- independence (static)

@pytest.mark.parametrize('tool', ['oracle10.py', 'oracle10_plan.py', 'compare10.py', 'equivalence10.py'])
def test_oracle_side_never_imports_runtime(tool):
    found = imports(ROOT/'tools'/tool)
    assert 'inwime_runtime' not in found and 'inwime_reference' not in found


def test_oracle10_relative_imports_are_oracle_only():
    allowed = {'.oracle_precision9', '.', '.oracle10_plan', '.compare10', '.oracle10'}
    for tool in ('oracle10.py', 'equivalence10.py'):
        rel = {m for m in imports(ROOT/'tools'/tool) if m.startswith('.')}
        assert rel <= allowed, (tool, rel)
    # the reused FP64 profile itself depends only on the increment-7 profile, not on InWiMe
    assert imports(ROOT/'tools'/'oracle_precision9.py') - {'contextlib', 'types', 'torch', 'transformers', '.oracle_precision7'} == set()
    assert 'inwime_runtime' not in imports(ROOT/'tools'/'oracle_precision7.py')


def test_runtime_never_imports_torch_or_transformers():
    for path in (ROOT/'inwime_runtime').glob('*.py'):
        assert not imports(path) & {'torch', 'transformers', 'safetensors', 'numpy'}, path


def test_compare_and_plan_are_numpy_or_stdlib_only():
    assert imports(ROOT/'tools'/'compare10.py') <= {'numpy'}
    assert imports(ROOT/'tools'/'oracle10_plan.py') <= {'json', 'pathlib', 'struct'}


# ---------------------------------------------------------------- explicit memory plan (14B, real headers)

@pytest.fixture(scope='module')
def q14(tmp_path_factory):
    return fixture_qwen10.sparse_checkpoint(METADATA, tmp_path_factory.mktemp('q14')/'ckpt')


@pytest.fixture(scope='module')
def config():
    return json.loads((METADATA/'file-config.json').read_text())


def test_fp64_head_never_materialized_and_plans_fit_scope(q14, config):
    protocol = json.loads((ROOT/'results'/'increment10'/'protocol.json').read_text())['numerical_contract']['oracle10']
    scope = protocol['scope_memory_max_bytes']
    for contract, entry in protocol['contracts'].items():
        p = oracle10_plan.plan(q14, config, entry['dtype'], 8, entry['head_mode'], entry['block_rows'])
        assert p['planned_peak_bytes'] <= protocol['memory_budget_bytes'] < scope, contract
        if entry['dtype'] == 'float64':
            assert entry['head_mode'] == 'blocked'
            assert p['components']['head'] < p['full_head_in_dtype_bytes']/7
    assert oracle10_plan.plan(q14, config, 'float64', 8)['full_head_in_dtype_bytes'] == 152064*5120*8
    with pytest.raises(ValueError, match='excluded by policy'):
        oracle10_plan.plan(q14, config, 'float64', 8, 'resident')


def test_oracle9_style_module_would_not_fit(q14, config):
    """oracle9 materializes the embedding/head module in FP64: 6.23 GB + 1.56 GB BF16 (preflight R1)."""
    full_module = 152064*5120*8 + 152064*5120*2
    assert full_module == 7_785_676_800
    p = oracle10_plan.plan(q14, config, 'float64', 8)
    assert p['planned_peak_bytes'] < full_module/1.8


def test_budget_is_explicit_and_refuses(q14, config):
    p = oracle10_plan.plan(q14, config, 'float64', 8)
    with pytest.raises(MemoryError, match='no fallback'):
        oracle10_plan.check_budget(p, p['planned_peak_bytes']-1)
    with pytest.raises(ValueError):
        oracle10_plan.check_budget(p, None)
    assert oracle10_plan.check_budget(p, p['planned_peak_bytes'])['headroom_bytes'] == 0


def test_plan_rejects_non_bf16(tmp_path):
    d = fixture_qwen10.write(tmp_path/'q')
    shard = d/'model-00001-of-00003.safetensors'
    raw = bytearray(shard.read_bytes())
    i = raw.index(b'"BF16"')
    raw[i:i+6] = b'"F16" '
    shard.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match='not BF16'):
        oracle10_plan.read_headers(d)


# ---------------------------------------------------------------- coverage classification

SHAPES = compare10.required_shapes(layers=2, positions=3, width=4, vocab=5, kv_width=2)


def arrays(seed=0, scale=1.0):
    rng = np.random.default_rng(seed)
    return {k: rng.normal(size=s)*scale for k, s in SHAPES.items()}


def test_full_pass():
    ref = arrays()
    r = compare10.compare({k: v+1e-7 for k, v in ref.items()}, ref, SHAPES, 'fp64')
    assert (r['coverage'], r['verdict']) == ('FULL', 'PASS')
    assert r['kinds']['logits']['elements'] == 15 and r['kinds']['kv']['elements'] == 24


def test_single_violation_fails():
    ref = arrays()
    act = {k: v.copy() for k, v in ref.items()}
    act['kv'][1, 0, 2, 1] += 1e-3
    r = compare10.compare(act, ref, SHAPES, 'fp64')
    assert r['verdict'] == 'FAIL' and r['kinds']['kv']['failures'] == 1


def test_fp32_threshold_boundary():
    ref = arrays()
    inside = {k: v + 0.999*(2e-3 + 2e-4*np.abs(v)) for k, v in ref.items()}
    assert compare10.compare(inside, ref, SHAPES, 'fp32')['verdict'] == 'PASS'
    outside = dict(inside, logits=ref['logits'].copy())
    outside['logits'][0, 0] += 1.001*(2e-3 + 2e-4*abs(ref['logits'][0, 0]))
    r = compare10.compare(outside, ref, SHAPES, 'fp32')
    assert r['verdict'] == 'FAIL' and r['kinds']['logits']['failures'] == 1
    assert compare10.compare(outside, ref, SHAPES, 'fp64')['verdict'] == 'FAIL'


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_non_finite_fails(bad):
    ref = arrays()
    act = {k: v.copy() for k, v in ref.items()}
    act['hidden'][0, 0, 0] = bad
    assert compare10.compare(act, ref, SHAPES, 'fp64')['verdict'] == 'FAIL'


def test_missing_kind_is_partial_not_pass():
    ref = arrays()
    r = compare10.compare({'logits': ref['logits'], 'hidden': ref['hidden']}, ref, SHAPES, 'fp64')
    assert (r['coverage'], r['verdict']) == ('PARTIAL', 'NOT PROVEN')
    assert r['kinds']['kv']['state'] == 'missing'


def test_declared_subset_is_partial():
    ref = arrays()
    act = dict(ref, logits=ref['logits'][-1:])
    r = compare10.compare(act, ref, SHAPES, 'fp64', selections={'logits': slice(-1, None)})
    assert (r['coverage'], r['verdict']) == ('PARTIAL', 'NOT PROVEN')


def test_undeclared_subset_is_contradiction():
    ref = arrays()
    act = dict(ref, logits=ref['logits'][-1:])
    r = compare10.compare(act, ref, SHAPES, 'fp64')
    assert r['verdict'] == 'FAIL' and r['kinds']['logits']['state'] == 'shape-contradiction'


def test_nothing_is_not_proven_and_incomplete_reference_never_passes():
    ref = arrays()
    assert compare10.compare(None, ref, SHAPES, 'fp64')['verdict'] == 'NOT PROVEN'
    truncated = dict(ref, logits=ref['logits'][:, :4])
    r = compare10.compare(arrays(), truncated, SHAPES, 'fp64')
    assert r['verdict'] != 'PASS' and r['kinds']['logits']['state'] == 'reference-incomplete'


def test_equivalence_rule():
    old = arrays()
    new = {k: v.copy() for k, v in old.items()}
    assert compare10.equivalence(new, old)['passed']
    new['logits'] = old['logits'] + 5e-10
    assert compare10.equivalence(new, old)['passed']
    assert not compare10.equivalence(new, old, require_bitwise_logits=True)['passed']
    new['logits'] = old['logits'] + 2e-9
    assert not compare10.equivalence(new, old)['passed']
    new = {k: v.copy() for k, v in old.items()}
    new['hidden'][0, 0, 0] = np.nextafter(new['hidden'][0, 0, 0], np.inf)
    assert not compare10.equivalence(new, old)['passed']           # residuals must be bit-identical
    del new['kv']
    assert not compare10.equivalence(new, old)['passed']

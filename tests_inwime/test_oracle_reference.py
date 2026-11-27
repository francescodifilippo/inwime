"""Independent numerical regression of the Percorso additivo (#41): equivalence with the oracle.

`reference/additive41_oracle.json` holds every logit, residual and K/V entry that stock
Transformers computed offline (`tools/additive41_oracle.py`, in the separate oracle environment)
on the tiny tied and untied fixtures, under the FP32 oracle and the FP64 reference contracts,
with its provenance and digests. Here, without Transformers or PyTorch, the digests are checked,
the fixtures are regenerated and must have the recorded SHA256, and the additive runtime must
agree with the oracle at the preregistered M10 thresholds on every element: an attestation
distinct from the bit-identity with the Percorso congelato (test_additive.py).
"""
import base64
import hashlib
import json
from array import array
from pathlib import Path

import pytest

import tinyqwen
from inwime import ops as operators
from inwime.additive import AdditiveQwen2Runtime
from inwime.qwen2_tied import qwen2_catalog
from inwime_runtime.math_cpu import ScalarMath
from inwime_runtime.safetensors_source import SafetensorsSource

REFERENCE = json.loads((Path(__file__).resolve().parent/'reference'/'additive41_oracle.json').read_text())
OPS_KERNEL = tinyqwen.ops_kernel_path()
IMPLEMENTATIONS = ['python'] + (['native'] if OPS_KERNEL else [])


def decode(encoded):
    values = array('d')
    values.frombytes(base64.b64decode(encoded['float64_le_base64']))
    return encoded['shape'], list(values)


def test_the_reference_is_the_one_its_provenance_describes():
    assert REFERENCE['schema'] == 'inwime-oracle-reference/1'
    assert REFERENCE['produced_by'] == 'tools/additive41_oracle.py'
    assert not REFERENCE['git_worktree_dirty'] and len(REFERENCE['git_commit']) == 40
    canonical = json.dumps(REFERENCE['values'], sort_keys=True, separators=(',', ':')).encode()
    assert hashlib.sha256(canonical).hexdigest() == REFERENCE['values_sha256']
    assert {c: (v['atol'], v['rtol']) for c, v in REFERENCE['contracts'].items()} == {
        'stock_fp32': (2e-3, 2e-4), 'full64': (2e-5, 2e-6)}      # the M10 thresholds


@pytest.fixture(scope='module')
def fixtures(tmp_path_factory):
    root = tmp_path_factory.mktemp('oracle-fixtures')
    out = {}
    for head, provenance in REFERENCE['fixtures'].items():
        directory = tinyqwen.generator(head).write(root/head)
        files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(directory.iterdir())}
        assert files == provenance['files'], 'the fixture generator no longer gives the oracle\'s fixture'
        out[head] = directory
    return out


class Capture:
    def __init__(self):
        self.logits, self.hidden = {}, {}

    def residual(self, layer, position, values):
        self.hidden[layer, position] = list(values)

    def logit(self, position, values):
        self.logits[position] = list(values)


def additive_values(directory, implementation):
    sequence, chunks = REFERENCE['sequence'], REFERENCE['chunks']
    native = operators.NativeOps(OPS_KERNEL) if implementation == 'native' else None
    catalog = qwen2_catalog(SafetensorsSource(directory))
    with AdditiveQwen2Runtime(catalog, len(sequence), block=3, all_logits=True, integrity='full',
                              math_backend=ScalarMath(), tile_elements=7, read_elements=7,
                              ops={n: native for n in operators.NAMES} if native else None) as r:
        c = r.spec
        capture, start = Capture(), 0
        for size in chunks:
            with r.prefill(sequence[start:start+size], trace=capture) as out:
                kv = [list(v) for pair in out.kv for v in pair]
            start += size
    positions = len(sequence)
    return dict(logits=[x for p in range(positions) for x in capture.logits[p]],
                hidden=[x for layer in range(c.layers) for p in range(positions) for x in capture.hidden[layer, p]],
                kv=[x for rows in kv for x in rows])


def failures(actual, reference, atol, rtol):
    return [i for i, (a, r) in enumerate(zip(actual, reference, strict=True)) if not abs(a-r) <= atol+rtol*abs(r)]


@pytest.mark.parametrize('implementation', IMPLEMENTATIONS)
@pytest.mark.parametrize('head', ['untied', 'tied'])
@pytest.mark.parametrize('contract', ['stock_fp32', 'full64'])
def test_the_additive_path_agrees_with_the_oracle_on_every_element(fixtures, head, contract, implementation):
    actual = additive_values(fixtures[head], implementation)
    limits = REFERENCE['contracts'][contract]
    for kind in ('logits', 'hidden', 'kv'):
        shape, reference = decode(REFERENCE['values'][head][contract][kind])
        assert len(actual[kind]) == len(reference)
        assert failures(actual[kind], reference, limits['atol'], limits['rtol']) == [], (head, contract, kind)


def test_the_comparison_sees_the_wrong_model(fixtures):
    """Sensitivity: the untied run against the tied fixture's FP64 values fails."""
    actual = additive_values(fixtures['untied'], 'python')
    _, reference = decode(REFERENCE['values']['tied']['full64']['logits'])
    assert failures(actual['logits'], reference, 2e-5, 2e-6)

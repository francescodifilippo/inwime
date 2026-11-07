"""Increment 1: numerical correctness, not performance or out-of-core claims.

Revised FP64 policies are defined below before running the revised suite.
They are specific to the fixture, operators and comparison, never FP32 policies.
"""

from dataclasses import replace
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from inwime_reference.checkpoint import (
    CONFIG, SHARDS, decode_bf16_le, generate_checkpoint, load_checkpoint,
)
from inwime_reference.decoder import KVState, ReferenceDecoder, _ffn, _rms, _softmax, route
from reference_oracle import full_prefix


EPS64 = np.finfo(np.float64).eps
# (atol, rtol): small operator budgets; larger budgets for accumulated error
# across projections, nonlinearities and two residual blocks. These are fixed
# acceptance margins for this finite fixture, not formal universal bounds.
TOLERANCES = {
    "primitive": (16 * EPS64, 32 * EPS64),
    "router": (32 * EPS64, 64 * EPS64),
    "partition": (128 * EPS64, 256 * EPS64),
    "oracle_kv": (512 * EPS64, 512 * EPS64),
    "oracle_hidden": (1024 * EPS64, 1024 * EPS64),
    "oracle_logits": (2048 * EPS64, 2048 * EPS64),
    "analytic_decoder": (512 * EPS64, 1024 * EPS64),
}
ERRORS = {"logits": 0.0, "hidden": 0.0, "kv": 0.0, "router": 0.0}


@pytest.fixture(scope="session", autouse=True)
def numerical_error_report():
    yield
    # Diagnostic only, not an additional passing numerical test.
    print("Maximum observed absolute errors:", json.dumps(ERRORS, sort_keys=True))


def close(actual, expected, kind="logits", policy=None):
    a, b = np.asarray(actual), np.asarray(expected)
    ERRORS[kind] = max(ERRORS.get(kind, 0.), float(np.max(np.abs(a - b))))
    policy = policy or ("router" if kind == "router" else "oracle_" + kind)
    tolerance, relative = TOLERANCES[policy]
    np.testing.assert_allclose(a, b, atol=tolerance, rtol=relative)


@pytest.fixture(params=["dense", "moe"])
def checkpoint(request, tmp_path):
    directory = tmp_path / request.param
    generate_checkpoint(directory, request.param)
    return load_checkpoint(directory)


def compare(result, expected, offset=0):
    close(result.logits, expected["logits"][offset:])
    for index, layer in enumerate(result.state.layers):
        close(layer.keys, expected["kv"][index][0], "kv")
        close(layer.values, expected["kv"][index][1], "kv")
        close(result.hidden[index], expected["hidden"][index][offset:], "hidden")
    for index, routing in enumerate(result.routing):
        np.testing.assert_array_equal(routing.indices, expected["routing"][index][0][offset:])
        close(routing.probabilities, expected["routing"][index][1][offset:], "router")


def test_prefill_and_each_decode_against_independent_scalar_oracle(checkpoint):
    model = ReferenceDecoder(checkpoint)
    tokens = [1, 7, 4, 7, 2, 11]
    result = model.prefill(tokens[:3])
    compare(result, full_prefix(tokens[:3], CONFIG, checkpoint.weights, checkpoint.variant))
    for count in range(4, len(tokens) + 1):
        previous = result.state
        snapshots = [(kv.keys.copy(), kv.values.copy()) for kv in previous.layers]
        result = model.decode(tokens[count - 1], previous)
        assert result.state.length == count
        expected = full_prefix(tokens[:count], CONFIG, checkpoint.weights, checkpoint.variant)
        compare(result, expected, offset=count - 1)
        for old, snap, new in zip(previous.layers, snapshots, result.state.layers):
            np.testing.assert_array_equal(old.keys, snap[0])
            np.testing.assert_array_equal(old.values, snap[1])
            np.testing.assert_array_equal(new.keys[:, :-1], old.keys)
            np.testing.assert_array_equal(new.values[:, :-1], old.values)
            assert not np.shares_memory(new.keys, old.keys)
            assert not np.shares_memory(new.values, old.values)
    full = model.prefill(tokens)
    close(result.logits, full.logits[-1:], policy="partition")
    for incremental, batch in zip(result.state.layers, full.state.layers):
        close(incremental.keys, batch.keys, "kv", policy="partition")
        close(incremental.values, batch.values, "kv", policy="partition")


def test_chunked_prefill_and_all_single_token_steps(checkpoint):
    model = ReferenceDecoder(checkpoint)
    tokens = [3, 2, 9, 5, 6]
    full = model.prefill(tokens)
    first = model.prefill(tokens[:2])
    rest = model.prefill(tokens[2:], first.state)
    close(rest.logits, full.logits[2:], policy="partition")
    result = model.prefill(tokens[:1])
    rows = [result.logits]
    for token in tokens[1:]:
        result = model.decode(token, result.state)
        rows.append(result.logits)
    close(np.concatenate(rows), full.logits, policy="partition")
    for incremental, batch in zip(rest.state.layers, full.state.layers):
        close(incremental.keys, batch.keys, "kv", policy="partition")
        close(incremental.values, batch.values, "kv", policy="partition")


def test_future_tokens_do_not_change_causal_prefix(checkpoint):
    model = ReferenceDecoder(checkpoint)
    first = model.prefill([1, 2, 3, 4, 5])
    changed = model.prefill([1, 2, 3, 20, 21])
    close(first.logits[:3], changed.logits[:3], policy="partition")
    for a, b in zip(first.hidden, changed.hidden):
        close(a[:3], b[:3], "hidden", policy="partition")
    for a, b in zip(first.state.layers, changed.state.layers):
        close(a.keys[:, :3], b.keys[:, :3], "kv", policy="partition")
        close(a.values[:, :3], b.values[:, :3], "kv", policy="partition")


def test_router_ties_selection_and_selected_softmax():
    indices, probabilities = route([[2., -1., 2.], [0., 0., 0.], [-1000., 1000., 999.]], 2)
    np.testing.assert_array_equal(indices, [[0, 2], [0, 1], [1, 2]])
    close(probabilities[:2], [[.5, .5], [.5, .5]], "router")
    p = 1 / (1 + np.exp(-1.))
    close(probabilities[2], [p, 1 - p], "router")


def test_all_finite_bf16_codes_reconstruct_exactly():
    words = np.arange(65536, dtype=np.uint16)
    words = words[(words & 0x7F80) != 0x7F80]
    assert len(words) == 65280
    decoded = decode_bf16_le(words.astype("<u2").tobytes())
    # Independent interpretation of sign/exponent/fraction, without bit-view
    # expansion into FP32 or importing the decoder's implementation.
    expected = []
    for word in words.tolist():
        exponent, fraction = (word >> 7) & 255, word & 127
        magnitude = (math.ldexp(fraction, -133) if exponent == 0
                     else math.ldexp(128 + fraction, exponent - 134))
        expected.append(-magnitude if word & 0x8000 else magnitude)
    np.testing.assert_array_equal(decoded, expected)
    reconstructed = decoded.astype(np.float32).view(np.uint32) >> 16
    np.testing.assert_array_equal(reconstructed, words)
    # Includes the exact little-endian byte pattern, negative zero and subnormal.
    special = decode_bf16_le(bytes.fromhex("0000 0080 0100 803f 20c0"))
    np.testing.assert_array_equal(special, [0., -0., math.ldexp(1., -133), 1., -2.5])
    np.testing.assert_array_equal(np.signbit(special), [False, True, False, False, True])


@pytest.mark.parametrize("raw", [b"\x00", b"\x80\x7f", b"\x80\xff", b"\xc0\x7f"])
def test_bf16_rejects_partial_words_and_nonfinite_values(raw):
    with pytest.raises(ValueError):
        decode_bf16_le(raw)


def test_analytic_operator_witnesses():
    value = np.array([[2., -2., 2., -2.]])
    expected = value / math.sqrt(4. + CONFIG.rms_epsilon)
    close(_rms(value, np.ones(4), CONFIG.rms_epsilon), expected,
          kind="rms", policy="primitive")
    scores = np.array([[math.log(2.), 0., math.log(3.)]])
    close(_softmax(scores), [[1/3, 1/6, 1/2]], kind="softmax", policy="primitive")
    close(_softmax(scores + 7.), [[1/3, 1/6, 1/2]], kind="softmax", policy="primitive")
    weights = {"gate": np.array([[1., 0.], [0., 0.]]),
               "up": np.array([[3., 0.], [0., 0.]]),
               "down": np.array([[5., 0.], [0., 0.]])}
    close(_ffn(np.array([2., 0.]), weights, ""), [60/(1 + math.exp(-2.)), 0.],
          kind="ffn", policy="primitive")


def test_analytic_uniform_causal_attention(checkpoint):
    # A closed-form witness: Q=K=0 gives a uniform mean over each causal
    # prefix. V=O=identity in block 0; block 1 and both FFNs are identity
    # residuals. Expected values contain no attention loop or mask algorithm.
    weights = {name: np.zeros_like(value) for name, value in checkpoint.weights.items()}
    for name in weights:
        if "norm" in name:
            weights[name].fill(1.)
    scalars = np.array([1., 3., -1.])
    for token, value in enumerate(scalars):
        weights["embedding"][token].fill(value)
    weights["layers.0.v"] = np.eye(CONFIG.width)
    weights["layers.0.o"] = np.eye(CONFIG.width)
    weights["lm_head"][:CONFIG.width] = np.eye(CONFIG.width)
    model = ReferenceDecoder(replace(checkpoint, weights=weights))
    u = [a / math.sqrt(a*a + CONFIG.rms_epsilon) for a in scalars]
    means = [u[0], (u[0] + u[1])/2, (u[0] + u[1] + u[2])/3]
    residual = scalars + means
    expected_hidden = np.repeat(residual[:, None], CONFIG.width, axis=1)
    expected_logits = np.zeros((3, CONFIG.vocab))
    for i, value in enumerate(residual):
        expected_logits[i, :CONFIG.width] = value / math.sqrt(value*value + CONFIG.rms_epsilon)
    result = model.prefill([0, 1, 2])
    close(result.logits, expected_logits, policy="analytic_decoder")
    for hidden in result.hidden:
        close(hidden, expected_hidden, "hidden", policy="analytic_decoder")
    expected_v = np.broadcast_to(np.array(u)[None, :, None], result.state.layers[0].values.shape)
    close(result.state.layers[0].values, expected_v, "kv", policy="analytic_decoder")
    np.testing.assert_array_equal(result.state.layers[0].keys, 0.)
    state = model.prefill([0]).state
    for token in (1, 2):
        step = model.decode(token, state)
        close(step.logits, expected_logits[token:token+1], policy="analytic_decoder")
        state = step.state


def test_unselected_expert_does_not_contribute(tmp_path):
    generate_checkpoint(tmp_path / "moe", "moe")
    checkpoint = load_checkpoint(tmp_path / "moe")
    baseline = ReferenceDecoder(checkpoint).prefill([5])
    changed = dict(checkpoint.weights)
    for layer, routing in enumerate(baseline.routing):
        unused, = set(range(CONFIG.experts)) - set(routing.indices[0].tolist())
        for name in ("gate", "up", "down"):
            key = f"layers.{layer}.experts.{unused}.{name}"
            changed[key] = np.full_like(changed[key], 3.)
    result = ReferenceDecoder(replace(checkpoint, weights=changed)).prefill([5])
    np.testing.assert_array_equal(result.logits, baseline.logits)
    for a, b in zip(result.routing, baseline.routing):
        np.testing.assert_array_equal(a.indices, b.indices)


def test_invalid_inputs_state_and_context_leave_old_state_unchanged(checkpoint):
    model = ReferenceDecoder(checkpoint)
    state = model.prefill([2, 3]).state
    snapshot = state.layers[0].keys.copy()
    for tokens in ([], [-1], [CONFIG.vocab], [1.0], [[1]], [True]):
        with pytest.raises(ValueError):
            model.prefill(tokens, state)
    for token in (1.0, True, [1]):
        with pytest.raises(ValueError):
            model.decode(token, state)
    with pytest.raises(ValueError, match="incompatible"):
        model.decode(1, replace(state, checkpoint_identity="other"))
    with pytest.raises(ValueError, match="shape"):
        bad = replace(state.layers[0], keys=state.layers[0].keys[:, :1])
        model.decode(1, replace(state, layers=(bad, state.layers[1])))
    with pytest.raises(ValueError, match="incompatible"):
        model.decode(1, KVState(checkpoint.identity, 2, ()))
    full = model.prefill([1] * CONFIG.context)
    with pytest.raises(ValueError, match="context"):
        model.decode(1, full.state)
    np.testing.assert_array_equal(state.layers[0].keys, snapshot)
    assert not state.layers[0].keys.flags.writeable
    assert not checkpoint.weights["embedding"].flags.writeable


@pytest.mark.parametrize("variant", ["dense", "moe"])
def test_checkpoint_reproducibility_native_bf16_and_no_overwrite(tmp_path, variant):
    a, b = tmp_path / "a", tmp_path / "b"
    manifest = generate_checkpoint(a, variant)
    generate_checkpoint(b, variant)
    assert set(p.name for p in a.iterdir()) == {*SHARDS, "manifest.json"}
    for name in (*SHARDS, "manifest.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes()
        stored = Path(__file__).resolve().parents[1] / "fixtures" / "reference_decoder" / variant
        assert (a / name).read_bytes() == (stored / name).read_bytes()
    with pytest.raises(FileExistsError):
        generate_checkpoint(a, variant)
    loaded = load_checkpoint(a)
    assert loaded.payload_bytes < 64 * 1024
    generate_checkpoint(tmp_path / "different-seed", variant, seed=17)
    other = load_checkpoint(tmp_path / "different-seed")
    assert other.identity != loaded.identity
    assert not np.array_equal(other.weights["embedding"], loaded.weights["embedding"])
    with pytest.raises(ValueError, match="incompatible"):
        ReferenceDecoder(other).decode(1, ReferenceDecoder(loaded).prefill([1]).state)
    for name, descriptor in manifest["tensors"].items():
        raw = (a / descriptor["shard"]).read_bytes()
        bits, = struct.unpack_from("<H", raw, descriptor["offset"])
        # Independent scalar decode of the first BF16 word in each tensor.
        value, = struct.unpack("<f", struct.pack("<I", bits << 16))
        assert loaded.weights[name].flat[0] == value
        assert descriptor["dtype"] == "BF16"
        if "norm" in name:
            assert value == 1.0


@pytest.mark.parametrize("damage", ["corruption", "truncation", "descriptor"])
def test_checkpoint_rejects_damage(tmp_path, damage):
    directory = tmp_path / "checkpoint"
    generate_checkpoint(directory, "dense")
    path = directory / SHARDS[0]
    if damage == "corruption":
        raw = bytearray(path.read_bytes())
        raw[0] ^= 1
        path.write_bytes(raw)
    elif damage == "truncation":
        path.write_bytes(path.read_bytes()[:-2])
    else:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["tensors"]["embedding"]["offset"] = 2
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_checkpoint(directory)


def test_cpu_reference_does_not_import_accelerator_or_runtime():
    root = Path(__file__).resolve().parents[1]
    code = """
import sys
from inwime_reference.decoder import ReferenceDecoder
assert not any(n.split('.')[0] in {'torch', 'cupy', 'cuda', 'inwime_runtime'} for n in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], cwd=root, check=True)

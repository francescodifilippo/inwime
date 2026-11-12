"""Increment 2: independent numerical comparisons and resource/failure tests.

FP64 policies are fixed in tools.verify_increment2 before the first comparison.
No tolerances are inherited from FP32/BF16 execution (not implemented here).
"""

import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import pytest

from inwime_runtime import CPURuntime, Budget, BudgetError, CheckpointError, ReadError
from inwime_runtime.core import FileBackend
from inwime_reference.checkpoint import load_checkpoint
from inwime_reference.decoder import ReferenceDecoder
from tools.verify_increment2 import TOKENS, compare, runtime_export


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=['dense', 'moe'])
def checkpoint(request):
    return ROOT / 'fixtures' / 'reference_decoder' / request.param


def export_reference(r):
    return {'length': r.state.length, 'logits': r.logits.tolist(),
            'hidden': [a.tolist() for a in r.hidden],
            'kv': [[a.keys.tolist(), a.values.tolist()] for a in r.state.layers],
            'routing': [[a.indices.tolist(), a.probabilities.tolist()] for a in r.routing]}


@pytest.mark.parametrize('tile,read', [(3, 1), (5, 2), (7, 3)])
def test_prefill_decode_matches_reference_under_exact_budget(checkpoint, tile, read):
    plan = CPURuntime.minimum_plan(checkpoint, tile_elements=tile, read_elements=read)
    reference = ReferenceDecoder(load_checkpoint(checkpoint))
    expected = reference.prefill(TOKENS[:3])
    with CPURuntime(checkpoint, tile_elements=tile, read_elements=read, budget=plan.budget()) as runtime:
        with runtime.prefill(TOKENS[:3]) as actual:
            compare([runtime_export(actual)], [export_reference(expected)])
        for token in TOKENS[3:]:
            expected = reference.decode(token, expected.state)
            with runtime.decode(token) as actual:
                compare([runtime_export(actual)], [export_reference(expected)])
        stats = runtime.stats()
        assert stats['reserved_peak_bytes'] == plan.total
        assert stats['required_by_category'] == plan.required
        assert stats['audited_retained_metadata_bytes'] <= plan.required['metadata']
        for category in plan.required:
            if category == 'metadata':
                continue
            observed = sum(len(runtime.control.buffers[n])*runtime.control.buffers[n].itemsize
                           for n, (cat, _, _) in plan.arrays.items() if cat == category)
            assert observed == plan.required[category]
        assert stats['weight_slot_bytes'] == tile*8 < stats['payload_bytes']
        assert stats['peak_region_elements'] <= tile
        assert stats['max_backend_read_bytes'] <= read*2
        assert stats['integrity_read_bytes'] == stats['payload_bytes']
        assert stats['inference_read_bytes'] > stats['payload_bytes']
        assert stats['requests'] == stats['completed'] == stats['released']
        assert not stats['active_region'] and stats['failed'] == 0
    assert not runtime.control.fds and not runtime.control.buffers


class FaultBackend(FileBackend):
    def __init__(self):
        self.mode, self.remaining = None, 0
        self.calls = 0
        self.fds_seen = set()

    def readinto(self, fd, offset, destination):
        self.calls += 1
        self.fds_seen.add(fd)
        if self.mode is not None:
            if self.remaining == 0:
                if self.mode == 'error':
                    raise OSError('injected read failure')
                return super().readinto(fd, offset, destination[:-1])
            self.remaining -= 1
        return super().readinto(fd, offset, destination)


def test_minimum_budget_rejects_insufficient_global_and_category_limits(checkpoint):
    plan = CPURuntime.minimum_plan(checkpoint)
    bad = [Budget(plan.total-1, dict(plan.required))]
    for category in plan.required:
        caps = dict(plan.required)
        caps[category] -= 1
        bad.append(Budget(plan.total, caps))
    for budget in bad:
        backend = FaultBackend()
        with pytest.raises(BudgetError):
            CPURuntime(checkpoint, budget=budget, backend=backend)
        assert backend.calls == 0  # no weight reads, including integrity scan


def test_regions_and_lease_lifecycle(checkpoint):
    reference = load_checkpoint(checkpoint)
    runtime = CPURuntime(checkpoint)
    control = runtime.control
    for name, start, count, error in [('missing', 0, 1, CheckpointError),
                                      ('embedding', -1, 1, CheckpointError),
                                      ('embedding', 496, 1, CheckpointError),
                                      ('embedding', 0, 0, CheckpointError),
                                      ('embedding', 0, 6, BudgetError)]:
        calls = control.read_calls
        with pytest.raises(error):
            control.acquire(name, start, count)
        assert control.read_calls == calls and not control.active
    lease = control.acquire('layers.0.q', 15, 3)  # crosses a matrix row boundary
    assert control.request_state == 'READY'
    np.testing.assert_array_equal([lease.value(i) for i in range(3)],
                                  reference.weights['layers.0.q'].reshape(-1)[15:18])
    assert control.request_state == 'IN_USE'
    with pytest.raises(RuntimeError):
        control.acquire('embedding', 0, 1)
    with pytest.raises(RuntimeError):
        runtime.prefill([1])
    with pytest.raises(RuntimeError):
        runtime.close()
    lease.release()
    lease.release()  # idempotent consumer release
    assert control.request_state == 'RELEASED'
    with pytest.raises(RuntimeError):
        lease.value(0)
    with control.acquire('embedding', 0, 1) as new:
        with pytest.raises(RuntimeError):
            lease.value(0)
        assert new.value(0) == reference.weights['embedding'][0, 0]
    runtime.close()


def test_result_lease_and_input_errors_do_not_advance_state(checkpoint):
    with CPURuntime(checkpoint) as runtime:
        initial = runtime.state_digest()
        for invalid in ([], [-1], [31], [True], [1.0], list(range(7))):
            with pytest.raises(ValueError):
                runtime.prefill(invalid)
            assert runtime.state_digest() == initial
        with pytest.raises(ValueError):
            runtime.decode(1)
        result = runtime.prefill([1])
        assert result.logits.readonly
        with pytest.raises(RuntimeError):
            runtime.decode(2)
        result.release()
        with pytest.raises(RuntimeError):
            _ = result.logits
        with runtime.prefill([2, 3, 4, 5, 6]):
            pass
        digest = runtime.state_digest()
        with pytest.raises(ValueError):
            runtime.decode(2)
        assert runtime.state_digest() == digest


@pytest.mark.parametrize('mode', ['error', 'short'])
def test_read_failure_preserves_published_kv_and_invalidates_session(checkpoint, mode):
    backend = FaultBackend()
    with CPURuntime(checkpoint, backend=backend) as runtime:
        with runtime.prefill([1, 2]):
            pass
        before = runtime.state_digest()
        backend.mode, backend.remaining = mode, 100
        with pytest.raises(OSError):
            runtime.decode(3)
        assert runtime.state_digest() == before
        assert runtime.failed and not runtime.control.active
        assert runtime.control.request_state == 'FAILED'
        assert runtime.control.failed == 1
        with pytest.raises(RuntimeError, match='failed'):
            runtime.decode(3)
    for fd in backend.fds_seen:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_compute_failure_after_complete_candidate_token_is_not_published(checkpoint, monkeypatch):
    with CPURuntime(checkpoint) as runtime:
        with runtime.prefill([1, 2]):
            pass
        before = runtime.state_digest()
        original = runtime._token
        def failure(*args):
            original(*args)
            raise ArithmeticError('injected failure after a full candidate token')
        monkeypatch.setattr(runtime, '_token', failure)
        with pytest.raises(ArithmeticError):
            runtime.prefill([3, 4])
        assert runtime.state_digest() == before
        assert runtime.failed and not runtime.control.active and not runtime.result_open


@pytest.mark.parametrize('damage', ['corruption', 'truncation', 'overlap'])
def test_altered_checkpoint_rejected_without_descriptor_leaks(checkpoint, tmp_path, damage):
    copy = tmp_path / 'checkpoint'
    shutil.copytree(checkpoint, copy)
    manifest = json.loads((copy/'manifest.json').read_text())
    shard = copy / next(iter(manifest['files']))
    if damage == 'corruption':
        raw = bytearray(shard.read_bytes())
        raw[0] ^= 1
        shard.write_bytes(raw)
    elif damage == 'truncation':
        shard.write_bytes(shard.read_bytes()[:-2])
    else:
        manifest['tensors']['embedding']['offset'] = 2
        (copy/'manifest.json').write_text(json.dumps(manifest))
    backend = FaultBackend()
    with pytest.raises(CheckpointError):
        CPURuntime(copy, backend=backend)
    for fd in backend.fds_seen:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_source_change_after_opening_rejected_before_publication(checkpoint, tmp_path):
    copy = tmp_path / 'checkpoint'
    shutil.copytree(checkpoint, copy)
    with CPURuntime(copy) as runtime:
        with runtime.prefill([1]):
            pass
        before = runtime.state_digest()
        shard = copy / runtime.catalog.tensors['embedding'].shard
        with shard.open('r+b') as f:
            f.seek(0)
            f.write(b'\x00\x00')
        with pytest.raises(CheckpointError, match='changed'):
            runtime.decode(2)
        assert runtime.state_digest() == before
        assert runtime.failed


def test_moe_reads_only_selected_experts_during_inference():
    path = ROOT/'fixtures/reference_decoder/moe'
    with CPURuntime(path) as runtime:
        with runtime.prefill([5]) as result:
            trace = [(list(ids), list(probs)) for ids, probs in result.routing]
        stats = runtime.stats()
        for layer, (selected, _) in enumerate(trace):
            assert len(set(selected)) == runtime.spec.top_k
            for expert in range(runtime.spec.experts):
                for name in ('gate', 'up', 'down'):
                    read = stats['tensor_read_bytes'][f'layers.{layer}.experts.{expert}.{name}']
                    assert (read > 0) == (expert in selected)
        assert stats['integrity_read_bytes'] == stats['payload_bytes']


def test_runtime_router_ties_have_explicit_low_index_priority():
    path = ROOT/'fixtures/reference_decoder/moe'
    with CPURuntime(path) as runtime:
        # A zero normalized input makes all router logits exactly zero.
        runtime._route('layers.0.')
        assert list(runtime.control.buffers['selected']) == [0, 1]
        assert list(runtime.control.buffers['probabilities']) == [.5, .5]


def test_self_consistent_nonfinite_checkpoint_cannot_publish_state(tmp_path):
    copy = tmp_path/'checkpoint'
    shutil.copytree(ROOT/'fixtures/reference_decoder/dense', copy)
    manifest = json.loads((copy/'manifest.json').read_text())
    descriptor = manifest['tensors']['embedding']
    shard = copy/descriptor['shard']
    raw = bytearray(shard.read_bytes())
    raw[descriptor['offset']:descriptor['offset']+2] = b'\x80\x7f'  # +Inf BF16
    shard.write_bytes(raw)
    manifest['files'][descriptor['shard']]['sha256'] = hashlib.sha256(raw).hexdigest()
    (copy/'manifest.json').write_text(json.dumps(manifest))
    with CPURuntime(copy) as runtime:
        before = runtime.state_digest()
        with pytest.raises(CheckpointError, match='nonfinite'):
            runtime.prefill([0])
        assert runtime.state_digest() == before and runtime.failed


def test_chunked_prefill_matches_full_prefix(checkpoint):
    with CPURuntime(checkpoint) as a, CPURuntime(checkpoint) as b:
        with a.prefill(TOKENS) as result:
            full = runtime_export(result)
        with b.prefill(TOKENS[:2]):
            pass
        with b.prefill(TOKENS[2:]) as result:
            chunk = runtime_export(result)
        full['logits'] = full['logits'][2:]
        full['hidden'] = [v[2:] for v in full['hidden']]
        full['routing'] = [[ids[2:], probs[2:]] for ids, probs in full['routing']]
        compare([chunk], [full])


def test_runtime_imports_no_numerical_reference_or_accelerator():
    script = """
import sys
import inwime_runtime
assert not any(name.split('.')[0] in {'inwime_reference','numpy','torch','cupy','cuda'} for name in sys.modules)
"""
    subprocess.run([sys.executable, '-c', script], cwd=ROOT, check=True)


def test_foreign_thread_is_rejected_without_session_mutation(checkpoint):
    with CPURuntime(checkpoint) as runtime:
        digest = runtime.state_digest()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(runtime.prefill, [1])
            with pytest.raises(RuntimeError, match='owning thread'):
                pending.result()
        assert runtime.state_digest() == digest and not runtime.failed


@pytest.mark.parametrize('stage', ['reading', 'after_commit'])
def test_keyboard_interrupt_does_not_publish_candidate_state(checkpoint, monkeypatch, stage):
    with CPURuntime(checkpoint) as runtime:
        with runtime.prefill([1, 2]):
            pass
        before = runtime.state_digest()
        published = runtime._published
        interrupted = False
        old_trace = sys.gettrace()
        if stage == 'reading':
            original = runtime.control.backend.readinto
            remaining = 100
            def cancel(*args):
                nonlocal remaining, interrupted
                if remaining == 0:
                    interrupted = True
                    raise KeyboardInterrupt('injected cancellation during I/O')
                remaining -= 1
                return original(*args)
            monkeypatch.setattr(runtime.control.backend, 'readinto', cancel)
        else:
            def trace(frame, event, arg):
                nonlocal interrupted
                if (frame.f_code is CPURuntime.prefill.__code__ and event == 'line'
                        and runtime._published is not published and not interrupted):
                    interrupted = True
                    raise KeyboardInterrupt('injected cancellation immediately after commit')
                return trace
            sys.settrace(trace)
        try:
            with pytest.raises(KeyboardInterrupt):
                runtime.decode(3)
        finally:
            sys.settrace(old_trace)
        assert interrupted
        assert runtime.state_digest() == before
        assert runtime.failed and not runtime.running and not runtime.result_open
        assert not runtime.control.active


@pytest.mark.parametrize('backend_method', ['preadv', 'readv'])
def test_tensor_edges_and_shard_switch_do_not_spill(checkpoint, monkeypatch, backend_method):
    if backend_method == 'readv':
        monkeypatch.delattr(os, 'preadv', raising=False)
    reference = load_checkpoint(checkpoint)
    with CPURuntime(checkpoint) as runtime:
        sources_seen = set()
        for name, tensor in runtime.catalog.tensors.items():
            sources_seen.add(tensor.shard)
            for start in (0, tensor.elements-1):
                with runtime.control.acquire(name, start, 1) as lease:
                    assert lease.value(0) == reference.weights[name].reshape(-1)[start]
            calls = runtime.control.read_calls
            with pytest.raises(CheckpointError, match='outside'):
                runtime.control.acquire(name, tensor.elements-1, 2)
            assert runtime.control.read_calls == calls
        assert len(sources_seen) == 2
        assert runtime.control.integrity_bytes == runtime.catalog.payload_bytes


@pytest.mark.parametrize('kv_parts', [0, 1])
def test_numerical_comparison_rejects_missing_key_or_value(kv_parts):
    reference = {'length': 1, 'logits': [[0.]], 'hidden': [[[0.]]],
                 'kv': [[[[0.]], [[1.]]]], 'routing': []}
    incomplete = deepcopy(reference)
    incomplete['kv'][0] = incomplete['kv'][0][:kv_parts]
    with pytest.raises(AssertionError):
        compare([incomplete], [reference])

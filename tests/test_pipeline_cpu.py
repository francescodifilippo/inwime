"""Increment 3: public runtime, region leases and injected file I/O boundaries."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import os
import threading
import time

import pytest

from inwime_runtime import (CPURuntime, Budget, BudgetError, BackpressureError,
                            CancelledError, ReadError)
from inwime_runtime.core import FileBackend
from tools.verify_increment2 import TOKENS, compare, runtime_export


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('variant', ['dense', 'moe'])
@pytest.mark.parametrize('slots', [1, 2, 3])
def test_pipeline_matches_sync_with_exact_budget(variant, slots):
    path = ROOT / 'fixtures/reference_decoder' / variant
    expected = []
    with CPURuntime(path) as sync:
        with sync.prefill(TOKENS[:3]) as result:
            expected.append(runtime_export(result))
        for token in TOKENS[3:]:
            with sync.decode(token) as result:
                expected.append(runtime_export(result))
        sync_bytes = sync.stats()['inference_read_bytes']
    options = dict(io_mode='pipeline', in_flight=slots, io_workers=min(slots, 2))
    plan = CPURuntime.minimum_plan(path, **options)
    with CPURuntime(path, budget=plan.budget(), **options) as engine:
        actual = []
        with engine.prefill(TOKENS[:3]) as result:
            actual.append(runtime_export(result))
        for token in TOKENS[3:]:
            with engine.decode(token) as result:
                actual.append(runtime_export(result))
        assert compare(actual, expected) == dict.fromkeys(('logits', 'hidden', 'kv', 'probabilities'), 0.)
        assert actual == expected  # the scalar reduction order must stay unchanged
        stats = engine.stats()
        assert stats['inference_read_bytes'] == sync_bytes
        assert stats['requests'] == stats['completed'] == stats['released']
        assert stats['occupied_slots'] == 0
        assert stats['peak_occupied_slots'] == slots
        assert stats['reserved_peak_bytes'] == plan.total
        assert stats['audited_retained_metadata_bytes'] <= plan.required['metadata'] + plan.required['requests']
        for category in plan.required:
            if category in ('metadata', 'requests'):
                continue
            assert sum(len(engine.control.buffers[n])*engine.control.buffers[n].itemsize
                       for n, (cat, _, _) in plan.arrays.items() if cat == category) == plan.required[category]


class ControlledBackend(FileBackend):
    """Delay/failure injection happens at the actual file boundary, after startup."""
    def __init__(self):
        self.enabled = False
        self.first_offset = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.later_done = threading.Event()
        self.failure = None
        self.seen = []
        self.lock = threading.Lock()

    def readinto(self, fd, offset, destination):
        if self.enabled:
            with self.lock:
                self.seen.append((fd, offset, len(destination)))
            if offset == self.first_offset:
                self.entered.set()
                if not self.release.wait(3):
                    raise TimeoutError('test barrier was not released')
                if self.failure == 'error':
                    raise OSError('injected read failure')
                if self.failure == 'short':
                    return super().readinto(fd, offset, destination[:-1])
                if self.failure == 'invalid':
                    return len(destination)+1
                if self.failure == 'interrupt':
                    raise KeyboardInterrupt('injected I/O interruption')
            else:
                n = super().readinto(fd, offset, destination)
                self.later_done.set()
                return n
        return super().readinto(fd, offset, destination)


def make_pipeline(backend=None, **options):
    return CPURuntime(ROOT/'fixtures/reference_decoder/moe', io_mode='pipeline',
                      backend=backend, **options)


def test_out_of_order_completion_backpressure_and_live_leases():
    backend = ControlledBackend()
    with make_pipeline(backend, io_workers=2, trace_capacity=100) as engine:
        c = engine.control
        backend.enabled = True
        backend.first_offset = engine.catalog.tensors['embedding'].offset
        first = c.submit('embedding', 0, 5)
        second = c.submit('embedding', 5, 5)
        assert backend.entered.wait(1) and backend.later_done.wait(1)
        # The backend event marks one syscall, not the complete multi-read job.
        assert second.job.done.wait(1)
        c.poll()
        assert first.state == 'READING' and second.state == 'READY'
        assert c.stats()['audited_retained_metadata_bytes'] <= engine.plan.required['metadata'] + engine.plan.required['requests']
        with pytest.raises(BackpressureError):
            c.submit('embedding', 10, 5)
        # A later completion can be consumed; the earlier slot remains reserved.
        lease = c.wait(second)
        expected = lease.value(0)
        with pytest.raises(RuntimeError, match='consumer'):
            c.cancel(second)
        with pytest.raises(RuntimeError, match='in use'):
            c.close()
        backend.release.set()
        with c.wait(first) as other:
            assert lease.value(0) == expected
            assert other.value(0) != expected
        # first's arena can be reused without invalidating second's consumer.
        with c.acquire('embedding', 10, 5):
            assert lease.value(0) == expected
        lease.release()
        lease.release()
        with pytest.raises(RuntimeError, match='expired'):
            lease.value(0)
        assert c.stats()['occupied_slots'] == 0


def test_cancellation_of_running_and_queued_reads_keeps_slots_until_physical_completion():
    backend = ControlledBackend()
    with make_pipeline(backend, in_flight=2, io_workers=1) as engine:
        c = engine.control
        backend.first_offset = engine.catalog.tensors['embedding'].offset
        backend.enabled = True
        first = c.submit('embedding', 0, 5)
        assert backend.entered.wait(1)
        second = c.submit('embedding', 5, 5)
        c.cancel(first)
        c.cancel(second)
        c.poll()
        assert c.stats()['occupied_slots'] == 2
        with pytest.raises(BackpressureError):
            c.submit('embedding', 10, 5)
        backend.release.set()
        c.drain()
        assert first.state == second.state == 'CANCELLED'
        assert c.stats()['cancelled'] == 2 and not c.active
        # Only the syscall already running was allowed to write/read bytes.
        assert len(backend.seen) == 1
        backend.enabled = False
        for _ in range(30):
            with c.acquire('embedding', 0, 5):
                pass  # mandatory work progresses after cancelled prefetch


@pytest.mark.parametrize('failure,error', [('error', OSError), ('short', ReadError),
                                         ('invalid', ReadError), ('interrupt', KeyboardInterrupt)])
def test_failed_read_drains_prefetch_and_preserves_published_state(failure, error):
    backend = ControlledBackend()
    with make_pipeline(backend, io_workers=2) as engine:
        with engine.prefill([1, 2]):
            pass
        before = engine.state_digest()
        backend.first_offset = engine.catalog.tensors['embedding'].offset + 3*engine.spec.width*2
        backend.failure = failure
        backend.release.set()
        backend.enabled = True
        with pytest.raises(error):
            engine.decode(3)
        assert engine.state_digest() == before
        assert engine.failed and not engine.running and not engine.result_open
        assert not engine.control.active and engine.stats()['occupied_slots'] == 0
    for fd, _, _ in backend.seen:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize('mode', ['sync', 'pipeline'])
def test_session_cancellation_during_io_is_atomic(mode):
    backend = ControlledBackend()
    event = threading.Event()
    with CPURuntime(ROOT/'fixtures/reference_decoder/dense', backend=backend, io_mode=mode) as engine:
        with engine.prefill([1, 2]):
            pass
        before = engine.state_digest()
        backend.first_offset = engine.catalog.tensors['embedding'].offset + 3*engine.spec.width*2
        backend.enabled = True
        def cancel_when_running():
            assert backend.entered.wait(1)
            event.set()
            backend.release.set()
        with ThreadPoolExecutor(max_workers=1) as pool:
            action = pool.submit(cancel_when_running)
            with pytest.raises(CancelledError):
                engine.decode(3, cancel_event=event)
            action.result(timeout=2)
        assert engine.state_digest() == before
        assert engine.failed and not engine.control.active


def test_stream_close_discards_unneeded_reads_and_preserves_lease_boundaries():
    with make_pipeline(io_workers=2) as engine:
        c = engine.control
        regions = (('embedding', start, 5) for start in range(0, 100, 5))
        with closing(c.stream(regions)) as stream:
            _, _, lease = next(stream)
            assert lease.value(0) != 0
        assert lease.released and not c.active
        assert c.stats()['requests'] == 2  # lazy bounded lookahead, not 20 requests
        with c.acquire('embedding', 0, 1):
            pass


def test_pipeline_moe_does_not_prefetch_unrouted_experts():
    with make_pipeline() as engine:
        with engine.prefill([5]) as output:
            routes = [list(ids) for ids, _ in output.routing]
        stats = engine.stats()
        for layer, ids in enumerate(routes):
            for expert in range(engine.spec.experts):
                for name in ('gate', 'up', 'down'):
                    assert (stats['tensor_read_bytes'][f'layers.{layer}.experts.{expert}.{name}'] > 0) == (expert in ids)


def test_pipeline_budget_rejected_before_any_io():
    path = ROOT/'fixtures/reference_decoder/dense'
    options = dict(io_mode='pipeline', in_flight=3, io_workers=2, trace_capacity=17)
    plan = CPURuntime.minimum_plan(path, **options)
    class NoReads(FileBackend):
        def readinto(self, *args):
            pytest.fail('infeasible budget must be rejected before integrity reads')
    for category in [None, *plan.required]:
        caps = dict(plan.required)
        if category:
            caps[category] -= 1
        with pytest.raises(BudgetError):
            CPURuntime(path, budget=Budget(plan.total-(category is None), caps),
                       backend=NoReads(), **options)


def test_concurrent_seek_fallback_is_numerically_identical(monkeypatch):
    monkeypatch.delattr(os, 'preadv', raising=False)
    test_pipeline_matches_sync_with_exact_budget('moe', 3)


def trace_overlap(events):
    """Independent interval sweep over exported events (not summed durations)."""
    readers = computers = overlap = 0
    previous = events[0][0]
    for stamp, kind, *_ in events:
        assert stamp >= previous
        if readers and computers:
            overlap += stamp-previous
        if kind == 1:
            readers += 1
        elif kind == 2:
            readers -= 1
        elif kind == 3:
            computers += 1
        elif kind == 4:
            computers -= 1
        assert readers >= 0 and computers >= 0
        previous = stamp
    assert readers == computers == 0
    return overlap


@pytest.mark.parametrize('mode', ['sync', 'pipeline'])
def test_measured_overlap_agrees_with_actual_read_and_compute_intervals(mode):
    class SlowRead(FileBackend):
        delay = 0
        def readinto(self, *args):
            time.sleep(self.delay) if self.delay else None
            return super().readinto(*args)
    backend = SlowRead()
    with CPURuntime(ROOT/'fixtures/reference_decoder/dense', io_mode=mode,
                    backend=backend, tile_elements=8, read_elements=8,
                    trace_capacity=500) as engine:
        backend.delay = .004
        regions = (('embedding', i, 8) for i in range(0, 64, 8))
        total = 0.
        with closing(engine.control.stream(regions)) as stream:
            for _, count, lease in stream:
                for j in range(count):
                    total += lease.value(j)
                time.sleep(.004)  # controlled compute occupancy, explicitly synthetic
        stats = engine.stats()
        assert stats['trace_dropped'] == 0
        assert trace_overlap(engine.control.trace()) == stats['overlap_ns']
        assert (stats['overlap_ns'] > 0) == (mode == 'pipeline')
        assert stats['inference_read_bytes'] == 128
        assert stats['read_service_ns'] >= 8*.003*1e9
        assert stats['compute_ns'] >= 8*.003*1e9
        assert stats['occupied_slots'] == 0


def test_diagnostic_trace_is_bounded_without_losing_aggregate_counters():
    with make_pipeline(trace_capacity=3) as engine:
        with engine.prefill([1]):
            pass
        stats = engine.stats()
        assert len(engine.control.trace()) == 3
        assert stats['trace_dropped'] > 0
        assert stats['read_service_ns'] > 0 and stats['compute_ns'] > 0
        assert stats['occupied_slots'] == 0


def test_cancellation_immediately_after_kv_commit_rolls_back(monkeypatch):
    import sys
    with make_pipeline() as engine:
        with engine.prefill([1, 2]):
            pass
        before, published = engine.state_digest(), engine._published
        interrupted = False
        def trace(frame, event, arg):
            nonlocal interrupted
            if (frame.f_code is CPURuntime.prefill.__code__ and event == 'line'
                    and engine._published is not published and not interrupted):
                interrupted = True
                raise KeyboardInterrupt('cancel after candidate commit')
            return trace
        previous_trace = sys.gettrace()
        try:
            sys.settrace(trace)
            with pytest.raises(KeyboardInterrupt):
                engine.decode(3)
        finally:
            sys.settrace(previous_trace)
        assert interrupted and engine.state_digest() == before
        assert engine.failed and not engine.running and not engine.result_open
        assert engine.stats()['occupied_slots'] == 0


def test_compute_error_releases_current_lease_and_prefetch():
    with make_pipeline() as engine:
        c = engine.control
        with pytest.raises(ArithmeticError):
            with closing(c.stream(('embedding', start, 5) for start in range(0, 100, 5))) as stream:
                for _, _, lease in stream:
                    lease.value(0)
                    raise ArithmeticError('injected consumer failure')
        assert not c.active and c.stats()['occupied_slots'] == 0
        assert c.stats()['released'] == 1


@pytest.mark.parametrize('options', [{'in_flight': 0}, {'io_workers': 0}, {'io_mode': 'bad'},
                                   {'in_flight': 1, 'io_workers': 2}, {'trace_capacity': -1}])
def test_invalid_pipeline_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        CPURuntime.minimum_plan(ROOT/'fixtures/reference_decoder/dense', **options)


def test_nonfinite_bf16_is_rejected_before_a_lease_or_kv_can_be_published(tmp_path):
    import hashlib
    import json
    import shutil
    copy = tmp_path/'checkpoint'
    shutil.copytree(ROOT/'fixtures/reference_decoder/dense', copy)
    manifest = json.loads((copy/'manifest.json').read_text())
    entry = manifest['tensors']['embedding']
    shard = copy/entry['shard']
    data = bytearray(shard.read_bytes())
    data[entry['offset']:entry['offset']+2] = b'\x80\x7f'
    shard.write_bytes(data)
    manifest['files'][entry['shard']]['sha256'] = hashlib.sha256(data).hexdigest()
    (copy/'manifest.json').write_text(json.dumps(manifest))
    from inwime_runtime import CheckpointError
    with CPURuntime(copy, io_mode='pipeline') as engine:
        before = engine.state_digest()
        with pytest.raises(CheckpointError, match='nonfinite'):
            engine.prefill([0])
        assert before == engine.state_digest()
        assert engine.failed and not engine.control.active


def test_metadata_and_request_records_do_not_grow_with_the_stream_length():
    with make_pipeline(trace_capacity=10) as engine:
        c = engine.control
        with c.acquire('embedding', 0, 5):
            pass
        initial = c.stats()['audited_retained_metadata_bytes']
        with closing(c.stream(('embedding', i % 80, 5) for i in range(512))) as stream:
            for _, _, lease in stream:
                lease.value(0)
        stats = c.stats()
        assert stats['live_request_records'] == 0
        assert len(c.slots) == 2 and len(c.trace()) == 10
        # Integer counters can gain a few interpreter words; no per-request log.
        assert stats['audited_retained_metadata_bytes'] < initial+1024
        assert stats['audited_retained_metadata_bytes'] <= engine.plan.required['metadata'] + engine.plan.required['requests']


def test_close_releases_the_trace_arena():
    import weakref
    engine = make_pipeline(trace_capacity=8192)
    trace = weakref.ref(engine.control.buffers['trace'])
    assert trace() is not None
    engine.close()
    assert trace() is None
    assert engine.control.timeline.trace is None


def test_live_metadata_audit_does_not_count_staging_payload_twice():
    backend = ControlledBackend()
    with make_pipeline(backend, tile_elements=65536, read_elements=5) as engine:
        backend.first_offset = engine.catalog.tensors['embedding'].offset
        backend.enabled = True
        request = engine.control.submit('embedding', 0, 5)
        try:
            assert backend.entered.wait(1)
            stats = engine.stats()
            assert stats['audited_retained_metadata_bytes'] <= engine.plan.required['metadata'] + engine.plan.required['requests']
            assert stats['required_by_category']['staging'] == 2*65536*2
        finally:
            backend.release.set()
            engine.control.cancel(request)
            engine.control.drain()


def test_worker_start_failure_happens_before_admitting_any_read(monkeypatch):
    calls = []
    class AuditReads(FileBackend):
        def readinto(self, fd, offset, destination):
            calls.append((fd, offset, len(destination)))
            return super().readinto(fd, offset, destination)
    original = threading.Thread.start
    def start(thread):
        if thread.name == 'inwime-read-1':
            raise RuntimeError('injected worker start failure')
        return original(thread)
    monkeypatch.setattr(threading.Thread, 'start', start)
    with pytest.raises(RuntimeError, match='worker start'):
        make_pipeline(AuditReads(), io_workers=2)
    assert not any(t.name.startswith('inwime-read-') for t in threading.enumerate())
    for fd, _, _ in calls:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize('started', [False, True])
def test_dispatch_error_after_enqueue_cannot_orphan_a_read(monkeypatch, started):
    from inwime_runtime.pipeline import ReadWorkers
    original = ReadWorkers.dispatch
    backend = ControlledBackend()
    with make_pipeline(backend, read_elements=5, io_workers=2) as engine:
        c = engine.control
        before = c.stats()['inference_read_bytes']
        backend.enabled = started
        backend.first_offset = engine.catalog.tensors['embedding'].offset
        def dispatch(workers, job):
            if started:
                original(workers, job)
                assert backend.entered.wait(1)
                backend.release.set()
            else:
                # Force failure in notify(), after append but while the queue
                # lock still prevents a worker from accepting the destination.
                def fail(*args):
                    raise RuntimeError('injected enqueue failure')
                with monkeypatch.context() as patch:
                    patch.setattr(workers.condition, 'notify', fail)
                    original(workers, job)
            raise RuntimeError('injected dispatch failure after acceptance')
        with monkeypatch.context() as patch:
            patch.setattr(ReadWorkers, 'dispatch', dispatch)
            with pytest.raises(RuntimeError, match='injected'):
                c.submit('embedding', 0, 5)
        assert not c.active and c.stats()['occupied_slots'] == 0
        assert c.stats()['inference_read_bytes']-before == (10 if started else 0)
        backend.enabled = False
        with c.acquire('embedding', 10, 5) as lease:
            values = [lease.value(i) for i in range(5)]
        with CPURuntime(ROOT/'fixtures/reference_decoder/moe') as baseline:
            with baseline.control.acquire('embedding', 10, 5) as lease:
                assert values == [lease.value(i) for i in range(5)]


def test_dispatch_and_read_error_count_one_failed_request(monkeypatch):
    from inwime_runtime.pipeline import ReadWorkers
    original = ReadWorkers.dispatch
    backend = ControlledBackend()
    with make_pipeline(backend, read_elements=5) as engine:
        backend.enabled = True
        backend.first_offset = engine.catalog.tensors['embedding'].offset
        backend.failure = 'error'
        backend.release.set()
        def dispatch(workers, job):
            original(workers, job)
            assert job.done.wait(1)
            raise RuntimeError('dispatch error after failed read')
        with monkeypatch.context() as patch:
            patch.setattr(ReadWorkers, 'dispatch', dispatch)
            with pytest.raises(RuntimeError):
                engine.control.submit('embedding', 0, 5)
        stats = engine.stats()
        assert stats['requests'] == stats['failed'] == 1
        assert stats['cancelled'] == 1 and stats['occupied_slots'] == 0

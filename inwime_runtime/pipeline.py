"""Bounded read workers; only the owning thread changes request/slot state.

Workers receive one exclusively reserved staging arena and return a completion.
BF16 expansion and scalar computation stay on the owner; no reference imports.
"""

from concurrent.futures import CancelledError
from collections import deque
from dataclasses import dataclass
import time
import threading

from .core import (Coordinator, BudgetError, CheckpointError, ReadError, signature,
                   deep_size, expand_bf16)


class BackpressureError(RuntimeError):
    """All admitted slots are reserved. Release or drain before submitting."""


@dataclass(slots=True, eq=False)
class Request:
    owner: object
    number: int
    slot: int
    name: str
    start: int
    count: int
    state: str = 'READING'
    job: object = None
    error: object = None
    cancelled: bool = False
    stop: object = None
    failure_counted: bool = False


@dataclass(slots=True)
class Completion:
    read_bytes: int = 0
    calls: int = 0
    max_read: int = 0
    error: object = None


@dataclass(slots=True, eq=False)
class ReadJob:
    args: tuple
    done: object
    result: object = None
    started: bool = False


class ReadWorkers:
    """Fixed workers and bounded transport, including failed dispatch recovery.

    Transport acknowledgments are separate from the owner's request states.
    No lazy thread creation can strand an accepted read without a completion.
    """

    def __init__(self, count, capacity, load):
        self.capacity, self.load = capacity, load
        self.condition = threading.Condition()
        self.pending, self.threads = deque(), []
        self.closed = False
        try:
            for i in range(count):
                thread = threading.Thread(target=self._run, name=f'inwime-read-{i}')
                thread.start()
                self.threads.append(thread)
        except BaseException:
            self.close()
            raise

    def dispatch(self, job):
        try:
            with self.condition:
                if self.closed or len(self.pending) >= self.capacity:
                    raise BackpressureError('read transport is closed or full')
                job.done.clear()
                self.pending.append(job)
                self.condition.notify()
        except BaseException:
            # Covers interruptions both before and AFTER enqueue. The worker
            # marks started under this same lock; no accepted job is orphaned.
            with self.condition:
                if job in self.pending:
                    self.pending.remove(job)
                    job.args = ()
                    job.done.set()
                elif not job.started:
                    job.args = ()
                    job.done.set()
            raise

    def _run(self):
        while True:
            with self.condition:
                while not self.pending and not self.closed:
                    self.condition.wait()
                if not self.pending:
                    return
                job = self.pending.popleft()
                job.started = True
            try:
                job.result = self.load(*job.args)
            except BaseException as error:
                job.result = Completion(error=error.with_traceback(None))
            finally:
                job.args = ()  # release all destination references BEFORE ack
                job.done.set()

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        for thread in self.threads:
            thread.join()


class Lease:
    __slots__ = ('owner', 'request', 'released')

    def __init__(self, owner, request):
        self.owner, self.request, self.released = owner, request, False

    def value(self, index):
        self.owner.check_thread()
        r = self.request
        if self.released or self.owner.closed or r.state != 'IN_USE' or not 0 <= index < r.count:
            raise RuntimeError('expired or invalid region reference')
        return self.owner.buffers[self.owner.weight_name(r.slot)][index]

    def release(self):
        self.owner.check_thread()
        if not self.released:
            self.owner.timeline.event('compute_end', self.request.number, self.request.slot)
            self.owner.released += 1
            self.owner._free(self.request, 'RELEASED')
            self.released = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class PipelineCoordinator(Coordinator):
    def __init__(self, *args, **kwargs):
        # Base constructor verifies all budgets BEFORE allocating or reading.
        self.pool = None
        self.slots = []
        self.cancelled = self.backpressure = 0
        super().__init__(*args, **kwargs)
        self.slots = [None]*self.plan.io_slots
        try:
            self.pool = ReadWorkers(self.plan.io_workers, self.plan.io_slots, self._load)
        except BaseException:
            self.close()
            raise
        self.weight_peak = self.plan.required['weights']

    @staticmethod
    def weight_name(slot):
        return 'weights' + (str(slot) if slot else '')

    @staticmethod
    def staging_name(slot):
        return 'staging' + (str(slot) if slot else '')

    def _validate(self, r):
        if (not isinstance(r, Request) or r.owner is not self or self.closed
                or r.slot >= len(self.slots) or self.slots[r.slot] is not r):
            raise RuntimeError('expired or foreign request')

    def submit(self, name, start, count):
        self.check_thread()
        self.check_cancelled()
        if self.closed:
            raise RuntimeError('closed coordinator')
        tensor, offset = self.catalog.region(name, start, count)
        if count > self.plan.tile_elements:
            raise BudgetError('region exceeds the reserved weight slot')
        try:
            slot = self.slots.index(None)
        except ValueError:
            self.backpressure += 1
            raise BackpressureError('all weight/staging/request slots are reserved') from None
        self.generation += 1
        r = Request(self, self.generation, slot, name, start, count)
        r.stop = threading.Event()
        r.job = ReadJob((self.fds[tensor.shard], self.stamps[tensor.shard], offset, count,
                         self.buffers[self.staging_name(slot)], r.number, slot, r.stop),
                        threading.Event())
        r.job.done.set()  # not yet dispatched: no physical user of the destination
        self.slots[slot] = r
        self.active = True
        self.issued += 1
        try:
            self.timeline.event('reserve', r.number, slot)
            self.pool.dispatch(r.job)
        except BaseException:
            self.cancel(r)
            while self.slots[r.slot] is r:
                r.job.done.wait(.02)
                self.poll()
            self._fail(r)
            raise
        return r

    def _load(self, fd, stamp, offset, count, staging, number, slot, stop):
        """Physical read only. No request, counters, KV or slot transitions."""
        result = Completion()
        try:
            if signature(fd) != stamp:
                raise CheckpointError('source changed after opening')
            for done in range(0, count, self.plan.read_elements):
                if stop.is_set():
                    break
                size = min(self.plan.read_elements, count-done)*2
                view = memoryview(staging)[done*2:done*2+size]
                result.calls += 1
                result.max_read = max(result.max_read, size)
                self.timeline.event('read_begin', number, slot)
                try:
                    n = self.backend.readinto(fd, offset+done*2, view)
                finally:
                    self.timeline.event('read_end', number, slot)
                if type(n) is not int or not 0 <= n <= size:
                    raise ReadError('invalid backend completion')
                result.read_bytes += n
                if n != size:
                    raise ReadError('short read: region is incomplete')
            if signature(fd) != stamp:
                raise CheckpointError('source changed during reading')
        except BaseException as error:
            # Drop worker traceback: it can retain views and arbitrary backend
            # locals. The completion itself remains bounded by the slot count.
            result.error = error.with_traceback(None)
        return result

    def poll(self):
        self.check_thread()
        for r in self.slots:
            if r is None or r.state != 'READING' or not r.job.done.is_set():
                continue
            c = r.job.result or Completion()
            r.job = None  # discard work/completion references before slot reuse
            self.read_calls += c.calls
            self.inference_bytes += c.read_bytes
            self.tensor_bytes[r.name] += c.read_bytes
            self.max_read = max(self.max_read, c.max_read)
            r.error = c.error
            if c.error is not None:
                self._fail(r)
            if r.cancelled:
                self._free(r, 'CANCELLED')
            elif c.error is not None:
                r.state = 'FAILED'
            else:
                r.state = 'READY'
                self.completed += 1
                self.timeline.event('ready', r.number, r.slot)

    def wait(self, r):
        self.check_thread()
        self._validate(r)
        began = time.monotonic_ns()
        self.timeline.event('wait_begin', r.number, r.slot)
        try:
            while r.state == 'READING':
                self.check_cancelled()
                r.job.done.wait(.02)
                self.poll()
            self.check_cancelled()
            if r.cancelled or r.state == 'CANCELLED':
                raise CancelledError('request cancelled')
            if r.state == 'FAILED':
                error = r.error
                r.error = None
                self._free(r, 'FAILED')
                raise error
            if r.state != 'READY':
                raise RuntimeError('request is already in use')
        finally:
            self.timeline.event('wait_end', r.number, r.slot, time.monotonic_ns()-began)
        # Only the owner publishes expanded weights. Partial expansion cannot
        # become a lease if BF16 validation fails.
        try:
            expand_began = time.monotonic_ns()
            raw = self.buffers[self.staging_name(r.slot)]
            weights = self.buffers[self.weight_name(r.slot)]
            expand_bf16(raw, r.count, weights)
        except BaseException:
            self._fail(r)
            self._free(r, 'FAILED')
            raise
        finally:
            self.expand_ns += time.monotonic_ns()-expand_began
        r.state = 'IN_USE'
        self.request_state = 'IN_USE'
        self.region_peak = max(self.region_peak, r.count)
        self.timeline.event('compute_begin', r.number, r.slot)
        return Lease(self, r)

    def acquire(self, name, start, count):
        return self.wait(self.submit(name, start, count))

    def _fail(self, r):
        if not r.failure_counted:
            self.failed += 1
            r.failure_counted = True

    def _free(self, r, state):
        r.state = state
        self.slots[r.slot] = None
        self.timeline.event('free', r.number, r.slot)
        self.active = any(x is not None for x in self.slots)
        self.request_state = state

    def cancel(self, r):
        self.check_thread()
        self._validate(r)
        if r.state == 'IN_USE':
            raise RuntimeError('release the consumer before cancelling')
        if not r.cancelled:
            r.cancelled = True
            self.cancelled += 1
            self.timeline.event('cancel', r.number, r.slot)
        if r.state == 'READING':
            r.stop.set()  # a started syscall still retains its arena until return
        else:
            r.error = None
            self._free(r, 'CANCELLED')

    def drain(self):
        self.check_thread()
        if any(r is not None and r.state == 'IN_USE' for r in self.slots):
            raise RuntimeError('cannot drain while a region is in use')
        for r in self.slots:
            if r is not None:
                self.cancel(r)
        while any(r is not None for r in self.slots):
            for r in self.slots:
                if r is not None and r.job is not None:
                    r.job.done.wait(.02)
            self.poll()

    def abort(self):
        self.drain()

    def stream(self, regions):
        self.check_thread()
        if self.active:
            raise RuntimeError('stream requires exclusive ownership of the region slots')
        source, pending = iter(regions), deque()
        exhausted = False
        try:
            while pending or not exhausted:
                while not exhausted and len(pending) < self.plan.io_slots:
                    try:
                        region = next(source)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.append(self.submit(*region))
                if pending:
                    r = pending.popleft()
                    with self.wait(r) as lease:
                        yield r.start, r.count, lease
        finally:
            # Covers generator.close(), compute errors and cancellation. No
            # queued/running read may outlive its owning operation unnoticed.
            self.drain()

    def stats(self):
        self.poll()
        stats = super().stats()
        live = [r for r in self.slots if r is not None]
        stats['audited_retained_metadata_bytes'] += deep_size((
            self.slots, [(r.job, vars(r.job.done)) for r in live if r.job is not None],
            [vars(r.stop) for r in live],
            [vars(t) for t in self.pool.threads] if self.pool is not None else []),
            seen={id(buffer) for buffer in self.buffers.values()})
        return stats | {'io_mode': 'pipeline', 'in_flight_capacity': self.plan.io_slots,
                        'io_workers': self.plan.io_workers, 'cancelled': self.cancelled,
                        'live_request_records': len(live),
                        'staging_slot_bytes': self.plan.required['staging'],
                        'backpressure_rejections': self.backpressure}

    def close(self):
        self.check_thread()
        self.drain()
        if self.pool is not None:
            self.pool.close()
        super().close()

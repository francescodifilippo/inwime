"""Actual-checkpoint admission and transactional decode fault probes (bounded)."""
import argparse
from concurrent.futures import CancelledError
from dataclasses import replace
import json
from pathlib import Path
import resource
import threading

from inwime_runtime.core import BudgetError, FileBackend, ReadError
from inwime_runtime.llama import LlamaRuntime
from inwime_runtime.math_cpu import NativeMath


def probe(checkpoint, mode, fault):
    event = threading.Event()

    class Backend(FileBackend):
        armed = False
        reads = 0
        total_reads = 0

        def readinto(self, fd, offset, destination):
            self.total_reads += 1
            n = super().readinto(fd, offset, destination)
            if self.armed:
                self.reads += 1
                if self.reads == 100:
                    if fault == 'io':
                        raise OSError('increment8 injected I/O failure')
                    if fault == 'short':
                        return n-1
                    event.set()
            return n

    backend = Backend()
    plan = LlamaRuntime.minimum_plan(checkpoint, capacity=8, io_mode=mode)
    rejected = False
    try:
        LlamaRuntime(checkpoint, capacity=8, io_mode=mode, backend=backend,
                     budget=replace(plan.budget(), total=plan.total-1))
    except BudgetError:
        rejected = True
    assert rejected and backend.total_reads == 0
    with LlamaRuntime(checkpoint, capacity=8, io_mode=mode, backend=backend,
                      math_backend=NativeMath('build/libinwime_cpu.so')) as runtime:
        with runtime.prefill([504, 3575]):
            pass
        before = runtime.state_digest()
        published = runtime._published
        backend.armed = True
        caught = None
        try:
            runtime.decode(282, cancel_event=event)
        except (CancelledError, ReadError, OSError) as error:
            caught = type(error).__name__
        expected = {'io': 'OSError', 'short': 'ReadError', 'cancel': 'CancelledError'}[fault]
        assert caught == expected
        assert runtime.state_digest() == before and runtime._published == published
        assert runtime.failed and not runtime.control.active and not runtime.result_open
        candidate = runtime.control.buffers['kv'+str(1-runtime.active_bank)]
        active = runtime.control.buffers['kv'+str(runtime.active_bank)]
        # Ensure the failure came after actual new KV work, not before the call.
        assert candidate.tobytes() != active.tobytes()
        stats = runtime.stats()
        if mode == 'pipeline':
            assert all(slot is None for slot in runtime.control.slots)
    assert not runtime.control.fds and not runtime.control.buffers
    return dict(mode=mode, fault=fault, caught=caught, budget_rejected_before_payload=True,
                old_kv_all_bytes_preserved=True, descriptor_preserved=True,
                candidate_was_modified=True, drained=True, released=True, stats=stats)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS, (256*2**20, 256*2**20))
    rows = [probe(a.checkpoint, mode, fault) for mode in ('sync', 'pipeline')
            for fault in ('io', 'short', 'cancel')]
    report = dict(passed=True, rlimit_as_bytes=resource.getrlimit(resource.RLIMIT_AS),
                  cases=rows, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    with a.output.open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(a.output)

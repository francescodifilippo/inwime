"""Increment-10 fault probes N1-N4 on the real Qwen2.5-14B checkpoint, one I/O mode per process.

Same probe as increment 9 (tools/fault_increment9.probe, unchanged): budget one byte
below the plan refused before any payload read (N4), then a one-token prefill and a
decode faulted inside layer 0 with an injected I/O error (N1), a short read (N2) or a
cancellation (N3). Each probe asserts that the old KV bank and the
(bank, length, generation) descriptor are intact, the candidate bank was modified,
the session is fail-stop, slots are drained and FDs/arenas released.

Differences from tools/fault_increment9.py (which stays unchanged):
  * RLIMIT_AS is set per mode as preregistered (protocol memory_limits.rlimit_as_bytes:
    sync 256 MiB, pipeline 384 MiB), so the two modes run in separate processes;
  * the fault position is derived from the geometry instead of the fixed read 100.
    M9 placed the fault "inside layer 0, after real new KV work": on the 7B, read 100
    falls in o_proj of layer 0 (embedding 1 + norm 1 + q 49 + bias 1 + k 7 + bias 1 +
    v 7 + bias 1 = 68 reads before o). On the 14B, q_proj alone is 100 reads of
    262,144 elements, so read 100 precedes any K/V write and the probe's own check
    "candidate bank was modified" fails (F-sync attempt 1, 2026-11-19, kept). Here the
    fault is placed at the same structural point: midway through layer-0 o_proj
    (7B: read 92; 14B: read 195). The check that the candidate bank was modified is
    kept unchanged, so a misplaced fault is still a failure, never a weaker pass.
Integrity 'metadata' as in M9 (the full SHA is verified by R1-R3 and the final re-hash).
"""
import argparse
import json
from pathlib import Path
import resource
import time

from concurrent.futures import CancelledError
from dataclasses import replace
import math
import threading

from inwime_runtime.core import BudgetError, FileBackend, ReadError
from inwime_runtime.qwen2 import Qwen2Runtime
from inwime_runtime.math_cpu import NativeMath

RLIMIT = {'sync': 268435456, 'pipeline': 402653184}
FAULTS = ('io', 'short', 'cancel')
LAYER0_BEFORE_O = ('layers.0.norm1', 'layers.0.q', 'layers.0.q.bias', 'layers.0.k', 'layers.0.k.bias',
                   'layers.0.v', 'layers.0.v.bias')


def fault_read(catalog, read_elements):
    """Read index (1-based, counted from the start of the decode) midway through layer-0 o_proj."""
    reads = lambda name: math.ceil(catalog.tensors[name].elements/read_elements)
    before = 1 + sum(reads(n) for n in LAYER0_BEFORE_O)            # 1 = the embedding row
    return before + reads('layers.0.o')//2, before


def probe(checkpoint, mode, fault):
    """tools/fault_increment9.probe with the fault at fault_read() instead of read 100."""
    event = threading.Event()
    plan = Qwen2Runtime.minimum_plan(checkpoint, capacity=8, io_mode=mode)
    state = {}

    class Backend(FileBackend):
        armed = False
        reads = total_reads = 0

        def readinto(self, fd, offset, destination):
            self.total_reads += 1
            n = super().readinto(fd, offset, destination)
            if self.armed:
                self.reads += 1
                if self.reads == state['at']:
                    if fault == 'io':
                        raise OSError('increment10 injected I/O failure')
                    if fault == 'short':
                        return n-1
                    event.set()
            return n

    backend = Backend()
    try:
        Qwen2Runtime(checkpoint, capacity=8, io_mode=mode, backend=backend, integrity='metadata',
                     budget=replace(plan.budget(), total=plan.total-1))
        rejected = False
    except BudgetError:
        rejected = True
    assert rejected and backend.total_reads == 0
    with Qwen2Runtime(checkpoint, capacity=8, io_mode=mode, backend=backend, integrity='metadata',
                      math_backend=NativeMath('build/libinwime_cpu.so')) as runtime:
        state['at'], state['before_o'] = fault_read(runtime.catalog, runtime.plan.read_elements)
        with runtime.prefill([785]):
            pass
        before, published = runtime.state_digest(), runtime._published
        backend.armed = True
        caught = None
        try:
            runtime.decode(6722, cancel_event=event)
        except (CancelledError, ReadError, OSError) as error:
            caught = type(error).__name__
        expected = {'io': 'OSError', 'short': 'ReadError', 'cancel': 'CancelledError'}[fault]
        assert caught == expected, caught
        assert runtime.state_digest() == before and runtime._published == published
        assert runtime.failed and not runtime.control.active and not runtime.result_open
        candidate = runtime.control.buffers['kv'+str(1-runtime.active_bank)]
        active = runtime.control.buffers['kv'+str(runtime.active_bank)]
        assert candidate.tobytes() != active.tobytes()  # fault came after real new KV work (unchanged check)
        try:
            runtime.prefill([13])
            resumed = True
        except RuntimeError:
            resumed = False
        assert not resumed
        if mode == 'pipeline':
            assert all(slot is None for slot in runtime.control.slots)
        stats = runtime.stats()
    assert not runtime.control.fds and not runtime.control.buffers
    return dict(mode=mode, fault=fault, caught=caught, budget_rejected_before_payload=True,
                old_kv_all_bytes_preserved=True, descriptor_preserved=True, candidate_was_modified=True,
                fail_stop_no_resume=True, drained=True, released=True, fault_at_read=state['at'],
                reads_before_layer0_o=state['before_o'], decode_reads_before_fault=backend.reads,
                inference_read_bytes=stats['inference_read_bytes'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--mode', choices=sorted(RLIMIT), required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError(a.output)
    resource.setrlimit(resource.RLIMIT_AS, (RLIMIT[a.mode], RLIMIT[a.mode]))
    rows = []
    for fault in FAULTS:
        began = time.monotonic()
        row = probe(a.checkpoint, a.mode, fault)          # asserts every property; raises on violation
        row['wall_s'] = time.monotonic()-began
        rows.append(row)
        print(json.dumps(dict(mode=a.mode, fault=fault, caught=row['caught'], wall_s=row['wall_s'])), flush=True)
    report = dict(schema='inwime-fault10/1', checkpoint=str(a.checkpoint), mode=a.mode, passed=True,
                  ids={'io': 'N1', 'short': 'N2', 'cancel': 'N3', 'budget': 'N4'},
                  rlimit_as_bytes=list(resource.getrlimit(resource.RLIMIT_AS)), integrity='metadata', cases=rows,
                  peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    with a.output.open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(a.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

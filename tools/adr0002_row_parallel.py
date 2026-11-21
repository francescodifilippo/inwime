"""Evidence for ADR 0002: Python-driven row parallelism over the unchanged pinned C kernel.

Calls `build/libinwime_cpu.so` (`make cpu-kernel`, `src/cpu_kernel.c` untouched) from a Python
thread pool. Each thread owns a disjoint contiguous range of output rows, so the per-row
accumulation order is that of one thread. ctypes releases the GIL during each call.

Reports the per-call overhead of `NativeMath.linear`, and for 1..12 threads the time of BF16
expansion and of applying one expanded tile to B positions (the chunked-prefill pattern),
checking that weights and outputs are bit-identical to the single-thread run.

Synthetic deterministic weights (`inwime_fill`), no checkpoint, no I/O, warm memory.
Usage: python3 -m tools.adr0002_row_parallel [B]
"""
import sys
import time
from array import array
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from inwime_runtime.math_cpu import NativeMath

ROOT = Path(__file__).resolve().parent.parent
ROWS, COLUMNS = 4864, 896          # Qwen2.5-0.5B MLP gate/up projection
THREADS = (1, 2, 4, 6, 8, 12)


def bounds(rows, threads):
    return [rows*k//threads for k in range(threads+1)]


def expand_rows(math, pool, threads, raw, rows, columns, out):
    edges, source, target = bounds(rows, threads), memoryview(raw), memoryview(out)

    def work(k):
        first, last = edges[k], edges[k+1]
        math.expand(source[2*first*columns:2*last*columns], (last-first)*columns, target, first*columns)
    list(pool.map(work, range(threads)))


def linear_rows(math, pool, threads, weights, rows, columns, inputs, outputs):
    edges, view = bounds(rows, threads), memoryview(weights)
    targets = [memoryview(y) for y in outputs]

    def work(k):
        first, last = edges[k], edges[k+1]
        if first == last:
            return
        part = view[first*columns:last*columns]
        for x, y in zip(inputs, targets):
            math.linear(part, (last-first)*columns, x, y, columns, 0, first)
    list(pool.map(work, range(threads)))


def main(positions=32):
    math = NativeMath(ROOT/'build'/'libinwime_cpu.so')
    w8, x8, y8 = array('d', [0.5]*8), array('d', [1.0]*8), array('d', [0.0])
    calls = 200_000
    began = time.perf_counter()
    for _ in range(calls):
        math.linear(w8, 8, x8, y8, 8, 0)
    print('NativeMath.linear overhead: %.2f us/call' % ((time.perf_counter()-began)/calls*1e6))

    raw = bytearray(2*ROWS*COLUMNS)
    math.fill(raw, ROWS*COLUMNS, 0, 7, 120)
    weights = array('d', bytes(8*ROWS*COLUMNS))
    inputs = [array('d', [((k*7919+p*31) % 211-105)/64 for k in range(COLUMNS)])
              for p in range(positions)]
    reference = None
    for threads in THREADS:
        with ThreadPoolExecutor(threads) as pool:
            outputs = [array('d', bytes(8*ROWS)) for _ in range(positions)]
            expand = linear = float('inf')
            for _ in range(3):
                began = time.perf_counter()
                expand_rows(math, pool, threads, raw, ROWS, COLUMNS, weights)
                expand = min(expand, time.perf_counter()-began)
                began = time.perf_counter()
                linear_rows(math, pool, threads, weights, ROWS, COLUMNS, inputs, outputs)
                linear = min(linear, time.perf_counter()-began)
        result = weights.tobytes()+b''.join(y.tobytes() for y in outputs)
        reference = reference or result
        print('threads=%2d  expand %6.1f ms  linear x%d %6.1f ms  bit-identical to 1 thread: %s'
              % (threads, expand*1e3, positions, linear*1e3, result == reference))


if __name__ == '__main__':
    main(*(int(a) for a in sys.argv[1:2]))

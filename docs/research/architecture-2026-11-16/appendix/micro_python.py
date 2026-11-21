"""Scratch micro-measurement of the architecture review (NOT part of the repository).

Per-element cost of the Python loops that the runtime uses for the KV double-bank
copy (decoder.py/llama.py prefill) and for attention (llama.py::_attention),
at Qwen2.5-7B geometry. Synthetic values, no checkpoint, no I/O.
"""
import math
import time
from array import array

n = 458752  # 28 layers * 2 * capacity 16 * kv_width 512
a = array('d', [1.0])*n
c = array('d', [0.0])*n
t = time.perf_counter()
for i in range(len(a)):
    c[i] = a[i]
dt = time.perf_counter()-t
print(f'KV copy python loop: {n} elems {dt*1e3:.1f} ms -> {dt/n*1e9:.1f} ns/elem')


def attention(T, heads=28, kvh=4, dim=128, cap=16):
    kvw = kvh*dim
    q = array('d', [0.01])*(heads*dim)
    cand = array('d', [0.01])*(2*cap*kvw)
    acc = array('d', [0.0])*(dim+1)
    out = array('d', [0.0])*(heads*dim)
    position = T-1
    for head in range(heads):
        kh = head//(heads//kvh)

        def score(token):
            base = token*kvw+kh*dim
            total = 0.
            for d in range(dim):
                total += q[head*dim+d]*cand[base+d]
            return total/math.sqrt(dim)
        maximum = max(score(t) for t in range(position+1))
        den = 0.
        for d in range(dim):
            acc[d] = 0.
        for token in range(position+1):
            p = math.exp(score(token)-maximum)
            den += p
            base = cap*kvw+token*kvw+kh*dim
            for d in range(dim):
                acc[d] += p*cand[base+d]
        for d in range(dim):
            out[head*dim+d] = acc[d]/den


for T in (4, 8, 16):
    t = time.perf_counter()
    attention(T)
    dt = time.perf_counter()-t
    print(f'attention 1 layer T={T}: {dt*1e3:.1f} ms -> per layer-position {dt/T*1e3:.2f} ms; x28 layers = {dt*28:.2f} s/token')
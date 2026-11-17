"""Scratch check of the architecture review (NOT part of the repository).

Compares, bit for bit, the prototype 8-row fused BF16->FP64 kernel (e8lib.c) with the
repository's compiled kernel build/libinwime_cpu.so (inwime_expand + inwime_linear),
calling the latter exactly like the runtime does: flat tiles of 262144 elements that
cut rows, partial sums carried in y[row]. Synthetic random BF16 words (finite, with
zeros, negative zeros and subnormals) and wide-range FP64 activations. No checkpoint.

Build:  gcc -O3 -mavx2 -mfma -fPIC -shared -fno-fast-math -ffp-contract=off -o libe8.so e8lib.c
Run:    python3 compare_with_repo_kernel.py <repo>/build/libinwime_cpu.so ./libe8.so
"""
import ctypes
import hashlib
import os
import random
import sys
from array import array

repo = ctypes.CDLL(sys.argv[1])
e8 = ctypes.CDLL(sys.argv[2])
P, S = ctypes.c_void_p, ctypes.c_size_t
repo.inwime_expand.argtypes = [P, S, P]
repo.inwime_expand.restype = ctypes.c_int
repo.inwime_linear.argtypes = [P, S, P, P, S, S]
repo.inwime_linear.restype = ctypes.c_int
e8.e8_gemv.argtypes = [P, S, S, P, P]
e8.e8_gemv.restype = None
random.seed(20261116)
for rows, cols in ((2048, 3584), (1027, 5120), (515, 18944), (64, 13824+3)):
    n = rows*cols
    words = array('H')
    words.frombytes(os.urandom(n*2))
    for i in range(0, n, 97):
        words[i] = (0x0000, 0x8000, 0x0001, 0x807f)[(i//97) % 4]
    b = bytearray(words.tobytes())
    for i in range(1, len(b), 2):          # clear exponent 255 (non-finite)
        if (b[i] & 0x7f) == 0x7f and (b[i-1] & 0x80):
            b[i-1] &= 0x7f
    w = (ctypes.c_char*len(b)).from_buffer(b)
    x = array('d', [random.uniform(-4, 4)*(10.0**random.randint(-6, 6)) for _ in range(cols)])
    xa = (ctypes.c_double*cols).from_buffer(x)
    arena = (ctypes.c_double*n)()
    assert repo.inwime_expand(ctypes.byref(w), n, ctypes.byref(arena)) == 0
    y_repo = (ctypes.c_double*rows)()
    tile = 262144
    for start in range(0, n, tile):
        count = min(tile, n-start)
        assert repo.inwime_linear(ctypes.byref(arena, start*8), count, ctypes.byref(xa),
                                  ctypes.byref(y_repo), cols, start) == 0
    y_e8 = (ctypes.c_double*rows)()
    e8.e8_gemv(ctypes.byref(w), rows, cols, ctypes.byref(xa), ctypes.byref(y_e8))
    print('rows=%d cols=%d bit-identical: %s sha %s / %s' % (
        rows, cols, bytes(y_repo) == bytes(y_e8),
        hashlib.sha256(bytes(y_repo)).hexdigest()[:16], hashlib.sha256(bytes(y_e8)).hexdigest()[:16]))
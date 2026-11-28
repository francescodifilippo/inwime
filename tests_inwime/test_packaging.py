"""Seam: the installed package's native kernel, compared with the historical `make cpu-kernel` build."""
import random
import struct
from array import array
from pathlib import Path

import pytest

from inwime import kernel
from inwime_runtime.math_cpu import NativeMath

REFERENCE = Path(__file__).resolve().parents[1]/'build/libinwime_cpu.so'


def test_packaged_kernel_is_bit_identical_to_the_historical_build():
    packaged = kernel.locate(env={})
    if packaged is None:
        pytest.skip('no packaged native kernel (source checkout or install without compiler)')
    if not REFERENCE.is_file():
        pytest.skip('historical kernel not built (make cpu-kernel)')
    rng = random.Random(20261120)
    rows, columns = 64, 1531
    count = rows*columns
    raw = bytearray()
    for _ in range(count):
        word = rng.getrandbits(16)
        if (word >> 7) & 255 == 255:
            word &= ~(1 << 7)                     # keep values finite
        raw += struct.pack('<H', word)
    x = array('d', [rng.uniform(-4, 4) for _ in range(columns)])
    outputs = []
    for library in (packaged, REFERENCE):
        math = NativeMath(library)
        weights, y = array('d', bytes(8*count)), array('d', bytes(8*rows))
        math.expand(raw, count, weights)
        try:
            math.linear(weights, count, x, y, columns, 0)
        except ArithmeticError:
            pass                                  # same behaviour required, compared below
        outputs.append((weights.tobytes(), y.tobytes()))
    assert outputs[0] == outputs[1]


def test_disabled_kernel_is_never_located():
    assert kernel.locate(env={'INWIME_CPU_KERNEL': 'off'}) is None


OPS_REFERENCE = Path(__file__).resolve().parents[1]/'build/libinwime_ops.so'


def test_packaged_operators_give_the_bits_of_the_make_build():
    """setuptools prepends Python's CFLAGS: the packaged operators must still equal `make ops-kernel`."""
    from inwime import gate, ops
    packaged = ops.locate(env={})
    if packaged is None:
        pytest.skip('no packaged operator kernel (source checkout or install without compiler)')
    if not OPS_REFERENCE.is_file():
        pytest.skip('operator kernel not built (make ops-kernel)')
    report = gate.check(ops.NativeOps(packaged), reference=ops.NativeOps(OPS_REFERENCE))
    assert all(r['passed'] for r in report.values()), report


def test_disabled_operators_are_never_located():
    from inwime import ops
    assert ops.locate(env={'INWIME_OPS_KERNEL': 'off'}) is None

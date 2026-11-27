"""Seam: the operator gate of the Percorso additivo (#41).

When the Motore loads, `inwime.gate` runs every native operator and its Python reference on the
same deterministic cases, edge values included, and compares status and output bit for bit. An
operator that differs is not declared equivalent: the Motore uses the Python reference for it.
"""
from array import array
import ctypes
import math

import pytest

import tinyqwen
from inwime import gate, ops as operators

OPS_KERNEL = operators.locate({}) or next(
    (p for p in [tinyqwen.TOOLS.parent/'build'/'libinwime_ops.so'] if p.is_file()), None)
native_only = pytest.mark.skipif(OPS_KERNEL is None, reason='operator kernel not built (make ops-kernel)')


class RopeByProduct(operators.PythonOps):
    """A plausible but different RoPE: pos * theta**(-2j/dim) instead of pos / theta**(2j/dim)."""
    name = 'rope-by-product'

    def rope(self, v, vo, vs, positions, first, heads, dim, theta):
        half = dim//2
        for p in range(positions):
            for head in range(heads):
                for j in range(half):
                    angle = (first+p) * theta**(-2*j/dim)
                    a = vo+p*vs+head*dim+j
                    x, y = v[a], v[a+half]
                    v[a], v[a+half] = x*math.cos(angle)-y*math.sin(angle), y*math.cos(angle)+x*math.sin(angle)
        return operators.OK


class Raising(operators.PythonOps):
    name = 'raising'

    def silu_mul(self, gate, up, count):
        raise OverflowError('math range error')


def test_an_operator_that_differs_is_replaced_by_the_reference():
    candidate = RopeByProduct()
    chosen, report = gate.select(candidate)
    assert report['rope']['passed'] is False
    assert chosen['rope'].name == operators.PythonOps.name
    assert all(report[name]['passed'] and chosen[name] is candidate
               for name in operators.NAMES if name != 'rope')


def test_an_operator_that_raises_fails_the_gate():
    chosen, report = gate.select(Raising())
    assert report['silu_mul']['passed'] is False and 'OverflowError' in report['silu_mul']['detail']
    assert chosen['silu_mul'].name == operators.PythonOps.name


def test_the_gate_sees_a_one_ulp_difference():
    class OneUlp(operators.PythonOps):
        def add(self, y, yo, ys, x, xo, xs, positions, count):
            code = super().add(y, yo, ys, x, xo, xs, positions, count)
            y[yo] = math.nextafter(y[yo], math.inf)
            return code
    _, report = gate.select(OneUlp())
    assert report['add']['passed'] is False


def test_without_a_library_every_operator_is_the_reference():
    described = gate.describe({'INWIME_OPS_KERNEL': 'off'})
    assert described['native'] is False and described['status'] == 'warn'
    assert set(described['operators']) == set(operators.NAMES)
    assert {o['implementation'] for o in described['operators'].values()} == {operators.PythonOps.name}


def test_a_library_that_is_not_the_operator_kernel_is_refused(tmp_path):
    bogus = tmp_path/'not-a-library.so'
    bogus.write_bytes(b'\x7fELF garbage')
    described = gate.describe({'INWIME_OPS_KERNEL': str(bogus)})
    assert described['status'] == 'fail' and described['native'] is False
    assert {o['implementation'] for o in described['operators'].values()} == {operators.PythonOps.name}


@native_only
def test_the_native_operators_pass_the_gate_on_this_host():
    described = gate.describe({'INWIME_OPS_KERNEL': str(OPS_KERNEL)})
    assert described['status'] == 'ok', described
    assert all(o['passed'] and o['implementation'] == operators.NativeOps.name
               for o in described['operators'].values())


@pytest.mark.parametrize('implementation', ['python'] + (['native'] if OPS_KERNEL else []))
def test_linear_and_attention_stop_inside_the_loop_when_the_flag_is_set(implementation):
    ops = operators.NativeOps(OPS_KERNEL) if implementation == 'native' else operators.PythonOps()
    cancel = ctypes.c_int(1)
    w, x, y = array('d', [1.]*12), array('d', [2.]*12), array('d', [7.]*6)
    assert ops.linear(w, 12, x, 0, 6, y, 0, 3, 2, 6, 0, cancel) == operators.CANCELLED
    assert list(y) == [7.]*6                       # stopped before the first row
    q, kv, out = array('d', [1.]*8), array('d', [1.]*32), array('d', [9.]*8)
    assert ops.attention(q, 0, 4, 2, 1, kv, 0, 16, 4, 2, 1, 2, out, 0, 4, cancel) == operators.CANCELLED
    assert list(out) == [9.]*8
    cancel.value = 0
    assert ops.linear(w, 12, x, 0, 6, y, 0, 3, 2, 6, 0, cancel) == operators.OK
    assert list(y) == [12., 12., 7., 12., 12., 7.]   # rows 0-1 of each position, stride 3


@native_only
def test_native_arenas_are_checked_before_the_call():
    native, cancel = operators.NativeOps(OPS_KERNEL), ctypes.c_int(0)
    w, x, y = array('d', [1.]*12), array('d', [2.]*6), array('d', [0.]*2)
    with pytest.raises(ValueError):            # the second position's input row is missing
        native.linear(w, 12, x, 0, 6, y, 0, 1, 2, 6, 0, cancel)
    with pytest.raises(ValueError):            # a read-only arena
        native.add(memoryview(y).toreadonly(), 0, 1, x, 0, 1, 1, 1)
    assert native.rope(array('d', [1.]*3), 0, 3, 1, 0, 1, 3, 1e4) == operators.INVALID   # odd dim

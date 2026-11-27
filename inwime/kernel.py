"""Locate the packaged native CPU kernel and check it bit for bit against Python.

The wheel ships ``src/cpu_kernel.c`` (the pinned M10 kernel, unmodified) compiled
as ``inwime/_cpu_kernel*.so`` and loaded through ctypes. If it is missing (no
compiler at install) or disabled, the pure-Python FP64 path stays available.
"""
import importlib.machinery
import os
import struct
from array import array
from pathlib import Path

from inwime_runtime.core import expand_bf16
from inwime_runtime.math_cpu import NativeMath, ScalarMath

ENV = 'INWIME_CPU_KERNEL'  # explicit library path, or 'off' to force the Python fallback
FALLBACK = ScalarMath.name


def find_library(env, variable, stem):
    """`variable` of `env` (a path, or 'off' for none), else the library `stem` packaged here."""
    env = os.environ if env is None else env
    override = env.get(variable)
    if override:
        return None if override == 'off' else Path(override)
    here = Path(__file__).resolve().parent
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = here/(stem+suffix)
        if candidate.is_file():
            return candidate
    return None


def locate(env=None):
    return find_library(env, ENV, '_cpu_kernel')


def _fixture(rows=3, columns=37, seed=0x1F2E3D4C):
    """Deterministic finite BF16 words, both signs, normal and subnormal exponents."""
    raw, state = bytearray(), seed
    for i in range(rows*columns):
        state = (state*6364136223846793005+1442695040888963407) % 2**64
        exponent = 0 if i % 29 == 0 else 100+(state >> 33) % 50
        word = ((state >> 63) << 15) | (exponent << 7) | ((state >> 40) & 127)
        raw += struct.pack('<H', word)
    x = array('d', [((k*7919) % 211-105)/64 for k in range(columns)])
    return raw, x


def self_test(math_backend, rows=3, columns=37):
    """Compare expand and linear (whole and split tiles) with the Python reference, bitwise."""
    count = rows*columns
    raw, x = _fixture(rows, columns)
    reference, candidate = array('d', bytes(8*count)), array('d', bytes(8*count))
    expand_bf16(raw, count, reference)
    math_backend.expand(raw, count, candidate)
    if reference.tobytes() != candidate.tobytes():
        return dict(passed=False, detail='expand differs from the Python reference')
    expected = array('d', bytes(8*rows))
    ScalarMath().linear(reference, count, x, expected, columns, 0)
    whole = array('d', bytes(8*rows))
    math_backend.linear(candidate, count, x, whole, columns, 0)
    split, cut = array('d', bytes(8*rows)), columns+columns//2
    math_backend.linear(candidate[:cut], cut, x, split, columns, 0)
    math_backend.linear(candidate[cut:], count-cut, x, split, columns, cut)
    for name, result in (('linear', whole), ('linear split tile', split)):
        if result.tobytes() != expected.tobytes():
            return dict(passed=False, detail=name+' differs from the Python reference')
    return dict(passed=True, detail=f'self-test passed: expand + linear equal bit for bit to the Python '
                                    f'reference on a {count}-value fixture')


def describe(env=None):
    path = locate(env)
    out = dict(path=str(path) if path else None, implementation=FALLBACK, native=False, self_test=None)
    if path is None:
        out['status'] = 'warn'
        out['detail'] = 'native kernel absent or disabled: Python FP64 fallback (very slow)'
        return out
    try:
        native = NativeMath(path)
        out['self_test'] = result = self_test(native)
    except (OSError, AttributeError) as error:   # not a library, or one without the kernel symbols
        out.update(status='fail', detail='native kernel not loadable: '+type(error).__name__)
        return out
    except (ValueError, ArithmeticError) as error:
        out.update(status='fail', self_test=dict(passed=False, detail=type(error).__name__),
                   detail='native kernel self-test raised '+type(error).__name__)
        return out
    if result['passed']:
        out.update(implementation=NativeMath.name, native=True, status='ok', detail=result['detail'])
    else:
        out.update(status='fail', detail=result['detail'])
    return out

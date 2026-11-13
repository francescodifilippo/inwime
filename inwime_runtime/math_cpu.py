"""Replaceable synchronous CPU math; no ownership of weights, I/O or allocation.

Both implementations expand original BF16 exactly and accumulate in FP64,
in increasing column order. Native code uses no heap and releases Python's GIL.
"""
import ctypes
import math

from .core import CheckpointError, expand_bf16


class ScalarMath:
    name = 'python-fp64'
    expand = staticmethod(expand_bf16)

    def linear(self, weights, count, source, destination, columns, start, offset=0):
        for i in range(count):
            row, column = divmod(start+i, columns)
            target = offset+row
            if column == 0:
                destination[target] = 0.
            destination[target] += weights[i]*source[column]
            if column+1 == columns and not math.isfinite(destination[target]):
                raise ArithmeticError('nonfinite linear output')


class NativeMath:
    name = 'c-fp64-ordered'

    def __init__(self, library):
        self.library = ctypes.CDLL(str(library))
        pointer = ctypes.c_void_p
        size = ctypes.c_size_t
        self.library.inwime_expand.argtypes = [pointer, size, pointer]
        self.library.inwime_expand.restype = ctypes.c_int
        self.library.inwime_linear.argtypes = [pointer, size, pointer, pointer, size, size]
        self.library.inwime_linear.restype = ctypes.c_int
        self.library.inwime_fill.argtypes = [pointer, size, ctypes.c_uint64,
                                            ctypes.c_uint64, ctypes.c_uint32, ctypes.c_int]
        self.library.inwime_fill.restype = None

    @staticmethod
    def _arena(buffer, code, count, offset=0):
        view = memoryview(buffer)
        if (view.readonly or not view.c_contiguous or view.format != code
                or offset < 0 or count < 0 or offset+count > len(view)):
            raise ValueError('kernel requires a bounded writable contiguous arena')
        itemsize = 8 if code == 'd' else 1
        if view.itemsize != itemsize:
            raise ValueError('unexpected native arena item size')
        # Keep the exported buffer alive across CDLL's GIL-releasing call.
        # A bare address would allow another thread to resize the underlying array.
        return ctypes.c_char.from_buffer(buffer, offset*itemsize)

    def expand(self, raw, count, destination, offset=0):
        a = self._arena(raw, 'B', count*2)
        b = self._arena(destination, 'd', count, offset)
        if self.library.inwime_expand(ctypes.byref(a), count, ctypes.byref(b)):
            raise CheckpointError('nonfinite BF16 weight')

    def linear(self, weights, count, source, destination, columns, start, offset=0):
        if type(columns) is not int or columns <= 0 or start < 0 or count <= 0:
            raise ValueError('invalid linear dimensions')
        a = self._arena(weights, 'd', count)
        x = self._arena(source, 'd', columns)
        y = self._arena(destination, 'd', (start+count-1)//columns+1, offset)
        if self.library.inwime_linear(ctypes.byref(a), count, ctypes.byref(x),
                                      ctypes.byref(y), columns, start):
            raise ArithmeticError('nonfinite linear output')

    def fill(self, destination, count, start, seed, exponent, norm=False):
        """Synthetic native BF16 generator, unrelated to original checkpoint conversion."""
        if not (0 <= start < 2**64 and 0 <= seed < 2**64 and 1 <= exponent <= 253):
            raise ValueError('invalid deterministic generator parameters')
        raw = self._arena(destination, 'B', count*2)
        self.library.inwime_fill(ctypes.byref(raw), count, start, seed, exponent, norm)

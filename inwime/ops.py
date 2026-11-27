"""Operators of the Percorso additivo (#41): a Python reference and native C leaf kernels.

Each operator applies one step of the frozen Qwen2 forward (`inwime_runtime.llama` and
`inwime_runtime.decoder`) to a block of positions at once, with the same operations in the same
order on each element, so that every result keeps the bits of the Percorso congelato:

    linear             y[p][row] = sum_j w[row][j]*x[p][j], one FP64 sum from 0.0 in increasing
                       column order, carried in y across tiles; tiles may cut rows (as the frozen
                       streamed GEMV); nonfinite on a finished row is an error
    rms_denominators   d[p] = sqrt(sum_i x[p][i]*x[p][i] / width + eps), the sum in index order
    rms_scale          y[p][i] = x[p][i] / d[p] * w[i]
    add                y[p][i] = y[p][i] + x[p][i] (residuals; a bias has stride 0)
    rope               split-half RoPE at positions first+p: angle = pos / theta**(2j/dim)
    attention          causal GQA attention over the KV rows 0..first+p: scores, maximum (the
                       first maximum wins), exp(score - maximum), sum, weighted values, division;
                       no buffer that grows with the context: each score is computed twice, as
                       the frozen path does
    silu_mul           gate[i] = silu(gate[i]) * up[i], silu split on the sign of z

Both implementations share one calling convention: buffers are flat `array('d')` arenas owned by
the caller, with an element offset and a row stride, and each call returns a status code (below)
instead of raising, so that the operator gate (`gate.py`) compares them result for result.

`linear` and `attention` read a cancellation flag (a `ctypes.c_int`, nonzero = cancelled) at a
declared frequency: before each row segment of a weight tile (at most positions x columns
multiply-adds between two checks) and before each (position, head) pair. The other operators are
linear in the block and are not interrupted: the runtime checks between calls.
"""
import ctypes
import math

from .kernel import find_library

OK, NONFINITE, CANCELLED, INVALID = 0, 1, 2, 3
NAMES = ('linear', 'rms_denominators', 'rms_scale', 'add', 'rope', 'attention', 'silu_mul')
ABI_VERSION = 1            # INWIME_OPS_ABI_VERSION of native/inwime_ops.h
ENV = 'INWIME_OPS_KERNEL'  # explicit library path, or 'off' to force the Python reference


def bad_rope(dim):
    return dim == 0 or dim % 2 == 1


def bad_attention(heads, kv_heads, dim):
    return dim == 0 or kv_heads == 0 or heads % kv_heads != 0


class PythonOps:
    """The reference: the frozen Python loops, rearranged for a block of positions."""
    name = 'python-fp64'

    def linear(self, w, count, x, xo, xs, y, yo, ys, positions, columns, start, cancel):
        if not columns:
            return INVALID
        row, column = divmod(start, columns)
        i = 0
        while i < count:
            if cancel.value:
                return CANCELLED
            number = min(columns-column, count-i)
            for p in range(positions):
                xb, target = xo+p*xs+column, yo+p*ys+row
                total = y[target] if column else 0.
                for j in range(number):
                    total += w[i+j]*x[xb+j]
                y[target] = total
                if column+number == columns and not math.isfinite(total):
                    return NONFINITE
            i += number
            column += number
            if column == columns:
                row, column = row+1, 0
        return OK

    def rms_denominators(self, x, xo, xs, positions, width, eps, out):
        if not width:
            return INVALID
        for p in range(positions):
            squared = 0.
            for i in range(xo+p*xs, xo+p*xs+width):
                squared += x[i]*x[i]
            denominator = math.sqrt(squared/width + eps)
            if not math.isfinite(denominator):
                return NONFINITE
            out[p] = denominator
        return OK

    def rms_scale(self, x, xo, xs, denominators, w, count, start, y, yo, ys, positions):
        for p in range(positions):
            a, b, d = xo+p*xs+start, yo+p*ys+start, denominators[p]
            for i in range(count):
                y[b+i] = x[a+i]/d*w[i]
        return OK

    def add(self, y, yo, ys, x, xo, xs, positions, count):
        for p in range(positions):
            a, b = yo+p*ys, xo+p*xs
            for i in range(count):
                y[a+i] += x[b+i]
        return OK

    def rope(self, v, vo, vs, positions, first, heads, dim, theta):
        if bad_rope(dim):
            return INVALID
        half = dim//2
        for p in range(positions):
            position = first+p
            for head in range(heads):
                for j in range(half):
                    angle = position / theta**(2*j/dim)
                    cosine, sine = math.cos(angle), math.sin(angle)
                    a = vo+p*vs+head*dim+j
                    b = a+half
                    x, y = v[a], v[b]
                    v[a], v[b] = x*cosine-y*sine, y*cosine+x*sine
        return OK

    def attention(self, q, qo, qs, positions, first, kv, ko, vo, kvs, heads, kv_heads, dim,
                  out, oo, out_stride, cancel):
        if bad_attention(heads, kv_heads, dim):
            return INVALID
        group = heads//kv_heads
        for p in range(positions):
            position = first+p
            for head in range(heads):
                if cancel.value:
                    return CANCELLED
                kh, qb, ob = head//group, qo+p*qs+head*dim, oo+p*out_stride+head*dim

                def score(token):
                    base, total = ko+token*kvs+kh*dim, 0.
                    for d in range(dim):
                        total += q[qb+d]*kv[base+d]
                    return total/math.sqrt(dim)
                maximum = max(score(t) for t in range(position+1))
                denominator = 0.
                for d in range(dim):
                    out[ob+d] = 0.
                for token in range(position+1):
                    probability = math.exp(score(token)-maximum)
                    denominator += probability
                    base = vo+token*kvs+kh*dim
                    for d in range(dim):
                        out[ob+d] += probability*kv[base+d]
                for d in range(dim):
                    out[ob+d] = out[ob+d]/denominator
        return OK

    def silu_mul(self, gate, up, count):
        for i in range(count):
            z = gate[i]
            if z >= 0:
                silu = z/(1+math.exp(-z))
            else:
                e = math.exp(z)
                silu = z*e/(1+e)
            gate[i] = silu*up[i]
        return OK


def _rows(positions, stride, length):
    """Elements spanned by `positions` rows of `length` elements, `stride` apart."""
    return 0 if positions == 0 else (positions-1)*stride+length


class NativeOps:
    """The C leaf kernels of `native/inwime_ops.c` through ctypes; ctypes releases the GIL.

    Every pointer is checked against its arena before the call: an FP64 `array`, writable and
    contiguous, long enough for every row the call touches. The kernel checks the dimensions again."""
    name = 'c-fp64-ordered-ops'

    def __init__(self, library):
        self.path = str(library)
        lib = self.library = ctypes.CDLL(self.path)
        size, pointer, flag = ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)
        lib.inwime_ops_abi_version.argtypes, lib.inwime_ops_abi_version.restype = [], ctypes.c_int
        if lib.inwime_ops_abi_version() != ABI_VERSION:
            raise OSError('operator kernel ABI %d, expected %d'
                          % (lib.inwime_ops_abi_version(), ABI_VERSION))
        signatures = {
            'linear': [pointer, size, pointer, size, pointer, size, size, size, size, flag],
            'rms_denominators': [pointer, size, size, size, ctypes.c_double, pointer],
            'rms_scale': [pointer, size, pointer, pointer, size, size, pointer, size, size],
            'add': [pointer, size, pointer, size, size, size],
            'rope': [pointer, size, size, size, size, size, ctypes.c_double],
            'attention': [pointer, size, size, size, pointer, pointer, size, size, size, size,
                          pointer, size, flag],
            'silu_mul': [pointer, pointer, size]}
        self.kernels = {}
        for name, argtypes in signatures.items():
            function = getattr(lib, 'inwime_ops_'+name)
            function.argtypes, function.restype = argtypes, ctypes.c_int
            self.kernels[name] = function

    @staticmethod
    def _arena(buffer, offset, needed):
        view = memoryview(buffer)
        if (view.readonly or not view.c_contiguous or view.format != 'd' or view.itemsize != 8
                or type(offset) is not int or type(needed) is not int or offset < 0 or needed < 0
                or offset+needed > len(view)):
            raise ValueError('operator requires a bounded writable contiguous FP64 arena')
        # The ctypes object keeps the export alive across the GIL-releasing call.
        return ctypes.byref(ctypes.c_char.from_buffer(buffer, offset*8)) if len(view) else None

    @staticmethod
    def _sizes(*values):
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError('operator dimensions must be nonnegative integers')

    def linear(self, w, count, x, xo, xs, y, yo, ys, positions, columns, start, cancel):
        self._sizes(count, xs, ys, positions, columns, start)
        if not columns or not count or not positions:
            return INVALID if not columns else OK
        last_row = (start+count-1)//columns
        return self.kernels['linear'](
            self._arena(w, 0, count), count, self._arena(x, xo, _rows(positions, xs, columns)), xs,
            self._arena(y, yo, _rows(positions, ys, last_row+1)), ys, positions, columns, start,
            ctypes.byref(cancel))

    def rms_denominators(self, x, xo, xs, positions, width, eps, out):
        self._sizes(xs, positions, width)
        if not width:
            return INVALID
        if not positions:
            return OK
        return self.kernels['rms_denominators'](self._arena(x, xo, _rows(positions, xs, width)), xs,
                                                positions, width, eps, self._arena(out, 0, positions))

    def rms_scale(self, x, xo, xs, denominators, w, count, start, y, yo, ys, positions):
        self._sizes(xs, count, start, ys, positions)
        if not positions or not count:
            return OK
        return self.kernels['rms_scale'](
            self._arena(x, xo, _rows(positions, xs, start+count)), xs,
            self._arena(denominators, 0, positions), self._arena(w, 0, count), count, start,
            self._arena(y, yo, _rows(positions, ys, start+count)), ys, positions)

    def add(self, y, yo, ys, x, xo, xs, positions, count):
        self._sizes(ys, xs, positions, count)
        if not positions or not count:
            return OK
        return self.kernels['add'](self._arena(y, yo, _rows(positions, ys, count)), ys,
                                   self._arena(x, xo, _rows(positions, xs, count)), xs, positions, count)

    def rope(self, v, vo, vs, positions, first, heads, dim, theta):
        self._sizes(vs, positions, first, heads, dim)
        if bad_rope(dim):
            return INVALID
        if not positions or not heads:
            return OK
        return self.kernels['rope'](self._arena(v, vo, _rows(positions, vs, heads*dim)), vs, positions,
                                    first, heads, dim, theta)

    def attention(self, q, qo, qs, positions, first, kv, ko, vo, kvs, heads, kv_heads, dim,
                  out, oo, out_stride, cancel):
        self._sizes(qs, positions, first, kvs, heads, kv_heads, dim, out_stride)
        if bad_attention(heads, kv_heads, dim):
            return INVALID
        if not positions or not heads:
            return OK
        context = _rows(first+positions, kvs, kv_heads*dim)
        return self.kernels['attention'](
            self._arena(q, qo, _rows(positions, qs, heads*dim)), qs, positions, first,
            self._arena(kv, ko, context), self._arena(kv, vo, context), kvs, heads, kv_heads, dim,
            self._arena(out, oo, _rows(positions, out_stride, heads*dim)), out_stride, ctypes.byref(cancel))

    def silu_mul(self, gate, up, count):
        self._sizes(count)
        if not count:
            return OK
        return self.kernels['silu_mul'](self._arena(gate, 0, count), self._arena(up, 0, count), count)


def locate(env=None):
    """The operator library: `INWIME_OPS_KERNEL` (a path, or 'off'), else the packaged one."""
    return find_library(env, ENV, '_ops_kernel')

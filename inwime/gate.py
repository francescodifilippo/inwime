"""Operator gate of the Percorso additivo (#41): native operators are used only where they give
the bits of their Python reference on this host.

Bit-identity of C with Python is not implied by the same libm and the same apparent order of
operations (ADR 0001): compiler options, contraction into FMA, the libm entry points, rounding
of intermediates and edge values can each change a bit. So when the Motore loads, every operator
of `inwime.ops` runs here on deterministic cases, native and reference on identical inputs, and
the status codes and every output byte must agree. The cases cover tiles that cut rows, several
positions with row strides, GQA groups, positions up to 10^6, two RoPE bases, exp underflow and
ties in attention, the sign branches and the saturating ends of SiLU, subnormals, signed zeros,
overflow to the nonfinite status, and a set cancellation flag.

An operator that fails a case is not declared equivalent: `select` gives the Python reference for
it and keeps the native implementation for the others. The end-to-end bit-identity of the whole
additive path with the Percorso congelato is proved by the product tests, on tiny tied and untied
Checkpoint, for the certified configuration only.
"""
from array import array
import ctypes

from . import ops as operators

EDGES = (0., -0., 5e-324, -5e-324, 2.2250738585072014e-308, -1e-300, 1e-160, 1., -1., .5,
         3.0000000000000004, -7.25)


def _values(count, seed, scale=1.):
    """Deterministic FP64 values: mostly within [-4, 4) times `scale`, with edge values mixed in."""
    out, state = array('d'), seed
    for i in range(count):
        state = (state*6364136223846793005+1442695040888963407) % 2**64
        if i % 11 == 5:
            out.append(EDGES[(state >> 40) % len(EDGES)])
        else:
            out.append(((state >> 11)/2**53*8-4)*scale)
    return out


def _linear(impl, scale=1., cancel=0):
    rows, columns, positions, xs, ys = 5, 37, 3, 40, 7
    w, x = _values(rows*columns, 11, scale), _values(positions*xs, 12)
    y, flag, cut = array('d', [0.])*(positions*ys), ctypes.c_int(cancel), columns+columns//2
    codes = [impl.linear(w[:cut], cut, x, 0, xs, y, 0, ys, positions, columns, 0, flag),
             impl.linear(w[cut:], rows*columns-cut, x, 0, xs, y, 0, ys, positions, columns, cut, flag)]
    single = array('d', [0.])*rows
    codes.append(impl.linear(w, rows*columns, x, 3, xs, single, 0, 1, 1, columns, 0, flag))
    return codes, (y, single)


def _rms(impl, scale, width=80):
    positions, xs = 64, 86
    x = _values(positions*xs, 21, scale)
    out = array('d', [0.])*positions
    return [impl.rms_denominators(x, 2, xs, positions, width, 1e-5, out)], (out,)


def _rms_zero(impl):
    x, out = array('d', [0., -0.]*40), array('d', [0.])
    return [impl.rms_denominators(x, 0, 80, 1, 80, 1e-6, out)], (out,)


def _rms_scale(impl):
    positions, width = 3, 29
    x, w = _values(positions*width, 31), _values(20, 32)
    denominators = array('d', [1.7, 3e-3, 1e10])
    y = array('d', [5.])*(positions*31)
    return [impl.rms_scale(x, 0, width, denominators, w, 20, 3, y, 1, 31, positions)], (y,)


def _add(impl):
    y, x = _values(3*24, 41, 4.4e307), _values(3*24, 42, 4.4e307)
    bias, z = _values(16, 43), _values(3*20, 44)
    return [impl.add(y, 0, 24, x, 0, 24, 3, 24), impl.add(z, 2, 20, bias, 0, 0, 3, 16)], (y, z)


def _rope(impl, first, theta, dim, heads):
    positions, stride = 3, heads*dim+2
    v = _values(positions*stride, 51+first)
    return [impl.rope(v, 1, stride, positions, first, heads, dim, theta)], (v,)


def _attention(impl, first, scale, ties=False, cancel=0):
    positions, heads, kv_heads, dim = 3, 10, 2, 8
    kvs = kv_heads*dim
    context = first+positions
    q = _values(positions*heads*dim, 61, scale)
    kv = _values(2*context*kvs, 62, scale)
    if ties:
        for t in range(1, context):
            kv[t*kvs:(t+1)*kvs] = kv[0:kvs]
    out, flag = array('d', [0.])*(positions*heads*dim), ctypes.c_int(cancel)
    return [impl.attention(q, 0, heads*dim, positions, first, kv, 0, context*kvs, kvs, heads,
                           kv_heads, dim, out, 0, heads*dim, flag)], (out,)


def _silu(impl):
    special = array('d', [0., -0., 5e-324, -5e-324, 1e-300, -1e-300, 1., -1., 36.7, -36.7, 709.78,
                          -709.78, 745.1, -745.1, 800., -800., 1e300, -1e300])
    gate = special+_values(64, 71, 6.)
    up = _values(len(gate), 72)
    return [impl.silu_mul(gate, up, len(gate))], (gate,)


CASES = {
    'linear': {'tiles cutting rows': _linear, 'overflow': lambda i: _linear(i, 1e300),
               'cancelled': lambda i: _linear(i, cancel=1)},
    'rms_denominators': {'unit': lambda i: _rms(i, 1.), 'odd width': lambda i: _rms(i, 1., 83), 'eps scale': lambda i: _rms(i, 1e-3),
                         'overflow': lambda i: _rms(i, 1e200), 'zeros': _rms_zero},
    'rms_scale': {'strided': _rms_scale},
    'add': {'residual and overflow': _add},
    'rope': {'start': lambda i: _rope(i, 0, 1e6, 64, 2), 'qwen2.5': lambda i: _rope(i, 7, 1e6, 64, 14),
             'far': lambda i: _rope(i, 32760, 1e6, 128, 2), 'llama base': lambda i: _rope(i, 1000003, 1e4, 6, 3)},
    'attention': {'gqa': lambda i: _attention(i, 5, 1.), 'exp underflow': lambda i: _attention(i, 40, 30.),
                  'ties': lambda i: _attention(i, 6, 1., ties=True),
                  'cancelled': lambda i: _attention(i, 2, 1., cancel=1)},
    'silu_mul': {'branches and saturation': _silu},
}


def _outcome(case, impl):
    try:
        codes, outputs = case(impl)
    except Exception as error:      # an implementation that raises is not equivalent
        return 'raised %s: %s' % (type(error).__name__, error)
    return codes, tuple(o.tobytes() for o in outputs)


def check(candidate, reference=None):
    """Per operator: passed, and the first case that differs."""
    reference = reference or operators.PythonOps()
    report = {}
    for name in operators.NAMES:
        failed = None
        for title, case in CASES[name].items():
            expected, actual = _outcome(case, reference), _outcome(case, candidate)
            if isinstance(expected, str):
                raise RuntimeError('the reference %s failed its own case %r: %s' % (name, title, expected))
            if actual != expected:
                failed = '%s: %s' % (title, actual if isinstance(actual, str) else 'differs from the Python reference')
                break
        report[name] = dict(passed=failed is None, cases=len(CASES[name]),
                            detail=failed or 'bit-identical to the Python reference on %d cases' % len(CASES[name]))
    return report


def select(candidate, reference=None):
    """Operator name -> implementation: `candidate` where it passes, else the reference."""
    reference = reference or operators.PythonOps()
    report = check(candidate, reference)
    return {name: candidate if report[name]['passed'] else reference for name in operators.NAMES}, report


def evaluate(env=None):
    """(operators to use, JSON report) for the library `ops.locate(env)` finds."""
    reference = operators.PythonOps()
    chosen = {name: reference for name in operators.NAMES}
    path = operators.locate(env)
    out = dict(path=str(path) if path else None, native=False)

    def described(passed, detail):
        return {name: dict(implementation=chosen[name].name, passed=passed(name), detail=detail(name))
                for name in operators.NAMES}
    if path is None:
        detail = 'native operators absent or disabled: Python reference (very slow)'
        return chosen, dict(out, status='warn', detail=detail,
                            operators=described(lambda n: None, lambda n: detail))
    try:
        native = operators.NativeOps(path)
    except (OSError, AttributeError) as error:   # not a library, a wrong ABI, missing symbols
        detail = 'native operators not loadable: %s: %s' % (type(error).__name__, error)
        return chosen, dict(out, status='fail', detail=detail,
                            operators=described(lambda n: None, lambda n: detail))
    chosen, report = select(native, reference)
    failed = [name for name in operators.NAMES if not report[name]['passed']]
    out.update(native=any(chosen[n] is native for n in operators.NAMES),
               status='fail' if failed else 'ok',
               detail=('operators not equivalent, Python reference used: ' + ', '.join(failed)) if failed
               else 'every operator bit-identical to its Python reference',
               operators=described(lambda n: report[n]['passed'], lambda n: report[n]['detail']))
    return chosen, out


def describe(env=None):
    return evaluate(env)[1]

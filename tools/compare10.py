"""Coverage-aware numerical comparison for increment 10 (NumPy only, no torch, no InWiMe import).

Classes of evidence, never conflated:
  FULL        every required element of logits (all positions x full vocabulary),
              residuals (all layers x positions x width) and K/V (all layers, K and V,
              all positions) was compared
  PARTIAL     some elements were compared (a declared selection or a missing kind)
  STRUCTURAL  no element was compared numerically (shapes, digests or argmax only)
  NOT PROVEN  nothing usable was provided

Verdicts: PASS only with FULL coverage and zero violations; any violation, a
non-finite value or a shape that contradicts the requirement is FAIL; PARTIAL or
STRUCTURAL without violations is NOT PROVEN. There is no fallback that turns a
missing kind into a smaller requirement.
"""
import numpy as np

KINDS = ('logits', 'hidden', 'kv')
THRESHOLDS = {'fp32': (2e-3, 2e-4), 'fp64': (2e-5, 2e-6)}


def required_shapes(layers, positions, width, vocab, kv_width):
    return dict(logits=(positions, vocab), hidden=(layers, positions, width), kv=(layers, 2, positions, kv_width))


def compare_kind(actual, reference, atol, rtol, selection=None):
    ref = np.asarray(reference, dtype=np.float64)
    act = np.asarray(actual, dtype=np.float64)
    if selection is not None:
        ref = ref[selection]
    if act.shape != ref.shape:
        return dict(state='shape-contradiction', actual_shape=list(act.shape), required_shape=list(ref.shape))
    finite = bool(np.isfinite(act).all() and np.isfinite(ref).all())
    err = np.abs(act-ref)
    bound = atol + rtol*np.abs(ref)
    failures = int(np.count_nonzero(~(err <= bound)))
    out = dict(state='compared', elements=int(act.size), failures=failures, finite=finite,
               max_abs_error=float(err.max()) if err.size else 0.0,
               max_threshold_utilization=float((err/bound).max()) if err.size else 0.0)
    if act.ndim >= 1 and act.shape[-1] > 1 and selection is None and act.ndim == 2:
        out['argmax_equal'] = bool((act.argmax(-1) == ref.argmax(-1)).all())
    return out


def compare(actual, reference, shapes, precision, selections=None):
    """actual/reference: dict kind -> array. shapes: required full shapes. selections: kind -> index."""
    atol, rtol = THRESHOLDS[precision]
    selections = selections or {}
    kinds, full, any_compared, fail = {}, True, False, False
    for kind in KINDS:
        req = tuple(shapes[kind])
        ref = reference.get(kind)
        if ref is None or tuple(np.shape(ref)) != req:
            kinds[kind] = dict(state='reference-incomplete')
            full = False
            continue
        if actual is None or actual.get(kind) is None:
            kinds[kind] = dict(state='missing', required_elements=int(np.prod(req)))
            full = False
            continue
        r = compare_kind(actual[kind], ref, atol, rtol, selections.get(kind))
        r['required_elements'] = int(np.prod(req))
        kinds[kind] = r
        if r['state'] != 'compared':
            fail = True
            full = False
            continue
        any_compared = True
        if r['failures'] or not r['finite']:
            fail = True
        if kind in selections or r['elements'] != r['required_elements']:
            full = False
    coverage = 'FULL' if full and any_compared else 'PARTIAL' if any_compared else (
        'STRUCTURAL' if actual else 'NOT PROVEN')
    verdict = 'FAIL' if fail else 'PASS' if coverage == 'FULL' else 'NOT PROVEN'
    if 'logits' in kinds and kinds['logits'].get('state') == 'compared' and 'argmax_equal' in kinds['logits']:
        if not kinds['logits']['argmax_equal']:
            # argmax disagreement with elementwise PASS is reported, not hidden
            kinds['logits']['note'] = 'argmax differs inside the elementwise thresholds'
    return dict(precision=precision, atol=atol, rtol=rtol, coverage=coverage, verdict=verdict, kinds=kinds)


def identical(a, b):
    """Bit identity of float64 arrays (equivalence check of oracle10 against oracle9 captures)."""
    a, b = np.ascontiguousarray(a, dtype=np.float64), np.ascontiguousarray(b, dtype=np.float64)
    return a.shape == b.shape and a.tobytes() == b.tobytes()


def equivalence(new, old, logits_max_abs=1e-9, require_bitwise_logits=False):
    """Preregistered oracle10 vs oracle9 rule (protocol.numerical_contract.oracle10_equivalence)."""
    out = {}
    for kind in KINDS:
        if new.get(kind) is None or old.get(kind) is None:
            out[kind] = dict(state='missing')
            continue
        same = identical(new[kind], old[kind])
        a, b = np.asarray(new[kind], np.float64), np.asarray(old[kind], np.float64)
        d = float(np.abs(a-b).max()) if a.shape == b.shape else None
        out[kind] = dict(bit_identical=same, max_abs_difference=d)
    logits = out.get('logits', {})
    argmax = (np.asarray(new['logits']).argmax(-1) == np.asarray(old['logits']).argmax(-1)).all() \
        if 'bit_identical' in logits and np.shape(new['logits']) == np.shape(old['logits']) else False
    ok = (all('bit_identical' in out[k] for k in KINDS)
          and out['hidden']['bit_identical'] and out['kv']['bit_identical']
          and (logits['bit_identical'] if require_bitwise_logits
               else logits['max_abs_difference'] is not None and logits['max_abs_difference'] <= logits_max_abs
               and bool(argmax)))
    return dict(kinds=out, logits_argmax_identical=bool(argmax), require_bitwise_logits=require_bitwise_logits,
                logits_max_abs_allowed=logits_max_abs, passed=bool(ok))

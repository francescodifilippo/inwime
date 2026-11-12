"""Correctness/resource evidence in separate runtime and reference processes."""

import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
# Fixed before the first runtime comparison. FP64 scalar accumulation differs
# from the vectorized reference and compounds through two nonlinear blocks.
EPS64 = 2.0**-52
TOLERANCES = {
    "logits": (8192*EPS64, 8192*EPS64),
    "hidden": (4096*EPS64, 4096*EPS64),
    "kv": (2048*EPS64, 2048*EPS64),
    "probabilities": (128*EPS64, 256*EPS64),
}
TOKENS = [1, 7, 4, 7, 2, 11]


def compare(runtime, reference):
    import numpy as np
    errors = dict.fromkeys(TOLERANCES, 0.)
    def check(a, b, kind):
        a, b = np.asarray(a), np.asarray(b)
        assert a.shape == b.shape, (kind, a.shape, b.shape)
        errors[kind] = max(errors[kind], float(np.max(np.abs(a-b))))
        atol, rtol = TOLERANCES[kind]
        np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)
    assert len(runtime) == len(reference)
    for a, b in zip(runtime, reference):
        assert a["length"] == b["length"]
        check(a["logits"], b["logits"], "logits")
        assert len(a["hidden"]) == len(b["hidden"])
        for x, y in zip(a["hidden"], b["hidden"]):
            check(x, y, "hidden")
        assert len(a["kv"]) == len(b["kv"])
        for x, y in zip(a["kv"], b["kv"]):
            assert len(x) == len(y) == 2, 'each layer must contain both K and V'
            for u, v in zip(x, y):
                check(u, v, "kv")
        assert len(a["routing"]) == len(b["routing"])
        for x, y in zip(a["routing"], b["routing"]):
            np.testing.assert_array_equal(x[0], y[0])
            check(x[1], y[1], "probabilities")
    return errors


def runtime_export(result):
    """Explicit caller-owned diagnostic copies, outside the engine arenas."""
    owner = result.owner
    c = owner.spec
    def rows(view, width):
        return [list(view[i:i+width]) for i in range(0, len(view), width)]
    kv = []
    dim = c.width//c.heads
    for layer in result.kv:
        kv.append([[[list(view[t*c.width+h*dim:t*c.width+(h+1)*dim])
                     for t in range(owner.length)] for h in range(c.heads)] for view in layer])
    return {"length": owner.length, "logits": rows(result.logits, c.vocab),
            "hidden": [rows(view, c.width) for view in result.hidden], "kv": kv,
            "routing": [[rows(ids, c.top_k), rows(probs, c.top_k)]
                        for ids, probs in result.routing]}


def process_io():
    p = Path('/proc/self/io')
    if not p.exists():
        return None
    return {k: int(v) for k, v in (line.split(':') for line in p.read_text().splitlines())}


def process_memory():
    path = Path('/proc/self/status')
    if not path.exists():
        return None
    return {k: int(v.split()[0]) for k, v in (line.split(':', 1) for line in path.read_text().splitlines())
            if k in ('VmRSS', 'VmHWM')}


def worker(kind, variant):
    directory = ROOT / 'fixtures' / 'reference_decoder' / variant
    outputs = []
    if kind == 'runtime':
        from inwime_runtime import CPURuntime
        before = process_io()
        with CPURuntime(directory) as engine:
            with engine.prefill(TOKENS[:3]) as result:
                outputs.append(runtime_export(result))
            for token in TOKENS[3:]:
                with engine.decode(token) as result:
                    outputs.append(runtime_export(result))
            stats = engine.stats()
            assert stats['weight_slot_bytes'] < stats['payload_bytes']
            assert stats['requests'] == stats['completed'] == stats['released']
        after = process_io()
        stats['numeric_buffers_after_close'] = len(engine.control.buffers)
        stats['linux_process_io_delta'] = ({k: after[k]-before[k] for k in after}
                                            if before is not None and after is not None else None)
        assert 'inwime_reference.decoder' not in sys.modules
        assert 'numpy' not in sys.modules
    else:
        from inwime_reference.checkpoint import load_checkpoint
        from inwime_reference.decoder import ReferenceDecoder
        model = ReferenceDecoder(load_checkpoint(directory))
        result = model.prefill(TOKENS[:3])
        def export(r):
            return {'length': r.state.length, 'logits': r.logits.tolist(),
                    'hidden': [a.tolist() for a in r.hidden],
                    'kv': [[layer.keys.tolist(), layer.values.tolist()] for layer in r.state.layers],
                    'routing': [[r.indices.tolist(), r.probabilities.tolist()] for r in r.routing]}
        outputs.append(export(result))
        for token in TOKENS[3:]:
            result = model.decode(token, result.state)
            outputs.append(export(result))
        stats = {}
    return {'outputs': outputs, 'stats': stats,
            'linux_status_memory_kib': process_memory(),
            'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', choices=('runtime', 'reference'))
    parser.add_argument('--variant', choices=('dense', 'moe'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.variant), allow_nan=False))
        return
    evidence = {'scope': 'increment2 CPU correctness, not a performance benchmark',
                'tokens': TOKENS, 'prefill_tokens': 3, 'tolerances': TOLERANCES, 'cases': {}}
    collected = {}
    # Launch every child before importing NumPy for comparison in the parent.
    # This avoids inheriting the parent's larger RSS high-water mark on fork.
    for variant in ('dense', 'moe'):
        values = {}
        for kind in ('runtime', 'reference'):
            completed = subprocess.run([sys.executable, '-m', 'tools.verify_increment2',
                                        '--worker', kind, '--variant', variant], cwd=ROOT,
                                       check=True, text=True, capture_output=True)
            values[kind] = json.loads(completed.stdout)
        collected[variant] = values
    for variant, values in collected.items():
        errors = compare(values['runtime']['outputs'], values['reference']['outputs'])
        evidence['cases'][variant] = {'max_absolute_errors': errors,
            'runtime': {k: v for k, v in values['runtime'].items() if k != 'outputs'},
            'reference_peak_rss_kib': values['reference']['process_peak_rss_kib']}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(evidence, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()

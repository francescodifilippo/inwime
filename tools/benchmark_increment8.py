"""Real Llama forward under RLIMIT_AS; stdlib runtime, optional official tokenizer.

Oracle/capture comparison runs elsewhere. Captures are bounded FP64 outputs,
never weights. This tool intentionally does not import NumPy or torch.
"""
import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

from inwime_runtime.llama import LlamaRuntime, LlamaCatalog
from inwime_runtime.safetensors_source import SafetensorsSource
from inwime_runtime.math_cpu import NativeMath
from inwime_runtime.tokenizer import LlamaTokenizer

COUNTERS = ('inference_read_bytes', 'gemv_ns', 'bf16_expand_ns', 'compute_ns',
            'read_service_ns', 'data_wait_ns', 'read_union_ns', 'overlap_ns')


def write_json(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, indent=2); f.write('\n')


def proc_status():
    result = {}
    for line in Path('/proc/self/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key.startswith('Vm'):
            result[key+'_bytes'] = int(value.split()[0])*1024
    return result


def io_counters():
    return {k: int(v) for k, v in (s.split(':') for s in Path('/proc/self/io').read_text().splitlines())}


def delta(after, before):
    return {k: after[k]-before[k] for k in before}


def digest_views(views):
    h = hashlib.sha256()
    for view in views:
        h.update(view.cast('B'))
    return h.hexdigest()


def kv_prefix(result, length, width):
    return digest_views(v[:length*width] for pair in result.kv for v in pair)


def save_values(directory, prefix, kind, views, shape):
    path = directory/(prefix+'.'+kind+'.f64')
    h = hashlib.sha256()
    with path.open('xb') as f:
        for view in views:
            raw = view.cast('B')
            f.write(raw); h.update(raw)
    return dict(file=path.name, shape=shape, dtype='<f8', nbytes=path.stat().st_size,
                sha256=h.hexdigest())


def expected_reads(catalog, token_count):
    return {name: token_count*2*(catalog.spec.width if name == 'embedding' else t.elements)
            for name, t in catalog.tensors.items()}


def run(a):
    if sys.byteorder != 'little':
        raise RuntimeError('capture contract requires little endian')
    if a.as_mib <= 0:
        raise ValueError('positive AS limit required')
    resource.setrlimit(resource.RLIMIT_AS, (a.as_mib*2**20, a.as_mib*2**20))
    report = dict(mode=a.mode, rlimit_as_bytes=resource.getrlimit(resource.RLIMIT_AS),
                  cache_state='buffered/uncontrolled; SHA scans precede inference',
                  checkpoint=str(a.checkpoint), tile_elements=a.tile_elements,
                  capacity=a.capacity, gpu_allocations_bytes=0, weight_cache=False,
                  process_before=proc_status(), phases=[], captures={}, scenario=a.scenario,
                  cgroup_path=Path('/proc/self/cgroup').read_text(),
                  cgroup_memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip())
    # Tokenization happens inside the limited process; Rayon parallelism disabled
    # by the launcher to avoid a global thread pool and many virtual stacks.
    began = time.monotonic()
    tokenizer = LlamaTokenizer(a.checkpoint)
    tokens = tokenizer.encode(a.text)
    if len(tokens) < 2 or len(tokens)+3 > a.capacity:
        raise ValueError('need >=2 prefill tokens and capacity for three decodes')
    report.update(text=a.text, input_ids=tokens, tokenizer_s=time.monotonic()-began)
    began = time.monotonic()
    catalog = LlamaCatalog(SafetensorsSource(a.checkpoint))
    report['open_index_s'] = time.monotonic()-began
    report['config'] = catalog.config
    report['source_identity'] = catalog.identity
    report['unique_active_bf16_bytes'] = catalog.payload_bytes
    # An independently timed, bounded SHA, in addition to runtime full integrity.
    before = io_counters(); began = time.monotonic()
    for name, (_, expected) in catalog.files.items():
        with (a.checkpoint/name).open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError('SHA256 mismatch')
    report['sha256_s'] = time.monotonic()-began
    report['sha256_bytes'] = sum(s for s, _ in catalog.files.values())
    report['sha256_proc_io_delta'] = delta(io_counters(), before)
    if a.capture:
        a.capture.mkdir(parents=True, exist_ok=False)
    before = io_counters(); began = time.monotonic()
    with LlamaRuntime(a.checkpoint, capacity=a.capacity, tile_elements=a.tile_elements,
                      read_elements=a.tile_elements, io_mode=a.mode,
                      math_backend=NativeMath('build/libinwime_cpu.so')) as runtime:
        report['startup_full_integrity_s'] = time.monotonic()-began
        report['startup_proc_io_delta'] = delta(io_counters(), before)
        report['arena_by_category'] = runtime.plan.required
        report['arena_plan_bytes'] = runtime.plan.total
        report['bootstrap_estimate_bytes'] = runtime.catalog.bootstrap_bytes
        report['process_after_startup'] = proc_status()
        array_sizes = {n: len(v) for n, v in runtime.control.buffers.items()}
        previous_kv = None
        continuation = []
        if a.scenario == 'extended':
            if not a.sequence_report:
                raise ValueError('extended scenario requires a sequence report')
            source = json.loads(a.sequence_report.read_text())
            if source['input_ids'] != tokens:
                raise ValueError('different prompt')
            chunks = [tokens+source['continuation'][:3]]
        elif a.scenario == 'chunks':
            chunks = [tokens[:2], tokens[2:]]
            chunks = [chunk for chunk in chunks if chunk]
        else:
            chunks = [tokens, None, None, None]
        inference_io = io_counters()
        for index, ids in enumerate(chunks):
            if ids is None:
                ids = [continuation[-1]]
            previous = runtime.stats(); before = io_counters()
            old_length, version = runtime.length, runtime.version
            began = time.monotonic()
            with runtime.prefill(ids) as result:
                elapsed = time.monotonic()-began
                after = io_counters(); stats = runtime.stats()
                counters = {k: stats[k]-previous[k] for k in COUNTERS}
                if previous_kv is not None:
                    assert kv_prefix(result, old_length, runtime.spec.kv_width) == previous_kv
                previous_kv = kv_prefix(result, runtime.length, runtime.spec.kv_width)
                assert runtime.length == old_length+len(ids) and runtime.version == version+1
                c = runtime.spec
                chosen = [max(range(c.vocab), key=result.logits[row*c.vocab:(row+1)*c.vocab].__getitem__)
                          for row in range(len(ids))]
                continuation.append(chosen[-1])
                digests = dict(logits=digest_views([result.logits]),
                               hidden=digest_views(result.hidden), kv=previous_kv)
                prefix = f'{a.scenario}.{index}'
                if a.capture:
                    report['captures'][prefix] = {
                        'logits': save_values(a.capture, prefix, 'logits', [result.logits], [len(ids), c.vocab]),
                        'hidden': save_values(a.capture, prefix, 'hidden', result.hidden, [c.layers, len(ids), c.width]),
                        'kv': save_values(a.capture, prefix, 'kv', (v for pair in result.kv for v in pair),
                                          [c.layers, 2, runtime.length, c.kv_width])}
            assert array_sizes == {n: len(v) for n, v in runtime.control.buffers.items()}
            report['phases'].append(dict(name=prefix, ids=ids, greedy=chosen, wall_s=elapsed,
                tokens_per_s=len(ids)/elapsed, counters_delta=counters,
                host_residual_s=elapsed-(counters['compute_ns']+counters['bf16_expand_ns']+
                    counters['read_union_ns']-counters['overlap_ns'])/1e9,
                proc_io_delta=delta(after, before), process=proc_status(),
                kv_prefix_preserved=True, length=runtime.length, generation=runtime.version,
                digests=digests))
        report['stats'] = runtime.stats()
        report['inference_proc_io_delta'] = delta(io_counters(), inference_io)
        count = sum(len(p['ids']) for p in report['phases'])
        expected = expected_reads(runtime.catalog, count)
        assert expected == report['stats']['tensor_read_bytes']
        assert sum(expected.values()) == report['stats']['inference_read_bytes']
        linear = sum(t.elements for n, t in runtime.catalog.tensors.items()
                     if n != 'embedding' and len(t.shape) == 2)
        assert report['stats']['gemv_weight_elements'] == count*linear
        assert report['unique_active_bf16_bytes'] > a.as_mib*2**20
        report['all_tensors_used'] = all(v > 0 for v in expected.values())
        report['coverage_exact'] = True
        report['expected_tensor_read_bytes'] = expected
    report['released_fds'] = not runtime.control.fds
    report['released_arenas'] = not runtime.control.buffers
    assert report['released_fds'] and report['released_arenas']
    report['continuation'] = continuation
    report['generated_text'] = tokenizer.decode(tokens+continuation)
    report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report['process_after_close'] = proc_status()
    report['heavy_modules_loaded'] = sorted(n for n in ('torch', 'numpy', 'transformers') if n in sys.modules)
    assert not report['heavy_modules_loaded']
    write_json(a.output, report)
    print(a.output, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--mode', choices=['sync', 'pipeline'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--capture', type=Path)
    p.add_argument('--as-mib', type=int, default=1024)
    p.add_argument('--capacity', type=int, default=16)
    p.add_argument('--tile-elements', type=int, default=262144)
    p.add_argument('--text', default='The capital of France is')
    p.add_argument('--scenario', choices=['decode', 'extended', 'chunks'], default='decode')
    p.add_argument('--sequence-report', type=Path)
    run(p.parse_args())

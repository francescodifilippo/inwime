"""Real Llama/Qwen2 forward for increment 9, optionally inside a verified cgroup scope.

Stdlib runtime plus the optional official tokenizer; never NumPy/torch. Captures
are bounded FP64 outputs (logits, residuals, KV), never weights. When launched by
tools.cgroup9 the worker proves its own scope membership and limits BEFORE it
opens the tokenizer or any checkpoint file. Unique active BF16 bytes are derived
from per-tensor read counters, never from file sizes.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

from inwime_runtime.llama import LlamaRuntime
from inwime_runtime.qwen2 import Qwen2Runtime
from inwime_runtime.math_cpu import NativeMath
from inwime_runtime.tokenizer import LlamaTokenizer
from .benchmark_increment8 import (COUNTERS, write_json, proc_status, io_counters, delta,
                                   digest_views, kv_prefix, save_values)
from .sanitize9 import portable

FAMILIES = {'llama': LlamaRuntime, 'qwen2': Qwen2Runtime}


def repository_path(path):
    """Repository-relative when inside the checkout: evidence carries no absolute home path."""
    try:
        return str(Path(os.path.abspath(path)).relative_to(Path.cwd()))
    except ValueError:
        return portable(str(path))


def meminfo():
    out = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, _, value = line.partition(':')
        out[key] = int(value.split()[0])*1024
    return {k: out[k] for k in ('MemTotal', 'MemAvailable', 'Cached', 'SwapTotal', 'SwapFree')}


def own_cgroup():
    path = Path('/proc/self/cgroup').read_text().strip().split('::', 1)[1]
    return path, Path('/sys/fs/cgroup'+path)


def cgroup_snapshot():
    path, directory = own_cgroup()
    snap = {'path': path}
    for name in ('memory.current', 'memory.peak', 'memory.max', 'memory.swap.current', 'memory.swap.max'):
        f = directory/name
        if f.exists():
            value = f.read_text().strip()
            snap[name] = None if value == 'max' else int(value)
    for name in ('memory.events', 'memory.swap.events'):
        f = directory/name
        if f.exists():
            snap[name] = {k: int(v) for k, v in (l.split() for l in f.read_text().splitlines())}
    f = directory/'memory.stat'
    if f.exists():
        stat = {k: int(v) for k, v in (l.split() for l in f.read_text().splitlines())}
        snap['memory.stat'] = {k: stat[k] for k in ('anon', 'file', 'kernel', 'file_mapped', 'shmem',
                                                    'active_file', 'inactive_file', 'pgmajfault',
                                                    'workingset_refault_file') if k in stat}
    return snap


def verify_scope():
    """Fail before any model file is opened if the scope/limits are not the expected ones."""
    unit = os.environ.get('INWIME9_SCOPE_UNIT')
    if not unit:
        return dict(required=False)
    path, directory = own_cgroup()
    snap = cgroup_snapshot()
    expected_max = int(os.environ['INWIME9_MEMORY_MAX'])
    expected_swap = int(os.environ['INWIME9_SWAP_MAX'])
    member = path.endswith('/'+unit+'.scope') and str(os.getpid()) in (directory/'cgroup.procs').read_text().split()
    ok = member and snap.get('memory.max') == expected_max and snap.get('memory.swap.max') == expected_swap
    result = dict(required=True, unit=unit, pid=os.getpid(), ppid=os.getppid(), member=member,
                  observed=snap, expected_memory_max=expected_max, expected_swap_max=expected_swap,
                  verified_before_opening_weights=ok, open_fds_before=sorted(os.listdir('/proc/self/fd')))
    if not ok:
        raise RuntimeError('worker is not in the expected limited cgroup scope: '+json.dumps(result))
    return result


def unique_active_bytes(catalog, reads, tokens):
    """Union of BF16 bytes actually consumed. Aliases (tied head) count once."""
    width = catalog.spec.width
    by_physical = {}
    for name, tensor in catalog.tensors.items():
        key = (tensor.shard, tensor.offset)
        full = name != 'embedding' and reads[name] > 0
        entry = by_physical.setdefault(key, dict(elements=tensor.elements, full=False, names=[]))
        entry['full'] |= full
        entry['names'].append(name)
    rows = len(set(tokens))
    total = 0
    for entry in by_physical.values():
        if entry['full']:
            total += entry['elements']*2
        elif 'embedding' in entry['names']:
            total += rows*width*2
    return total, {','.join(e['names']): (e['elements']*2 if e['full'] else rows*width*2)
                   for e in by_physical.values()}


def expected_reads(catalog, token_count):
    return {name: token_count*2*(catalog.spec.width if name == 'embedding' else t.elements)
            for name, t in catalog.tensors.items()}


def run(a):
    began_process = time.monotonic()
    scope = verify_scope()  # before tokenizer, catalog, SHA or runtime construction
    if a.as_mib:
        resource.setrlimit(resource.RLIMIT_AS, (a.as_mib*2**20, a.as_mib*2**20))
    report = dict(family=a.family, mode=a.mode, scenario='decode', checkpoint=repository_path(a.checkpoint),
                  tile_elements=a.tile_elements, capacity=a.capacity, decodes=a.decodes,
                  rlimit_as_bytes=resource.getrlimit(resource.RLIMIT_AS), scope=scope,
                  meminfo_start=meminfo(), cgroup_start=cgroup_snapshot(),
                  cache_state=a.cache_state, gpu_allocations_bytes=0, weight_cache=False,
                  process_before=proc_status(), phases=[], captures={})
    began = time.monotonic()
    tokenizer = LlamaTokenizer(a.checkpoint)
    tokens = tokenizer.encode(a.text)
    if len(tokens) < 2 or len(tokens)+a.decodes > a.capacity:
        raise ValueError('need >=2 prefill tokens and capacity for the decodes')
    report.update(text=a.text, input_ids=tokens, tokenizer_s=time.monotonic()-began,
                  tokenizer_class='tokenizers.Tokenizer(tokenizer.json)')
    if a.capture:
        a.capture.mkdir(parents=True, exist_ok=False)
    runtime_class = FAMILIES[a.family]
    before = io_counters(); began = time.monotonic()
    with runtime_class(a.checkpoint, capacity=a.capacity, tile_elements=a.tile_elements,
                       read_elements=a.tile_elements, io_mode=a.mode,
                       math_backend=NativeMath('build/libinwime_cpu.so')) as runtime:
        report['startup_full_integrity_s'] = time.monotonic()-began
        report['startup_proc_io_delta'] = delta(io_counters(), before)
        startup = runtime.stats()
        report['integrity_read_bytes'] = startup['integrity_read_bytes']
        report['integrity_read_ns'] = startup['integrity_read_ns']
        report['config'] = runtime.catalog.config
        report['source_identity'] = runtime.catalog.identity
        report['payload_bytes'] = runtime.catalog.payload_bytes
        report['source_files'] = {n: dict(nbytes=s, sha256=h) for n, (s, h) in runtime.catalog.files.items()}
        report['arena_by_category'] = runtime.plan.required
        report['arena_plan_bytes'] = runtime.plan.total
        report['budget_bytes'] = runtime.control.budget.total
        report['bootstrap_estimate_bytes'] = runtime.catalog.bootstrap_bytes
        report['process_after_startup'] = proc_status()
        report['cgroup_after_startup'] = cgroup_snapshot()
        array_sizes = {n: len(v) for n, v in runtime.control.buffers.items()}
        previous_kv, continuation = None, []
        chunks = [tokens] + [None]*a.decodes
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
                prefix = f'decode.{index}'
                if a.capture:
                    report['captures'][prefix] = {
                        'logits': save_values(a.capture, prefix, 'logits', [result.logits], [len(ids), c.vocab]),
                        'hidden': save_values(a.capture, prefix, 'hidden', result.hidden, [c.layers, len(ids), c.width]),
                        'kv': save_values(a.capture, prefix, 'kv', (v for pair in result.kv for v in pair),
                                          [c.layers, 2, runtime.length, c.kv_width])}
            assert array_sizes == {n: len(v) for n, v in runtime.control.buffers.items()}
            report['phases'].append(dict(
                name=prefix, ids=ids, greedy=chosen, wall_s=elapsed, tokens_per_s=len(ids)/elapsed,
                counters_delta=counters,
                host_residual_s=elapsed-(counters['compute_ns']+counters['bf16_expand_ns']+
                                         counters['read_union_ns']-counters['overlap_ns'])/1e9,
                proc_io_delta=delta(after, before), process=proc_status(), cgroup=cgroup_snapshot(),
                meminfo=meminfo(), kv_prefix_preserved=True, length=runtime.length,
                generation=runtime.version, digests=digests))
            print(f'phase {prefix} ids={ids} greedy={chosen[-1]} wall={elapsed:.2f}s', flush=True)
        report['stats'] = runtime.stats()
        report['inference_proc_io_delta'] = delta(io_counters(), inference_io)
        sequence = [t for p in report['phases'] for t in p['ids']]
        expected = expected_reads(runtime.catalog, len(sequence))
        assert expected == report['stats']['tensor_read_bytes']
        assert sum(expected.values()) == report['stats']['inference_read_bytes']
        linear = sum(t.elements for n, t in runtime.catalog.tensors.items()
                     if n != 'embedding' and len(t.shape) == 2)
        assert report['stats']['gemv_weight_elements'] == len(sequence)*linear
        unique, parts = unique_active_bytes(runtime.catalog, report['stats']['tensor_read_bytes'], sequence)
        report['unique_active_bf16_bytes'] = unique
        report['unique_active_by_physical_tensor'] = parts
        report['logical_bytes_per_token'] = sum(expected_reads(runtime.catalog, 1).values())
        report['all_tensors_used'] = all(v > 0 for v in expected.values())
        report['coverage_exact'] = True
        report['expected_tensor_read_bytes'] = expected
        report['memtotal_criterion'] = dict(
            memtotal_at_start=report['meminfo_start']['MemTotal'],
            unique_active_bf16_bytes=unique,
            exceeds=unique > report['meminfo_start']['MemTotal'])
    report['released_fds'] = not runtime.control.fds
    report['released_arenas'] = not runtime.control.buffers
    assert report['released_fds'] and report['released_arenas']
    report['open_fds_after_close'] = sorted(os.listdir('/proc/self/fd'))
    report['continuation'] = continuation
    report['generated_text'] = tokenizer.decode(tokens+continuation)
    report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report['process_after_close'] = proc_status()
    report['cgroup_end'] = cgroup_snapshot()
    report['meminfo_end'] = meminfo()
    report['process_wall_s'] = time.monotonic()-began_process
    report['heavy_modules_loaded'] = sorted(n for n in ('torch', 'numpy', 'transformers') if n in sys.modules)
    assert not report['heavy_modules_loaded']
    write_json(a.output, report)
    print(a.output, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--family', choices=sorted(FAMILIES), required=True)
    p.add_argument('--mode', choices=['sync', 'pipeline'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--capture', type=Path)
    p.add_argument('--as-mib', type=int, default=0)
    p.add_argument('--capacity', type=int, default=16)
    p.add_argument('--decodes', type=int, default=3)
    p.add_argument('--tile-elements', type=int, default=262144)
    p.add_argument('--text', default='The capital of France is')
    p.add_argument('--cache-state', required=True,
                   help='declared condition, e.g. evicted-under-scope / warm-charged-elsewhere')
    run(p.parse_args())

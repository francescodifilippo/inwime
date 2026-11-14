"""Small A/B/C cache-warm comparisons, fresh supervised processes, no downloads.

A is an exported f304ee9 tree, B is candidate/v2, C is candidate/safetensors.
Fixture writing is explicit; existing input directories are never overwritten.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
BASE = 'f304ee9f19b6a89907d4ad10cbdce435640352fb'


def memory(pid='self'):
    return {k: int(v.split()[0])*1024 for k, v in
            (line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines())
            if k in ('VmRSS', 'VmHWM', 'VmSwap', 'VmSize')}


def worker(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    # Resolve runtime imports from the requested code tree BEFORE importing any
    # InWiMe modules. Tools used for diagnostics do not implement the runtime.
    sys.path.insert(0, str(args.code_root.resolve()))
    from inwime_runtime import CPURuntime
    import inwime_runtime.decoder as module
    from inwime_runtime.math_cpu import NativeMath
    from tools.verify_increment2 import process_io
    from tools.verify_increment3 import delta, host_state
    from tools.verify_increment5 import summarize_output
    resource.setrlimit(resource.RLIMIT_AS, (512*2**20, 512*2**20))
    native = NativeMath(ROOT/'build/libinwime_cpu.so')
    parse, parse_io = [], []
    symbol = 'open_catalog' if hasattr(module, 'open_catalog') else 'Catalog'
    original = getattr(module, symbol)
    def observed(*values, **options):
        began, io = time.monotonic_ns(), process_io()
        result = original(*values, **options)
        parse.append(time.monotonic_ns()-began)
        parse_io.append(delta(process_io(), io))
        return result
    setattr(module, symbol, observed)
    before, host_before = process_io(), host_state()
    began = time.monotonic_ns()
    options = dict(capacity=6, tile_elements=73, read_elements=13,
                   math_backend=native, io_mode=args.mode)
    if args.source == 'safetensors':
        options['source'] = args.source
    engine = CPURuntime(args.fixture, **options)
    startup = time.monotonic_ns()-began
    after_open = process_io()
    assert engine.plan.total <= 128*2**20
    outputs, phases, old = [], [], None
    try:
        began = time.monotonic_ns()
        for tokens in ([1, 7, 4], [7], [2], [11]):
            t = time.monotonic_ns()
            result = engine.prefill(tokens)
            forward_ns = time.monotonic_ns()-t
            with result:
                output = summarize_output(engine, result, old)
                output['routing_sha256'] = [[hashlib.sha256(v).hexdigest() for v in parts]
                                            for parts in result.routing]
            old = engine.length, output['kv_sha256']
            outputs.append(output)
            phases.append(forward_ns)
        wall = time.monotonic_ns()-began
        io_after = process_io()
        stats = engine.stats()
        assert stats['audited_retained_metadata_bytes'] <= engine.plan.required['metadata']
        assert stats['requests'] == stats['completed'] == stats['released']
        assert not stats['active_region'] and not stats['session_failed']
        observed_memory = memory()
        result = dict(code_module=str(Path(module.__file__).resolve()), source=args.source,
                      mode=args.mode, checkpoint_identity=engine.catalog.identity,
                      payload_bytes=engine.catalog.payload_bytes,
                      catalog_bootstrap_bytes=engine.catalog.bootstrap_bytes,
                      catalog_retained_bytes=engine.catalog.retained_bytes,
                      wall_ns=wall, startup_ns=startup, index_parse_ns=sum(parse),
                      index_process_io=parse_io, startup_process_io=delta(after_open, before),
                      workload_process_io=delta(io_after, after_open), stats=stats,
                      prefill_ns=phases[0], decode_ns=phases[1:], outputs=outputs,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      process_memory=observed_memory, gpu_allocated_bytes=0,
                      host_before=host_before, host_after=host_state(),
                      address_limit_bytes=resource.getrlimit(resource.RLIMIT_AS)[0])
    finally:
        engine.close()
    result['after_close'] = dict(fds=len(engine.control.fds), buffers=len(engine.control.buffers),
                                live_requests=sum(r is not None for r in getattr(engine.control, 'slots', [])))
    assert not any(result['after_close'].values())
    args.output.write_text(json.dumps(result, indent=2)+'\n')


def supervised(command, log):
    peak = 0
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   env=os.environ | {'OPENBLAS_NUM_THREADS': '1'})
        try:
            while process.poll() is None:
                try:
                    peak = max(peak, memory(process.pid).get('VmRSS', 0))
                except (FileNotFoundError, ProcessLookupError):
                    pass
                if peak > 2**30:
                    process.kill()
                    raise MemoryError('RSS safety stop at 1 GiB')
                time.sleep(.01)
            if process.returncode:
                raise RuntimeError(f'failed ({process.returncode}): {command}; see {log}')
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    return peak


def run(args):
    from tools.checkpoint4 import generate, audit
    from tools.fixture_safetensors6 import write_fixture
    if args.output.exists() or args.fixture.exists() or args.code_root.exists():
        raise FileExistsError('output, fixture and baseline tree must all be new')
    args.output.mkdir(parents=True)
    args.fixture.mkdir(parents=True)
    args.code_root.mkdir(parents=True)
    # Export only the small baseline source, never old checkpoint corpora.
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', BASE,
                                     'inwime_runtime'], cwd=ROOT, text=True).splitlines()
    for name in paths:
        p = args.code_root/name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(subprocess.check_output(['git', 'show', f'{BASE}:{name}'], cwd=ROOT))
    provenance = dict(base=BASE, environment=audit(ROOT), python=sys.version,
                      platform=platform.platform(), library_sha256=hashlib.sha256(
                          (ROOT/'build/libinwime_cpu.so').read_bytes()).hexdigest(),
                      baseline_sha256={name: hashlib.sha256((args.code_root/name).read_bytes()).hexdigest()
                                       for name in paths})
    for variant in ('dense', 'moe'):
        original = ROOT/'fixtures/reference_decoder'/variant
        generate(args.fixture/(variant+'-v2'), source=original,
                 shard_bytes=4096, segment_bytes=1022, execute=True)
        write_fixture(original, args.fixture/(variant+'-safe'))
    def all_hashes():
        return {str(p.relative_to(args.fixture)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(args.fixture.rglob('*')) if p.is_file()}
    provenance['fixture_sha256_before'] = all_hashes()
    provenance['fixture_disk_bytes'] = sum(p.stat().st_size for p in args.fixture.rglob('*') if p.is_file())
    assert provenance['fixture_disk_bytes'] <= 64*2**20
    cases, order = {}, []
    for variant in ('dense', 'moe'):
        for mode in ('sync', 'pipeline'):
            groups = {arm: [] for arm in 'ABC'}
            # One discarded warm-up per arm, followed by five counterbalanced
            # permutations. Full integrity also warms each fresh process's files.
            sequences = ['ABC', 'ABC', 'BCA', 'CAB', 'CBA', 'BAC']
            for repetition, sequence in enumerate(sequences):
                for arm in sequence:
                    label = f'{variant}-{mode}-{arm}-{repetition}'
                    output = args.output/(label+'.json')
                    source = 'safetensors' if arm == 'C' else 'fixture'
                    fixture = args.fixture/(variant+('-safe' if arm == 'C' else '-v2'))
                    command = [sys.executable, '-m', 'tools.benchmark_increment6', '--worker',
                               '--code-root', str(args.code_root if arm == 'A' else ROOT),
                               '--fixture', str(fixture), '--source', source, '--mode', mode,
                               '--output', str(output)]
                    peak = supervised(command, args.output/(label+'.log'))
                    sample = json.loads(output.read_text())
                    sample['supervisor_sampled_peak_rss_bytes'] = peak
                    output.write_text(json.dumps(sample, indent=2)+'\n')
                    order.append(label)
                    if repetition:
                        groups[arm].append(sample)
            reference = groups['A'][0]
            for samples in groups.values():
                for sample in samples:
                    assert sample['outputs'] == reference['outputs']
                    assert sample['stats']['tensor_read_bytes'] == reference['stats']['tensor_read_bytes']
                    assert sample['stats']['gemv_weight_elements'] == reference['stats']['gemv_weight_elements']
            assert groups['A'][0]['stats']['required_by_category'] == groups['B'][0]['stats']['required_by_category']
            summary = {}
            for arm, samples in groups.items():
                summary[arm] = {key: dict(median=statistics.median(s[key] for s in samples),
                                          minimum=min(s[key] for s in samples),
                                          maximum=max(s[key] for s in samples),
                                          samples=[s[key] for s in samples])
                                for key in ('wall_ns', 'startup_ns', 'index_parse_ns', 'peak_rss_kib',
                                            'catalog_bootstrap_bytes', 'catalog_retained_bytes')}
                summary[arm]['stats'] = samples[0]['stats']
                summary[arm]['workload_read_bytes_samples'] = [s['workload_process_io']['read_bytes'] for s in samples]
            summary['B_vs_A_wall_percent'] = (summary['B']['wall_ns']['median']/summary['A']['wall_ns']['median']-1)*100
            summary['C_vs_B_wall_percent'] = (summary['C']['wall_ns']['median']/summary['B']['wall_ns']['median']-1)*100
            cases[variant+'-'+mode] = summary
    provenance['fixture_sha256_after'] = all_hashes()
    assert provenance['fixture_sha256_before'] == provenance['fixture_sha256_after']
    (args.output/'summary.json').write_text(json.dumps(dict(
        provenance=provenance, execution_order=order, cases=cases,
        cache='cache-warm; full SHA before inference, no cache eviction',
        caveat='small fixtures; Windows/device caches and temperature/power uncontrolled',
        gpu_allocated_bytes=0), indent=2)+'\n')
    print(json.dumps({k: {n: v[n] for n in ('B_vs_A_wall_percent', 'C_vs_B_wall_percent')}
                      for k, v in cases.items()}, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--code-root', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', choices=['fixture', 'safetensors'], default='fixture')
    p.add_argument('--mode', choices=['sync', 'pipeline'], default='sync')
    args = p.parse_args()
    for key in ('code_root', 'fixture', 'output'):
        setattr(args, key, getattr(args, key).resolve())
    worker(args) if args.worker else run(args)


if __name__ == '__main__':
    main()

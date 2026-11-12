"""Increment 3 evidence; separate processes, paired conditions, no scratch weights.

The controlled workload uses delayed real file reads and repeated real scalar
arithmetic, not a storage bandwidth claim or decoder acceleration. Historical
benchmark tools and artifacts are untouched.
"""

import argparse
from contextlib import closing
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time

from tools.verify_increment2 import (ROOT, TOKENS, TOLERANCES, compare, runtime_export,
                                     process_io, process_memory)


def host_state():
    paths = [Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor')]
    paths += sorted(Path('/sys/class/thermal').glob('thermal_zone*/temp'))
    paths += sorted(Path('/sys/class/power_supply').glob('*/online'))
    result = {}
    for path in paths:
        try:
            result[str(path)] = path.read_text().strip()
        except OSError:
            pass
    return result


def delta(after, before):
    return None if before is None or after is None else {k: after[k]-before[k] for k in after}


def worker(args):
    from inwime_runtime import CPURuntime
    from inwime_runtime.core import FileBackend
    class DelayedFile(FileBackend):
        delay = 0.
        def readinto(self, *values):
            if self.delay:
                time.sleep(self.delay)
            return super().readinto(*values)
    backend = DelayedFile()
    controlled = args.scenario == 'controlled'
    path = ROOT/'fixtures/reference_decoder'/('dense' if controlled else args.scenario)
    # Controlled case: one physical read per tile. Decoder: same requested
    # dimensions and numerical algorithm for both modes.
    read = args.tile_elements if controlled else args.read_elements
    io_before = process_io()
    host_before = host_state()
    startup_began = time.perf_counter_ns()
    outputs, checksum = [], 0.
    with CPURuntime(path, tile_elements=args.tile_elements, read_elements=read,
                    io_mode=args.mode, in_flight=args.in_flight, io_workers=args.io_workers,
                    trace_capacity=4096 if controlled else 0, backend=backend) as engine:
        startup_ns = time.perf_counter_ns()-startup_began
        io_after_startup = process_io()
        backend.delay = args.read_delay if controlled else 0.
        began = time.perf_counter_ns()
        if controlled:
            tensor = engine.catalog.tensors['embedding']
            count = min(args.tile_elements, tensor.elements)
            regions = (('embedding', (i % (tensor.elements//count))*count, count)
                       for i in range(args.regions))
            with closing(engine.control.stream(regions)) as stream:
                for _, size, weights in stream:
                    for _ in range(args.compute_repeats):
                        for j in range(size):
                            checksum += weights.value(j)*(j+1)
        else:
            with engine.prefill(TOKENS[:3]) as output:
                outputs.append(runtime_export(output))
            for token in TOKENS[3:]:
                with engine.decode(token) as output:
                    outputs.append(runtime_export(output))
        wall_ns = time.perf_counter_ns()-began
        io_after_inference = process_io()
        stats = engine.stats()
        stats['owner_phase_remainder_ns'] = (wall_ns-stats['data_wait_ns']
                                             -stats['bf16_expand_ns']-stats['compute_ns'])
        trace = engine.control.trace() if controlled else []
        assert not stats['active_region'] and stats['occupied_slots'] == 0
        assert stats['requests'] == stats['completed'] == stats['released']
        assert stats['required_by_category']['weights'] < stats['payload_bytes']
    assert not engine.control.fds and not engine.control.buffers
    assert not any(name.split('.')[0] in {'torch', 'numpy', 'inwime_reference', 'cuda'}
                   for name in sys.modules)
    return {'wall_ns': wall_ns, 'startup_ns': startup_ns, 'stats': stats,
            'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'linux_status_memory_kib': process_memory(),
            'startup_process_io_delta': delta(io_after_startup, io_before),
            'inference_process_io_delta': delta(io_after_inference, io_after_startup),
            'gpu_memory_bytes': 0, 'gpu_backend_initialized': False,
            'host_before': host_before, 'host_after': host_state(),
            'outputs': outputs, 'controlled_checksum': checksum, 'trace': trace}


def verify_trace(events):
    """Reconstruct intersection of real intervals, independently of telemetry."""
    readers = computers = overlap = 0
    last = events[0][0]
    for stamp, kind, *_ in events:
        assert stamp >= last
        if readers and computers:
            overlap += stamp-last
        if kind == 1:
            readers += 1
        elif kind == 2:
            readers -= 1
        elif kind == 3:
            computers += 1
        elif kind == 4:
            computers -= 1
        assert readers >= 0 and computers >= 0
        last = stamp
    assert readers == computers == 0
    return overlap


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', action='store_true')
    p.add_argument('--scenario', choices=['dense', 'moe', 'controlled'])
    p.add_argument('--mode', choices=['sync', 'pipeline'])
    p.add_argument('--samples', type=int, default=5)
    p.add_argument('--tile-elements', type=int, default=32)
    p.add_argument('--read-elements', type=int, default=16)
    p.add_argument('--in-flight', type=int, default=2)
    p.add_argument('--io-workers', type=int, default=1)
    p.add_argument('--regions', type=int, default=48)
    p.add_argument('--read-delay', type=float, default=.004)
    p.add_argument('--compute-repeats', type=int, default=512)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if min(args.samples, args.regions, args.compute_repeats) < 1 or args.read_delay < 0:
        p.error('positive sample/region/repeat counts and nonnegative read delay required')
    if args.worker:
        print(json.dumps(worker(args), allow_nan=False))
        return
    evidence = {'scope': 'CPU pipeline: BF16 fixture correctness and controlled overlap',
                'python': sys.version, 'platform': platform.platform(),
                'cache': 'warm after mandatory SHA256 scan; no eviction or direct I/O',
                'power_thermal': 'not pinned; sysfs observations where exposed; WSL may hide sensors',
                'controlled_read_delay_is_synthetic': True,
                'controlled_compute': 'fixed count of real scalar multiply-add operations',
                'tokens': TOKENS,
                'tolerances_unchanged': TOLERANCES,
                'options': {k: v for k, v in vars(args).items()
                            if k not in ('output', 'worker', 'mode', 'scenario')},
                'execution_order': [], 'cases': {}}
    references = {}
    for variant in ('dense', 'moe'):
        result = subprocess.run([sys.executable, '-m', 'tools.verify_increment2', '--worker',
                                 'reference', '--variant', variant], cwd=ROOT, check=True,
                                capture_output=True, text=True, timeout=60)
        references[variant] = json.loads(result.stdout)['outputs']
    # Launch all measurements before importing NumPy in the parent, avoiding
    # inherited RSS high-water effects. Alternate order to reduce drift bias.
    for scenario in ('dense', 'moe', 'controlled'):
        cases = {mode: [] for mode in ('sync', 'pipeline')}
        for repetition in range(args.samples):
            modes = ('sync', 'pipeline') if repetition % 2 == 0 else ('pipeline', 'sync')
            for mode in modes:
                command = [sys.executable, '-m', 'tools.verify_increment3', '--worker',
                           '--scenario', scenario, '--mode', mode]
                for key in ('tile_elements', 'read_elements', 'in_flight', 'io_workers',
                            'regions', 'read_delay', 'compute_repeats'):
                    command.extend(['--'+key.replace('_', '-'), str(getattr(args, key))])
                result = subprocess.run(command, cwd=ROOT, check=True, capture_output=True,
                                        text=True, timeout=120)
                cases[mode].append(json.loads(result.stdout))
                evidence['execution_order'].append([scenario, repetition, mode])
        evidence['cases'][scenario] = cases
    for scenario, cases in evidence['cases'].items():
        for mode, runs in cases.items():
            for i, run in enumerate(runs):
                if scenario != 'controlled':
                    run['reference_max_absolute_errors'] = compare(run['outputs'], references[scenario])
                    run['sync_max_absolute_errors'] = compare(run['outputs'], cases['sync'][i]['outputs'])
                    assert run['outputs'] == cases['sync'][i]['outputs']
                else:
                    assert run['controlled_checksum'] == cases['sync'][i]['controlled_checksum']
                    assert run['stats']['trace_dropped'] == 0
                    assert verify_trace(run['trace']) == run['stats']['overlap_ns']
                if mode == 'sync':
                    assert run['stats']['overlap_ns'] == 0
                assert run['stats']['inference_read_bytes'] == cases['sync'][i]['stats']['inference_read_bytes']
        # Drop caller-owned exports after all comparisons; retain raw traces.
        for runs in cases.values():
            for run in runs:
                del run['outputs']
        sync = statistics.median(r['wall_ns'] for r in cases['sync'])
        pipeline = statistics.median(r['wall_ns'] for r in cases['pipeline'])
        cases['summary'] = {'sync_median_ns': sync, 'pipeline_median_ns': pipeline,
                            'sync_over_pipeline_wall_ratio': sync/pipeline,
                            'pipeline_wall_change_percent': (pipeline/sync-1)*100}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as f:
            json.dump(evidence, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')
    print(json.dumps({k: v['summary'] for k, v in evidence['cases'].items()}, indent=2))


if __name__ == '__main__':
    main()

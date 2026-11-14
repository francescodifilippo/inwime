"""Review-only independent PID/limit observation around the unchanged benchmark."""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def worker(mode):
    from types import SimpleNamespace
    from tools import benchmark_increment8 as benchmark
    checkpoint = Path('checkpoints/smollm2-1.7b-d4c18cc').resolve()
    expected = 128*2**20
    opens = {}
    path_open, os_open = Path.open, os.open

    def check(path):
        if isinstance(path, (str, os.PathLike)):
            path = Path(path).absolute()
            if path.is_relative_to(checkpoint):
                limits = resource.getrlimit(resource.RLIMIT_AS)
                if limits != (expected, expected):
                    raise AssertionError(('checkpoint open outside required AS limit', path, limits))
                opens[str(path.relative_to(checkpoint))] = list(limits)

    def checked_path_open(path, *args, **kwargs):
        check(path)
        return path_open(path, *args, **kwargs)

    def checked_os_open(path, *args, **kwargs):
        check(path)
        return os_open(path, *args, **kwargs)

    Path.open, os.open = checked_path_open, checked_os_open
    directory = Path('results/increment8/review')
    benchmark.run(SimpleNamespace(checkpoint=checkpoint, mode=mode, as_mib=128,
        capacity=16, tile_elements=262144, text='The capital of France is',
        scenario='decode', sequence_report=None, capture=None,
        output=directory/f'memory-{mode}.json'))
    Path.open, os.open = path_open, os_open
    output = dict(pid=os.getpid(), parent_pid=os.getppid(), rlimit_as=list(resource.getrlimit(resource.RLIMIT_AS)),
                  checked_checkpoint_opens=opens, all_opens_under_limit=bool(opens))
    with (directory/f'worker-{mode}.json').open('x') as f:
        json.dump(output, f, indent=2); f.write('\n')


def parent(mode):
    from tools.supervise8 import status
    directory = Path('results/increment8/review')
    env = dict(os.environ, TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    command = ['.venv-oracle7/bin/python', '-m', 'results.increment8.review.check_memory', '--worker', mode]
    peaks, limits_seen, samples = {}, set(), 0
    with (directory/f'memory-{mode}.log').open('x') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        child_pid = child.pid
        start_ticks = None
        while child.poll() is None:
            try:
                sample = status(child_pid)
                stat = Path(f'/proc/{child_pid}/stat').read_text()
                if start_ticks is None:
                    start_ticks = stat.rsplit(')', 1)[1].split()[19]
                limit = next(s for s in Path(f'/proc/{child_pid}/limits').read_text().splitlines()
                             if s.startswith('Max address space'))
            except FileNotFoundError:
                continue
            limits_seen.add(limit)
            for key, value in sample.items():
                peaks[key] = max(peaks.get(key, 0), value)
            samples += 1
            time.sleep(.02)
        code = child.wait()
    if code:
        raise RuntimeError(('memory review worker failed', mode, code))
    observed = json.loads((directory/f'worker-{mode}.json').read_text())
    run = json.loads((directory/f'memory-{mode}.json').read_text())
    assert observed['pid'] == child_pid and observed['parent_pid'] == os.getpid()
    assert observed['all_opens_under_limit'] and observed['rlimit_as'] == [134217728]*2
    assert any('134217728' in line for line in limits_seen)
    assert peaks['VmSize'] <= 134217728
    assert run['coverage_exact'] and len(run['phases']) == 4
    report = dict(parent_pid=os.getpid(), child_pid=child_pid, child_start_ticks=start_ticks,
                  worker_pid_matched=True, observed_limits=sorted(limits_seen), samples=samples,
                  peaks_bytes=peaks, exit_code=code, checkpoint_opens=observed['checked_checkpoint_opens'],
                  all_checkpoint_opens_under_limit=True, full_prefill_and_three_decode=True,
                  attribution='Only /proc/<Popen.pid> and child RUSAGE_SELF; oracle not launched',
                  timing_limitation='Review instrumentation, excluded from original benchmark medians')
    with (directory/f'memory-{mode}-audit.json').open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(mode, 'PID/AS audit PASS', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', choices=['sync', 'pipeline'])
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        for mode in ('sync', 'pipeline'):
            parent(mode)

"""Read-only /proc sampler and per-child watchdog; never changes system limits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def status(pid):
    data = {}
    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
        key, _, value = line.partition(':')
        if key.startswith('Vm'):
            data[key] = int(value.split()[0])*1024
    return data


def supervise(command, output, rss_stop, available_stop=0):
    env = dict(os.environ, TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='2')
    began = time.monotonic()
    peaks, samples, minimum = {}, 0, None
    reason = None
    with output.with_suffix('.log').open('x') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                try:
                    sample = status(child.pid)
                except FileNotFoundError:
                    break
                samples += 1
                for k, v in sample.items():
                    peaks[k] = max(peaks.get(k, 0), v)
                available = next(int(s.split()[1])*1024 for s in
                                 Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
                minimum = available if minimum is None else min(minimum, available)
                if sample.get('VmRSS', 0) > rss_stop:
                    reason = 'RSS watchdog threshold exceeded'
                if available < available_stop:
                    reason = 'MemAvailable watchdog threshold crossed'
                if reason:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                    break
                time.sleep(.02)
        finally:
            if child.poll() is None:
                child.kill()
            code = child.wait()
    report = dict(command=command, exit_code=code, watchdog_reason=reason,
                  rss_stop_bytes=rss_stop, memavailable_stop_bytes=available_stop,
                  sample_interval_s=.02, samples=samples, sampled_peaks_bytes=peaks,
                  minimum_memavailable_bytes=minimum, wall_s=time.monotonic()-began,
                  limitation='20 ms sampling can miss transients; kernel VmHWM/ru_maxrss reported by child separately')
    with output.open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    print(json.dumps(report, indent=2))
    return code


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--rss-stop-mib', type=int, default=1024)
    p.add_argument('--available-stop-mib', type=int, default=0)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if not a.command:
        p.error('child command required')
    raise SystemExit(supervise(a.command, a.output, a.rss_stop_mib*2**20, a.available_stop_mib*2**20))

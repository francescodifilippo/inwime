#!/usr/bin/env python3
"""Bounded, resumable V2 sweep; Python standard library and existing nvidia-smi only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
MIB = 1024**2


def proc_status(pid):
    try:
        values = {}
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            key, _, val = line.partition(':')
            if key in ['VmRSS', 'VmHWM', 'VmSwap', 'VmPin', 'VmLck']:
                values[key] = int(val.split()[0]) * 1024
        return values
    except FileNotFoundError:
        return {}


def vmstat():
    data = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    return {k: int(data[k]) for k in ['pswpin', 'pswpout']}


def run_case(args, env, config, directory, executable_sha):
    n, mib, cache, mode = config
    name = f'n{n}-b{mib}-c{cache}-{mode}'
    base = directory / name
    complete = base.with_suffix('.json')
    if complete.exists():
        record = json.loads(complete.read_text())
        if record.get('executable_sha256') != executable_sha or record['repeats'] != args.repeats:
            raise RuntimeError(f'{name}: existing result from a different binary/repetition count')
        return
    total = args.file.stat().st_size
    block = mib*MIB
    if total % block:
        raise ValueError('dataset must contain an integral number of blocks')
    command = [str(ROOT/'build/bf16_stream'), '--file', str(args.file), '--m', str(block//8192),
               '--k', '4096', '--n', str(n), '--blocks', str(total//block), '--cache-mib', str(cache),
               '--mode', mode, '--threads', str(args.threads), '--repeats', str(args.repeats)]
    record = {'command': command, 'repeats': args.repeats, 'executable_sha256': executable_sha,
              'dataset_bytes': total, 'dataset_stat': list(args.file.stat())[:10],
              'start_time': time.time(), 'swap_before': vmstat()}
    peaks = {key: 0 for key in ['VmRSS', 'VmHWM', 'VmSwap', 'VmPin', 'VmLck']}
    with base.with_suffix('.csv').open('w') as output, base.with_suffix('.log').open('w') as log:
        p = subprocess.Popen(command, env=env, stdout=output, stderr=log)
        with base.with_suffix('.memory.csv').open('w') as memory:
            writer = csv.DictWriter(memory, fieldnames=['time', *peaks])
            writer.writeheader()
            while p.poll() is None:
                sample = proc_status(p.pid)
                writer.writerow({'time': time.time(), **sample})
                for key in peaks:
                    peaks[key] = max(peaks[key], sample.get(key, 0))
                if peaks['VmSwap']:
                    p.terminate()
                    p.wait()
                    raise RuntimeError(f'{name}: process swap detected; stopped without changing swap settings')
                time.sleep(0.1)
        if p.returncode:
            raise RuntimeError(f'{name}: benchmark exit {p.returncode}; see {base}.log')
    with base.with_suffix('.csv').open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != args.repeats or any(float(r['max_abs_error']) != 0 for r in rows):
        raise RuntimeError(f'{name}: missing repetitions or numerical error')
    record.update(end_time=time.time(), swap_after=vmstat(), process_peaks_bytes=peaks)
    complete.write_text(json.dumps(record, indent=2)+'\n')
    print(f'DONE {name}: {len(rows)} passes, peak RSS {max(int(r["rss_peak_bytes"]) for r in rows)/MIB:.1f} MiB', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, default=ROOT/'data/weights-5GiB.bf16')
    parser.add_argument('--output', type=Path, default=ROOT/'results/v2/sweep')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--stage', choices=['buffers', 'cache', 'all'], default='all')
    parser.add_argument('--buffers', nargs='+', type=int, choices=[4,8,16,32,64], default=[4,8,16,32,64])
    parser.add_argument('--modes', nargs='+', choices=['io','cpu','sync','async'], default=['io','cpu','sync','async'])
    parser.add_argument('--ns', nargs='+', type=int, choices=[1,4,16], default=[1,4,16])
    args = parser.parse_args()
    if args.repeats < 10:
        parser.error('V2 requires at least 10 repetitions')
    args.file = args.file.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib' + (':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    configurations = []
    if args.stage in ['buffers', 'all']:
        for mib in args.buffers:
            if 'io' in args.modes:
                configurations.append((1, mib, 0, 'io'))
            for n in args.ns:
                configurations += [(n, mib, 0, mode) for mode in ['cpu', 'sync', 'async'] if mode in args.modes]
    if args.stage in ['cache', 'all']:
        for n in args.ns:
            for cache in [64, 256]:
                configurations += [(n, 8, cache, mode) for mode in ['sync', 'async'] if mode in args.modes]
    random.Random(20261105).shuffle(configurations)
    sha = hashlib.sha256((ROOT/'build/bf16_stream').read_bytes()).hexdigest()
    print(f'{len(configurations)} configurations x {args.repeats} passes; dataset reused read-only', flush=True)
    telemetry = args.output / f'gpu-{int(time.time())}.csv'
    monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,temperature.gpu,power.draw,memory.used,clocks.current.sm,clocks.current.memory,pstate',
                                '--format=csv,nounits', '-lms', '200', '-f', str(telemetry)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    stop = threading.Event()

    def safety():
        # Read telemetry without changing GPU clocks/power/fan configuration.
        while not stop.wait(2):
            try:
                with telemetry.open() as f:
                    rows = list(csv.DictReader(f))
                if rows and float(rows[-1][' temperature.gpu']) >= 85:
                    print('WARNING: GPU >=85 C; interrupt sweep and investigate cooling', flush=True)
            except (OSError, ValueError, KeyError, TypeError):
                pass

    thread = threading.Thread(target=safety, daemon=True)
    thread.start()
    try:
        for config in configurations:
            run_case(args, env, config, args.output, sha)
    finally:
        stop.set()
        monitor.terminate()
        _, errors = monitor.communicate()
        if errors:
            print('nvidia-smi:', errors.decode(), flush=True)


if __name__ == '__main__':
    main()

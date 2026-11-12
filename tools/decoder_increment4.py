"""Complete one-token scalable fixture workload; output for sync/pipeline comparison."""
import argparse
import json
import resource
import time
from pathlib import Path

from inwime_runtime import CPURuntime
from tools.verify_increment2 import runtime_export, process_io, process_memory
from tools.verify_increment3 import delta, host_state


def run(path, mode):
    io0 = process_io()
    t0 = time.monotonic_ns()
    with CPURuntime(path, capacity=1, tile_elements=4096, read_elements=4096,
                    io_mode=mode) as engine:
        startup = time.monotonic_ns()-t0
        io1 = process_io()
        began = time.monotonic_ns()
        with engine.prefill([1]) as result:
            output = runtime_export(result)
        wall = time.monotonic_ns()-began
        stats, io2 = engine.stats(), process_io()
        return dict(path=str(path), mode=mode, output=output, stats=stats,
                    wall_ns=wall, startup_ns=startup,
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    process_memory=process_memory(), startup_io=delta(io1, io0),
                    workload_io=delta(io2, io1), host=host_state(),
                    cache='post-integrity; Linux buffered, host caches uncontrolled',
                    gpu_allocated_bytes=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--mode', choices=['sync', 'pipeline'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = run(args.path, args.mode)
    with args.output.open('x') as stream:
        json.dump(evidence, stream, indent=2)


if __name__ == '__main__':
    main()

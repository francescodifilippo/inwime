"""Large progressive scan and bounded CPU tensor workload (never token/s)."""
import argparse
from array import array
from contextlib import closing
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import struct
import time

from inwime_runtime.core import Catalog, Coordinator, Plan, CATEGORIES
from inwime_runtime.pipeline import PipelineCoordinator
from tools.checkpoint4 import audit
from tools.verify_increment2 import process_io, process_memory
from tools.verify_increment3 import delta, host_state


def scan(path, tensor, tile=262144, mode='sync', raw=False, cache='post-integrity',
         integrity='full', stride=1):
    before = audit(path)
    c = Catalog(path)
    # Tensor-access experiment, no decoder/output arenas proportional to vocab.
    slots = 2 if mode == 'pipeline' else 1
    arrays = {}
    for i in range(slots):
        suffix = str(i) if i else ''
        arrays['weights'+suffix] = ('weights', tile, 'd')
        arrays['staging'+suffix] = ('staging', tile*2, 'B')
    required = dict.fromkeys(CATEGORIES, 0)
    for category, count, code in arrays.values():
        required[category] += count*array(code).itemsize
    required['metadata'] = c.bootstrap_bytes+65536
    required['requests'] = 32768 if slots == 2 else 0
    p = Plan(arrays, required, 1, tile, tile, 2 if slots == 2 else 0, 1 if slots == 2 else 0)
    control = Coordinator if mode == 'sync' else PipelineCoordinator
    io0 = process_io()
    t0 = time.monotonic_ns()
    with closing(control(c, p, integrity=integrity)) as owner:
        startup = time.monotonic_ns()-t0
        io1 = process_io()
        if cache == 'advised-cold':
            for fd in owner.fds.values():
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        began = time.monotonic_ns()
        count = c.tensors[tensor].elements
        checksum = 0.
        visited = logical = 0
        digest = hashlib.sha256()
        if raw:
            # Full sequential read and SHA processing of one logical tensor;
            # no BF16 expansion of the full tensor. Independent digest below.
            owner.logical_requested_bytes += count*2
            for shard, offset, n, _ in c.spans(tensor, 0, count, tile):
                owner.check_cancelled()
                data = owner._read(shard, offset, n*2)
                with owner.compute():
                    digest.update(data)
                logical += n*2
                owner.tensor_bytes[tensor] += n*2
            visited = count
        else:
            regions = ((tensor, start, min(tile, count-start))
                       for start in range(0, count, tile*stride))
            with closing(owner.stream(regions)) as stream:
                for start, n, lease in stream:
                    # Every expanded value checked against a closed-form oracle.
                    for j in range(n):
                        i = (start+j) % 256
                        expected = (-1 if i >= 128 else 1)*(1+(i % 128)/128)/16
                        value = lease.value(j)
                        assert value == expected
                        checksum += value
                    visited += n
                    logical += n*2
        wall = time.monotonic_ns()-began
        io2 = process_io()
        if raw:
            independent = hashlib.sha256()
            # Independent IEEE BF16 encoding, period generated via float32 bits.
            period = b''.join(struct.pack('<I', struct.unpack('<I', struct.pack('<f',
                (-1 if i >= 128 else 1)*(1+(i % 128)/128)/16))[0])[2:]
                              for i in range(256))
            chunk = period*2048
            for start in range(0, count*2, len(chunk)):
                independent.update(chunk[:min(len(chunk), count*2-start)])
            assert digest.digest() == independent.digest()

        stats = owner.stats()
        assert not owner.active
        result = dict(checkpoint_payload_bytes=c.payload_bytes,
                      tensor_bytes=count*2, visited_bytes=visited*2,
                      logical_requested_bytes=logical, stats=stats,
                      wall_ns=wall, startup_ns=startup, checksum=checksum,
                      sha256=digest.hexdigest() if raw else None,
                      startup_process_io=delta(io1, io0), workload_process_io=delta(io2, io1),
                      audit_before=before, audit_after=audit(path),
                      process_memory=process_memory(),
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      allocated_payload_bytes=sum((Path(path)/f).stat().st_blocks*512 for f in c.files),
                      tile_elements=tile, in_flight=slots, mode=mode, raw=raw, stride=stride,
                      cache=("uncontrolled" if cache == "post-integrity" and integrity == "metadata" else cache), cache_control='Linux advisory only; host/device caches uncontrolled',
                      gpu_allocated_bytes=0, host=host_state())
        result['owner_phase_remainder_ns'] = wall-stats['data_wait_ns']-stats['bf16_expand_ns']-stats['compute_ns']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('path', type=Path)
    p.add_argument('--tensor', default='embedding')
    p.add_argument('--tile', type=int, default=262144)
    p.add_argument('--mode', choices=['sync','pipeline'], default='sync')
    p.add_argument('--raw', action='store_true')
    p.add_argument('--cache', choices=['post-integrity','advised-cold'], default='post-integrity')
    p.add_argument('--integrity', choices=['full','metadata'], default='full')
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    args = vars(p.parse_args())
    output = args.pop('output')
    if args['tile'] <= 0 or args['stride'] <= 0:
        p.error('positive tile and stride required')
    if args['raw'] and (args['mode'] != 'sync' or args['stride'] != 1):
        p.error('raw full scan uses sync stride=1')
    result = scan(**args)
    with output.open('x') as f:
        json.dump(result, f, indent=2)


if __name__ == '__main__':
    main()

"""Bounded source/GEMV probe: 4 MiB original BF16, all rows checked with fsum."""
import argparse
from array import array
import hashlib
import json
import math
from pathlib import Path
import resource
import struct
import time

from inwime_runtime.gemv import StreamedGEMV
from inwime_runtime.math_cpu import NativeMath
from tools.benchmark_increment6 import memory
from tools.verify_increment2 import process_io
from tools.verify_increment3 import delta, host_state

ROWS, COLUMNS = 513, 4099


def generate(path):
    path.mkdir(parents=True, exist_ok=False)
    raw = json.dumps({'matrix': dict(dtype='BF16', shape=[ROWS, COLUMNS],
                     data_offsets=[0, ROWS*COLUMNS*2])}, separators=(',', ':')).encode()
    raw += b' '*((-len(raw)) % 8)
    period = b''.join(struct.pack('<H', (0x8000 if i & 128 else 0) | 0x3d80 | (i & 127))
                      for i in range(256))
    digest = hashlib.sha256()
    name = 'model.safetensors'
    with (path/name).open('xb') as f:
        for data in (struct.pack('<Q', len(raw)), raw):
            f.write(data)
            digest.update(data)
        for start in range(0, ROWS*COLUMNS, 32768):
            number = min(32768, ROWS*COLUMNS-start)
            offset = start % 256
            data = (period*((offset+number+255)//256))[offset*2:(offset+number)*2]
            f.write(data)
            digest.update(data)
    descriptor = dict(format='inwime-safetensors-bf16-v1', index=None,
                      files={name: dict(nbytes=8+len(raw)+ROWS*COLUMNS*2,
                                        sha256=digest.hexdigest())})
    (path/'safetensors.json').write_text(json.dumps(descriptor, indent=2)+'\n')


def probe(args):
    resource.setrlimit(resource.RLIMIT_AS, (512*2**20, 512*2**20))
    native = NativeMath(Path('build/libinwime_cpu.so').resolve())
    source = array('d', [(i%97-48)/97 for i in range(COLUMNS)])
    output = array('d', [0.])*ROWS
    io0, t0 = process_io(), time.monotonic_ns()
    with StreamedGEMV(args.path, source='safetensors', io_mode=args.mode,
                      tile_elements=8191, read_elements=1021, math_backend=native) as op:
        startup, io1 = time.monotonic_ns()-t0, process_io()
        t0 = time.monotonic_ns()
        op.run('matrix', source, output)
        wall, io2 = time.monotonic_ns()-t0, process_io()
        stats = op.stats()
        bootstrap, retained = op.control.catalog.bootstrap_bytes, op.control.catalog.retained_bytes
    assert stats['inference_read_bytes'] == ROWS*COLUMNS*2
    assert stats['max_backend_read_bytes'] <= 2042
    assert stats['reserved_peak_bytes'] < 128*2**20
    assert stats['audited_retained_metadata_bytes'] <= stats['required_by_category']['metadata']
    assert not op.control.fds and not op.control.buffers
    # Independent addressing: fixture layout and header length read here, without
    # the new parser, Catalog.spans, expansion helper or GEMV implementation.
    t0, io3 = time.monotonic_ns(), process_io()
    max_error = 0.
    with (args.path/'model.safetensors').open('rb') as f:
        header_size, = struct.unpack('<Q', f.read(8))
        f.seek(8+header_size)
        for row in range(ROWS):
            raw = f.read(COLUMNS*2)
            weights = (struct.unpack('<f', struct.pack('<I', w << 16))[0]
                       for w, in struct.iter_unpack('<H', raw))
            expected = math.fsum(w*x for w, x in zip(weights, source))
            error = abs(output[row]-expected)
            assert error <= 1e-12+1e-12*abs(expected), (row, error)
            max_error = max(max_error, error)
        assert not f.read(1)
    validation_ns, io4 = time.monotonic_ns()-t0, process_io()
    with (args.path/'model.safetensors').open('rb') as f:
        after_sha = hashlib.file_digest(f, 'sha256').hexdigest()
    expected_sha = json.loads((args.path/'safetensors.json').read_text())['files']['model.safetensors']['sha256']
    assert after_sha == expected_sha
    report = dict(mode=args.mode, rows=ROWS, columns=COLUMNS, all_rows_validated=ROWS,
                  reference='independent IEEE BF16 expansion and math.fsum, every row',
                  reference_max_abs_error=max_error, startup_ns=startup, wall_ns=wall,
                  validation_ns=validation_ns, stats=stats, catalog_bootstrap_bytes=bootstrap,
                  catalog_retained_bytes=retained, caller_numeric_bytes=(ROWS+COLUMNS)*8,
                  oracle_max_row_raw_bytes=COLUMNS*2,
                  startup_process_io=delta(io1, io0), workload_process_io=delta(io2, io1),
                  validation_process_io=delta(io4, io3),
                  post_validation_hash_read_bytes=(args.path/'model.safetensors').stat().st_size,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  process_memory=memory(), host_state=host_state(), gpu_allocated_bytes=0,
                  checkpoint_sha256=after_sha, output_sha256=hashlib.sha256(output).hexdigest(),
                  cache='cache-warm after full integrity scan; no eviction')
    args.output.write_text(json.dumps(report, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('path', type=Path)
    p.add_argument('--generate', action='store_true')
    p.add_argument('--mode', choices=['sync', 'pipeline'], default='sync')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.generate:
        generate(args.path)
    else:
        if args.output is None or args.output.exists():
            p.error('new --output required')
        probe(args)


if __name__ == '__main__':
    main()

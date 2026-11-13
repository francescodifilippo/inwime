"""Complete inference/GEMV evidence in a fresh, optionally address-limited process.

Use independent small oracle tests first. Large validation checks selected complete
rows after inference, using independent segment addressing and BF16 decoding.
It is deliberately NOT a full independent large-decoder oracle.
"""
import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import struct
import time

from inwime_runtime import CPURuntime
from inwime_runtime.core import Catalog
from inwime_runtime.gemv import StreamedGEMV
from inwime_runtime.math_cpu import NativeMath
from tools.checkpoint4 import audit
from tools.verify_increment2 import process_io
from tools.verify_increment3 import delta, host_state


def process_memory():
    return {k: int(v.split()[0])*1024
            for k, v in (line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
            if k in ('VmRSS', 'VmHWM', 'VmSwap', 'VmSize')}


def independent_row(path, manifest, name, row):
    """No Catalog.spans, math adapter, runtime expansion or reference import."""
    d = manifest['tensors'][name]
    columns = d['shape'][1]
    low, high = row*columns, (row+1)*columns
    values = array('d')
    read_bytes = 0
    descriptors = d.get('segments', [dict(start=0, shard=d.get('shard'),
                                         offset=d.get('offset'), nbytes=d['nbytes'])])
    for s in descriptors:
        left, right = max(low, s['start']), min(high, s['start']+s['nbytes']//2)
        if left >= right:
            continue
        with (Path(path)/s['shard']).open('rb') as stream:
            stream.seek(s['offset']+(left-s['start'])*2)
            raw = stream.read((right-left)*2)
        if len(raw) != (right-left)*2:
            raise OSError('short independent validation read')
        read_bytes += len(raw)
        for word, in struct.iter_unpack('<H', raw):
            values.append(struct.unpack('<f', struct.pack('<I', word << 16))[0])
    if len(values) != columns:
        raise AssertionError('independent row incomplete')
    return values, read_bytes


def validate_rows(path, snapshots):
    manifest = json.loads((Path(path)/'manifest.json').read_text())
    began = time.monotonic_ns()
    records, total_bytes = [], 0
    max_abs = max_bound_ratio = 0.
    for name, source, samples in snapshots:
        for row, actual in samples:
            weights, size = independent_row(path, manifest, name, row)
            expected = math.fsum(w*x for w, x in zip(weights, source))
            magnitude = math.fsum(abs(w*x) for w, x in zip(weights, source))
            # Standard sequential FP64 dot-product bound, fixed before execution.
            unit = 2.**-53
            gamma = (2*len(source)*unit)/(1-2*len(source)*unit)
            bound = gamma*magnitude+4*unit*abs(expected)+1e-300
            error = abs(actual-expected)
            if not (math.isfinite(actual) and math.isfinite(expected) and error <= bound):
                raise AssertionError((name, row, actual, expected, error, bound))
            total_bytes += size
            max_abs = max(max_abs, error)
            max_bound_ratio = max(max_bound_ratio, error/bound)
            records.append(dict(tensor=name, row=row, actual=actual, reference=expected,
                                absolute_error=error, error_bound=bound))
    return dict(scope='selected full dot products, not a full large-decoder oracle',
                rows=len(records), tensor_count=len(snapshots), read_bytes=total_bytes,
                wall_ns=time.monotonic_ns()-began, max_absolute_error=max_abs,
                max_error_bound_ratio=max_bound_ratio, checks=records)


class ObservedRuntime(CPURuntime):
    """Caller diagnostics: bounded first-invocation samples, outside engine arenas."""
    def __init__(self, *args, **kwargs):
        self.snapshots, self.seen = [], set()
        self.snapshot_copy_ns = 0
        super().__init__(*args, **kwargs)

    def _linear(self, name, source, destination, offset=0):
        super()._linear(name, source, destination, offset)
        if name not in self.seen:
            began = time.monotonic_ns()
            rows = self.catalog.tensors[name].shape[0]
            sample = [(row, destination[offset+row]) for row in sorted({0, rows//2, rows-1})]
            self.snapshots.append((name, array('d', source), sample))
            self.seen.add(name)
            self.snapshot_copy_ns += time.monotonic_ns()-began


def advise_drop(control, cache):
    if cache == 'advised-cold':
        for fd in control.fds.values():
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def summarize_output(engine, result, previous):
    c = engine.spec
    logits = result.logits
    start = (result.count-1)*c.vocab
    next_token = max(range(c.vocab), key=lambda i: (logits[start+i], -i))
    kv_hashes, preserved = [], True
    for layer, parts in enumerate(result.kv):
        pair = []
        for kind, view in enumerate(parts):
            if previous is not None:
                old_length, old = previous
                preserved &= hashlib.sha256(view[:old_length*c.width]).hexdigest() == old[layer][kind]
            pair.append(hashlib.sha256(view).hexdigest())
        kv_hashes.append(pair)
    if not preserved:
        raise AssertionError('published KV prefix changed')
    return dict(length=engine.length, generated_token=next_token,
                logits_sha256=hashlib.sha256(logits).hexdigest(),
                logits_last=list(logits[start:start+c.vocab]), kv_sha256=kv_hashes,
                kv_prefix_preserved=preserved,
                hidden_sha256=[hashlib.sha256(view).hexdigest() for view in result.hidden])


def metrics(stats, wall):
    return dict(effective_weight_GB_s=stats['inference_read_bytes']/wall,
                effective_GFLOP_s=stats['gemv_flops']/wall,
                kernel_GFLOP_s=stats['gemv_flops']/max(1, stats['gemv_ns']),
                owner_phase_remainder_ns=wall-stats['data_wait_ns']-stats['bf16_expand_ns']-stats['compute_ns'])


def decoder(path, mode, native, tile, read, integrity, cache, decode_steps):
    if decode_steps < 2:
        raise ValueError('evidence requires at least two decode calls')
    t0, io0 = time.monotonic_ns(), process_io()
    engine = ObservedRuntime(path, capacity=2+decode_steps, tile_elements=tile,
                             read_elements=read, io_mode=mode, math_backend=native,
                             integrity=integrity)
    startup = time.monotonic_ns()-t0
    io1 = process_io()
    try:
        advise_drop(engine.control, cache)
        prompt, phases, consumed = [1, 7], [], [1, 7]
        old = None
        began = time.monotonic_ns()
        for step in range(1+decode_steps):
            t = time.monotonic_ns()
            result = engine.prefill(prompt) if step == 0 else engine.decode(next_token)
            forward = time.monotonic_ns()-t
            with result:
                output = summarize_output(engine, result, old)
            phases.append(dict(kind='prefill' if step == 0 else 'decode', forward_ns=forward,
                               output=output, process_memory=process_memory()))
            if step:
                consumed.append(next_token)
            next_token = output['generated_token']
            old = engine.length, output['kv_sha256']
            print(f'{mode}: {phases[-1]["kind"]} length={engine.length} '
                  f'{forward/1e9:.3f}s next={next_token}', file=__import__('sys').stderr, flush=True)
        wall = time.monotonic_ns()-began
        io2, stats = process_io(), engine.stats()
        matrix_bytes = sum(t.elements*2 for n, t in engine.catalog.tensors.items()
                           if len(t.shape) == 2 and n not in ('embedding', 'position'))
        norm_bytes = sum(t.elements*2 for n, t in engine.catalog.tensors.items() if 'norm' in n)
        unique = matrix_bytes+norm_bytes+2*engine.spec.width*(len(set(consumed))+len(consumed))
        expected_per_tensor = {n: (engine.spec.width*2 if n in ('embedding', 'position')
                                    else t.elements*2)*len(consumed)
                               for n, t in engine.catalog.tensors.items()}
        assert stats['tensor_read_bytes'] == expected_per_tensor
        assert stats['gemv_weight_elements']*2 == matrix_bytes*len(consumed)
        assert not stats['active_region'] and stats['requests'] == stats['completed'] == stats['released']
        assert not stats['session_failed'] and stats['failed'] == 0
        memory_before_close = process_memory()
    finally:
        engine.close()
    closed = dict(fds=len(engine.control.fds), numeric_buffers=len(engine.control.buffers),
                  live_requests=len([r for r in getattr(engine.control, 'slots', []) if r is not None]))
    assert closed == dict(fds=0, numeric_buffers=0, live_requests=0)
    validation_io0 = process_io()
    validation = validate_rows(path, engine.snapshots)
    validation['process_io'] = delta(process_io(), validation_io0)
    decode_ns = [p['forward_ns'] for p in phases[1:]]
    return dict(kind='complete-decoder', mode=mode, prompt=prompt, consumed_tokens=consumed,
                generated_tokens=[p['output']['generated_token'] for p in phases],
                prefill_ns=phases[0]['forward_ns'], decode_ns=decode_ns,
                decode_tokens_s=len(decode_ns)*1e9/sum(decode_ns),
                generation_tokens_s_including_prefill=len(phases)*1e9/wall,
                wall_ns=wall, startup_ns=startup, startup_process_io=delta(io1, io0),
                workload_process_io=delta(io2, io1), stats=stats, phases=phases,
                unique_forward_weight_bytes=unique, full_matrix_bytes_per_token=matrix_bytes,
                sampled_source_copy_bytes=sum(len(s)*8 for _, s, _ in engine.snapshots),
                snapshot_copy_ns=engine.snapshot_copy_ns, memory_before_close=memory_before_close,
                resources_after_close=closed, independent_validation=validation,
                **metrics(stats, wall))


def gemv(path, tensor, mode, native, tile, read, integrity, cache):
    catalog = Catalog(path)
    rows, columns = catalog.tensors[tensor].shape
    source = array('d', [((i*17)%97-48)/97 for i in range(columns)])
    output = array('d', [0.])*rows
    io0, t0 = process_io(), time.monotonic_ns()
    with StreamedGEMV(path, io_mode=mode, tile_elements=tile, read_elements=read,
                      math_backend=native, integrity=integrity) as op:
        startup, io1 = time.monotonic_ns()-t0, process_io()
        advise_drop(op.control, cache)
        began = time.monotonic_ns()
        op.run(tensor, source, output)
        wall = time.monotonic_ns()-began
        stats, io2 = op.stats(), process_io()
        samples = [(r, output[r]) for r in sorted({0, rows//3, rows//2, rows-1})]
    assert stats['gemv_weight_elements'] == rows*columns
    validation = validate_rows(path, [(tensor, source, samples)])
    return dict(kind='complete-gemv', tensor=tensor, mode=mode, shape=[rows, columns],
                wall_ns=wall, startup_ns=startup, startup_process_io=delta(io1, io0),
                workload_process_io=delta(io2, io1), stats=stats,
                caller_numeric_bytes=(rows+columns)*8, independent_validation=validation,
                output_sha256=hashlib.sha256(output).hexdigest(), **metrics(stats, wall))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--mode', choices=['sync', 'pipeline'], default='sync')
    parser.add_argument('--tensor', help='run standalone GEMV instead of decoder')
    parser.add_argument('--tile', type=int, default=262144)
    parser.add_argument('--read', type=int, default=262144)
    parser.add_argument('--integrity', choices=['full', 'metadata'], default='full')
    parser.add_argument('--cache', choices=['post-integrity', 'advised-cold'], default='advised-cold')
    parser.add_argument('--decode-steps', type=int, default=3)
    parser.add_argument('--address-limit-mib', type=int, default=512)
    parser.add_argument('--library', type=Path, default=Path('build/libinwime_cpu.so'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.address_limit_mib:
        limit = args.address_limit_mib*2**20
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    before = audit(args.path)
    native = NativeMath(args.library.resolve())
    options = (args.path, args.mode, native, args.tile, args.read, args.integrity, args.cache)
    result = (gemv(args.path, args.tensor, *options[1:]) if args.tensor
              else decoder(*options, args.decode_steps))
    catalog = Catalog(args.path)
    result.update(checkpoint_identity=catalog.identity, payload_bytes=catalog.payload_bytes,
                  allocated_payload_bytes=sum((args.path/n).stat().st_blocks*512 for n in catalog.files),
                  audit_before=before, audit_after=audit(args.path), host=host_state(),
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  process_memory=process_memory(), tile_elements=args.tile, read_elements=args.read,
                  cache=args.cache, cache_caveat='Linux buffered; DONTNEED advisory, Windows/device caches uncontrolled',
                  gpu_allocated_bytes=0, native_library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest())
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: result[k] for k in ('kind', 'mode', 'wall_ns', 'peak_rss_kib', 'payload_bytes')}))


if __name__ == '__main__':
    main()

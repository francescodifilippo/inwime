"""Bounded deterministic BF16 writer/repacker; no resident reference dependency."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import stat
import struct
import time

from inwime_runtime.core import Catalog, signature


def audit(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    memory = {line.split(':')[0]: int(line.split()[1])*1024
              for line in Path('/proc/meminfo').read_text().splitlines()}
    limits = {}
    group = Path('/sys/fs/cgroup')
    membership = Path('/proc/self/cgroup').read_text().strip()
    relative = next((line.split(':', 2)[2] for line in membership.splitlines()
                     if line.startswith('0::')), '/')
    current = group/relative.lstrip('/')
    while current.is_relative_to(group):
        for name in ('memory.max', 'memory.current', 'memory.swap.max'):
            p = current/name
            if p.exists():
                limits[str(p)] = p.read_text().strip()
        if current == group:
            break
        current = current.parent
    return dict(meminfo=memory, cgroup=limits, disk_free=shutil.disk_usage(path).free,
                filesystem=str(path.resolve()),
                rlimit_as=resource.getrlimit(resource.RLIMIT_AS),
                rlimit_nofile=resource.getrlimit(resource.RLIMIT_NOFILE))


def pattern(start, count):
    """Period 256: signed native BF16 (1 + (index % 128)/128)/16."""
    period = b''.join(struct.pack('<H', (0x8000 if i & 128 else 0) | 0x3d80 | (i & 127))
                      for i in range(256))
    offset = start % 256
    return (period*((offset+count+255)//256))[offset*2:(offset+count)*2]


def generate(output, variant='dense', *, vocab=31, hidden=24, experts=3,
             shard_bytes=64*1024*1024, segment_bytes=16*1024*1024, source=None,
             execute=False):
    if any(type(n) is not int or n <= 0 for n in (vocab, hidden, experts)):
        raise ValueError("positive decoder dimensions required")
    if variant == "moe" and experts < 2:
        raise ValueError("MoE requires at least top_k=2 experts")
    # Shape template only; checkpoint originals never modified.
    if experts > 10000:
        raise ValueError("expert descriptors exceed generator metadata policy")
    if source:
        catalog = Catalog(source)
    template = Path('fixtures/reference_decoder')/variant
    with ((Path(source) if source else template)/'manifest.json').open('rb') as manifest:
        raw_manifest = manifest.read(16*1024*1024+1)
    if len(raw_manifest) > 16*1024*1024:
        raise ValueError('manifest exceeds input bound')
    m = json.loads(raw_manifest)
    if source and hashlib.sha256(raw_manifest).hexdigest() != catalog.identity:
        raise ValueError('source manifest changed during repack')
    if not source:
        old = m['config'].copy()
        m['config'].update(vocab=vocab, hidden=hidden, experts=experts)
        # Dimension substitution on this adapter's known fixture shapes.
        tensors = m['tensors']
        if variant == 'moe':
            for layer in range(old['layers']):
                for e in range(experts):
                    for kind in ('gate', 'up', 'down'):
                        key = f'layers.{layer}.experts.{e}.{kind}'
                        tensors[key] = dict(tensors[f'layers.{layer}.experts.0.{kind}'])
                for e in range(experts, old['experts']):
                    for kind in ('gate', 'up', 'down'):
                        del tensors[f'layers.{layer}.experts.{e}.{kind}']
        for name, d in tensors.items():
            if name in ('embedding', 'lm_head'):
                d['shape'] = [vocab, old['width']]
            elif name.endswith(('gate', 'up')):
                d['shape'] = [hidden, old['width']]
            elif name.endswith('down'):
                d['shape'] = [old['width'], hidden]
            elif name.endswith('router'):
                d['shape'] = [experts, old['width']]
            d['nbytes'] = math.prod(d['shape'])*2
    if min(shard_bytes, segment_bytes) < 2 or shard_bytes % 2 or segment_bytes % 2:
        raise ValueError('even positive shard/segment sizes required')
    total = sum(d['nbytes'] for d in m['tensors'].values())
    # Conservative admission before payload allocation or writes. Each physical
    # boundary can add at most one segment; cap descriptor count independently.
    descriptor_bound = sum((d['nbytes']+segment_bytes-1)//segment_bytes
                           for d in m['tensors'].values()) + (total+shard_bytes-1)//shard_bytes
    if descriptor_bound > 32768:
        raise ValueError('segment descriptors exceed bounded generator policy')
    state = audit(output)
    estimate = dict(payload_bytes=total, audit=state, estimated_io_bytes=total*3,
                    time_estimate_seconds_at_100_MiB_s=total*3/(100*2**20),
                    writer_chunk_bytes=1024*1024, descriptor_bound=descriptor_bound,
                    metadata_estimate_bytes=descriptor_bound*1024, checkpoint_path=str(output))
    if not execute:
        return estimate
    if total+2*2**30 > state['disk_free']:
        raise ValueError('insufficient free disk including 2 GiB reserve')
    if total//shard_bytes+len(m['tensors']) > min(4096, state['rlimit_nofile'][0]-64):
        raise ValueError('too many shards for bounded current file-handle policy')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    m.update(format='inwime-segmented-bf16-v2', files={},
             generator='repack-original' if source else 'period256-native-bf16-v1')
    current = None
    source_fds = {}
    began = time.monotonic()
    try:
        if source:
            for n in catalog.files:
                source_fds[n] = os.open(Path(source)/n, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            stamps = {n: signature(fd) for n, fd in source_fds.items()}
            for n, fd in source_fds.items():
                h = hashlib.sha256()
                size, expected = catalog.files[n]
                source_stat = os.fstat(fd)
                if not stat.S_ISREG(source_stat.st_mode) or source_stat.st_size != size:
                    raise ValueError('source is not a regular file of the expected size')
                for offset_source in range(0, size, 1024*1024):
                    count = min(1024*1024, size-offset_source)
                    raw = os.pread(fd, count, offset_source)
                    if len(raw) != count:
                        raise OSError('short source integrity read')
                    h.update(raw)
                if signature(fd) != stamps[n] or h.hexdigest() != expected:
                    raise ValueError('source checksum or identity mismatch')
        for name, d in sorted(m['tensors'].items()):
            segments = []
            for start in range(0, d['nbytes']//2, segment_bytes//2):
                left = min(segment_bytes, d['nbytes']-start*2)
                cursor = start
                while left:
                    if current is None or offset == shard_bytes:
                        if current:
                            current.flush(); os.fsync(current.fileno()); current.close()
                            m['files'][filename] = dict(nbytes=offset, sha256=digest.hexdigest())
                        filename = f'part-{len(m["files"]):06d}.bf16.bin'
                        current = (output/filename).open('xb')
                        offset, digest = 0, hashlib.sha256()
                    size = min(left, shard_bytes-offset)
                    segments.append(dict(start=cursor, shard=filename, offset=offset, nbytes=size))
                    for done in range(0, size, 1024*1024):
                        n = min(1024*1024, size-done)
                        if source:
                            pieces = []
                            for shard, pos, count, _ in catalog.spans(name, cursor+done//2, n//2, n//2):
                                if signature(source_fds[shard]) != stamps[shard]:
                                    raise ValueError("source changed during repack")
                                raw = os.pread(source_fds[shard], count*2, pos)
                                if signature(source_fds[shard]) != stamps[shard]:
                                    raise ValueError("source changed during repack")
                                if len(raw) != count*2:
                                    raise OSError('short source read')
                                pieces.append(raw)
                            raw = b''.join(pieces)
                        elif 'norm' in name:
                            raw = b'\x80\x3f'*(n//2)
                        else:
                            raw = pattern(cursor+done//2, n//2)
                        current.write(raw); digest.update(raw)
                    offset += size; cursor += size//2; left -= size
            m['tensors'][name] = {k: d[k] for k in ('dtype', 'shape', 'nbytes')} | dict(segments=segments)
        current.flush(); os.fsync(current.fileno()); current.close()
        m['files'][filename] = dict(nbytes=offset, sha256=digest.hexdigest())
        with (output/'manifest.json').open('x') as f:
            json.dump(m, f, indent=2, sort_keys=True)
        Catalog(output)
    finally:
        if current:
            current.close()
        for fd in source_fds.values():
            os.close(fd)
    allocated = sum((output/n).stat().st_blocks*512 for n in m['files'])
    return estimate | dict(allocated_bytes=allocated, wall_seconds=time.monotonic()-began,
                           files=len(m['files']), peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--source', type=Path)
    p.add_argument('--variant', choices=('dense','moe'), default='dense')
    p.add_argument('--vocab', type=int, default=31)
    p.add_argument('--hidden', type=int, default=24)
    p.add_argument('--experts', type=int, default=3)
    p.add_argument('--shard-bytes', type=int, default=64*2**20)
    p.add_argument('--segment-bytes', type=int, default=16*2**20)
    p.add_argument('--execute', action='store_true', help='otherwise audit/estimate only')
    args = vars(p.parse_args())
    print(json.dumps(generate(**args), indent=2))


if __name__ == '__main__':
    main()

"""Dense scalable native-BF16 decoder. Audit only unless --execute is explicit.

No payload padding, sparse files, unused experts or dependency on the reference.
Writes at most one configurable chunk at once; never overwrites an existing path.
"""
import argparse
from array import array
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

from inwime_runtime.core import Catalog, Spec
from inwime_runtime.math_cpu import NativeMath
from tools.checkpoint4 import audit


def tensor_shapes(c):
    shapes = {'embedding': (c.vocab, c.width), 'position': (c.context, c.width),
              'lm_head': (c.vocab, c.width), 'final_norm': (c.width,)}
    for layer in range(c.layers):
        p = f'layers.{layer}.'
        for name in ('norm1', 'norm2'):
            shapes[p+name] = (c.width,)
        for name in ('q', 'k', 'v', 'o'):
            shapes[p+name] = (c.width, c.width)
        for name in ('gate', 'up'):
            shapes[p+'ffn.'+name] = (c.hidden, c.width)
        shapes[p+'ffn.down'] = (c.width, c.hidden)
    return dict(sorted(shapes.items()))


def generate(output, *, layers=24, width=4096, heads=32, hidden=16384, vocab=257,
             context=16, shard_bytes=64*2**20, segment_bytes=16*2**20,
             chunk_bytes=2**20, math_backend=None, execute=False):
    dimensions = (layers, width, heads, hidden, vocab, context)
    if (any(type(n) is not int or n <= 0 for n in dimensions) or width % heads
            or layers > 1024 or any(type(n) is not int or n < 2 or n % 2
                                   for n in (shard_bytes, segment_bytes, chunk_bytes))
            or chunk_bytes > 16*2**20):
        raise ValueError('invalid dimensions or bounded writer chunk')
    c = Spec(layers, width, heads, hidden, vocab, context, 1, 1, 1e-5)
    shapes = tensor_shapes(c)
    payload = sum(math.prod(shape)*2 for shape in shapes.values())
    descriptors = sum((math.prod(s)*2+segment_bytes-1)//segment_bytes for s in shapes.values())
    descriptors += (payload+shard_bytes-1)//shard_bytes
    state = audit(output)
    if descriptors > 32768 or (payload+shard_bytes-1)//shard_bytes > min(4096, state['rlimit_nofile'][0]-64):
        raise ValueError('checkpoint exceeds metadata/file-handle policy')
    if payload+2*2**30 > state['disk_free']:
        raise ValueError('insufficient filesystem space including 2 GiB reserve')
    estimate = dict(payload_bytes=payload, config=asdict(c), audit=state,
                    descriptor_bound=descriptors, writer_chunk_bytes=chunk_bytes,
                    checkpoint_path=str(output))
    if not execute:
        return estimate
    backend = NativeMath(Path('build/libinwime_cpu.so').resolve()) if math_backend is None else math_backend
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    m = dict(format='inwime-segmented-bf16-v2', byte_order='little', layout='C', variant='dense',
             config=asdict(c), generator='splitmix64-indexed-native-bf16-v1',
             generation={}, tensors={}, files={})
    raw = array('B', [0])*chunk_bytes
    current = None
    began = time.monotonic()
    try:
        for name, shape in shapes.items():
            elements = math.prod(shape)
            # Native BF16 scale follows fan-in; embeddings use smaller fixed amplitudes.
            exponent = (123 if name == 'embedding' else 121 if name == 'position'
                        else 126-((shape[-1]-1).bit_length()+1)//2)
            seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little')
            m['generation'][name] = dict(seed=seed, exponent=exponent, norm='norm' in name)
            segments = []
            cursor = 0
            while cursor < elements:
                if current is None or offset == shard_bytes:
                    if current:
                        current.flush()
                        os.fsync(current.fileno())
                        current.close()
                        m['files'][filename] = dict(nbytes=offset, sha256=digest.hexdigest())
                    filename = f'part-{len(m["files"]):06d}.bf16.bin'
                    current = (output/filename).open('xb')
                    offset, digest = 0, hashlib.sha256()
                number = min(elements-cursor, segment_bytes//2, (shard_bytes-offset)//2)
                segments.append(dict(start=cursor, shard=filename, offset=offset, nbytes=number*2))
                done = 0
                while done < number:
                    n = min(chunk_bytes//2, number-done)
                    backend.fill(raw, n, cursor+done, seed, exponent, 'norm' in name)
                    view = memoryview(raw)[:n*2]
                    if current.write(view) != n*2:
                        raise OSError('short checkpoint write')
                    digest.update(view)
                    done += n
                cursor += number
                offset += number*2
            m['tensors'][name] = dict(dtype='BF16', shape=list(shape), nbytes=elements*2,
                                       segments=segments)
        current.flush()
        os.fsync(current.fileno())
        current.close()
        m['files'][filename] = dict(nbytes=offset, sha256=digest.hexdigest())
        allocated = sum((output/n).stat().st_blocks*512 for n in m['files'])
        if any((output/n).stat().st_blocks*512 < d['nbytes'] for n, d in m['files'].items()):
            raise OSError('filesystem did not report fully allocated payload files')
        with (output/'manifest.json').open('x') as stream:
            json.dump(m, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        Catalog(output)
    finally:
        if current:
            current.close()
    return estimate | dict(allocated_payload_bytes=allocated, files=len(m['files']),
                           wall_seconds=time.monotonic()-began,
                           peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    for name, value in dict(layers=24, width=4096, heads=32, hidden=16384, vocab=257,
                            context=16, shard_bytes=64*2**20, segment_bytes=16*2**20).items():
        parser.add_argument('--'+name.replace('_', '-'), type=int, default=value)
    parser.add_argument('--execute', action='store_true')
    print(json.dumps(generate(**vars(parser.parse_args())), indent=2))


if __name__ == '__main__':
    main()

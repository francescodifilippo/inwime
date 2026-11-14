"""Small controlled test fixtures, copying original BF16 words without conversion.

Not a checkpoint downloader or production repacker. Total new payload <=32 MiB.
The runtime itself never invokes this writer. Source v1/v2 files stay immutable.
"""
import hashlib
import json
from pathlib import Path
import struct


def write_fixture(source, destination, *, shards=2):
    source, destination = Path(source), Path(destination)
    manifest = json.loads((source/'manifest.json').read_text())
    tensors = manifest['tensors']
    total = sum(t['nbytes'] for t in tensors.values())
    if total > 32*2**20 or not 1 <= shards <= min(64, len(tensors)):
        raise ValueError('small fixture size/shard bound exceeded')
    # Validate trusted fixture inputs before creating a destination.
    for name, d in manifest['files'].items():
        path = source/name
        if path.stat().st_size != d['nbytes']:
            raise ValueError('source size mismatch')
        with path.open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest() != d['sha256']:
                raise ValueError('source checksum mismatch')
    destination.mkdir(parents=True, exist_ok=False)
    files, weight_map, bindings = {}, {}, {}
    # Deliberately use physical names and order unlike the decoder schema.
    groups = [list(reversed(list(tensors)))[i::shards] for i in range(shards)]
    for i, names in enumerate(groups):
        filename = f'model-{i+1:05d}-of-{shards:05d}.safetensors'
        header, end = {'__metadata__': {'format': 'pt', 'fixture': 'original BF16 words'}}, 0
        for name in names:
            d, physical = tensors[name], 'checkpoint.'+name
            header[physical] = dict(dtype='BF16', shape=d['shape'],
                                    data_offsets=[end, end+d['nbytes']])
            end += d['nbytes']
            weight_map[physical], bindings[name] = filename, physical
        raw = json.dumps(header, separators=(',', ':')).encode()
        raw += b' '*((-len(raw)) % 8)
        digest = hashlib.sha256()
        with (destination/filename).open('xb') as out:
            def write(data):
                out.write(data)
                digest.update(data)
            write(struct.pack('<Q', len(raw)))
            write(raw)
            for name in names:
                d = tensors[name]
                parts = d.get('segments', [dict(shard=d.get('shard'), offset=d.get('offset'),
                                                 nbytes=d['nbytes'])])
                for part in parts:
                    with (source/part['shard']).open('rb') as f:
                        f.seek(part['offset'])
                        left = part['nbytes']
                        while left:
                            data = f.read(min(left, 65536))
                            if not data:
                                raise OSError('incomplete fixture source')
                            write(data)
                            left -= len(data)
        files[filename] = dict(nbytes=8+len(raw)+end, sha256=digest.hexdigest())
    index = dict(metadata={'total_size': total}, weight_map=weight_map)
    raw = json.dumps(index, indent=2).encode()
    index_name = 'model.safetensors.index.json'
    (destination/index_name).write_bytes(raw)
    descriptor = dict(format='inwime-safetensors-bf16-v1', index=index_name,
                      index_sha256=hashlib.sha256(raw).hexdigest(), files=files,
                      config=manifest['config'], variant=manifest['variant'], bindings=bindings)
    (destination/'safetensors.json').write_text(json.dumps(descriptor, indent=2)+'\n')
    return descriptor

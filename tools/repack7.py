"""Offline, bounded byte-preserving whole-tensor safetensors repacker.

Never invoked by runtime. Creates one explicitly requested copy; source stays
immutable. Uses at most 64 KiB payload scratch, verifies input hashes first and
verifies the copied bytes against independent per-tensor source digests.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
from inwime_runtime.safetensors_source import SafetensorsSource
from inwime_runtime.core import signature


def repack(source, destination, shards=3):
    source, destination = Path(source), Path(destination)
    catalog = SafetensorsSource(source)
    if type(shards) is not int or not 2 <= shards <= min(64,len(catalog.tensors)):
        raise ValueError('invalid shard count')
    for name, (_, sha) in catalog.files.items():
        with (source/name).open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest() != sha:
                raise ValueError('source checksum mismatch')
    destination.mkdir(parents=True, exist_ok=False)
    groups = [[] for _ in range(shards)]
    sizes = [0]*shards
    for name, tensor in sorted(catalog.tensors.items(), key=lambda item: -item[1].elements):
        i = min(range(shards),key=sizes.__getitem__)
        groups[i].append(name)
        sizes[i] += tensor.elements*2
    files, weight_map, hashes = {}, {}, {}
    for i, names in enumerate(groups):
        filename = f'model-{i+1:05d}-of-{shards:05d}.safetensors'
        header, offset = {'__metadata__': {'format':'pt'}}, 0
        for name in names:
            tensor = catalog.tensors[name]
            header[name] = dict(dtype='BF16',shape=tensor.shape,data_offsets=[offset,offset+tensor.elements*2])
            offset += tensor.elements*2
            weight_map[name] = filename
        raw = json.dumps(header,separators=(',',':')).encode()
        raw += b' '*(-len(raw)%8)
        with (destination/filename).open('xb') as out:
            out.write(struct.pack('<Q',len(raw))); out.write(raw)
            for name in names:
                tensor = catalog.tensors[name]
                digest = hashlib.sha256()
                with (source/tensor.shard).open('rb') as inp:
                    if signature(inp.fileno()) != catalog.source_stamps[tensor.shard]:
                        raise ValueError('source changed after verification')
                    inp.seek(tensor.offset)
                    left = tensor.elements*2
                    while left:
                        data = inp.read(min(left,65536))
                        if not data: raise OSError('short source read')
                        out.write(data); digest.update(data); left -= len(data)
                    if signature(inp.fileno()) != catalog.source_stamps[tensor.shard]:
                        raise ValueError('source changed while copying')
                hashes[name] = digest.hexdigest()
        with (destination/filename).open('rb') as inp:
            sha = hashlib.file_digest(inp,'sha256').hexdigest()
        files[filename] = dict(nbytes=(destination/filename).stat().st_size,sha256=sha)
    index = dict(metadata={'total_size':catalog.payload_bytes},weight_map=weight_map)
    raw = (json.dumps(index,indent=2)+'\n').encode()
    (destination/'model.safetensors.index.json').write_bytes(raw)
    descriptor = dict(format='inwime-safetensors-bf16-v1',files=files,index='model.safetensors.index.json',index_sha256=hashlib.sha256(raw).hexdigest())
    (destination/'safetensors.json').write_text(json.dumps(descriptor,indent=2)+'\n')
    result = SafetensorsSource(destination)
    for name,tensor in result.tensors.items():
        digest = hashlib.sha256()
        with (destination/tensor.shard).open('rb') as inp:
            inp.seek(tensor.offset); left=tensor.elements*2
            while left:
                data=inp.read(min(left,65536))
                if not data: raise OSError('short destination read')
                digest.update(data); left-=len(data)
        if digest.hexdigest()!=hashes[name]: raise ValueError('tensor bytes changed')
    for name in ['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','provenance.json']:
        if (source/name).exists(): shutil.copyfile(source/name,destination/name)
    evidence = dict(source_identity=catalog.identity,destination_identity=result.identity,
                    tensor_sha256=hashes, tensor_bytes_equal=True,descriptor=descriptor)
    (destination/'repack-provenance.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return evidence


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source');p.add_argument('destination');p.add_argument('--shards',type=int,default=3)
    a=p.parse_args(); print(json.dumps(repack(a.source,a.destination,a.shards),indent=2))

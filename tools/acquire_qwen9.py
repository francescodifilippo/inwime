"""Preflight and explicit acquisition of the pinned original Qwen2.5-7B BF16 checkpoint.

No Hugging Face cache, no conversion, one persistent copy. Without --execute only
the official API, config and the four safetensors headers (HTTP Range) are read.
Shard digests are computed over the local bytes (while streaming, or by re-hashing
a file or .part that is already present) and recorded in the evidence next to the
official LFS digest; .part files are renamed in place only after size and digest
match. Space is rechecked during download.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import urllib.request

MODEL = 'Qwen/Qwen2.5-7B'
REVISION = 'd149729398750b98c0af14eb82c78cfe92750796'
SHARDS = {f'model-0000{i}-of-00004.safetensors': None for i in range(1, 5)}
SMALL = ('config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
         'vocab.json', 'merges.txt', 'LICENSE', 'README.md', 'model.safetensors.index.json')
INDEX = 'model.safetensors.index.json'
AGENT = {'User-Agent': 'inwime-acquire/9'}
CHUNK = 2**20
MARGIN = 8*2**30


def fetch(url, headers=None, limit=64*2**20):
    request = urllib.request.Request(url, headers=AGENT | (headers or {}))
    with urllib.request.urlopen(request, timeout=120) as f:
        data = f.read(limit+1)
    if len(data) > limit:
        raise ValueError('response exceeds bound: '+url)
    return data


def resolve(name):
    return f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}'


def header(name, size):
    prefix = fetch(resolve(name), {'Range': 'bytes=0-7'})
    if len(prefix) != 8:
        raise ValueError('range request not honoured')
    length, = struct.unpack('<Q', prefix)
    if not 2 <= length <= min(size-8, 16*2**20):
        raise ValueError('invalid safetensors header length')
    raw = fetch(resolve(name), {'Range': f'bytes=8-{7+length}'})
    if len(raw) != length:
        raise ValueError('truncated header range')
    parsed = json.loads(raw)
    parsed.pop('__metadata__', None)
    tensors = {}
    for tensor, d in parsed.items():
        begin, end = d['data_offsets']
        if d['dtype'] != 'BF16' or end-begin != 2*math.prod(d['shape']):
            raise ValueError('non-BF16 or inconsistent tensor: '+tensor)
        tensors[tensor] = dict(shape=d['shape'], dtype=d['dtype'], nbytes=end-begin)
    if sum(t['nbytes'] for t in tensors.values()) != size-8-length:
        raise ValueError('payload/header extent mismatch')
    return dict(header_bytes=8+length, tensors=tensors)


def blob_id(raw):
    return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def preflight(destination):
    api = f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true'
    metadata = json.loads(fetch(api))
    entries = {x['rfilename']: x for x in metadata['siblings']}
    if (metadata['sha'] != REVISION or metadata['cardData']['license'] != 'apache-2.0'
            or metadata.get('gated') not in (False, None) or set(SHARDS) - set(entries)
            or set(SMALL) - set(entries)):
        raise ValueError('official metadata differs from the pinned checkpoint')
    small = {}
    for name in ('config.json', 'generation_config.json', 'tokenizer_config.json', INDEX):
        raw = fetch(resolve(name))
        if len(raw) != entries[name]['size'] or blob_id(raw) != entries[name]['blobId']:
            raise ValueError('official Git blob mismatch: '+name)
        small[name] = json.loads(raw)
    config, index = small['config.json'], small[INDEX]
    shards = {}
    for name in SHARDS:
        e = entries[name]
        shards[name] = dict(nbytes=e['size'], sha256=e['lfs']['sha256'], **header(name, e['size']))
    tensors = {t: d | {'shard': s} for s, v in shards.items() for t, d in v['tensors'].items()}
    if {t: d['shard'] for t, d in tensors.items()} != index['weight_map']:
        raise ValueError('index weight_map disagrees with shard headers')
    payload = sum(d['nbytes'] for d in tensors.values())
    if payload != index['metadata']['total_size']:
        raise ValueError('index total_size disagrees with headers')
    width, vocab = config['hidden_size'], config['vocab_size']
    embedding = tensors['model.embed_tokens.weight']['nbytes']
    if embedding != 2*vocab*width or 'lm_head.weight' not in tensors:
        raise ValueError('unexpected embedding/head layout')
    total = sum(entries[n]['size'] for n in (*SMALL, *SHARDS))
    usage = shutil.disk_usage(destination.parent)
    backing = shutil.disk_usage('/mnt/c')
    return dict(model=MODEL, revision=REVISION, api_url=api, license='apache-2.0',
                gated=metadata.get('gated'), safetensors_parameters=metadata.get('safetensors'),
                config=config, generation_config=small['generation_config.json'],
                tokenizer_config={k: v for k, v in small['tokenizer_config.json'].items()
                                  if k not in ('added_tokens_decoder', 'chat_template')},
                files={n: dict(nbytes=entries[n]['size'], blob_id=entries[n]['blobId'],
                               lfs_sha256=entries[n].get('lfs', {}).get('sha256')) for n in (*SMALL, *SHARDS)},
                shards={n: {k: v for k, v in d.items() if k != 'tensors'} for n, d in shards.items()},
                tensor_count=len(tensors), dtypes=sorted({d['dtype'] for d in tensors.values()}),
                tensors=tensors, payload_bytes=payload,
                embedding_bytes=embedding,
                upper_bound_unique_active_bytes=payload,
                lower_bound_unique_active_bytes_without_embedding=payload-embedding,
                download_bytes=total, free_bytes=usage.free, backing_volume_free_bytes=backing.free,
                required_free_bytes=total+MARGIN,
                existing=sorted(p.name for p in destination.iterdir()) if destination.exists() else [])


def hash_file(path):
    digest, done = hashlib.sha256(), 0
    with path.open('rb') as f:
        while data := f.read(CHUNK):
            digest.update(data)
            done += len(data)
    return digest, done


def download(name, size, expected, destination, report):
    """Fetch or accept one file; return the digest actually computed over its bytes.

    Nothing is accepted on trust: a file already in place is hashed again, a
    complete .part is hashed and verified without any request (a Range request
    at EOF would be answered 416), a partial .part is hashed and then resumed.
    """
    path = destination/name
    if path.exists():
        digest, done = hash_file(path)
        if done != size or (expected is not None and digest.hexdigest() != expected):
            raise ValueError('existing file differs from the official size/SHA256: '+name)
        return dict(method='existing file re-hashed', computed_sha256=digest.hexdigest(), resumed_from_bytes=0)
    temp = path.with_name(name+'.part')
    digest, done = hash_file(temp) if temp.exists() else (hashlib.sha256(), 0)
    resumed = done
    if done > size:
        raise ValueError('.part longer than official size: '+name)
    if done < size or not temp.exists():
        headers = {'Range': f'bytes={done}-'} if done else {}
        request = urllib.request.Request(resolve(name), headers=AGENT | headers)
        with urllib.request.urlopen(request, timeout=120) as inp, temp.open('ab') as out:
            if done and inp.status != 206:
                raise OSError('server ignored resume range; .part retained')
            since_check = 0
            while done < size:
                data = inp.read(min(size-done, CHUNK))
                if not data:
                    raise OSError('incomplete download; .part retained: '+name)
                out.write(data); digest.update(data)
                done += len(data); since_check += len(data)
                if since_check >= 256*2**20:
                    since_check = 0
                    free = min(shutil.disk_usage(destination).free, shutil.disk_usage('/mnt/c').free)
                    report['minimum_free_during_download'] = min(
                        report.get('minimum_free_during_download', free), free)
                    if free < MARGIN:
                        raise OSError('free space below 8 GiB margin; .part retained')
            if inp.read(1):
                raise ValueError('download longer than official size: '+name)
    computed = digest.hexdigest()
    if expected is not None and computed != expected:
        raise ValueError('official SHA256 mismatch; .part retained: '+name)
    temp.rename(path)
    method = ('streamed' if not resumed else 'complete .part re-hashed' if resumed == size
              else 'resumed .part re-hashed then streamed')
    return dict(method=method, computed_sha256=computed, resumed_from_bytes=resumed)


def acquire(destination, evidence, execute):
    report = preflight(destination)
    report['execute'] = execute
    if not execute:
        with evidence.open('x') as f:
            json.dump(report, f, indent=2); f.write('\n')
        return report
    if min(report['free_bytes'], report['backing_volume_free_bytes']) < report['required_free_bytes']:
        raise OSError('insufficient space for checkpoint plus 8 GiB margin')
    destination.mkdir(parents=True, exist_ok=True)
    computed = {}
    for name in (*SMALL, *SHARDS):
        f = report['files'][name]
        computed[name] = download(name, f['nbytes'], f['lfs_sha256'], destination, report)
    files = {}
    for name in (*SMALL, *SHARDS):
        path = destination/name
        f = report['files'][name]
        if path.stat().st_size != f['nbytes']:
            raise ValueError('size mismatch after download: '+name)
        if name in SHARDS:
            # sha256 is the digest computed over the local bytes in this run, recorded next
            # to the official one: their equality is a fact of the evidence, not a copy.
            digest = computed[name]['computed_sha256']
            if digest != f['lfs_sha256']:
                raise ValueError('official SHA256 mismatch: '+name)
            files[name] = dict(nbytes=f['nbytes'], sha256=digest, computed_sha256=digest,
                               official_lfs_sha256=f['lfs_sha256'], verification=computed[name]['method'],
                               resumed_from_bytes=computed[name]['resumed_from_bytes'])
        else:
            raw = path.read_bytes()
            if blob_id(raw) != f['blob_id']:
                raise ValueError('official Git blob mismatch: '+name)
            files[name] = dict(nbytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), blob_id=f['blob_id'])
    index_raw = (destination/INDEX).read_bytes()
    descriptor = dict(format='inwime-safetensors-bf16-v1', index=INDEX,
                      index_sha256=hashlib.sha256(index_raw).hexdigest(),
                      files={n: dict(nbytes=files[n]['nbytes'], sha256=files[n]['sha256']) for n in SHARDS})
    # How each file was verified belongs to this run's evidence, not to the checkpoint
    # directory: provenance keeps only stable facts, so a later run can compare it.
    stable = lambda entries: {n: {k: e[k] for k in ('nbytes', 'sha256')} for n, e in entries.items()}
    provenance = dict(model=MODEL, revision=REVISION, license='apache-2.0',
                      files={n: {k: v for k, v in e.items() if k not in ('verification', 'resumed_from_bytes')}
                             for n, e in files.items()})
    for name, data in (('safetensors.json', descriptor), ('provenance.json', provenance)):
        path = destination/name
        if path.exists():
            old = json.loads(path.read_text())
            if name == 'provenance.json':
                old, data = ({k: (stable(d[k]) if k == 'files' else d[k]) for k in ('model', 'revision', 'license', 'files')}
                             for d in (old, data))
            if old != data:
                raise ValueError('existing descriptor/provenance differs: '+name)
        else:
            path.write_text(json.dumps(data, indent=2)+'\n')
    report['downloaded'] = files
    report['free_bytes_after'] = shutil.disk_usage(destination).free
    report['backing_volume_free_bytes_after'] = shutil.disk_usage('/mnt/c').free
    with evidence.open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination', type=Path)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    r = acquire(a.destination, a.evidence, a.execute)
    print(json.dumps({k: r[k] for k in ('tensor_count', 'dtypes', 'payload_bytes', 'embedding_bytes',
                                        'lower_bound_unique_active_bytes_without_embedding',
                                        'download_bytes', 'backing_volume_free_bytes',
                                        'required_free_bytes', 'existing')}, indent=1))

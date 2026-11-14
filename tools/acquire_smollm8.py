"""Explicit acquisition of the pinned original 1.7B checkpoint; no HF cache copy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

MODEL = 'HuggingFaceTB/SmolLM2-1.7B'
REVISION = 'd4c18ccb8a25e6e637e0609aac08a5d1a0436d1b'
EXPECTED = '1193528982f4ac0c0b707ce36fd7dc03a0ef6f3e1a432deb886dce2e90c300c0'
SIZE = 3422777952
NAMES = ('config.json', 'generation_config.json', 'tokenizer.json',
         'tokenizer_config.json', 'special_tokens_map.json', 'README.md', 'model.safetensors')


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def acquire(destination, evidence, execute=False):
    url = f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true'
    with urllib.request.urlopen(url, timeout=60) as f:
        metadata = json.load(f)
    entries = {x['rfilename']: x for x in metadata['siblings']}
    if (metadata['sha'] != REVISION or metadata['cardData']['license'] != 'apache-2.0'
            or entries['model.safetensors']['lfs']['sha256'] != EXPECTED
            or entries['model.safetensors']['size'] != SIZE
            or metadata['safetensors']['parameters'] != {'BF16': 1711376384}):
        raise ValueError('official metadata differs from requested checkpoint')
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = sum(entries[n]['size'] for n in NAMES)
    # Rename .part in place, no full-size temporary duplicate; one optional repack.
    maximum = total + SIZE + 256*2**20
    report = dict(model=MODEL, revision=REVISION, api_url=url, official=metadata,
                  download_bytes=total, maximum_additional_disk_bytes=maximum,
                  free_bytes=shutil.disk_usage(destination.parent).free,
                  backing_volume_free_bytes=shutil.disk_usage('/mnt/c').free,
                  existing=[n for n in NAMES if (destination/n).exists()], execute=execute)
    with evidence.open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    if not execute:
        return report
    if min(report['free_bytes'], report['backing_volume_free_bytes']) < maximum + 2**30:
        raise OSError('insufficient space including repack, captures and 1 GiB margin')
    destination.mkdir(exist_ok=True)
    files = {}
    for name in NAMES:
        path = destination/name
        url = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}'
        if not path.exists():
            temp = path.with_name(name+'.part')
            with urllib.request.urlopen(url, timeout=60) as inp, temp.open('xb') as out:
                left = entries[name]['size']
                while left:
                    data = inp.read(min(left, 2**20))
                    if not data:
                        raise OSError('incomplete download; .part retained')
                    out.write(data); left -= len(data)
                if inp.read(1):
                    raise ValueError('download longer than official size')
            digest = sha256(temp)
            if name == 'model.safetensors' and digest != EXPECTED:
                raise ValueError('checkpoint digest mismatch; .part retained')
            temp.rename(path)
        digest = sha256(path)
        if path.stat().st_size != entries[name]['size']:
            raise ValueError('file size mismatch')
        if name == 'model.safetensors':
            if digest != EXPECTED:
                raise ValueError('checkpoint digest mismatch')
        else:
            # Git blob identity independently authenticates the small official files.
            raw = path.read_bytes()
            blob = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            if blob != entries[name]['blobId']:
                raise ValueError('official Git blob mismatch: '+name)
        files[name] = dict(nbytes=path.stat().st_size, sha256=digest, url=url)
    provenance = dict(model=MODEL, revision=REVISION, license='apache-2.0', files=files)
    descriptor = dict(format='inwime-safetensors-bf16-v1', index=None,
                      files={'model.safetensors': {k: files['model.safetensors'][k]
                                                   for k in ('nbytes', 'sha256')}})
    for name, data in [('provenance.json', provenance), ('safetensors.json', descriptor)]:
        path = destination/name
        if path.exists():
            if json.loads(path.read_text()) != data:
                raise ValueError('existing provenance/descriptor differs')
        else:
            path.write_text(json.dumps(data, indent=2)+'\n')
    return provenance


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination', type=Path)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    print(json.dumps(acquire(a.destination, a.evidence, a.execute), indent=2))

"""Create at most two exclusive 5 GiB test copies; record ownership for later cleanup."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'results/v2/storage/datasets.json'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


if __name__=='__main__':
    if MANIFEST.exists():
        raise RuntimeError('dataset manifest already exists; preserve/reuse owned files')
    source=ROOT/'data/weights-5GiB.bf16'
    size=source.stat().st_size
    assert size==5*1024**3
    data={'source':str(source),'bytes':size,'sha256':digest(source),'copies':[]}
    for name,parent in [('internal-drvfs',Path('/mnt/c/Users/franc/AppData/Local/Temp')),
                        ('external-usb',Path('/mnt/d'))]:
        if shutil.disk_usage(parent).free<size+1024**3:
            raise RuntimeError('insufficient free space with 1 GiB margin')
        directory=Path(tempfile.mkdtemp(prefix='inwime-benchmark-',dir=parent))
        path=directory/'weights-5GiB.bf16'
        item={'name':name,'directory':str(directory),'file':str(path),'complete':False}
        data['copies'].append(item)
        MANIFEST.write_text(json.dumps(data,indent=2)+'\n')
        with source.open('rb') as src,path.open('xb') as dst:
            shutil.copyfileobj(src,dst,8*1024**2)
            dst.flush(); os.fsync(dst.fileno())
        if path.stat().st_size!=size or digest(path)!=data['sha256']:
            raise RuntimeError('copy size/hash mismatch')
        item['complete']=True
        MANIFEST.write_text(json.dumps(data,indent=2)+'\n')
        print(name,path,'5 GiB SHA256 verified',flush=True)

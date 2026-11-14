"""Explicit pinned 272 MB acquisition; never called by runtime or tests."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

MODEL='HuggingFaceTB/SmolLM2-135M'
REVISION='93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
EXPECTED='80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1'
NAMES=('config.json','generation_config.json','tokenizer.json','tokenizer_config.json',
       'special_tokens_map.json','README.md','model.safetensors')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('destination',type=Path)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true') as f:
        metadata=json.load(f)
    assert metadata['sha']==REVISION
    assert metadata['cardData']['license']=='apache-2.0'
    entries={x['rfilename']:x for x in metadata['siblings']}
    assert entries['model.safetensors']['lfs']['sha256']==EXPECTED
    total=sum(entries[n]['size'] for n in NAMES)
    a.destination.parent.mkdir(parents=True,exist_ok=True)
    free=shutil.disk_usage(a.destination.parent).free
    print(json.dumps(dict(model=MODEL,revision=REVISION,download_bytes=total,free_bytes=free,
                          existing=[n for n in NAMES if (a.destination/n).exists()],execute=a.execute)))
    if not a.execute:return
    if free < total+16*2**20: raise OSError('insufficient disk headroom')
    a.destination.mkdir(exist_ok=True)
    old=a.destination/'provenance.json'
    previous=json.loads(old.read_text()) if old.exists() else None
    if previous is not None and (previous['model']!=MODEL or previous['revision']!=REVISION):
        raise ValueError('existing provenance mismatch')
    files={}
    for name in NAMES:
        path=a.destination/name
        url=f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}'
        if path.exists():
            if previous is None: raise ValueError('existing copy without provenance: '+name)
            sha=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
            if sha!=previous['files'][name]['sha256']: raise ValueError('existing digest mismatch')
        else:
            temp=path.with_suffix(path.suffix+'.part')
            with urllib.request.urlopen(url) as inp,temp.open('xb') as out:
                shutil.copyfileobj(inp,out,length=65536)
            sha=hashlib.file_digest(temp.open('rb'),'sha256').hexdigest()
            if temp.stat().st_size!=entries[name]['size'] or (name=='model.safetensors' and sha!=EXPECTED):
                raise ValueError('download size/digest mismatch; .part retained for inspection')
            temp.rename(path)
        if path.stat().st_size!=entries[name]['size'] or (name=='model.safetensors' and sha!=EXPECTED):
            raise ValueError('size/digest mismatch')
        files[name]=dict(nbytes=path.stat().st_size,sha256=sha,url=url)
    provenance=dict(model=MODEL,revision=REVISION,license='apache-2.0',files=files)
    old.write_text(json.dumps(provenance,indent=2)+'\n')
    descriptor=dict(format='inwime-safetensors-bf16-v1',index=None,
                    files={'model.safetensors':{k:files['model.safetensors'][k] for k in ['nbytes','sha256']}})
    (a.destination/'safetensors.json').write_text(json.dumps(descriptor,indent=2)+'\n')


if __name__=='__main__':main()

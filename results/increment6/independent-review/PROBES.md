# Ricette riproducibili delle prove indipendenti

Eseguire dalla radice del repository con l’ambiente di test già disponibile.
I payload sono temporanei e minuscoli; nessun checkpoint scaricato.
Il limite FD viene modificato soltanto nel processo della prova e poi ripristinato.
Per una nuova acquisizione cambiare il percorso di output degli script, preservando
questa directory. Salvare ogni blocco in un file temporaneo ed eseguirlo con
`PYTHONPATH=. python3 /tmp/nome-prova.py`.

## Pipeline concorrente e ammissione FD

```python
from array import array
import hashlib, json, os, resource, struct, tempfile, threading, time
from pathlib import Path
from inwime_runtime import CPURuntime, CheckpointError
from inwime_runtime.core import Coordinator, FileBackend, Plan, CATEGORIES
from inwime_runtime.safetensors_source import SafetensorsSource
from inwime_reference.checkpoint import load_checkpoint
from inwime_reference.decoder import ReferenceDecoder
from tests.test_runtime_cpu import export_reference
from tools.fixture_safetensors6 import write_fixture
from tools.verify_increment2 import TOKENS, compare, runtime_export
ROOT=Path.cwd()
report={}
class Delayed(FileBackend):
    concurrent=peak=0
    armed=False
    lock=threading.Lock()
    def readinto(self, fd, offset, destination):
        if not self.armed: return super().readinto(fd,offset,destination)
        with self.lock:
            self.concurrent+=1; self.peak=max(self.peak,self.concurrent)
        try:
            time.sleep(.0001 if offset%3 else .0003)
            return super().readinto(fd,offset,destination)
        finally:
            with self.lock: self.concurrent-=1
with tempfile.TemporaryDirectory(prefix='inwime-independent-') as tmp:
    tmp=Path(tmp)
    for variant in ('dense','moe'):
        original=ROOT/'fixtures/reference_decoder'/variant
        safe=tmp/variant; write_fixture(original,safe)
        oracle=ReferenceDecoder(load_checkpoint(original)); state=None
        backend=Delayed(); phases=0
        with CPURuntime(safe,source='safetensors',io_mode='pipeline',in_flight=3,
                        io_workers=2,tile_elements=73,read_elements=13,backend=backend) as runtime:
            backend.armed=True
            for tokens in (TOKENS[:3], *([t] for t in TOKENS[3:])):
                expected=oracle.prefill(tokens,state); state=expected.state
                with runtime.prefill(tokens) as result:
                    compare([runtime_export(result)],[export_reference(expected)])
                phases+=1
            stats=runtime.stats()
            assert stats['requests']==stats['completed']==stats['released']
        assert not runtime.control.fds and not runtime.control.buffers
        assert not any(t.name.startswith('inwime-read-') for t in threading.enumerate())
        assert backend.peak==2
        report[variant]={'independent_oracle_phases':phases,'peak_concurrent_reads':backend.peak,
                         'inference_read_bytes':stats['inference_read_bytes'],
                         'integrity_read_bytes':stats['integrity_read_bytes'],
                         'reserved_peak_bytes':stats['reserved_peak_bytes'],'cleanup':True}
    many=tmp/'many'; many.mkdir()
    files={}; weights={}
    for i in range(65):
        name=f'shard-{i:03}.safetensors'; tensor=f'w{i}'
        header=json.dumps({tensor:dict(dtype='BF16',shape=[1,1],data_offsets=[0,2])}).encode()
        data=struct.pack('<Q',len(header))+header+b'\x80\x3f'
        (many/name).write_bytes(data)
        files[name]=dict(nbytes=len(data),sha256=hashlib.sha256(data).hexdigest()); weights[tensor]=name
    index=json.dumps(dict(weight_map=weights,metadata=dict(total_size=130))).encode()
    (many/'index.json').write_bytes(index)
    descriptor=dict(format='inwime-safetensors-bf16-v1',files=files,index='index.json',
                    index_sha256=hashlib.sha256(index).hexdigest())
    (many/'safetensors.json').write_text(json.dumps(descriptor))
    try: SafetensorsSource(many)
    except CheckpointError as e: report['default_65_shards']=str(e)
    else: raise AssertionError('default unexpectedly accepts 65 shards')
    source=SafetensorsSource(many,max_shards=65)
    arrays={'weights':('weights',1,'d'),'staging':('staging',2,'B')}
    required=dict.fromkeys(CATEGORIES,0); required.update(weights=8,staging=2,metadata=source.bootstrap_bytes+65536)
    plan=Plan(arrays,required,1,1,1)
    before=len(list(Path('/proc/self/fd').iterdir())); limits=resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE,(32,limits[1]))
        try: Coordinator(source,plan,integrity='metadata')
        except OSError as e: report['raised_shard_limit_low_fd']={'errno':e.errno,'message':str(e)}
        else: raise AssertionError('expected descriptor exhaustion')
    finally: resource.setrlimit(resource.RLIMIT_NOFILE,limits)
    after=len(list(Path('/proc/self/fd').iterdir())); assert before==after
    report['fd_counts_before_after']=[before,after]
    report['tiny_65_shards_payload_bytes']=130
report['ru_maxrss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
report['process_io']={k:int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}
report['gpu_allocations_bytes']=0
out=ROOT/'results/increment6/independent-review/probes.json'; out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
```

## Matrice vuota

```python
from array import array
import hashlib, json, struct, tempfile
from pathlib import Path
from inwime_runtime.gemv import StreamedGEMV
from inwime_runtime.safetensors_source import SafetensorsSource
results={}
with tempfile.TemporaryDirectory(prefix='inwime-empty-') as tmp:
    p=Path(tmp)
    h=json.dumps({'w':dict(dtype='BF16',shape=[1,0],data_offsets=[0,0])}).encode()
    raw=struct.pack('<Q',len(h))+h
    (p/'model.safetensors').write_bytes(raw)
    (p/'safetensors.json').write_text(json.dumps(dict(format='inwime-safetensors-bf16-v1', index=None,
        files={'model.safetensors':dict(nbytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())})))
    assert SafetensorsSource(p).tensors['w'].elements==0
    for mode in ('sync','pipeline'):
        out=array('d',[123.])
        with StreamedGEMV(p,source='safetensors',io_mode=mode,tile_elements=1,read_elements=1) as op:
            try: op.run('w',array('d'),out)
            except ValueError as e: results[mode]={'error':str(e),'output':list(out),'inference_read_bytes':op.stats()['inference_read_bytes']}
            else: raise AssertionError('expected empty buffer rejection')
        assert not op.control.fds and not op.control.buffers
Path('results/increment6/independent-review/empty-gemv.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
```


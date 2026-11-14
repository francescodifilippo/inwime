"""Small real-family fixtures; upstream checkpoint suite is a separate mandatory runner."""
import hashlib
import json
from pathlib import Path
import struct
import threading
from concurrent.futures import CancelledError
import pytest
from inwime_runtime.llama import LlamaCatalog, LlamaRuntime
from inwime_runtime.safetensors_source import SafetensorsSource
from inwime_runtime.core import CheckpointError, FileBackend, ReadError, BudgetError
from inwime_runtime.math_cpu import NativeMath
from tools.repack7 import repack


@pytest.fixture
def checkpoint(tmp_path):
    d=dict(model_type='llama',architectures=['LlamaForCausalLM'], hidden_act='silu',tie_word_embeddings=True,
           hidden_size=8,intermediate_size=12,num_hidden_layers=2,num_attention_heads=2,
           num_key_value_heads=1,vocab_size=16,max_position_embeddings=8,rms_norm_eps=1e-5,rope_theta=100000.)
    shapes={'model.embed_tokens.weight':[16,8],'model.norm.weight':[8]}
    for l in range(2):
        p=f'model.layers.{l}.'
        for name,shape in [('input_layernorm',[8]),('post_attention_layernorm',[8]),
             ('self_attn.q_proj',[8,8]),('self_attn.k_proj',[4,8]),('self_attn.v_proj',[4,8]),
             ('self_attn.o_proj',[8,8]),('mlp.gate_proj',[12,8]),('mlp.up_proj',[12,8]),('mlp.down_proj',[8,12])]:
            shapes[p+name+'.weight']=shape
    import math
    header, words, offset={},bytearray(),0
    for name,shape in shapes.items():
        n=math.prod(shape)
        header[name]=dict(dtype='BF16',shape=shape,data_offsets=[offset,offset+n*2]);offset+=n*2
        for i in range(n):
            value=1. if len(shape)==1 else ((i*17+len(name))%29-14)/64
            words.extend(struct.pack('<I',struct.unpack('<I',struct.pack('<f',value))[0])[2:])
    raw=json.dumps(header).encode(); data=struct.pack('<Q',len(raw))+raw+words
    (tmp_path/'model.safetensors').write_bytes(data)
    (tmp_path/'safetensors.json').write_text(json.dumps(dict(format='inwime-safetensors-bf16-v1',index=None,
        files={'model.safetensors':dict(nbytes=len(data),sha256=hashlib.sha256(data).hexdigest())})))
    (tmp_path/'config.json').write_text(json.dumps(d))
    return tmp_path


@pytest.mark.parametrize('field,value', [('model_type','qwen'),('hidden_act','gelu'),('tie_word_embeddings',False),
    ('rope_scaling',{'type':'linear','factor':2}),('rope_interleaved',True),('rope_theta',0),('num_key_value_heads',3),
    ('num_attention_heads',3),('head_dim',8),('attention_bias',True),('mlp_bias',True),('rms_norm_eps',-1),
    ('num_hidden_layers',True)])
def test_config_rejected(checkpoint,field,value):
    path=checkpoint/'config.json';d=json.loads(path.read_text());d[field]=value;path.write_text(json.dumps(d))
    with pytest.raises(CheckpointError): LlamaCatalog(SafetensorsSource(checkpoint))


@pytest.mark.parametrize('bad',['missing','shape','tying'])
def test_tensor_rejected(checkpoint,bad):
    source=SafetensorsSource(checkpoint)
    from dataclasses import replace
    if bad=='missing': del source.tensors['model.norm.weight']
    if bad=='shape': source.tensors['model.norm.weight']=replace(source.tensors['model.norm.weight'],shape=(7,))
    if bad=='tying': source.tensors['lm_head.weight']=source.tensors['model.embed_tokens.weight']
    with pytest.raises(CheckpointError): LlamaCatalog(source)


@pytest.mark.parametrize('mode',['sync','pipeline'])
def test_chunks_and_repack(checkpoint,tmp_path,mode):
    dest=tmp_path/'sharded';repack(checkpoint,dest,2)
    outputs=[]
    for directory,chunks in [(checkpoint,[[1,3,2,4]]),(checkpoint,[[1],[3,2],[4]]),(dest,[[1,3,2,4]])]:
        with LlamaRuntime(directory,capacity=4,tile_elements=7,read_elements=3,io_mode=mode) as r:
            logits=[]
            for chunk in chunks:
                with r.prefill(chunk) as result:
                    logits.extend(result.logits);kv=tuple(tuple(bytes(v) for v in pair) for pair in result.kv)
            outputs.append((logits,kv))
    assert outputs[0]==outputs[1]==outputs[2]


@pytest.mark.parametrize('tokens',[[],[-1],[16],[True],[1]*9])
def test_tokens(checkpoint,tokens):
    with LlamaRuntime(checkpoint,capacity=8) as r:
        before=r.state_digest()
        with pytest.raises(ValueError): r.prefill(tokens)
        assert before==r.state_digest()


@pytest.mark.parametrize('mode',['sync','pipeline'])
@pytest.mark.parametrize('phase',['prefill','decode'])
@pytest.mark.parametrize('fault',['cancel','short','io'])
def test_transaction(checkpoint,mode,phase,fault):
    event=threading.Event()
    class Backend(FileBackend):
        armed=False
        count=0
        def readinto(self,fd,offset,view):
            n=super().readinto(fd,offset,view)
            if self.armed:
                self.count+=1
                if self.count==10:
                    if fault=='cancel': event.set()
                    if fault=='short': return n-1
                    if fault=='io': raise OSError('injected')
            return n
    backend=Backend()
    with LlamaRuntime(checkpoint,capacity=8,tile_elements=7,read_elements=3,io_mode=mode,backend=backend) as r:
        if phase=='decode':
            with r.prefill([1,2]): pass
        before=r.state_digest();published=r._published
        backend.armed=True
        with pytest.raises((CancelledError,ReadError,OSError)):
            if phase=='decode': r.decode(3,cancel_event=event)
            else: r.prefill([1,2,3],cancel_event=event)
        assert r.state_digest()==before and r._published==published
        assert not r.control.active and not r.result_open and r.failed
    assert not r.control.fds and not r.control.buffers


@pytest.mark.parametrize('fault',['corrupt','truncate'])
def test_integrity(checkpoint,fault):
    p=checkpoint/'model.safetensors';raw=bytearray(p.read_bytes())
    if fault=='truncate': raw=raw[:-1]
    else: raw[-1]^=1
    p.write_bytes(raw)
    with pytest.raises(CheckpointError): LlamaRuntime(checkpoint,capacity=4)


def test_budget_and_lifetime(checkpoint):
    plan=LlamaRuntime.minimum_plan(checkpoint,capacity=4,tile_elements=7,read_elements=3)
    budget=plan.budget()
    from dataclasses import replace
    with pytest.raises(BudgetError):
        LlamaRuntime(checkpoint,capacity=4,tile_elements=7,read_elements=3,budget=replace(budget,total=1))
    with LlamaRuntime(checkpoint,capacity=4) as r:
        result=r.prefill([1])
        with pytest.raises(RuntimeError): r.decode(2)
        result.release()
        with pytest.raises(RuntimeError): _=result.logits
    assert not r.control.fds


def test_token_snapshot(checkpoint):
    tokens=[1,2]
    class Backend(FileBackend):
        armed=False
        def readinto(self,fd,offset,view):
            if self.armed:
                tokens.clear()
            return super().readinto(fd,offset,view)
    backend=Backend()
    with LlamaRuntime(checkpoint,capacity=4,backend=backend) as r:
        backend.armed=True
        with r.prefill(tokens) as result:
            assert len(result.logits)==32 and result.routing==()
        assert r.length==2 and tokens==[]


def test_greedy_eos(checkpoint):
    from inwime_runtime.tokenizer import greedy
    with LlamaRuntime(checkpoint,capacity=4) as r:
        with r.prefill([1]) as out:
            chosen=max(range(16),key=out.logits.__getitem__)
    class Tokenizer:
        bos=0
        eos={chosen}
        def encode(self,text): return [1]
        def decode(self,ids): return str(ids)
    with LlamaRuntime(checkpoint,capacity=4) as r:
        out=greedy(r,Tokenizer(),'test',3)
        assert out['stopped_on_eos'] and out['generated_ids']==[chosen] and r.length==1


def test_no_oracle_skip(tmp_path):
    import subprocess,sys
    result=subprocess.run([sys.executable,'-m','tools.verify_increment7',str(tmp_path),
        str(tmp_path/'missing.npz'),'--output',str(tmp_path/'out.json')],capture_output=True,text=True)
    assert result.returncode!=0 and not (tmp_path/'out.json').exists()


def test_import_cpu_only():
    import subprocess,sys
    subprocess.run([sys.executable,'-c',"from inwime_runtime.llama import LlamaRuntime; from inwime_runtime.tokenizer import greedy; import sys; assert not ({'torch','transformers','tokenizers','numpy'} & set(sys.modules))"],check=True)

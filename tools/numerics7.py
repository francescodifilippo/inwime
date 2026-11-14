"""RMSNorm isolation and first-divergence localization; verification only."""
import argparse
from array import array
from contextlib import nullcontext
from decimal import Decimal, localcontext
import hashlib
import inspect
import sys
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM
from transformers.models.llama import modeling_llama as llama
from inwime_runtime.llama import LlamaRuntime
from inwime_runtime.decoder import CPURuntime
from inwime_runtime.math_cpu import NativeMath
from .oracle_precision7 import precision_profile, rms64, FloatingDtypeAudit
from .verify_increment7 import metrics, TOLERANCES


def decimal_rms(x, w, eps):
    with localcontext() as ctx:
        ctx.prec = 80
        values = [Decimal.from_float(float(v)) for v in x]
        denominator = (sum(v*v for v in values)/len(values)+Decimal.from_float(eps)).sqrt()
        return np.array([float(v/denominator*Decimal.from_float(float(a))) for v,a in zip(values,w)])


def isolated_case(x, w, eps, inwime):
    t = torch.tensor(x, dtype=torch.float64)
    module = llama.LlamaRMSNorm(len(x), eps).double()
    with torch.no_grad(): module.weight.copy_(torch.tensor(w, dtype=torch.float64))
    # Stage replication is checked bitwise against the actual installed method.
    stages = {'input':t, 'cast_input':t.float()}
    stages['square'] = stages['cast_input'].pow(2)
    stages['mean'] = stages['square'].mean(-1,keepdim=True)
    stages['epsilon_add'] = stages['mean']+eps
    stages['rsqrt'] = torch.rsqrt(stages['epsilon_add'])
    stages['normalized_fp32'] = stages['cast_input']*stages['rsqrt']
    stages['back_to_input_dtype'] = stages['normalized_fp32'].to(t.dtype)
    stages['scale_weight'] = module.weight.detach()
    stages['output'] = module.weight.detach()*stages['back_to_input_dtype']
    audit = FloatingDtypeAudit()
    with torch.inference_mode(), audit:
        stock = module(t).numpy()
        double = rms64(module,t).numpy()
    assert np.array_equal(stock,stages['output'].numpy())
    # Same staged audit for the pure64 formula, including the output dtype.
    double_stages = {'input':t, 'square':t.square()}
    double_stages['mean'] = double_stages['square'].mean(-1,keepdim=True)
    double_stages['epsilon_add'] = double_stages['mean']+eps
    double_stages['rsqrt'] = torch.rsqrt(double_stages['epsilon_add'])
    double_stages['normalized'] = t*double_stages['rsqrt']
    double_stages['output'] = module.weight.detach()*double_stages['normalized']
    assert np.array_equal(double,double_stages['output'].numpy())
    independent = decimal_rms(x,w,eps)
    arrays = {'stock':stock, 'pure64':double, 'inwime':inwime, 'decimal80':independent}
    return dict(stages={k:dict(dtype=str(v.dtype),values=v.detach().tolist()) for k,v in stages.items()},
                stage_replica_equals_installed=True,
                pure64_stages={k:dict(dtype=str(v.dtype),values=v.detach().tolist()) for k,v in double_stages.items()},
                comparisons={f'{a}_vs_{b}':metrics(arrays[a],arrays[b],*TOLERANCES['float64'])
                             for a,b in [('stock','pure64'),('inwime','pure64'),('inwime','decimal80'),('pure64','decimal80')]},
                max_stock_cast_effect=float(np.abs(stock-double).max()),
                dispatcher=audit.summary())


class TraceRuntime(LlamaRuntime):
    def __init__(self,*a,**kw):
        self.ops = {}; self.norm_inputs = {}
        super().__init__(*a,**kw)

    def _embedding(self,name,row,destination,add=False):
        super()._embedding(name,row,destination,add)
        self.ops[name] = np.array(destination)

    def _norm(self,source,name,destination):
        self.norm_inputs[name] = np.array(source)
        super()._norm(source,name,destination)
        self.ops[name] = np.array(destination)

    def _linear(self,name,source,destination,offset=0):
        super()._linear(name,source,destination,offset)
        self.ops[name] = np.array(destination[offset:offset+self.catalog.tensors[name].shape[0]])


def norm_probe(checkpoint):
    with TraceRuntime(checkpoint,capacity=1,math_backend=NativeMath('build/libinwime_cpu.so')) as runtime:
        with runtime.prefill([504]) as result:
            residuals = [np.array(v) for v in result.hidden]
        own = {}
        for name,values in runtime.ops.items():
            own[name] = values
            if name.endswith('ffn.down'):
                layer = int(name.split('.')[1])
                own[f'layers.{layer}.residual'] = residuals[layer]
        x = runtime.norm_inputs['layers.0.norm1']
        # Acquire the original first norm scale using physical streaming as well.
        scale = array('d',[0])*runtime.spec.width
        runtime._embedding('layers.0.norm1',0,scale)
        isolation = isolated_case(x,np.array(scale),runtime.spec.rms_epsilon,own['layers.0.norm1'])
    return own,isolation


def analytic_probe():
    x = [3.,4.,-3.,-4.];w=[1.,2.,3.,4.];eps=.5
    class Weights:
        def value(self,i): return w[i]
    class Control:
        def compute(self): return nullcontext()
        def stream(self,requests):
            for _,start,count in requests: yield start,count,Weights()
    owner = SimpleNamespace(spec=SimpleNamespace(width=4,rms_epsilon=eps),
                            plan=SimpleNamespace(tile_elements=4),control=Control())
    output=array('d',[0])*4
    CPURuntime._norm(owner,array('d',x),'scale',output)
    report=isolated_case(x,w,eps,np.array(output))
    report['formula']='[3,8,-9,-16] / sqrt(13), epsilon=0.5'
    assert np.allclose(np.array(output),decimal_rms(x,w,eps),atol=1e-15,rtol=1e-15)
    return report


def trace_upstream(model, profile):
    trace={};inputs={};handles=[]
    def hook(name):
        def record(module,args,output):
            output = output[0] if isinstance(output, tuple) else output
            trace[name]=output.detach().numpy().reshape(-1).copy()
            if isinstance(module,llama.LlamaRMSNorm): inputs[name]=args[0].detach().numpy().reshape(-1).copy()
        return record
    bindings={'embedding':model.model.embed_tokens,'final_norm':model.model.norm,'lm_head':model.lm_head}
    for i,layer in enumerate(model.model.layers):
        p=f'layers.{i}.'
        bindings.update({p+'norm1':layer.input_layernorm,p+'norm2':layer.post_attention_layernorm,p+'residual':layer})
        for name in ('q','k','v','o'): bindings[p+name]=getattr(layer.self_attn,name+'_proj')
        for name in ('gate','up','down'): bindings[p+'ffn.'+name]=getattr(layer.mlp,name+'_proj')
    try:
        for name,module in bindings.items(): handles.append(module.register_forward_hook(hook(name)))
        with precision_profile(model,profile),torch.inference_mode(),torch.autocast('cpu',enabled=False):
            model(torch.tensor([[504]]),use_cache=True)
    finally:
        for h in handles:h.remove()
    return trace,inputs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2)
    norm_locals = {}
    def trace(frame,event,arg):
        if (frame.f_code is CPURuntime._norm.__code__ and event == 'return'
                and frame.f_locals.get('scale_name') == 'layers.0.norm1'):
            local = frame.f_locals
            norm_locals.update({k:dict(type=type(local[k]).__name__,value=local[k])
                                for k in ('squared','denominator')})
            norm_locals['source_itemsize'] = local['source'].itemsize
            norm_locals['destination_itemsize'] = local['destination'].itemsize
        return trace
    previous_trace = sys.gettrace()
    try:
        sys.settrace(trace)
        own,isolation=norm_probe(a.checkpoint)
    finally:
        sys.settrace(previous_trace)
    model=AutoModelForCausalLM.from_pretrained(a.checkpoint,local_files_only=True,trust_remote_code=False,
                     dtype=torch.bfloat16,attn_implementation='eager').eval().double()
    stock,stock_inputs=trace_upstream(model,'stock');pure,_=trace_upstream(model,'full64')
    comparisons={name:dict(stock=metrics(values,stock[name],*TOLERANCES['float64']),
                           full64=metrics(values,pure[name],*TOLERANCES['float64']),
                           stock_exact=bool(np.array_equal(values,stock[name]))) for name,values in own.items()}
    first='layers.0.norm1'
    assert np.array_equal(own['embedding'],stock['embedding'])
    assert np.array_equal(own['embedding'],stock_inputs[first])
    assert not np.array_equal(own[first],stock[first])
    assert all(v['full64']['failures']==0 for v in comparisons.values())
    # Replay installed RMSNorm with the *same* input at each norm; this isolates
    # local casting from the already-propagated differences in its actual input.
    replay={}
    for name,module in [('layers.0.norm1',model.model.layers[0].input_layernorm)]:
        with torch.inference_mode():
            stock_replay=module(torch.tensor(own['embedding'])).numpy()
        replay[name]=dict(stock_replay_equals_captured=bool(np.array_equal(stock_replay,stock[name])),
                         stock_replay_vs_inwime=metrics(own[name],stock_replay,*TOLERANCES['float64']))
    sources={}
    from transformers.modeling_rope_utils import _compute_default_rope_parameters
    for label,obj in [('LlamaRMSNorm',llama.LlamaRMSNorm.forward),('LlamaRotaryEmbedding',llama.LlamaRotaryEmbedding.forward),('eager_attention_forward',llama.eager_attention_forward),('default_rope_parameters',_compute_default_rope_parameters)]:
        src=inspect.getsource(obj)
        sources[label]=dict(path=inspect.getsourcefile(obj),source=src,sha256=hashlib.sha256(src.encode()).hexdigest())
    result=dict(versions=dict(torch=torch.__version__,transformers=transformers.__version__),
                source=sources,attention_implementation=model.config._attn_implementation,
                rms_actual_bound_implementation=str(model.model.layers[0].input_layernorm.forward.__func__),
                use_kernels=model.config.use_kernels if hasattr(model.config,'use_kernels') else False,
                input_dtype='torch.float64',inwime_actual_norm_locals=norm_locals,python_float_info=dict(mant_dig=sys.float_info.mant_dig,epsilon=sys.float_info.epsilon),checkpoint_rmsnorm=isolation,analytic=analytic_probe(),
                first_nonexact_operator=next(n for n,v in comparisons.items() if not v['stock_exact']),
                first_threshold_violation=next(n for n,v in comparisons.items() if v['stock']['failures']),
                operator_comparisons= comparisons,first_operator_replay=replay)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(a.output)


if __name__=='__main__': main()

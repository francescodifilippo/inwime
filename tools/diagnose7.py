"""Isolate the first RMSNorm precision difference using upstream code and weights."""
import argparse,json
from pathlib import Path
import torch
from safetensors import safe_open
from transformers.models.llama.modeling_llama import LlamaRMSNorm

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);a=p.parse_args()
    with safe_open(a.checkpoint/'model.safetensors',framework='pt',device='cpu') as f:
        x=f.get_slice('model.embed_tokens.weight')[504:505].double()
        scale=f.get_tensor('model.layers.0.input_layernorm.weight').double()
    norm=LlamaRMSNorm(x.shape[-1],1e-5).double()
    norm.weight.data.copy_(scale)
    with torch.inference_mode():
        stock=norm(x)
        exact64=scale*x*torch.rsqrt(x.square().mean(-1,keepdim=True)+norm.variance_epsilon)
        d=(stock-exact64).abs()
    print(json.dumps(dict(operator='layer0.input_layernorm',input_id=504,
        stock_internal_dtype='float32 despite double parameters/input',max_abs=float(d.max()),
        rms=float(d.square().mean().sqrt()),nonzero=int(torch.count_nonzero(d))),indent=2))

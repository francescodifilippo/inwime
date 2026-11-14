"""Small deterministic real-text demo, tokenizers is the only added engine dependency."""
import argparse
import json
from inwime_runtime.llama import LlamaRuntime
from inwime_runtime.math_cpu import NativeMath
from inwime_runtime.tokenizer import LlamaTokenizer,greedy

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('checkpoint');p.add_argument('--text',default='The capital of France is');p.add_argument('--tokens',type=int,default=3)
    a=p.parse_args(); tokenizer=LlamaTokenizer(a.checkpoint)
    ids=tokenizer.encode(a.text)
    with LlamaRuntime(a.checkpoint,capacity=max(1,len(ids))+a.tokens,
                      math_backend=NativeMath('build/libinwime_cpu.so')) as r:
        print(json.dumps(greedy(r,tokenizer,a.text,a.tokens),ensure_ascii=False,indent=2))

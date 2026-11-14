"""CPU runtime only, one measurement per process. No oracle/NumPy/tokenizer imports."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import time
from inwime_runtime.llama import LlamaRuntime
from inwime_runtime.math_cpu import NativeMath


def io_counters():
    return {k:int(v) for k,v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('--mode',choices=['sync','pipeline'],required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--as-mib',type=int,default=128)
    a=p.parse_args()
    resource.setrlimit(resource.RLIMIT_AS,(a.as_mib*2**20,a.as_mib*2**20))
    tokens=[504,3575,282,4649,314]  # Official tokenizer for "The capital of France is".
    report=dict(mode=a.mode,rlimit_as_bytes=a.as_mib*2**20,rss_acceptance_bytes=128*2**20,
                weight_cache=False,gpu_allocations_bytes=0,cache_state='buffered, uncontrolled OS cache; startup SHA scans all weights',
                platform=platform.platform(),python=platform.python_version(),phases=[])
    rss_before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    io_before=io_counters();started=time.monotonic()
    with LlamaRuntime(a.checkpoint,capacity=16,io_mode=a.mode,
                      math_backend=NativeMath('build/libinwime_cpu.so')) as r:
        report['startup_s']=time.monotonic()-started
        report['bootstrap_estimate_bytes']=r.catalog.bootstrap_bytes
        report['arena_plan_bytes']=r.plan.total
        report['arena_by_category']=r.plan.required
        for phase,ids in [('prefill',tokens),('decode1',[260]),('decode2',[3575]),('decode3',[282])]:
            previous=r.stats();before=io_counters();started=time.monotonic()
            with r.prefill(ids) as out:
                elapsed=time.monotonic()-started
                digest=hashlib.sha256(out.logits.cast('B')).hexdigest()
            stats=r.stats();after=io_counters()
            delta={k:stats[k]-previous[k] for k in ['inference_read_bytes','gemv_ns','bf16_expand_ns','compute_ns','read_service_ns','data_wait_ns','read_union_ns','overlap_ns'] if k in stats}
            # Wall residual is observation, not an exclusive coordinator timer.
            residual = elapsed-(delta['compute_ns']+delta['bf16_expand_ns']+delta['read_union_ns']-delta['overlap_ns'])/1e9
            report['phases'].append(dict(host_residual_s=residual, name=phase,ids=ids,wall_s=elapsed,s_per_token=elapsed/len(ids),
                logits_sha256=digest,counters_delta=delta,proc_io_delta={k:after[k]-before[k] for k in after}))
        report['stats']=r.stats()
    report['released_fds']=not r.control.fds
    report['released_arenas']=not r.control.buffers
    report['rss_before_bytes']=rss_before
    report['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report['proc_io_delta']={k:v-io_before[k] for k,v in io_counters().items()}
    assert report['peak_rss_bytes'] < report['rss_acceptance_bytes']
    assert report['released_fds'] and report['released_arenas']
    with a.output.open('x') as f: json.dump(report,f,indent=2);f.write('\n')
    print(a.output)


if __name__=='__main__':main()

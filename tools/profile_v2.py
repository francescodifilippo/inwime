#!/usr/bin/env python3
"""Separate diagnostic CUPTI acquisitions; run only after the unprofiled sweep."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from analyze_trace import analyze

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--buffers',nargs='+',type=int,choices=[4,8,16,32,64],default=[4,8,16,32,64])
    args=parser.parse_args()
    env=os.environ.copy()
    env['LD_LIBRARY_PATH']='/usr/lib/wsl/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    results=[]
    output=ROOT/'results/v2/activities'
    output.mkdir(exist_ok=True)
    for mib in args.buffers:
      for n in [1,4,16]:
        for mode in ['sync','async']:
            base=output/(f'n{n}-{mode}' if mib==8 else f'n{n}-b{mib}-{mode}')
            trace=base.with_suffix('.csv')
            env['BENCH_TRACE_FILE']=str(trace)
            command=[str(ROOT/'build/bf16_stream_trace'),'--file',str(ROOT/'data/weights-5GiB.bf16'),
                     '--m',str(mib*128),'--blocks',str(5120//mib),'--n',str(n),'--mode',mode,'--repeats','1','--profile']
            exit_code=0
            if not trace.exists():
                with base.with_suffix('.log').open('w') as log:
                    run=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                exit_code=run.returncode
            errors=base.with_suffix('.log').read_text()
            if exit_code:
                result={'file':str(trace),'verified':False,'reason':f'profiler exit {exit_code}; see {base.name}.log'}
            else:
                try:
                    result=analyze(trace)
                except (ValueError,KeyError) as e:
                    result={'file':str(trace),'verified':False,'reason':str(e)}
            if 'CUPTI callback status=' in errors:
                result.update(verified=False,reason='CUPTI callback error or dropped records; see log')
            result.update(n=n,buffer_mib=mib,mode=mode,command=command,exit_code=exit_code)
            results.append(result)
            (ROOT/'results/v2/trace-summary.json').write_text(json.dumps(results,indent=2)+'\n')
            print(f'n={n} {mode}: verified={result["verified"]} '+result.get('reason',''),flush=True)


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Matched 5 GiB storage paths; uses the existing benchmark and telemetry unchanged."""
import argparse
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'results/v2/storage'
PS='/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe'


def counters():
    command='Get-CimInstance Win32_PerfRawData_PerfDisk_PhysicalDisk | Select-Object Name,DiskReadBytesPersec,DiskWriteBytesPersec | ConvertTo-Json -Compress'
    result=subprocess.run([PS,'-NoProfile','-Command',command],capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    return json.loads(result.stdout.decode('utf-8-sig'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full',action='store_true',help='CPU/sync/async, n=1/4/16; much longer')
    args=parser.parse_args()
    manifest=json.loads((OUTPUT/'datasets.json').read_text())
    if not all(item['complete'] for item in manifest['copies']):
        raise RuntimeError('copies are incomplete')
    # The first USB I/O case is already running independently. Never run storage tests concurrently.
    initial=OUTPUT/'external-usb/sweep/n1-b64-c0-io.json'
    counters_done=OUTPUT/'external-usb/counters-after-io.json'
    if not initial.exists() and not initial.with_suffix('.csv').exists():
        directory=OUTPUT/'external-usb'
        directory.mkdir(exist_ok=True)
        file=next(x['file'] for x in manifest['copies'] if x['name']=='external-usb')
        command=['python3',str(ROOT/'tools/benchmark_v2.py'),'--file',file,'--output',str(directory/'sweep'),
                 '--buffers','64','--stage','buffers','--repeats','10','--ns','1','--modes','io']
        (directory/'counters-before-io.json').write_text(json.dumps(counters())+'\n')
        with (directory/'io-progress.log').open('w') as log:
            run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        counters_done.write_text(json.dumps(counters())+'\n')
        if run.returncode:
            raise RuntimeError('initial USB I/O calibration failed')
    if not initial.exists() or not counters_done.exists():
        print('Waiting for the independent USB I/O calibration to finish...',flush=True)
        deadline=time.monotonic()+1800
        while not initial.exists() or not counters_done.exists():
            if time.monotonic()>deadline:
                raise RuntimeError('USB calibration did not finish; inspect its log')
            time.sleep(2)
    paths=[('external-usb',next(x['file'] for x in manifest['copies'] if x['name']=='external-usb')),
           ('internal-drvfs',next(x['file'] for x in manifest['copies'] if x['name']=='internal-drvfs')),
           ('internal-ext4',manifest['source'])]
    for name,file in paths:
        directory=OUTPUT/name
        directory.mkdir(exist_ok=True)
        command=['python3',str(ROOT/'tools/benchmark_v2.py'),'--file',file,'--output',str(directory/'sweep'),
                 '--buffers','64','--stage','buffers','--repeats','10','--ns',*(['1','4','16'] if args.full else ['1']),
                 '--modes',*(['io','cpu','sync','async'] if args.full else ['io','async'])]
        record={'command':command,'source_sha256':manifest['sha256'],'start_time':time.time(),'before':counters()}
        with (directory/'comparison-progress.log').open('w') as log:
            run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        record.update(end_time=time.time(),after=counters(),exit_code=run.returncode)
        (directory/'physical-counters.json').write_text(json.dumps(record,indent=2)+'\n')
        if run.returncode:
            raise RuntimeError(f'{name}: benchmark failed; see comparison-progress.log')
        subprocess.run(['python3',str(ROOT/'tools/summarize_v2.py'),'--input',str(directory/'sweep'),
                        '--output',str(directory)],check=True)
        print(name,'complete',flush=True)


if __name__=='__main__':
    main()

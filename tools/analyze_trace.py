#!/usr/bin/env python3
"""Analyze actual CUPTI intervals; no inference of kernel overlap from wall-time gains."""
import csv
import json
from pathlib import Path
import sys


def union(intervals):
    result = []
    for a, b in sorted(intervals):
        if b <= a:
            raise ValueError('invalid profiling timestamps')
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def duration(intervals):
    return sum(b-a for a, b in union(intervals))


def intersection(a, b):
    a, b = union(a), union(b)
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1])-max(a[i][0], b[j][0]))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def analyze(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    kinds = {kind: [(int(r['start_ns']), int(r['end_ns'])) for r in rows if r['kind'] == kind]
             for kind in ['pread', 'h2d', 'kernel', 'd2h']}
    result = {'file': str(path), 'counts': {k: len(v) for k, v in kinds.items()}}
    if not all(kinds[k] for k in ['pread', 'h2d', 'kernel']):
        result['verified'] = False
        result['reason'] = 'missing actual GPU activity records; overlap not verified'
        return result
    read_start=min(a for a, _ in kinds['pread'])
    read_end=max(b for _, b in kinds['pread'])
    host_correlated=not any(a<read_start-1_000_000_000 or b>read_end+1_000_000_000
                           for key in ['h2d','kernel','d2h'] for a,b in kinds[key])
    result['host_gpu_correlated']=host_correlated
    result['verified'] = True
    copied=sum(int(r['bytes']) for r in rows if r['kind']=='h2d')
    read=sum(int(r['bytes']) for r in rows if r['kind']=='pread')
    result['h2d_bytes']=copied
    result['read_bytes']=read
    if copied!=read or len(kinds['h2d'])!=len(kinds['pread']) or len(kinds['d2h'])!=len(kinds['pread']):
        result['verified']=False
        result['reason']='incomplete activity coverage for the no-cache profiling pass'
        return result
    for kind, intervals in kinds.items():
        result[kind+'_sum_ms'] = sum(b-a for a, b in intervals)/1e6
        result[kind+'_union_ms'] = duration(intervals)/1e6
    for a, b in [('h2d', 'kernel'), ('pread', 'kernel'), ('pread', 'h2d')]:
        result[a+'_'+b+'_overlap_ms'] = (intersection(kinds[a], kinds[b])/1e6
                                        if a!='pread' or host_correlated else None)
    result['h2d_kernel_overlap_fraction_of_kernel'] = intersection(kinds['h2d'], kinds['kernel'])/duration(kinds['kernel'])
    visible={k:v for k,v in kinds.items() if k!='pread' or host_correlated}
    origin = min(a for v in visible.values() for a, _ in v)
    width_ns = 40_000_000
    colors = {'pread': '#2563eb', 'h2d': '#d97706', 'kernel': '#16a34a', 'd2h': '#9333ea'}
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="240" viewBox="0 0 1050 240">',
           '<rect width="1050" height="240" fill="white"/>',
           '<text x="15" y="22" font-family="sans-serif" font-size="16">CUPTI GPU timeline — first 40 ms</text>']
    for i, (kind, intervals) in enumerate(visible.items()):
        y = 45+i*40
        svg.append(f'<text x="10" y="{y+15}" font-family="sans-serif">{kind}</text>')
        for a, b in intervals:
            left, right = max(0, a-origin), min(width_ns, b-origin)
            if right > left:
                svg.append(f'<rect x="{110+900*left/width_ns}" y="{y}" width="{max(.2,900*(right-left)/width_ns)}" height="22" fill="{colors[kind]}"/>')
    for i in range(9):
        svg.append(f'<text x="{110+900*i/8}" y="225" font-family="sans-serif" font-size="11">{i*5} ms</text>')
    svg.append('</svg>')
    path.with_suffix('.svg').write_text('\n'.join(svg))
    return result


if __name__ == '__main__':
    assert duration([(0, 3), (2, 4)]) == 4
    assert intersection([(0, 3), (2, 5)], [(1, 2), (4, 6)]) == 2
    results=[]
    for p in sys.argv[1:]:
        try:
            results.append(analyze(Path(p)))
        except (ValueError, KeyError) as e:
            results.append({'file':p,'verified':False,'reason':str(e)})
    print(json.dumps(results, indent=2))

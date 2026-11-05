#!/usr/bin/env python3
"""Summarize completed V2 runs, preserving raw data and the original COMPARISON.md."""
import csv
from datetime import datetime
import json
import re
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT/'results/v2/sweep'
MIB = 1024**2


def main():
    gpu = []
    for file in SWEEP.glob('gpu-*.csv'):
        with file.open() as f:
            for r in csv.DictReader(f):
                try:
                    data = {k.strip(): v.strip() for k, v in r.items()}
                    data['time'] = datetime.strptime(data['timestamp'], '%Y/%m/%d %H:%M:%S.%f').timestamp()
                    for k in ['temperature.gpu', 'power.draw [W]', 'memory.used [MiB]', 'clocks.current.sm [MHz]', 'clocks.current.memory [MHz]']:
                        data[k] = float(data[k])
                    gpu.append(data)
                except (ValueError, TypeError, AttributeError):
                    continue
    summaries = []
    for file in sorted(SWEEP.glob('*.json')):
        record = json.loads(file.read_text())
        with file.with_suffix('.csv').open() as f:
            rows = list(csv.DictReader(f))
        first = rows[0]
        summary = {'name': file.stem, 'n': int(first['n']), 'buffer_mib': int(first['block_bytes'])//MIB,
                   'cache_mib': int(first['cache_mib']), 'mode': first['mode'], 'count': len(rows)}
        setup=re.search(r'cache_setup_bytes=(\d+) cache_setup_ms=([\d.]+)',file.with_suffix('.log').read_text())
        summary['cache_setup_bytes']=int(setup[1]) if setup else 0
        summary['cache_setup_ms']=float(setup[2]) if setup else 0
        for key in ['total_ms', 'read_ms', 'h2d_ms', 'compute_ms', 'd2h_ms', 'read_bytes', 'h2d_bytes', 'cache_hits']:
            values = [float(r[key]) for r in rows]
            q = st.quantiles(values, n=4, method='inclusive')
            for label, v in [('median', st.median(values)), ('min', min(values)), ('max', max(values)),
                             ('iqr', q[2]-q[0]), ('stdev', st.stdev(values))]:
                summary[key+'_'+label] = v
        for key in ['rss_peak_bytes', 'pinned_peak_bytes', 'device_alloc_bytes', 'gpu_used_bytes', 'max_abs_error']:
            summary[key] = max(float(r[key]) for r in rows)
        summary['swap_peak_bytes'] = record['process_peaks_bytes']['VmSwap']
        for key in ['pswpin', 'pswpout']:
            summary[key+'_delta'] = record['swap_after'][key]-record['swap_before'][key]
        samples = [g for g in gpu if record['start_time'] <= g['time'] <= record['end_time']]
        summary['gpu_samples'] = len(samples)
        for key, name in [('temperature.gpu', 'temperature_C'), ('power.draw [W]', 'power_W'),
                          ('memory.used [MiB]', 'gpu_memory_MiB'), ('clocks.current.sm [MHz]', 'sm_clock_MHz'),
                          ('clocks.current.memory [MHz]', 'memory_clock_MHz')]:
            values = [g[key] for g in samples]
            for label, fn in [('min', min), ('median', st.median), ('max', max)]:
                summary[name+'_'+label] = fn(values) if values else None
        summaries.append(summary)
    if not summaries:
        raise RuntimeError('no completed configurations')
    with (ROOT/'results/v2/summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (ROOT/'results/v2/summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    # Standalone SVG: grouped bars, one panel per n, default cache disabled.
    colors = {'cpu': '#64748b', 'sync': '#d97706', 'async': '#16a34a'}
    selected = [s for s in summaries if not s['cache_mib'] and s['mode'] in colors]
    ymax = max(s['total_ms_median'] for s in selected)*1.15
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="420" viewBox="0 0 1100 420">',
           '<rect width="1100" height="420" fill="white"/>',
           '<text x="25" y="25" font-family="sans-serif" font-size="18">5 GiB BF16 — median total time (seconds), 10 passes; cache=0</text>']
    for panel, n in enumerate([1, 4, 16]):
        x0 = 55+panel*350
        svg.append(f'<text x="{x0+100}" y="55" font-family="sans-serif">n={n}</text>')
        for tick in range(5):
            y = 330-260*tick/4
            svg.append(f'<path d="M{x0},{y}h300" stroke="#ddd"/>')
            svg.append(f'<text x="{x0-32}" y="{y+4}" font-size="11" font-family="sans-serif">{ymax*tick/4000:.1f}</text>')
        for group, mib in enumerate([4, 8, 16, 32, 64]):
            for mode_idx, mode in enumerate(colors):
                found = [s for s in selected if s['n'] == n and s['buffer_mib'] == mib and s['mode'] == mode]
                if not found:
                    continue
                s = found[0]
                x = x0+group*60+mode_idx*16
                height = 260*s['total_ms_median']/ymax
                svg.append(f'<rect x="{x}" y="{330-height}" width="14" height="{height}" fill="{colors[mode]}"/>')
                lo, hi = (330-260*s['total_ms_'+label]/ymax for label in ['min', 'max'])
                svg.append(f'<path d="M{x+7},{lo}V{hi}" stroke="#222"/>')
            svg.append(f'<text x="{x0+group*60}" y="350" font-size="12" font-family="sans-serif">{mib}</text>')
        svg.append(f'<text x="{x0+75}" y="375" font-size="12" font-family="sans-serif">buffer per slot (MiB)</text>')
    for i, (mode, color) in enumerate(colors.items()):
        svg.append(f'<rect x="{350+120*i}" y="395" width="12" height="12" fill="{color}"/><text x="{367+120*i}" y="406" font-family="sans-serif" font-size="12">{mode}</text>')
    svg.append('</svg>')
    (ROOT/'results/v2/buffer-comparison.svg').write_text('\n'.join(svg))
    print(f'{len(summaries)} configurations, {sum(s["count"] for s in summaries)} passes summarized')


if __name__ == '__main__':
    main()

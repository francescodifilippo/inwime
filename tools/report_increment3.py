"""Render actual read/compute intervals of one controlled sample per mode."""

import argparse
import json
from pathlib import Path


def intervals(events, begin, end):
    depth, start = 0, None
    for stamp, kind, *_ in events:
        if kind == begin:
            if depth == 0:
                start = stamp
            depth += 1
        elif kind == end:
            depth -= 1
            if depth == 0:
                yield start, stamp
    assert depth == 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('evidence', type=Path)
    p.add_argument('output', type=Path)
    args = p.parse_args()
    data = json.loads(args.evidence.read_text())
    case = data['cases']['controlled']
    span = max(case[mode][0]['wall_ns'] for mode in ('sync', 'pipeline'))
    scale = 850/span
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="325" viewBox="0 0 1100 325">',
           '<rect width="1100" height="325" fill="white"/>',
           '<g font-family="sans-serif" font-size="14" fill="#18304b">',
           '<text x="20" y="28" font-size="20">InWiMe increment 3 — observed intervals, sample 1</text>',
           f'<text x="20" y="52">Real file reads + {data["options"]["read_delay"]*1000:g} ms injected delay; real scalar computation.</text>',
           '<text x="20" y="75">Same time scale. Warm cache. This is a controlled pipeline test, not NVMe or LLM acceleration.</text>']
    for tick in range(0, int(span/1e6)+1, 50):
        x = 175+tick*1e6*scale
        svg.append(f'<path d="M{x:.2f},98 V285" stroke="#d5dde5"/>')
        svg.append(f'<text x="{x:.2f}" y="306">{tick} ms</text>')
    for row, mode in enumerate(('sync', 'pipeline')):
        run = case[mode][0]
        events = run['trace']
        assert not run['stats']['trace_dropped']
        origin, y = events[0][0], 122+row*100
        svg.append(f'<text x="20" y="{y}">{mode}</text>')
        for offset, begin, end, color, label in ((0, 1, 2, '#2478be', 'read'),
                                               (30, 3, 4, '#15966f', 'compute')):
            svg.append(f'<text x="95" y="{y+offset}">{label}</text>')
            for start, stop in intervals(events, begin, end):
                x, width = 175+(start-origin)*scale, (stop-start)*scale
                svg.append(f'<rect x="{x:.3f}" y="{y+offset-15}" width="{width:.3f}" height="21" fill="{color}"/>')
    svg.append('</g></svg>\n')
    with args.output.open('x') as target:
        target.write('\n'.join(svg))


if __name__ == '__main__':
    main()

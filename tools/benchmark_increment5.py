"""Sequential fresh-process comparisons; never generates or overwrites checkpoints.

Use --large for one expensive full decoder pair. Without it, three small pairs
provide a repeatable pipeline-overhead comparison using an existing checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


def summarize(paths):
    values = [json.loads(Path(p).read_text()) for p in paths]
    if not values:
        raise ValueError('no evidence to summarize')
    identities = {v['checkpoint_identity'] for v in values}
    if len(identities) != 1:
        raise ValueError('comparison requires the same checkpoint')
    # Bitwise hashes cover every produced logit, hidden output and published KV.
    first = values[0]
    for v in values[1:]:
        if v['kind'] != first['kind']:
            raise ValueError('cannot compare GEMV with a decoder')
        if v['kind'] == 'complete-gemv':
            assert v['output_sha256'] == first['output_sha256']
        else:
            assert [p['output'] for p in v['phases']] == [p['output'] for p in first['phases']]
    result = dict(checkpoint_identity=first['checkpoint_identity'], bitwise_outputs_equal=True,
                  payload_bytes=first['payload_bytes'], cases={})
    for mode in ('sync', 'pipeline'):
        group = [v for v in values if v['mode'] == mode]
        if not group:
            continue
        walls = [v['wall_ns'] for v in group]
        result['cases'][mode] = dict(samples=len(group), wall_median_ns=statistics.median(walls),
            wall_min_ns=min(walls), wall_max_ns=max(walls),
            peak_rss_max_kib=max(v['peak_rss_kib'] for v in group),
            reserved_peak_bytes=max(v['stats']['reserved_peak_bytes'] for v in group),
            read_bytes=[v['stats']['inference_read_bytes'] for v in group],
            process_read_bytes=[v['workload_process_io'].get('read_bytes')
                                if v['workload_process_io'] else None for v in group],
            overlap_median_ns=statistics.median(v['stats']['overlap_ns'] for v in group))
    if set(result['cases']) == {'sync', 'pipeline'}:
        a, b = (result['cases'][m]['wall_median_ns'] for m in ('sync', 'pipeline'))
        result['pipeline_wall_change_percent'] = 100*(b/a-1)
        result['sync_over_pipeline_speedup'] = a/b
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--large', action='store_true')
    parser.add_argument('--samples', type=int)
    parser.add_argument('--tile', type=int, default=262144)
    parser.add_argument('--read', type=int, default=262144)
    parser.add_argument('--tensor')
    parser.add_argument('--address-limit-mib', type=int, default=512)
    args = parser.parse_args()
    samples = args.samples if args.samples is not None else 1 if args.large else 3
    if samples < 1:
        raise ValueError('positive sample count required')
    args.output.mkdir(parents=True, exist_ok=False)
    paths = []
    for sample in range(samples):
        for mode in (('sync', 'pipeline') if sample % 2 == 0 else ('pipeline', 'sync')):
            target = args.output/f'{mode}-{sample+1}.json'
            command = [sys.executable, '-m', 'tools.verify_increment5', str(args.path),
                       '--mode', mode, '--tile', str(args.tile), '--read', str(args.read),
                       '--address-limit-mib', str(args.address_limit_mib), '--output', str(target)]
            if args.tensor:
                command.extend(['--tensor', args.tensor])
            subprocess.run(command, check=True)
            paths.append(target)
    result = summarize(paths)
    result['evidence_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with (args.output/'summary.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()

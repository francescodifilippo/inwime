"""Sequential paired experiments on already generated checkpoint4 datasets."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


def summarize(directory):
    directory = Path(directory)
    result = {}
    for label, glob in [('medium-16384', 'final-medium-16384-{}-*.json'),
                        ('medium-65536', 'final-medium-65536-{}-*.json'),
                        ('decoder-dense', 'final-decoder-dense-{}-*.json'),
                        ('decoder-moe', 'final-decoder-moe-{}-*.json')]:
        cases = {mode: [json.loads(p.read_text()) for p in sorted(directory.glob(glob.format(mode)))]
                 for mode in ('sync', 'pipeline')}
        assert len(cases['sync']) == len(cases['pipeline']) >= 3
        for a, b in zip(cases['sync'], cases['pipeline']):
            assert a.get('output', a.get('checksum')) == b.get('output', b.get('checksum'))
            assert a['stats']['inference_read_bytes'] == b['stats']['inference_read_bytes']
        result[label] = {}
        for mode, runs in cases.items():
            result[label][mode] = dict(
                wall_median_ms=statistics.median(r['wall_ns'] for r in runs)/1e6,
                wall_min_ms=min(r['wall_ns'] for r in runs)/1e6,
                wall_max_ms=max(r['wall_ns'] for r in runs)/1e6,
                peak_rss_kib=max(r['peak_rss_kib'] for r in runs),
                stats={k: statistics.median(r['stats'][k] for r in runs)
                       for k in ('read_service_ns', 'compute_ns', 'data_wait_ns', 'overlap_ns',
                                 'bf16_expand_ns', 'inference_read_bytes', 'backend_read_calls',
                                 'reserved_peak_bytes', 'weight_slot_bytes', 'peak_occupied_slots')})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-root', type=Path, default=Path('checkpoints'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--summarize-only', action='store_true')
    p.add_argument('--large', action='store_true')
    args = p.parse_args()
    if not args.summarize_only:
        args.output.mkdir(parents=True, exist_ok=False)
        def run(module, name, options, filename):
            subprocess.run([sys.executable, '-m', 'tools.'+module,
                            str(args.checkpoint_root/name), *options,
                            '--output', str(args.output/filename)], check=True)
        for tile in (16384, 65536):
            for rep in range(3):
                for mode in (('sync','pipeline') if rep % 2 == 0 else ('pipeline','sync')):
                    run('verify_increment4', 'inc4-medium', ['--tile',str(tile),'--mode',mode],
                        f'final-medium-{tile}-{mode}-{rep}.json')
        for variant in ('dense','moe'):
            for rep in range(3):
                for mode in (('sync','pipeline') if rep % 2 == 0 else ('pipeline','sync')):
                    run('decoder_increment4', 'inc4-'+variant, ['--mode',mode],
                        f'final-decoder-{variant}-{mode}-{rep}.json')
        if args.large:
            run('verify_increment4', 'inc4-large', ['--raw','--cache','advised-cold'],
                'large-progressive-final.json')
            run('verify_increment4', 'inc4-large', ['--tile','4096','--stride','65536',
                '--mode','pipeline','--integrity','metadata'], 'large-samples-final.json')
    result = summarize(args.output)
    with (args.output/'summary.json').open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

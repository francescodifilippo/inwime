"""GPU cache correctness, complete and partial hits, repeat reuse, and n=1/4/16."""
import csv
import io
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
binary = root / 'build/bf16_stream'

with tempfile.TemporaryDirectory(dir=root) as directory:
    file = Path(directory) / 'weights.bf16'
    # Three 1 MiB blocks: one fits in a 1 MiB cache, all fit in 4 MiB.
    common = ['--file', str(file), '--m', '32', '--k', '16384', '--blocks', '3', '--threads', '2']
    subprocess.run([str(binary), *common, '--prepare'], check=True, capture_output=True)
    for n in [1, 4, 16]:
        for mode in ['sync', 'async']:
            for mib, hits in [(0, 0), (1, 1), (4, 3)]:
                result = subprocess.run([str(binary), *common, '--n', str(n), '--mode', mode,
                                         '--cache-mib', str(mib), '--repeats', '2', '--full-check'],
                                        check=True, capture_output=True, text=True)
                rows = list(csv.DictReader(io.StringIO(result.stdout)))
                assert len(rows) == 2
                for r in rows:
                    assert int(r['checked']) == 32 * n * 3
                    assert float(r['max_abs_error']) == 0
                    assert int(r['cache_hits']) == hits
                    assert int(r['read_bytes']) == (3 - hits) * 1024**2
                    assert int(r['h2d_bytes']) == (3 - hits) * 1024**2
                    assert int(r['pinned_peak_bytes']) == 2 * (1024**2 + 32*n*4)
    print('PASS: full output checks n=1/4/16, sync/async, cache miss/partial/full, two passes')

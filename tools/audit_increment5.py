"""Read-only consolidation audit of captured increment-5 evidence and local payloads.

Recomputes sizes, forward coverage and units without importing the runtime.
Optional payload SHA scans are audit traffic, never inference performance evidence.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import statistics
import struct
import time


def read_json(path):
    return json.loads(Path(path).read_text())


def checkpoint(path, verify_payload):
    m = read_json(path/'manifest.json')
    intervals = {name: [] for name in m['files']}
    for name, tensor in m['tensors'].items():
        assert tensor['dtype'] == 'BF16'
        assert math.prod(tensor['shape'])*2 == tensor['nbytes']
        end = 0
        for s in tensor['segments']:
            assert s['start'] == end and s['nbytes'] > 0 and s['nbytes'] % 2 == 0
            assert s['offset'] >= 0 and s['offset'] % 2 == 0
            end += s['nbytes']//2
            intervals[s['shard']].append((s['offset'], s['offset']+s['nbytes']))
        assert end*2 == tensor['nbytes']
    payload = allocated = hashed = 0
    inodes = set()
    holes_checked = 0
    began = time.monotonic_ns()
    for name, entry in m['files'].items():
        p = path/name
        s = p.lstat()
        assert stat.S_ISREG(s.st_mode) and s.st_size == entry['nbytes']
        assert s.st_blocks*512 >= s.st_size
        assert (s.st_dev, s.st_ino) not in inodes
        inodes.add((s.st_dev, s.st_ino))
        end = 0
        for low, high in sorted(intervals[name]):
            assert low == end and high > low
            end = high
        assert end == s.st_size
        payload += s.st_size
        allocated += s.st_blocks*512
        with p.open('rb', buffering=0) as stream:
            if hasattr(os, 'SEEK_HOLE'):
                assert os.lseek(stream.fileno(), 0, os.SEEK_HOLE) == s.st_size
                holes_checked += 1
                stream.seek(0)
            if verify_payload:
                digest = hashlib.sha256()
                while raw := stream.read(2**20):
                    digest.update(raw)
                    hashed += len(raw)
                assert digest.hexdigest() == entry['sha256'], name
    # Independent, indexed word checks: no runtime loader or native generator.
    sampled = 0
    mask = (1 << 64)-1
    for name, tensor in m['tensors'].items():
        count = math.prod(tensor['shape'])
        for index in sorted({0, count//2, count-1}):
            segment = next(s for s in tensor['segments']
                           if s['start'] <= index < s['start']+s['nbytes']//2)
            with (path/segment['shard']).open('rb', buffering=0) as stream:
                stream.seek(segment['offset']+2*(index-segment['start']))
                word, = struct.unpack('<H', stream.read(2))
            if 'generation' in m:
                g = m['generation'][name]
                seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little')
                assert seed == g['seed']
                z = (index+seed+0x9e3779b97f4a7c15) & mask
                z = ((z ^ (z >> 30))*0xbf58476d1ce4e5b9) & mask
                z = ((z ^ (z >> 27))*0x94d049bb133111eb) & mask
                z ^= z >> 31
                expected = (0x3f80 if 'norm' in name else
                            ((z >> 48) & 0x8000) | g['exponent'] << 7 | (z & 127))
            else:
                expected = (0x3f80 if 'norm' in name else
                            (0x8000 if index % 256 >= 128 else 0) | 0x3d80 | (index % 128))
            assert word == expected, (name, index, word, expected)
            sampled += 1
    return dict(payload_bytes=payload, allocated_bytes=allocated, shards=len(inodes),
                non_sparse_files_checked=holes_checked, sampled_native_bf16_words=sampled,
                payload_sha256_verified=verify_payload, audit_payload_read_bytes=hashed,
                audit_wall_ns=time.monotonic_ns()-began)


def units(n):
    return dict(bytes=n, GB=n/10**9, GiB=n/2**30)


def audit(root, verify_payload=False):
    large = Path('checkpoints/inc5-dense-12g')
    legacy = Path('checkpoints/inc4-large')
    m = read_json(large/'manifest.json')
    c, tensors = m['config'], m['tensors']
    assert m['variant'] == 'dense' and len(tensors) == 4+9*c['layers']
    matrix_bytes = sum(t['nbytes'] for n, t in tensors.items()
                       if len(t['shape']) == 2 and n not in ('embedding', 'position'))
    norm_bytes = sum(t['nbytes'] for n, t in tensors.items() if 'norm' in n)
    results = {}
    pair = [read_json(root/f'large-{mode}-1.json') for mode in ('sync', 'pipeline')]
    assert pair[0]['phases'] != []
    assert [p['output'] for p in pair[0]['phases']] == [p['output'] for p in pair[1]['phases']]
    for r in pair:
        consumed = r['consumed_tokens']
        phases = r['phases']
        assert len(consumed) == 5 and len(phases) == 4
        assert [p['kind'] for p in phases] == ['prefill', 'decode', 'decode', 'decode']
        assert [p['output']['length'] for p in phases] == [2, 3, 4, 5]
        assert consumed == r['prompt']+r['generated_tokens'][:-1]
        assert all(p['output']['kv_prefix_preserved'] for p in phases)
        for p in phases:
            logits = p['output']['logits_last']
            assert len(logits) == c['vocab'] and all(math.isfinite(x) for x in logits)
            assert max(range(len(logits)), key=lambda i: (logits[i], -i)) == p['output']['generated_token']
        expected = {name: (2*c['width'] if name in ('embedding', 'position')
                          else t['nbytes'])*len(consumed) for name, t in tensors.items()}
        s = r['stats']
        assert s['tensor_read_bytes'] == expected
        assert s['inference_read_bytes'] == sum(expected.values())
        assert s['gemv_weight_elements']*2 == matrix_bytes*5
        assert s['gemv_flops'] == matrix_bytes*5
        assert s['requests'] == s['completed'] == s['released'] and s['failed'] == 0
        assert r['resources_after_close'] == dict(fds=0, numeric_buffers=0, live_requests=0)
        used = matrix_bytes+norm_bytes+2*c['width']*(len(set(consumed))+len(consumed))
        assert used == r['unique_forward_weight_bytes']
        total = r['audit_before']['meminfo']['MemTotal']
        assert total == r['audit_after']['meminfo']['MemTotal']
        assert used > total and matrix_bytes > total
        assert r['audit_before']['rlimit_as'] == [512*2**20]*2
        assert s['integrity_read_bytes'] == sum(f['nbytes'] for f in m['files'].values())
        validation = r['independent_validation']
        assert validation['tensor_count'] == 7*c['layers']+1 and validation['rows'] == 507
        assert all(v['absolute_error'] <= v['error_bound'] for v in validation['checks'])
        results[r['mode']] = dict(used_weights=units(used), full_matrices=units(matrix_bytes),
            wsl_memtotal=units(total), level_B_wsl=True, level_B_process=True,
            ru_maxrss_KiB=r['peak_rss_kib'], ru_maxrss_MiB=r['peak_rss_kib']/1024,
            VmHWM_bytes=r['memory_before_close']['VmHWM'],
            VmHWM_MiB=r['memory_before_close']['VmHWM']/2**20,
            workload_read_bytes=s['inference_read_bytes'], prefill_ns=r['prefill_ns'],
            decode_ns=r['decode_ns'], decode_tokens_s=3*10**9/sum(r['decode_ns']))
    gemv = [read_json(root/'gemv-6g'/f'{mode}-1.json') for mode in ('sync', 'pipeline')]
    assert gemv[0]['output_sha256'] == gemv[1]['output_sha256']
    for r in gemv:
        s = r['stats']
        assert math.prod(r['shape']) == s['gemv_weight_elements'] == 3221225472
        assert s['inference_read_bytes'] == 6*2**30 and s['gemv_flops'] == 6*2**30
        assert s['peak_region_elements'] <= 262144
        assert s['requests'] == s['completed'] == s['released']
        assert r['caller_numeric_bytes'] == 8*sum(r['shape'])
        assert r['audit_before']['rlimit_as'] == [2048*2**20]*2
        assert r['independent_validation']['rows'] == 4
    summaries = {}
    for sub in ('gemv-6g', 'gemv-reduction', 'small'):
        summary = read_json(root/sub/'summary.json')
        cases = [read_json(p) for p in sorted((root/sub).glob('*-[123].json'))]
        for mode in ('sync', 'pipeline'):
            values = [r['wall_ns'] for r in cases if r['mode'] == mode]
            assert statistics.median(values) == summary['cases'][mode]['wall_median_ns']
        for name, digest in summary['evidence_sha256'].items():
            assert hashlib.sha256((root/sub/name).read_bytes()).hexdigest() == digest
        summaries[sub] = summary['pipeline_wall_change_percent']
    sizes = {str(p): checkpoint(p, verify_payload) for p in (large, legacy)}
    for r in pair:
        assert sizes[str(large)]['payload_bytes'] == r['payload_bytes']
        assert sizes[str(large)]['allocated_bytes'] == r['allocated_payload_bytes']
    physical = read_json(root/'resource-audit.json')['host_total_physical_memory_bytes']
    assert pair[0]['unique_forward_weight_bytes'] < physical
    return dict(decoder=results, checkpoint_audits=sizes, level_C=False,
                host_physical_bytes=physical, standalone_gemv_bytes=6*2**30,
                pipeline_wall_change_percent=summaries,
                limitations=['Large oracle is partial: selected rows with runtime-produced inputs.',
                             'Page cache is outside RSS; RLIMIT_AS is not a system-memory limit.',
                             'Audit SHA traffic is separate from captured inference reads.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('results/increment5'))
    parser.add_argument('--verify-payload', action='store_true', help='reads existing 24 GiB once for SHA')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.root, args.verify_payload)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()

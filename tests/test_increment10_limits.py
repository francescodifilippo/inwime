"""Increment 10, C2: preregistered memory limits for sync (RLIMIT_AS 256 MiB) and pipeline (384 MiB).

1. The derivation in protocol.json is recomputed from the versioned increment-9
   evidence (VmPeak and arena plan of the 7B sync/pipeline runs) and the 14B plan.
2. Synthetic check on the 14B-shaped fixture, fresh processes: the address-space
   cost of the pipeline is a structural, model-size-independent overhead (I/O
   thread stack + its malloc arena), comparable to the 75.5 MB measured on the 7B,
   and the unchanged runtime completes in both modes under the proposed limits
   with byte-identical outputs.
"""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools import fixture_qwen10

ROOT = Path(__file__).resolve().parents[1]
MiB = 2**20


def limits():
    return json.loads((ROOT/'results'/'increment10'/'protocol.json').read_text())['memory_limits']


def m9(name):
    return json.loads((ROOT/'results'/'increment9'/name).read_text())


def test_derivation_recomputed_from_m9_evidence():
    LIMITS = limits()
    sync, pipe = m9('phaseB-sync-runtime.json'), m9('phaseB-pipeline-runtime.json')
    d = LIMITS['derivation']
    assert d['m9_7b_sync_vmpeak'] == sync['process_after_startup']['VmPeak_bytes'] == 142_110_720
    assert d['m9_7b_pipeline_vmpeak'] == pipe['process_after_startup']['VmPeak_bytes'] == 217_624_576
    assert d['m9_7b_sync_plan'] == sync['arena_plan_bytes'] and d['m9_7b_pipeline_plan'] == pipe['arena_plan_bytes']
    est_sync = d['m9_7b_sync_vmpeak'] + LIMITS['plan_bytes']['sync'] - d['m9_7b_sync_plan']
    est_pipe = d['m9_7b_pipeline_vmpeak'] + LIMITS['plan_bytes']['pipeline'] - d['m9_7b_pipeline_plan']
    assert (est_sync, est_pipe) == (d['estimated_14b_sync_vmpeak'], d['estimated_14b_pipeline_vmpeak'])
    sync_as, pipe_as = LIMITS['rlimit_as_bytes']['sync'], LIMITS['rlimit_as_bytes']['pipeline']
    assert (sync_as, pipe_as) == (256*MiB, 384*MiB)
    assert sync_as - est_sync >= 64*MiB                          # sync keeps the M9 limit with margin
    assert 256*MiB - est_pipe < 16*MiB                           # 256 MiB would leave < 16 MiB for the pipeline
    assert pipe_as - est_pipe >= 128*MiB                         # 384 MiB leaves >= 128 MiB
    assert LIMITS['anon_plus_kernel_max_bytes'] == LIMITS['vmhwm_max_bytes'] == 256*MiB
    assert LIMITS['scope']['memory_max_bytes'] == 2*2**30 and LIMITS['scope']['swap_max_bytes'] == 0


CHILD = r'''
import hashlib, json, resource, sys
from inwime_runtime.qwen2 import Qwen2Runtime
from inwime_runtime.math_cpu import NativeMath
directory, mode, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
def status():
    out = {}
    for line in open('/proc/self/status'):
        k, _, v = line.partition(':')
        if k in ('VmPeak', 'VmHWM'):
            out[k] = int(v.split()[0])*1024
    return out
before = status()
with Qwen2Runtime(directory, capacity=16, io_mode=mode, math_backend=NativeMath('build/libinwime_cpu.so')) as r:
    with r.prefill([3, 36, 7, 30, 5]) as out:
        digest = hashlib.sha256(bytes(out.logits) + b''.join(bytes(h) for h in out.hidden)).hexdigest()
    for t in (36, 12, 2):
        with r.decode(t) as out:
            digest = hashlib.sha256(digest.encode() + bytes(out.logits)).hexdigest()
    plan = r.plan.total
print(json.dumps(dict(before=before, after=status(), digest=digest, plan=plan)))
'''


def child(directory, mode, limit):
    out = subprocess.run([sys.executable, '-B', '-c', CHILD, str(directory), mode, str(limit)],
                         cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope='module')
def fixture_dir(tmp_path_factory):
    return fixture_qwen10.write(tmp_path_factory.mktemp('limits')/'q10')


def test_pipeline_overhead_is_structural_and_runs_under_limits(fixture_dir):
    LIMITS = limits()
    sync = child(fixture_dir, 'sync', LIMITS['rlimit_as_bytes']['sync'])
    pipe = child(fixture_dir, 'pipeline', LIMITS['rlimit_as_bytes']['pipeline'])
    assert sync['digest'] == pipe['digest']                      # same bytes in both modes, three decodes
    delta = pipe['after']['VmPeak'] - sync['after']['VmPeak']
    plan_delta = pipe['plan'] - sync['plan']
    structural = delta - plan_delta
    m9_structural = LIMITS['derivation']['m9_7b_pipeline_vmpeak'] - LIMITS['derivation']['m9_7b_sync_vmpeak'] - (
        LIMITS['derivation']['m9_7b_pipeline_plan'] - LIMITS['derivation']['m9_7b_sync_plan'])
    # thread stack (8 MiB) + per-thread malloc arena (64 MiB) + possible 64 MiB alignment transient,
    # whatever the model: between the M9 observation and that plus one transient
    assert 32*MiB <= structural <= m9_structural + 64*MiB + 16*MiB, (structural, m9_structural)
    d = LIMITS['derivation']
    assert d['estimated_14b_pipeline_vmpeak_worst_case'] == d['estimated_14b_pipeline_vmpeak'] + 64*MiB
    worst = d['m9_7b_sync_vmpeak'] + LIMITS['plan_bytes']['pipeline'] - d['m9_7b_sync_plan'] + structural
    assert worst > 256*MiB                                      # 256 MiB cannot hold the observed overhead
    assert LIMITS['rlimit_as_bytes']['pipeline'] - worst >= 48*MiB
    assert pipe['after']['VmPeak'] < LIMITS['rlimit_as_bytes']['pipeline']

"""Increment-9 tooling: unique-byte attribution, cgroup parsing, verifier and gate refusals."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.benchmark_increment9 import unique_active_bytes, expected_reads
from tools.cgroup9 import keyed, parse_final, disk_delta
from tools.report_increment9 import gate, NOT_PROVEN
from tools import verify_increment9
from tools.verify_increment9 import verify

T = lambda shard, offset, elements, shape: SimpleNamespace(shard=shard, offset=offset, elements=elements, shape=shape)


def catalog(tied):
    emb = T('a', 0, 16*4, (16, 4))
    head = emb if tied else T('a', 1000, 16*4, (16, 4))
    return SimpleNamespace(spec=SimpleNamespace(width=4), tensors={
        'embedding': emb, 'lm_head': head, 'layers.0.q': T('a', 500, 16, (4, 4)), 'final_norm': T('a', 900, 4, (4,))})


def test_unique_bytes_tied_counts_alias_once():
    c = catalog(tied=True)
    reads = expected_reads(c, 3)
    total, parts = unique_active_bytes(c, reads, [1, 2, 2])
    assert total == 16*4*2 + 16*2 + 4*2  # full shared table once, no extra row bytes
    assert len(parts) == 3


def test_unique_bytes_untied_counts_distinct_rows_only():
    c = catalog(tied=False)
    total, _ = unique_active_bytes(c, expected_reads(c, 3), [1, 2, 2])
    assert total == 2*4*2 + 16*4*2 + 16*2 + 4*2  # two distinct embedding rows + head


def test_unique_bytes_unread_tensor_not_counted():
    c = catalog(tied=False)
    reads = expected_reads(c, 1); reads['layers.0.q'] = 0
    total, _ = unique_active_bytes(c, reads, [1])
    assert total == 4*2 + 16*4*2 + 4*2


def test_finalizer_parsing_and_disk_delta(tmp_path):
    p = tmp_path/'f.txt'
    p.write_text('== cgroup.path\n/x\n== worker.exit\n0\n== memory.peak\n2147483648\n== memory.max\n2147483648\n'
                 '== memory.swap.max\n0\n== memory.events\nlow 0\nmax 7\noom 0\noom_kill 0\n')
    out = parse_final(p)
    assert out['memory.peak'] == 2147483648 and out['memory.events']['max'] == 7 and out['worker.exit'] == 0
    assert keyed('anon 5\nfile 9\nbad x\n') == {'anon': 5, 'file': 9}
    before = {'disks': [{'Name': '0 C:', 'DiskReadBytesPersec': 10, 'DiskReadsPersec': 1, 'DiskWriteBytesPersec': 0}]}
    after = {'disks': [{'Name': '0 C:', 'DiskReadBytesPersec': 30, 'DiskReadsPersec': 3, 'DiskWriteBytesPersec': 5}]}
    assert disk_delta(before, after) == {'0 C:': {'DiskReadBytesPersec': 20, 'DiskReadsPersec': 2, 'DiskWriteBytesPersec': 5}}
    assert disk_delta({'error': 'x'}, after) is None


def test_gate_missing_evidence_is_not_proven(tmp_path):
    (tmp_path/'protocol.json').write_text('{}')
    report = gate(tmp_path)
    assert set(report['verdicts'].values()) == {NOT_PROVEN}


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    """Minimal coherent runtime capture + oracle for the verifier."""
    capture = tmp_path/'cap'; capture.mkdir()
    rng = np.random.default_rng(0)
    seq, layers, vocab, width, kvw = [5, 7, 9], 2, 6, 4, 2
    full = dict(logits=rng.normal(size=(3, vocab)), hidden=rng.normal(size=(layers, 3, width)),
                kv=rng.normal(size=(layers, 2, 3, kvw)))
    full['logits'][0, 9 % vocab] = 50; full['logits'][1, 3] = 50; full['logits'][2, 1] = 50
    phases, captures, start = [], {}, 0
    for i, ids in enumerate([[5, 7], [9]]):
        stop = start+len(ids)
        arrays = dict(logits=full['logits'][start:stop], hidden=full['hidden'][:, start:stop], kv=full['kv'][:, :, :stop])
        captures[f'decode.{i}'] = {}
        for kind, values in arrays.items():
            raw = np.ascontiguousarray(values, dtype='<f8').tobytes()
            (capture/f'{i}.{kind}').write_bytes(raw)
            captures[f'decode.{i}'][kind] = dict(file=f'{i}.{kind}', shape=list(values.shape), dtype='<f8',
                                                 nbytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        phases.append(dict(ids=ids, greedy=arrays['logits'].argmax(-1).tolist()))
        start = stop
    runtime = tmp_path/'runtime.json'
    runtime.write_text(json.dumps(dict(phases=phases, captures=captures, input_ids=[5, 7], checkpoint='x')))
    oracle = tmp_path/'oracle.npz'
    with oracle.open('wb') as f:
        np.savez(f, **{f'full.{k}': v for k, v in full.items()})
    meta = dict(contract='stock_fp32', dtype='float32', precision_profile='stock', attention_implementation='eager',
                revision='r', sequence=seq, prompt_length=2, tokenizer=[], upstream_greedy_matches_sequence=True,
                capture_sha256=hashlib.sha256(oracle.read_bytes()).hexdigest())
    oracle.with_suffix('.json').write_text(json.dumps(meta))
    protocol = tmp_path/'protocol.json'
    protocol.write_text(json.dumps(dict(fp32_comparison=dict(atol=2e-3, rtol=2e-4),
                                        fp64_comparison=dict(atol=2e-5, rtol=2e-6), phase_b=dict(revision='r'))))
    monkeypatch.setattr(verify_increment9, 'LlamaTokenizer', lambda checkpoint: None)
    return SimpleNamespace(runtime=runtime, capture=capture, oracle=oracle, protocol=protocol, meta=meta)


def test_verifier_exact_pass(evidence):
    report = verify(evidence.runtime, evidence.capture, evidence.oracle, evidence.protocol)
    assert report['passed'] and report['compared_elements']['logits'] == 18


def rewrite(path, **changes):
    d = json.loads(path.read_text()); d.update(changes); path.write_text(json.dumps(d))


@pytest.mark.parametrize('case', ['threshold', 'dtype', 'profile', 'revision', 'sequence', 'digest', 'one_phase'])
def test_verifier_refuses_inconsistent_evidence(evidence, case):
    meta = evidence.oracle.with_suffix('.json')
    if case == 'threshold':
        rewrite(evidence.protocol, fp32_comparison=dict(atol=1e-2, rtol=2e-4))
    if case == 'dtype': rewrite(meta, dtype='float64')
    if case == 'profile': rewrite(meta, precision_profile='full64')
    if case == 'revision': rewrite(meta, revision='other')
    if case == 'sequence': rewrite(meta, sequence=[5, 7, 8])
    if case == 'digest': rewrite(meta, capture_sha256='0'*64)
    if case == 'one_phase':
        d = json.loads(evidence.runtime.read_text()); d['phases'] = d['phases'][:1]
        d['captures'] = {'decode.0': d['captures']['decode.0']}; rewrite(evidence.runtime, **d)
        rewrite(meta, sequence=[5, 7])
    with pytest.raises(ValueError):
        verify(evidence.runtime, evidence.capture, evidence.oracle, evidence.protocol)


@pytest.mark.parametrize('case', ['value', 'greedy', 'upstream_greedy'])
def test_verifier_detects_numerical_failures(evidence, case):
    d = json.loads(evidence.runtime.read_text())
    if case == 'value':
        desc = d['captures']['decode.1']['hidden']
        path = evidence.capture/desc['file']
        values = np.frombuffer(path.read_bytes(), '<f8').copy(); values[0] += 1e-2
        raw = values.tobytes(); path.write_bytes(raw)
        desc['sha256'] = hashlib.sha256(raw).hexdigest()
    if case == 'greedy':
        d['phases'][1]['greedy'] = [0]
    if case == 'upstream_greedy':
        rewrite(evidence.oracle.with_suffix('.json'), upstream_greedy_matches_sequence=False)
    evidence.runtime.write_text(json.dumps(d))
    assert not verify(evidence.runtime, evidence.capture, evidence.oracle, evidence.protocol)['passed']


EVIDENCE = Path('results/increment9')


@pytest.mark.skipif(not (EVIDENCE/'summary.json').exists(), reason='increment-9 evidence not acquired')
@pytest.mark.parametrize('case,condition,expected', [
    ('oom_kill', 'G', 'FAIL'), ('missing_full64', 'F', 'NOT PROVEN'), ('unique_bytes', 'D', 'FAIL'),
    ('warm_charged', 'C', 'FAIL'), ('not_evicted', 'B', 'FAIL'), ('peak_over', 'B', 'FAIL'),
    ('parity_fail', 'F', 'FAIL'), ('pytest_failed', 'H', 'FAIL'), ('one_decode_less', 'B', 'FAIL')])
def test_gate_rejects_altered_real_evidence(tmp_path, case, condition, expected):
    import shutil
    d = tmp_path/'e'
    shutil.copytree(EVIDENCE, d)
    edit = lambda name, fn: (lambda data: (fn(data), (d/name).write_text(json.dumps(data))))(json.loads((d/name).read_text()))
    if case == 'oom_kill':
        edit('phaseB-pipeline-supervisor.json', lambda x: x['finalizer']['memory.events'].update(oom_kill=1))
    if case == 'missing_full64': (d/'parity-full64.json').unlink()
    if case == 'unique_bytes': edit('phaseB-sync-runtime.json', lambda x: x.update(unique_active_bf16_bytes=1))
    if case == 'warm_charged':
        edit('phaseA-W-runtime.json', lambda x: x['inference_proc_io_delta'].update(read_bytes=10**10))
    if case == 'not_evicted':
        edit('phaseA-P-supervisor.json', lambda x: [v.update(resident_bytes=v['size_bytes'])
                                                    for v in x['residency_after_eviction'].values()])
    if case == 'peak_over': edit('phaseA-P-supervisor.json', lambda x: x['finalizer'].update({'memory.peak': 2**31+4096}))
    if case == 'parity_fail': edit('parity-stock_fp32.json', lambda x: x.update(passed=False))
    if case == 'pytest_failed': (d/'pytest-final.txt').write_text('1 failed, 474 passed in 1s\n')
    if case == 'one_decode_less':
        edit('phaseA-P-runtime.json', lambda x: x.update(phases=x['phases'][:3]))
    assert gate(d)['verdicts'][condition] == expected

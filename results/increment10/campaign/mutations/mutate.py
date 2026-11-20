"""Mutations of copies of the real campaign evidence: the gate must never return PASS for the mutated condition."""
import json, shutil, sys, os
from pathlib import Path
sys.path.insert(0, os.getcwd())
from tools import report_increment10 as g
REAL = Path('results/increment10/campaign'); ART = Path('artifacts/increment10')
TMP = Path(sys.argv[1])
def j(p): return json.loads(p.read_text())
def w(p, d): p.write_text(json.dumps(d))
M = {
 'D: W declared as C (R1 cache W)': ('D', lambda c: (lambda d: (d.update(cache='W', condition='as-is'), w(c/'R1-run.json', d)))(j(c/'R1-run.json'))),
 'D: unique bytes of the 7B': ('D', lambda c: (lambda d: (d.update(unique_active_bf16_bytes=14141295616), w(c/'R1-forward.json', d)))(j(c/'R1-forward.json'))),
 'D: host RAM missing': ('D', lambda c: (lambda d: (d['after']['windows'].pop('TotalPhysicalMemory'), w(c/'R1-host-start.json', d)))(j(c/'R1-host-start.json'))),
 'D: MemTotal as host RAM': ('D', lambda c: (lambda d: (d['after']['windows'].update(TotalPhysicalMemory=11483848704, DimmBytes=11483848704), w(c/'R1-host-start.json', d)))(j(c/'R1-host-start.json'))),
 'C: sustained paging annotation on R2': ('C', lambda c: (lambda d: (d['observer']['summary']['annotations'].append(dict(preregistered_action='annotate-io-attribution-invalid', quantity='host_pages_input_per_s')), w(c/'R2-run.json', d)))(j(c/'R2-run.json'))),
 'C: residency 5% after eviction R3': ('C', lambda c: (lambda d: ([v.update(resident_bytes=v['size_bytes']//20) for v in d['residency_after_eviction'].values()], w(c/'R3-scope.json', d)))(j(c/'R3-scope.json'))),
 'C: L3 below L1 on R1': ('C', lambda c: (lambda d: (d['l3_delta']['checkpoint'].update(read_bytes=10**9), w(c/'R1-run.json', d)))(j(c/'R1-run.json'))),
 'B: OOM event in R2': ('B', lambda c: (lambda d: (d['finalizer']['memory.events'].update(oom=1), w(c/'R2-scope.json', d)))(j(c/'R2-scope.json'))),
 'B: pipeline RLIMIT 256 MiB': ('B', lambda c: (lambda d: (d.update(rlimit_as_bytes=[268435456]*2), w(c/'R2-forward.json', d)))(j(c/'R2-forward.json'))),
 'E: R3 digest differs': ('E', lambda c: (lambda d: (d['phases'][2]['digests'].update(kv='0'*64), w(c/'R3-forward.json', d)))(j(c/'R3-forward.json'))),
 'E: tensor counter off by one': ('E', lambda c: (lambda d: (d['stats']['tensor_read_bytes'].update({'layers.7.q': d['stats']['tensor_read_bytes']['layers.7.q']+1}), w(c/'R1-forward.json', d)))(j(c/'R1-forward.json'))),
 'A: final re-hash differs': ('A', lambda c: (lambda d: (d['files']['qwen2.5-14b-97e1e76/model-00003-of-00008.safetensors'].update(sha256='0'*64), w(c/'integrity-final.json', d)))(j(c/'integrity-final.json'))),
 'F: stored full64 parity edited (smaller error)': ('F', lambda c: (lambda d: (d['comparison']['kinds']['hidden'].update(max_abs_error=1e-15), w(c/'parity-14b-full64.json', d)))(j(c/'parity-14b-full64.json'))),
 'F: stored stock_fp32 parity bound to another forward': ('F', lambda c: (lambda d: (d['inputs'].update(forward_sha256='0'*64), w(c/'parity-14b-stock_fp32.json', d)))(j(c/'parity-14b-stock_fp32.json'))),
 'F: 7B e2e full64 verdict FAIL': ('F', lambda c: (lambda d: (d.update(verdict='FAIL'), w(c/'equivalence-7b-e2e-full64.json', d)))(j(c/'equivalence-7b-e2e-full64.json'))),
 'F: 7B e2e record removed (only blocked file)': ('F', lambda c: [(c/x).unlink() for x in ('O7-stock_fp32-run.json',)] and (c/'blocked-O7-stock_fp32.json').write_text(json.dumps(dict(cause='host start check not GO', utc='2026-11-19T00:00:00Z')))),
 'G: unattributed OOM in N6 global counter': ('G', lambda c: (lambda d: (d.update(global_oom_kill_after=d['global_oom_kill_before']+2), w(c/'N6-scope.json', d)))(j(c/'N6-scope.json'))),
 'G: fault case flag false': ('G', lambda c: (lambda d: (d['cases'][1].update(old_kv_all_bytes_preserved=False), w(c/'F-pipeline-faults.json', d)))(j(c/'F-pipeline-faults.json'))),
 'G: observer killed a scope in R4': ('G', lambda c: (lambda d: (d['observer'].update(kills=[dict(scope='x')]), w(c/'R4-run.json', d)))(j(c/'R4-run.json'))),
 'G: R2 scope memory.peak edited after the run': ('G', lambda c: (lambda d: (d['finalizer'].update({'memory.peak': 1}), w(c/'R2-scope.json', d)))(j(c/'R2-scope.json'))),
 'G: unjustified retry of O14-full64': ('G', lambda c: [shutil.copy(c/f'O14-full64-{k}.json', c/f'O14-full64-a2-{k}.json') for k in ('run', 'scope', 'host-start')]),
 'C: sustained paging in R3 samples without annotation': ('C', lambda c: (lambda d: ([x.update(host_pages_input_per_s=5000) for x in d['observer']['summary']['last_samples'][3:12]], w(c/'R3-run.json', d)))(j(c/'R3-run.json'))),
 'F: 7B e2e stock_fp64 diagnostic verdict edited (amendment 007)': ('F', lambda c: (lambda d: (d.update(verdict='FAIL'), w(c/'equivalence-7b-e2e-stock_fp64.json', d)))(j(c/'equivalence-7b-e2e-stock_fp64.json'))),
 'F: 7B e2e stock_fp64 capture not bound to its run record (amendment 007)': ('F', lambda c: (lambda d: (d['produced_sha256'].update({'artifacts/increment10/oracle10-7b-stock_fp64.npz': '0'*64}), w(c/'O7-stock_fp64-a3-run.json', d)))(j(c/'O7-stock_fp64-a3-run.json'))),
 'G: O7-stock_fp64-a3 swap used (amendment 007)': ('G', lambda c: (lambda d: (d['finalizer'].update({'memory.swap.peak': 4096}), w(c/'O7-stock_fp64-a3-scope.json', d)))(j(c/'O7-stock_fp64-a3-scope.json'))),
 'E: R2 forward missing, blocked file present': ('E', lambda c: ((c/'R2-run.json').unlink(), (c/'blocked-R2.json').write_text(json.dumps(dict(cause='host start check not GO', utc='2026-11-19T00:00:00Z'))))),
}
rows = []
for name, (cond, mutate) in M.items():
    c = TMP/'c'
    if c.exists(): shutil.rmtree(c)
    shutil.copytree(REAL, c)
    mutate(c)
    v = g.gate(g.Evidence(campaign=c, artifacts=ART, root=Path('.')), only=[cond])['verdicts'][cond]
    rows.append((name, cond, v))
    print(f'{v:10s} {name}', flush=True)
bad = [r for r in rows if r[2] == 'PASS']
json.dump(dict(mutations=[dict(name=n, condition=c, verdict=v) for n, c, v in rows], any_pass=bool(bad)),
          open(sys.argv[2], 'x'), indent=1)
sys.exit(1 if bad else 0)

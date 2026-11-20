"""Second-review bypasses against the gate found on sys.path, on temporary copies of the evidence.

Green (current gate), from the repository root:
    python3 results/increment9/review/bypasses_followup.py
Red (gate of commit eb669f2): extract that commit's tools/ into an empty directory and pass it:
    python3 results/increment9/review/bypasses_followup.py <directory containing tools/>
Exit code 0 only when no bypass is left: every case must give the listed verdict on the
listed conditions, without an exception. The verdict of each condition on its own evidence,
before the preconditions, is printed when the gate reports it.
"""
import hashlib, json, shutil, sys, tempfile
from pathlib import Path
sys.path[:0] = [*sys.argv[1:2], '.']
from tools.report_increment9 import gate

EVIDENCE = Path('results/increment9')
SUP, RUN = 'phaseB-pipeline-supervisor.json', 'phaseB-sync-runtime.json'
INDEX = 'qwen2.5-7b-d149729/model.safetensors.index.json'


def edit(d, name, change):
    data = json.loads((d/name).read_text())
    change(data)
    (d/name).write_text(json.dumps(data, indent=2)+'\n')


def reseal(d):
    """A careful forger keeps the manifest digests consistent with the altered files."""
    edit(d, 'sanitization-manifest.json', lambda m: [e.update(
        sanitized_sha256=hashlib.sha256((d/n).read_bytes()).hexdigest()) for n, e in m['files'].items()])


def oom(d):
    edit(d, SUP, lambda x: x['finalizer']['memory.events'].update(oom_kill=1))


def number(d):
    edit(d, 'phaseA-W-supervisor.json', lambda x: x.update(wall_s=1.0))


def computed(d):
    edit(d, 'qwen-acquisition.json', lambda x: [e.update(computed_sha256='not a digest')
                                               for n, e in x['downloaded'].items() if 'of-00004' in n])


# id -> (mutation, conditions, verdict each of them must have)
CASES = {
    'P2-A number altered, manifest entry removed': (lambda d: (number(d), edit(
        d, 'sanitization-manifest.json', lambda m: m['files'].pop('phaseA-W-supervisor.json'))), 'ABCDEFGH', 'FAIL'),
    'P2-A number altered, manifest files = {}': (lambda d: (number(d), edit(
        d, 'sanitization-manifest.json', lambda m: m.update(files={}))), 'ABCDEFGH', 'FAIL'),
    'P2-A number altered, manifest digest re-sealed': (lambda d: (number(d), reseal(d)), 'ABCDEFGH', 'FAIL'),
    'P3-2 oom_kill=1, manifest missing': (lambda d: (oom(d), (d/'sanitization-manifest.json').unlink()), 'G', 'FAIL'),
    'P3-2 oom_kill=1, unrelated field missing': (lambda d: (oom(d), edit(d, SUP, lambda x: x.pop('samples')),
                                                            reseal(d)), 'G', 'FAIL'),
    'P3-3 protocol.json not valid JSON': (lambda d: (d/'protocol.json').write_text('{"increment": 9,'), 'ABCDEFGH', 'FAIL'),
    'P3-3 amendment not valid JSON': (lambda d: (d/'protocol-amendment-remediation.json').write_text('{'),
                                      'ABCDEFGH', 'FAIL'),
    'P3-3 manifest not valid JSON': (lambda d: (d/'sanitization-manifest.json').write_text('{'), 'ABCDEFGH', 'FAIL'),
    'P3-3 inference_read_bytes = 0': (lambda d: edit(d, RUN, lambda x: x['stats'].update(inference_read_bytes=0)),
                                      'CDG', 'FAIL'),
    'P3-4 integrity-final index = {"match": true}': (lambda d: edit(
        d, 'integrity-final.json', lambda x: x['files'].update({INDEX: dict(match=True)})), 'A', 'FAIL'),
    'P3-4 computed_sha256 not a digest': (computed, 'A', 'FAIL'),
    'extra: resident_bytes = -1': (lambda d: (edit(d, 'phaseB-sync-supervisor.json', lambda x: [
        v.update(resident_bytes=-1) for v in x['residency_after_eviction'].values()]), reseal(d)), 'CDG', 'FAIL'),
    'extra: oom_kill = false': (lambda d: (edit(d, SUP, lambda x: x['finalizer']['memory.events'].update(
        oom_kill=False)), reseal(d)), 'G', 'FAIL'),
}
options = dict(preservation=lambda baseline, intentional: dict(
    modified_tracked=['README.md'], deleted=[], unexpected_modifications=[]))
bypasses = 0
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)/'base'
    shutil.copytree(EVIDENCE, base, ignore=shutil.ignore_patterns('review'))
    (base/'pytest-final.txt').write_text('.'*72+' [100%]\n9999 passed in 1.00s\n')
    control = gate(base, originals=base/'no-originals', **options)['verdicts']
    print('control (unmutated copy):', ' '.join(f'{k}={v}' for k, v in control.items()))
    assert set(control.values()) == {'PASS'}
    for i, (case, (mutate, conditions, wanted)) in enumerate(CASES.items()):
        d = Path(tmp)/str(i)
        shutil.copytree(base, d)
        mutate(d)
        try:
            report = gate(d, originals=d/'no-originals', **options)
            v = report['verdicts']
            ok = all(v[c] == wanted for c in conditions)
            seen = ' '.join(f'{c}={v[c]}' for c in conditions) if len(set(v[c] for c in conditions)) > 1 else \
                f'{conditions}={v[conditions[0]]}'
            own = {report['conditions'][c].get('verdict_before_preconditions') for c in conditions}
            seen += f' (before preconditions: {", ".join(sorted(own))})' if None not in own else ''
        except Exception as error:
            ok, seen = False, 'CRASH '+type(error).__name__
        bypasses += not ok
        print(f'{"closed" if ok else "BYPASS":7s} {case:48s} wanted {wanted} on {conditions:8s} | {seen}')
print(f'{bypasses} bypasses, {len(CASES)-bypasses} closed, {len(CASES)} cases')
raise SystemExit(1 if bypasses else 0)

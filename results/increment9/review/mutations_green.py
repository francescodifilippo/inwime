"""Green run: every review mutation against the remediated gate, with the checks that reject it.

Run from the repository root: python3 results/increment9/review/mutations_green.py
"""
import shutil, sys, tempfile
from pathlib import Path
sys.path[:0] = ['.', 'tests']
import test_increment9_review as t
from tools.report_increment9 import gate

options = dict(preservation=lambda baseline, intentional: dict(t.CLEAN_PRESERVATION))
false_pass = 0
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)/'base'
    shutil.copytree(t.EVIDENCE, base, ignore=shutil.ignore_patterns('review'))
    (base/'pytest-final.txt').write_text(t.CLEAN_PYTEST)
    control = gate(base, originals=base/'no-originals', **options)['verdicts']
    print('control (unmutated copy, remediated gate):', ' '.join(f'{k}={v}' for k, v in control.items()))
    assert set(control.values()) == {'PASS'}
    for case in sorted(t.MUTATIONS):
        mutate, conditions = t.MUTATIONS[case]
        d = Path(tmp)/case
        shutil.copytree(base, d)
        mutate(d)
        t.reseal(d)
        report = gate(d, originals=d/'no-originals', **options)
        v = report['verdicts']
        still = ''.join(c for c in conditions if v[c] == 'PASS')
        false_pass += bool(still)
        reasons = []
        for c in conditions[:1]:
            facts = report['conditions'][c]['facts']
            reasons = [k for k, x in facts.get('checks', {}).items() if x is not True][:3]
            reasons += [f'{k}: {facts[k]}' for k in ('missing_evidence', 'incomplete_evidence', 'contradicted_evidence',
                                                      'precondition_not_met') if k in facts]
        line = ' '.join(f'{k}={x}' for k, x in v.items() if x != 'PASS')
        print(f'{"FALSE PASS" if still else "rejected":10s} {case:42s} {line:34s} | {"; ".join(map(str, reasons))[:150]}')
print(f'{false_pass} false PASS, {len(t.MUTATIONS)-false_pass} rejected, {len(t.MUTATIONS)} mutations')
raise SystemExit(1 if false_pass else 0)

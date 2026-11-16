"""Red run: apply every review mutation to a copy of the evidence and record the ORIGINAL gate's verdicts."""
import json, shutil, sys, tempfile
from pathlib import Path
sys.path[:0] = ['.', 'tests']
import test_increment9_review as t
from tools.report_increment9 import gate

false_pass = detected = 0
with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)/'base'
    shutil.copytree(t.EVIDENCE, base, ignore=shutil.ignore_patterns('review'))
    (base/'pytest-final.txt').write_text(t.CLEAN_PYTEST)
    control = gate(base)['verdicts']
    print('control (unmutated copy, original gate):', ''.join(k for k, v in control.items() if v == 'PASS'), 'PASS')
    for case in sorted(t.MUTATIONS):
        mutate, conditions = t.MUTATIONS[case]
        d = Path(tmp)/case
        shutil.copytree(base, d)
        try:
            mutate(d)
            v = gate(d)['verdicts']
            still = ''.join(c for c in conditions if v[c] == 'PASS')
            line = 'all A-H PASS' if set(v.values()) == {'PASS'} else ' '.join(f'{k}={x}' for k, x in v.items() if x != 'PASS')
        except Exception as error:  # the original gate crashed instead of returning a verdict
            still, line = '', 'gate raised '+type(error).__name__
        status = 'FALSE PASS' if still else 'detected'
        false_pass += bool(still); detected += not still
        print(f'{status:10s} {case:42s} expected not PASS: {conditions:8s} still PASS: {still or "-":8s} | {line}')
print(f'{false_pass} false PASS, {detected} detected, {len(t.MUTATIONS)} mutations')

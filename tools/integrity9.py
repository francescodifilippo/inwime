"""Final re-hash of increment-9 checkpoints and preservation check of tracked files.

integrity: SHA256 of every payload file against its descriptor and provenance.
preservation: files that existed at the baseline and changed since must be an
explicit, intentional list; files added later (committed or untracked) are listed
separately. Read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def integrity(checkpoints):
    files, ok = {}, True
    for directory in checkpoints:
        descriptor = json.loads((directory/'safetensors.json').read_text())
        for name, entry in descriptor['files'].items():
            with (directory/name).open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            match = digest == entry['sha256'] and (directory/name).stat().st_size == entry['nbytes']
            files[f'{directory.name}/{name}'] = dict(sha256=digest, expected=entry['sha256'], match=match)
            ok &= match
        if descriptor.get('index'):
            digest = hashlib.sha256((directory/descriptor['index']).read_bytes()).hexdigest()
            match = digest == descriptor['index_sha256']
            files[f'{directory.name}/{descriptor["index"]}'] = dict(sha256=digest, expected=descriptor['index_sha256'],
                                                                   match=match)
            ok &= match
    return dict(files=files, all_match=bool(ok))


def preservation(baseline, intentional, cwd=None):
    """Files that existed at the baseline: modified (M/T) and deleted (D) are judged.

    Files added after the baseline are listed separately whether already committed
    or still untracked, so the verdict does not depend on when the increment is
    committed. Renames are not followed: a renamed file counts as deleted.
    """
    run = lambda *a: subprocess.run(['git', *a], capture_output=True, text=True, check=True, cwd=cwd).stdout
    names = lambda *a: sorted(n for n in run(*a).split('\n') if n)
    diff = lambda kind: names('diff', '--name-only', '--no-renames', '--diff-filter='+kind, baseline)
    modified, deleted, added = diff('MT'), diff('D'), diff('A')
    untracked = names('ls-files', '--others', '--exclude-standard')
    unexpected = [m for m in modified if m not in intentional]
    return dict(baseline=baseline, head=run('rev-parse', 'HEAD').strip(), modified_tracked=modified,
                intentional=sorted(intentional), unexpected_modifications=unexpected, deleted=deleted,
                added_since_baseline=added, new_untracked=untracked,
                staged=names('diff', '--cached', '--name-only'), passed=not unexpected and not deleted)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    i = sub.add_parser('integrity'); i.add_argument('checkpoints', nargs='+', type=Path)
    s = sub.add_parser('preservation'); s.add_argument('--baseline', required=True)
    s.add_argument('--intentional', action='append', default=[])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report = integrity(a.checkpoints) if a.command == 'integrity' else preservation(a.baseline, a.intentional)
    with a.output.open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    key = 'all_match' if a.command == 'integrity' else 'passed'
    print(a.command, report[key])
    raise SystemExit(0 if report[key] else 1)

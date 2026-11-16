"""Mechanical, verifiable removal of personal paths from Git-bound increment-9 evidence.

Only path prefixes and local identifiers are replaced, and Hugging Face cache
entries unrelated to the project are dropped. No number is ever changed: JSON
files are rewritten structurally, every non-string value is carried over as is,
and `verify` proves it by re-deriving each sanitized file from its original.
Originals are copied first to a local, git-ignored directory and never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

MANIFEST = 'sanitization-manifest.json'
ORIGINALS = Path('artifacts/increment9/original-evidence')
PROJECT_MODELS = ('Qwen', 'SmolLM')
CACHE_KEY = 'huggingface_hub_cache_entries'
SUFFIXES = ('.json', '.txt', '.log')
# (id, pattern, replacement, description). Patterns are generic on purpose: the
# manifest that quotes them must not itself contain a personal path.
RULES = (
    ('repo-root', r'/home/[^/"\s]+/(?:[^/"\s]+/)*?ooc-inference(?![\w.-])', '<repo>',
     'absolute path of the local repository checkout -> <repo>'),
    ('session-scratch', r'/tmp/[a-z]+-[0-9]+/[^/"\s]+/[0-9a-f-]{36}/scratchpad', '<scratch>',
     'per-session scratch directory of the acquisition session -> <scratch>'),
    ('terminal-session', r'tmux-spawn-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', 'tmux-spawn-<session>',
     'random identifier of the local terminal session scope -> tmux-spawn-<session>'),
)
CACHE_RULE = ('unrelated-hf-cache', f'entries of {CACHE_KEY} naming none of {"/".join(PROJECT_MODELS)}', '(dropped)',
              'Hugging Face cache entries unrelated to the project are removed from the list')
# Anything matching these after sanitization is a leftover personal path or identifier.
RESIDUAL = (r'/home/[^/"\s<>]+', r'/tmp/[a-z]+-[0-9]+/', r'[A-Za-z]:\\+Users\\', r'/mnt/[a-z]/Users/',
            r'tmux-spawn-[0-9a-f]{8}-', r'models--(?!Qwen|HuggingFaceTB--SmolLM)[\w.-]+')


def text_rules(text, used=None):
    for name, pattern, replacement, _ in RULES:
        text, count = re.subn(pattern, replacement, text)
        if count and used is not None:
            used[name] = used.get(name, 0)+count
    return text


def clean(value, used=None):
    """Strings (keys included) go through the rules; every other value is returned untouched."""
    if isinstance(value, str):
        return text_rules(value, used)
    if isinstance(value, list):
        return [clean(v, used) for v in value]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key == CACHE_KEY and isinstance(item, list):
                kept = [v for v in item if any(model in v for model in PROJECT_MODELS)]
                if len(kept) != len(item) and used is not None:
                    used[CACHE_RULE[0]] = used.get(CACHE_RULE[0], 0)+len(item)-len(kept)
                item = kept
            out[clean(key, used)] = clean(item, used)
        if len(out) != len(value):
            raise ValueError('sanitization would merge two distinct keys')
        return out
    return value


def sanitize_bytes(name, raw, used=None):
    text = raw.decode()
    if name.endswith('.json'):
        data = json.loads(text)
        if json.dumps(data, indent=2)+'\n' != text:
            # Not written by the increment-9 tools: do not reformat, only replace text.
            return text_rules(text, used).encode()
        return (json.dumps(clean(data, used), indent=2)+'\n').encode()
    return text_rules(text, used).encode()


def portable(value, root=None):
    """For new reports: the checkout root becomes <repo>, then the generic rules apply."""
    root = str(Path.cwd() if root is None else root)
    if isinstance(value, str):
        return text_rules(re.sub(re.escape(root)+r'(?![\w.-])', '<repo>', value))
    if isinstance(value, (list, tuple)):
        return [portable(v, root) for v in value]
    if isinstance(value, dict):
        return {portable(k, root): portable(v, root) for k, v in value.items()}
    return value


def evidence_files(directory):
    return sorted(p for p in Path(directory).iterdir() if p.is_file() and p.suffix in SUFFIXES and p.name != MANIFEST)


def residual(directory):
    """Leftover personal paths or identifiers anywhere under the evidence directory."""
    found, directory = [], Path(directory)
    for path in sorted(directory.rglob('*')):
        if path.is_file() and path.name != MANIFEST and '__pycache__' not in path.parts:
            text = path.read_text(errors='replace')
            found += [f'{path.relative_to(directory)}: {pattern}' for pattern in RESIDUAL if re.search(pattern, text)]
    return found


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def apply(directory, originals):
    """Sanitize in place; originals are saved first with exclusive creation."""
    directory, originals = Path(directory), Path(originals)
    files = {}
    for path in evidence_files(directory):
        raw, used = path.read_bytes(), {}
        cleaned = sanitize_bytes(path.name, raw, used)
        if cleaned == raw:
            continue
        originals.mkdir(parents=True, exist_ok=True)
        with (originals/path.name).open('xb') as f:
            f.write(raw)
        path.write_bytes(cleaned)
        files[path.name] = dict(original_sha256=sha(raw), sanitized_sha256=sha(cleaned),
                                original_nbytes=len(raw), sanitized_nbytes=len(cleaned), replacements=used)
    return dict(
        note='mechanical sanitization of Git-bound evidence: path prefixes and local identifiers only; '
             'no number changed; originals kept locally, outside Git',
        originals_directory=str(originals), originals_in_git=False,
        rules=[dict(id=n, pattern=p, replacement=r, description=d) for n, p, r, d in (*RULES, CACHE_RULE)],
        files=files)


def verify(directory, originals, manifest):
    """PASS only if every listed file is exactly its original under the declared rules."""
    directory, originals = Path(directory), Path(originals)
    if not isinstance(manifest, dict) or not isinstance(manifest.get('files'), dict):
        return dict(verdict='NOT PROVEN', files=0, originals='unknown', problems=['sanitization manifest missing'])
    declared = [dict(id=n, pattern=p, replacement=r, description=d) for n, p, r, d in (*RULES, CACHE_RULE)]
    problems = [] if manifest.get('rules') == declared else ['declared rules differ from the rules of this tool']
    problems += ['personal path or identifier left: '+item for item in residual(directory)]
    present = 0
    for name, entry in manifest['files'].items():
        path = directory/name
        if Path(name).name != name or not path.is_file():
            problems.append('sanitized file missing: '+name)
            continue
        current = path.read_bytes()
        if sha(current) != entry.get('sanitized_sha256'):
            problems.append('sanitized file differs from the manifest: '+name)
        original = originals/name
        if original.is_file():
            present += 1
            raw = original.read_bytes()
            if sha(raw) != entry.get('original_sha256'):
                problems.append('local original differs from the manifest: '+name)
            elif sanitize_bytes(name, raw) != current:
                problems.append('file differs from its original beyond the declared rules: '+name)
    if present == len(manifest['files']):
        state = 'present: every sanitized file equals its original under the declared rules'
    elif present == 0:
        state = 'absent: originals are local and outside Git; only the manifest digests were checked'
    else:
        state = f'partial: {present} of {len(manifest["files"])} originals present'
        problems.append('only some originals are present')
    return dict(verdict='FAIL' if problems else 'PASS', files=len(manifest['files']), originals=state,
                problems=problems)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--originals', type=Path, default=ORIGINALS)
    p.add_argument('--verify', action='store_true', help='check only; never writes')
    a = p.parse_args()
    target = a.directory/MANIFEST
    if a.verify:
        result = verify(a.directory, a.originals, json.loads(target.read_text()) if target.exists() else None)
    else:
        manifest = apply(a.directory, a.originals)
        with target.open('x') as f:
            json.dump(manifest, f, indent=2); f.write('\n')
        result = verify(a.directory, a.originals, manifest)
    print(json.dumps(result, indent=1))
    raise SystemExit(0 if result['verdict'] == 'PASS' else 1)

"""Registro dei modelli: the local inventory of installed and imported Checkpoint.

Lives in the InWiMe data directory (`INWIME_HOME`, else `$XDG_DATA_HOME/inwime`, else
`~/.local/share/inwime`), never inside a Checkpoint or the Hugging Face cache:

- `installed/<org>--<name>@<sha>.json`: install manifest (schema inwime-install-manifest/1)
  of a Checkpoint downloaded into the standard Hugging Face cache;
- `imported/<digest>.json`: reference to a local Checkpoint directory (inwime-import/1),
  never a copy;
- `active.json`: the active model;
- `locks/install.lock`: held by the one install that may run at a time.

Every file is replaced atomically; none holds credentials.
"""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path

from datetime import datetime, timezone

from . import __version__, capability, catalog

SCHEMA_MANIFEST = 'inwime-install-manifest/1'
SCHEMA_IMPORT = 'inwime-import/1'
SCHEMA_LIST = 'inwime-model-list/1'


class ModelsError(catalog.CatalogError):
    """A model-management failure the user can act on; `kind` selects the exit status."""

    EXIT = dict(catalog.CatalogError.EXIT, declined=9, insufficient_space=10, storage_unknown=10,
                locked=11, verification_failed=12, download_failed=7, not_registered=3,
                not_usable=13, not_a_checkpoint=2)


def home(env=None):
    env = os.environ if env is None else env
    if env.get('INWIME_HOME'):
        return Path(env['INWIME_HOME'])
    data = Path(env['XDG_DATA_HOME']) if env.get('XDG_DATA_HOME') else \
        Path(env.get('HOME') or os.path.expanduser('~'))/'.local/share'
    return data/'inwime'


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def lock_path():
    return home()/'locks'/'install.lock'


def write_json(path, data):
    """Atomic replacement: readers see the old or the new file, never a partial one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=1)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def manifest_path(repo_id, revision):
    return home()/'installed'/('%s@%s.json' % (repo_id.replace('/', '--'), revision))


@contextlib.contextmanager
def install_lock(holder):
    """Exclusive, non-blocking: a second install fails at once instead of waiting. The kernel
    releases the lock when the holder exits, even if it is killed. Yields `announce(text)`,
    which records what the holder is installing once it is known."""
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            other = os.pread(fd, 4096, 0).decode('utf-8', 'replace').strip() or 'unknown'
            raise ModelsError('locked', 'another install is running (%s); retry when it ends'
                              % catalog.printable(other)) from None

        def announce(text):
            os.ftruncate(fd, 0)
            os.pwrite(fd, ('%s, pid %d\n' % (text, os.getpid())).encode(), 0)
        announce(holder)
        yield announce
    finally:
        os.close(fd)


def install_running(entry_id):
    """Whether an install of `entry_id` holds the lock now."""
    try:
        fd = os.open(lock_path(), os.O_RDONLY)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        return False
    except BlockingIOError:
        return os.pread(fd, 4096, 0).decode('utf-8', 'replace').startswith(entry_id+',')
    finally:
        os.close(fd)

def intact(manifest):
    """Whether a manifest records a verified install whose files are all still in the cache with
    their recorded sizes. Files are not hashed again here: hashing is the installer's job."""
    if not manifest or manifest.get('status') != 'verified':
        return False
    snapshot = Path(manifest['snapshot_dir'])
    for record in manifest['files']:
        try:
            if (snapshot/record['name']).stat().st_size != record['size']:
                return False
        except OSError:
            return False
    return True


def import_record_path(directory):
    return home()/'imported'/('%s.json' % hashlib.sha256(str(directory).encode()).hexdigest()[:24])


def scan(directory):
    """Regular files at the top of a Checkpoint directory (symlinks followed, as a loader would)."""
    files = []
    for entry in sorted(os.scandir(directory), key=lambda e: e.name):
        try:
            if entry.is_file():
                st = entry.stat()
                files.append({'name': entry.name, 'size': st.st_size, 'mtime_ns': st.st_mtime_ns})
        except OSError:
            continue
    return files


def import_checkpoint(path):
    """Register a local Checkpoint by reference: nothing is copied, written or changed in it."""
    directory = Path(os.path.realpath(os.path.expanduser(str(path))))
    if not directory.is_dir():
        raise ModelsError('not_a_checkpoint', '%s is not a directory' % catalog.printable(path))
    if not (directory/'config.json').is_file():
        raise ModelsError('not_a_checkpoint', '%s has no config.json: not a Checkpoint directory'
                          % catalog.printable(directory))
    files = scan(directory)
    record = {'schema': SCHEMA_IMPORT, 'inwime_version': __version__, 'path': str(directory),
              'imported_utc': utc_now(),
              'bytes': sum(f['size'] for f in files), 'files': files}
    write_json(import_record_path(directory), record)
    return record


# -- inventory -----------------------------------------------------------------------------------

def read_local_header(path):
    """{tensor: (dtype, shape)} of a safetensors file on disk, within the metadata bound."""
    try:
        with open(path, 'rb') as f:
            head = f.read(8)
            if len(head) < 8:
                raise catalog.HeaderError('file shorter than the 8-byte header length')
            length = int.from_bytes(head, 'little')
            if length > catalog.METADATA_FILE_LIMIT:
                raise catalog.HeaderError('%d-byte header above the %d-byte bound'
                                          % (length, catalog.METADATA_FILE_LIMIT))
            raw = f.read(length)
    except OSError as error:
        raise catalog.HeaderError(type(error).__name__) from None
    try:
        header = json.loads(raw)
    except ValueError:
        raise catalog.HeaderError('header is not JSON') from None
    if not isinstance(header, dict):
        raise catalog.HeaderError('header is not a JSON object')
    tensors = {}
    for name, meta in header.items():
        if name == '__metadata__':
            continue
        if not (isinstance(meta, dict) and isinstance(meta.get('dtype'), str)
                and isinstance(meta.get('shape'), list)
                and all(type(d) is int and d >= 0 for d in meta['shape'])):
            raise catalog.HeaderError('malformed entry for tensor %s' % name[:80])
        tensors[name] = (meta['dtype'], tuple(meta['shape']))
    return tensors


def local_metadata(directory, repo_id, revision):
    """What the capability assessment needs, read from a Checkpoint directory (no weights)."""
    directory = Path(directory)
    sizes = {f['name']: f['size'] for f in scan(directory)}

    def load(name):
        if name not in sizes:
            return None, None
        if sizes[name] > catalog.METADATA_FILE_LIMIT:
            return {capability.INVALID_JSON: True}, None
        raw = (directory/name).read_bytes()
        try:
            return json.loads(raw), raw
        except ValueError:
            return {capability.INVALID_JSON: True}, raw

    config, config_raw = load('config.json')
    index = load(catalog.SAFETENSORS_INDEX)[0]
    weights, tensors = catalog.summarize_weights(sizes, index, lambda n: read_local_header(directory/n))
    return capability.RevisionMetadata(
        repo_id=repo_id, revision=revision, files=frozenset(sizes),
        config=config if isinstance(config, dict) else {}, config_raw=config_raw,
        tokenizer_config=load('tokenizer_config.json')[0], index=index, weights=weights, tensors=tensors)


def installed_state(manifest):
    status = manifest.get('status')
    if status == 'verified':
        return 'verified' if intact(manifest) else 'missing'
    if status in ('installing', 'incomplete'):
        entry_id = '%s@%s' % (manifest['repo_id'], manifest['revision'])
        return 'installing' if status == 'installing' and install_running(entry_id) else 'incomplete'
    return 'failed'


def imported_state(record):
    directory = Path(record['path'])
    if not directory.is_dir():
        return 'missing'
    now = {f['name']: (f['size'], f['mtime_ns']) for f in scan(directory)}
    then = {f['name']: (f['size'], f['mtime_ns']) for f in record['files']}
    return 'imported' if now == then else 'changed'


USABLE = ('verified', 'imported')


def blobs(directory):
    """{(device, inode): size} of the files of a directory, symlinks followed: a blob shared by
    two snapshots is counted once in a total."""
    found = {}
    try:
        for entry in os.scandir(directory):
            with contextlib.suppress(OSError):
                if entry.is_file():
                    st = entry.stat()
                    found[(st.st_dev, st.st_ino)] = st.st_size
    except OSError:
        pass
    return found

def entries():
    """Installed and imported Checkpoint, each with its state, space and Stati di capacità."""
    found = []
    for path in sorted((home()/'installed').glob('*.json')):
        m = read_json(path)
        if not isinstance(m, dict) or m.get('schema') != SCHEMA_MANIFEST:
            continue
        found.append({'id': '%s@%s' % (m['repo_id'], m['revision']), 'source': 'installed',
                      'repo_id': m['repo_id'], 'revision': m['revision'], 'path': m['snapshot_dir'],
                      'state': installed_state(m), 'record': str(path)})
    for path in sorted((home()/'imported').glob('*.json')):
        r = read_json(path)
        if not isinstance(r, dict) or r.get('schema') != SCHEMA_IMPORT:
            continue
        found.append({'id': r['path'], 'source': 'imported', 'repo_id': None, 'revision': None,
                      'path': r['path'], 'state': imported_state(r), 'record': str(path)})
    active_id = active()
    for entry in found:
        entry['blobs'] = blobs(entry['path'])
        entry['bytes'] = sum(entry['blobs'].values())
        entry['active'] = entry['id'] == active_id
        assessment = None
        if entry['state'] in USABLE:
            assessment = capability.assess(local_metadata(
                entry['path'], entry['repo_id'] or entry['path'], entry['revision'] or 'local import'))
        reached = [s['state'] for s in assessment['states'] if s['status'] == 'reached'] if assessment else []
        entry.update(support=assessment['support'] if assessment else None,
                     reached=reached[-1] if reached else None, capability=assessment)
    return found


def inventory():
    models = entries()
    unique = {}
    for m in models:
        unique.update(m.pop('blobs'))
    return {'schema': SCHEMA_LIST, 'registry': str(home()), 'active': active(),
            'bytes_total': sum(unique.values()), 'models': models}


def active():
    record = read_json(home()/'active.json')
    return record.get('id') if isinstance(record, dict) else None


def render_list(result):
    rows = [('', 'ID', 'SOURCE', 'STATE', 'SUPPORT', 'SIZE')]
    for m in result['models']:
        support = '%s (%s)' % (m['support'], m['reached']) if m['support'] else '-'
        rows.append(('*' if m['active'] else ' ', m['id'], m['source'], m['state'], support,
                     catalog.format_bytes(m['bytes'])))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ['  '.join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
    if not result['models']:
        lines.append('no models: `inwime models install REPO@SHA` or `inwime models import PATH`')
    lines += ['', 'total %s; * active model%s' % (catalog.format_bytes(result['bytes_total']),
                                                  '' if result['active'] else ' (none selected)'),
              'registry: %s' % result['registry']]
    return catalog.render_lines(lines)


def use(name):
    """Make one registered, usable Checkpoint the active model. `name` is an id from `list`, a
    repository with exactly one installed revision, or the path of an import."""
    models = entries()
    candidates = [m for m in models if m['id'] == name]
    if not candidates and '@' not in name:
        candidates = [m for m in models if m['repo_id'] == name]
        if not candidates:
            path = os.path.realpath(os.path.expanduser(name))
            candidates = [m for m in models if m['source'] == 'imported' and m['id'] == path]
    if not candidates:
        raise ModelsError('not_registered', '%s is neither installed nor imported; see `inwime models list`'
                          % catalog.printable(name))
    if len(candidates) > 1:
        raise ModelsError('usage', '%s names %d installed revisions: choose one of %s'
                          % (catalog.printable(name), len(candidates),
                             ', '.join(m['id'] for m in candidates)))
    entry = candidates[0]
    if entry['state'] not in USABLE:
        raise ModelsError('not_usable', '%s is %s, not usable: %s' % (
            catalog.printable(entry['id']), entry['state'],
            're-run `inwime models install` to complete or repair it' if entry['source'] == 'installed'
            else 're-run `inwime models import` after checking the directory'))
    write_json(home()/'active.json', {'id': entry['id'], 'source': entry['source'], 'path': entry['path'],
                                      'selected_utc': utc_now()})
    return entry

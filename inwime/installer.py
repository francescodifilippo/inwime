"""`inwime models install`: a voluntary, confirmed download of one Checkpoint `repo@sha`.

Steps, in order:

1. resolve the requested branch, tag or SHA to the full commit SHA of the Revisione, with the
   per-file metadata the Hub declares (size, git blob id, LFS SHA256);
2. take the install lock (one install at a time, so two downloads cannot both pass the space
   check), measure the physical free space and print the summary;
3. refuse if less than the margin (default 30 GiB) would remain free on the physical volume
   after the download, or if that space cannot be determined (fail-closed);
4. ask for explicit confirmation (`--yes` is accepted only with a full SHA, which the user has
   then already seen), measure the space again, and once more before each file;
5. download each file through `huggingface_hub` into the standard, shared Hugging Face cache,
   then verify it: LFS files by SHA256 against the Hub's declaration, the others by size and
   git blob id. A file whose blob was already in the cache is not downloaded again, only
   verified: this is how an interrupted install resumes, file by file (within one file,
   `huggingface_hub` resumes a broken transfer with byte ranges);
6. record the outcome of every file in the install manifest; mark the weights read-only.

RAM and paging never stop an acquisition (policy #25): only storage is checked. Nothing is
converted or quantized. Tokens are read by `huggingface_hub` from its standard places and
never printed, logged or recorded.
"""
import contextlib
import hashlib
import os
import re
import stat
import sys
from pathlib import Path

from . import __version__, catalog, doctor, probe, registry, storage, wsl_host
from .registry import ModelsError

COMMIT = re.compile(r'^[0-9a-f]{40}$')
CHUNK = 8*2**20


class SpaceProbe:
    """Physical free space of the volume that receives the cache. The Windows host (under WSL)
    is queried once; each `measure` only reads the filesystems again."""

    def __init__(self, path, margin_bytes):
        system = probe.system('/', os.environ)
        table = storage.mounts('/')
        root_mount = storage.mount_for('/', table)
        host = wsl_host.query(table, system['wsl_distro']) if system['wsl'] else None
        self.path = path
        self.kwargs = dict(table=table, wsl=system['wsl'], host=host, margin_bytes=margin_bytes,
                           root_device=root_mount.device if root_mount else None)

    def measure(self):
        return storage.describe(self.path, **self.kwargs)


def cache_dir():
    return doctor.default_store(os.environ)


def storage_folder(repo_id):
    return cache_dir()/('models--'+repo_id.replace('/', '--'))


SHA256 = re.compile(r'^[0-9a-f]{64}$')


def safe_name(name):
    """A file name of the Revisione usable as a relative path inside the snapshot."""
    parts = name.split('/')
    return (bool(name) and not name.startswith('/') and '\\' not in name and '\0' not in name
            and all(part not in ('', '.', '..') for part in parts))


def make_plan(repo_id, requested):
    """What an install of `repo_id@requested` would download, from the Hub's metadata.

    Every identifier the Hub returns becomes a path component (registry file, snapshot, blob),
    so each is validated first: a hostile endpoint cannot steer writes outside the cache.
    """
    model = catalog.call(catalog.api().model_info, repo_id, revision=requested, files_metadata=True)
    revision = model.sha
    if not isinstance(revision, str) or not COMMIT.match(revision):
        raise ModelsError('hub_error', 'the Hub answered an invalid revision for %s' % repo_id)
    siblings = {s.rfilename: s for s in model.siblings or ()}
    for name, s in siblings.items():
        if not safe_name(name):
            raise ModelsError('hub_error', 'the Hub lists an invalid file name in %s@%s: %s'
                              % (repo_id, revision, catalog.printable(name)[:120]))
        if (s.lfs is not None and not SHA256.match(str(s.lfs.sha256))) or \
                (s.lfs is None and s.blob_id is not None and not COMMIT.match(str(s.blob_id))):
            raise ModelsError('hub_error', 'the Hub declares an invalid hash for %s in %s@%s'
                              % (catalog.printable(name)[:120], repo_id, revision))
    # Weight files the safetensors loader never reads (pickle, GGUF, ONNX... beside safetensors)
    # are not downloaded; without safetensors weights, every file is.
    index = catalog.read_json(repo_id, revision, siblings, catalog.SAFETENSORS_INDEX)
    weights, _ = catalog.summarize_weights({n: catalog.file_size(s) for n, s in siblings.items()}, index, None)
    excluded = weights['other_weight_files'] if weights['files'] else []
    folder = storage_folder(repo_id)
    files, duplicates = [], []
    for name in sorted(set(siblings) - set(excluded)):
        s = siblings[name]
        lfs = s.lfs is not None
        size = s.lfs.size if lfs else s.size
        expected = {'size': size}
        if lfs:
            expected['sha256'] = s.lfs.sha256
        elif s.blob_id:
            expected['git_sha1'] = s.blob_id
        etag = expected.get('sha256') or expected.get('git_sha1')
        cached = etag is not None and (folder/'blobs'/etag).exists()
        files.append({'name': name, 'kind': 'lfs' if lfs else 'git', 'expected': expected,
                      'cached': cached})
        if lfs and not cached:
            for other in sorted(cache_dir().glob('models--*/blobs/'+expected['sha256'])):
                owner = other.parent.parent.name[len('models--'):].replace('--', '/', 1)
                duplicates.append({'name': name, 'bytes': size, 'repo_id': owner})
                break
    total = sum(f['expected']['size'] or 0 for f in files)
    missing = sum(f['expected']['size'] or 0 for f in files if not f['cached'])
    return {'repo_id': repo_id, 'requested_ref': requested, 'revision': revision,
            'id': '%s@%s' % (repo_id, revision),
            'files': files, 'bytes_total': total, 'bytes_to_download': missing, 'duplicates': duplicates,
            'excluded_files': excluded,
            'cache_dir': str(cache_dir()), 'snapshot_dir': str(folder/'snapshots'/revision)}


def check_access(plan):
    """One HEAD request on a file of the Revisione: authentication and licence problems of a gated
    or private repository surface before the summary, not after the confirmation."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url
    if plan['files']:
        url = hf_hub_url(plan['repo_id'], plan['files'][0]['name'], revision=plan['revision'],
                         endpoint=catalog.endpoint())
        catalog.call(get_hf_file_metadata, url, endpoint=catalog.endpoint())


def check_space(space, needed, margin):
    free = space['physical_free_bytes']
    if free is None:
        raise ModelsError('storage_unknown', 'the physical free space behind %s is unknown (%s): download '
                          'refused (fail-closed); `inwime doctor` shows what could not be read'
                          % (space['path'], '; '.join(space['notes']) or space['backing']))
    if free-needed < margin:
        raise ModelsError('insufficient_space', 'after downloading %d bytes, %d bytes would remain free on the '
                          'physical volume, below the %d-byte margin: download refused'
                          % (needed, free-needed, margin))


def render_summary(plan, space, margin):
    size = catalog.format_bytes
    free = space['physical_free_bytes']
    cached = [f for f in plan['files'] if f['cached']]
    lines = ['Install %s' % plan['repo_id'],
             '  revision:        %s (requested: %s)' % (plan['revision'], plan['requested_ref']),
             '  files:           %d, %s' % (len(plan['files']), size(plan['bytes_total'])),
             '  to download:     %s' % size(plan['bytes_to_download']),
             '  already cached:  %d files (verified, not downloaded again)' % len(cached),
             '  destination:     %s' % plan['snapshot_dir'],
             '  storage:         %s on %s, physical free %s' % (space['backing'], space['mount_point'], size(free)),
             '  after download:  %s free, margin %s' % (size(None if free is None else
                                                           free-plan['bytes_to_download']), size(margin))]
    if plan['excluded_files']:
        lines.insert(3, '  not downloaded:  %s (weight formats the safetensors loader never reads)'
                     % ', '.join(plan['excluded_files']))
    if plan['duplicates']:
        lines.insert(5, '  duplicates:      %d files, %d bytes identical to blobs of %s, stored again '
                     '(the Hugging Face cache keeps one copy per repository)'
                     % (len(plan['duplicates']), sum(d['bytes'] for d in plan['duplicates']),
                        ', '.join(sorted({d['repo_id'] for d in plan['duplicates']}))))
    return catalog.render_lines(lines)


def confirm_from_stdin():
    sys.stdout.write("Type 'yes' to download: ")
    sys.stdout.flush()
    return sys.stdin.readline().strip().lower() in ('y', 'yes')


def digest(path, kind):
    """(size, sha256, git blob id or None) of a file, read once."""
    size = os.stat(path).st_size
    sha256 = hashlib.sha256()
    git = hashlib.sha1(b'blob %d\0' % size) if kind == 'git' else None
    with open(path, 'rb') as f:
        while chunk := f.read(CHUNK):
            sha256.update(chunk)
            if git is not None:
                git.update(chunk)
    return size, sha256.hexdigest(), git.hexdigest() if git else None


def verify(path, entry):
    expected = entry['expected']
    size, sha256, git = digest(path, entry['kind'])
    record = {'name': entry['name'], 'kind': entry['kind'], 'expected': expected, 'size': size,
              'sha256': sha256, 'action': 'reused' if entry['cached'] else 'downloaded'}
    if 'sha256' not in expected and 'git_sha1' not in expected:
        record.update(outcome='unverifiable', detail='the Hub declared no hash for this file')
    elif expected['size'] is not None and size != expected['size']:
        record.update(outcome='mismatch', detail='size %d, the Hub declares %d' % (size, expected['size']))
    elif 'sha256' in expected and sha256 != expected['sha256']:
        record.update(outcome='mismatch', detail='SHA256 differs from the Hub declaration')
    elif 'git_sha1' in expected and git != expected['git_sha1']:
        record.update(outcome='mismatch', detail='git blob id differs from the Hub declaration')
    else:
        record['outcome'] = 'verified'
    return record


def quarantine(local, entry):
    """Remove a blob that failed verification if this install downloaded it, so no tool loads it
    and a retry downloads it again. A blob that was already in the cache belongs to whoever put
    it there: it is reported, never deleted."""
    blob = os.path.realpath(local)
    if entry['cached']:
        return ('; the corrupt blob was already in the cache and is left untouched: %s. Remove it '
                '(for example with `hf cache rm`) before retrying' % blob)
    if os.path.islink(local):
        os.unlink(local)
    with contextlib.suppress(FileNotFoundError):
        os.unlink(blob)
    return '; the downloaded file was removed from the cache'


def read_only(path):
    """Remove the write permissions of a verified original (the blob the snapshot links to).
    Blobs are content-addressed and immutable by design, so this holds for blobs another tool
    downloaded too."""
    blob = os.path.realpath(path)
    os.chmod(blob, stat.S_IMODE(os.stat(blob).st_mode) & ~0o222)


def download(plan, entry):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(plan['repo_id'], entry['name'], revision=plan['revision'],
                           cache_dir=plan['cache_dir'], endpoint=catalog.endpoint())


def install(spec, *, margin_bytes=storage.DEFAULT_MARGIN_BYTES, yes=False, confirm=confirm_from_stdin,
            out=sys.stdout):
    repo_id, requested = catalog.parse_spec(spec)
    if yes and not COMMIT.match(requested):
        raise ModelsError('usage', '--yes needs the revision as a full 40-hex SHA (%s@<sha>); without --yes '
                          'the summary shows what %r resolves to' % (repo_id, requested))
    # The lock comes first: the plan records which blobs are already cached, and that must not
    # change before the download decides what it may delete.
    with registry.install_lock('%s@%s' % (repo_id, requested)) as announce:
        with catalog.translated(repo_id):
            plan = make_plan(repo_id, requested)
            announce(plan['id'])
            if COMMIT.match(requested) and plan['revision'] != requested:
                raise ModelsError('usage', '%s resolves to %s, not to itself: it names a branch or tag, not '
                                  'a commit; install %s if that is the Revisione wanted'
                                  % (requested, plan['revision'], plan['id']))
            previous = registry.read_json(registry.manifest_path(repo_id, plan['revision']))
            if registry.intact(previous):
                out.write(catalog.render_lines([
                    'already installed: %s (requested: %s), verified on %s; every file is still present '
                    'with its recorded size (not hashed again)' % (plan['id'], requested,
                                                                   previous['finished_utc'])]))
                return None
            check_access(plan)
        return _acquire(plan, margin_bytes, yes, confirm, out)


def _acquire(plan, margin_bytes, yes, confirm, out):
    space_probe = SpaceProbe(Path(plan['cache_dir']), margin_bytes)
    space = space_probe.measure()
    out.write(render_summary(plan, space, margin_bytes))
    out.flush()
    check_space(space, plan['bytes_to_download'], margin_bytes)
    if not yes and not confirm():
        raise ModelsError('declined', 'download not confirmed: nothing was downloaded')
    out.write('\n')
    check_space(space_probe.measure(), plan['bytes_to_download'], margin_bytes)
    manifest = {'schema': registry.SCHEMA_MANIFEST, 'inwime_version': __version__, 'client': catalog.client(),
                'endpoint': catalog.public_endpoint(), 'repo_id': plan['repo_id'],
                'requested_ref': plan['requested_ref'], 'revision': plan['revision'],
                'cache_dir': plan['cache_dir'], 'snapshot_dir': plan['snapshot_dir'],
                'started_utc': registry.utc_now(), 'finished_utc': None, 'status': 'installing',
                'failure': None, 'margin_bytes': margin_bytes, 'bytes_total': plan['bytes_total'],
                'read_only': False, 'duplicates': plan['duplicates'],
                'excluded_files': plan['excluded_files'], 'files': []}
    path = registry.manifest_path(plan['repo_id'], plan['revision'])
    registry.write_json(path, manifest)
    try:
        remaining = plan['bytes_to_download']
        for i, entry in enumerate(plan['files']):
            if i:
                check_space(space_probe.measure(), remaining, margin_bytes)
            try:
                with catalog.translated(plan['repo_id']):
                    local = download(plan, entry)
            except (OSError, ValueError) as error:   # disk full, permissions, Hub unreachable mid-install
                raise ModelsError('download_failed', 'cannot download %s: %s' % (
                    entry['name'], getattr(error, 'strerror', None) or type(error).__name__)) from None
            if not entry['cached']:
                remaining -= entry['expected']['size'] or 0
            record = verify(local, entry)
            manifest['files'].append(record)
            out.write(catalog.render_lines(['  %-10s %s' % (record['outcome'], entry['name'])]))
            if record['outcome'] != 'verified':
                raise ModelsError('verification_failed', '%s of %s: %s%s' % (
                    entry['name'], plan['id'], record['detail'], quarantine(local, entry)))
            read_only(local)
            record['read_only'] = True
    except BaseException as error:
        # An interruption leaves an install to resume; anything else is a failure.
        interrupted = isinstance(error, KeyboardInterrupt)
        manifest.update(status='incomplete' if interrupted else 'failed', finished_utc=registry.utc_now(),
                        failure='interrupted' if interrupted else catalog.printable(
                            error if isinstance(error, catalog.CatalogError) else type(error).__name__))
        registry.write_json(path, manifest)
        raise
    manifest.update(status='verified', finished_utc=registry.utc_now(), read_only=True)
    registry.write_json(path, manifest)
    out.write(catalog.render_lines(['installed and verified: %s' % plan['id'], '  manifest: %s' % path]))
    return manifest

"""Pinned, fail-closed acquisition of Qwen/Qwen2.5-14B BF16 (increment 10).

Independent of the increment-9 (7B) constants. The manifest below is pinned to
the immutable revision 97e1e76335b7017d8f67c08a19d103c0504298c9: exact sizes,
Git blob IDs, LFS SHA256 of the eight shards and SHA256 of each raw
safetensors header, all taken from the 2026-11-18 metadata preflight.

Modes
  --dry-run (default)  No payload byte is transferred and nothing is written
                       except the optional --evidence file (created, never
                       overwritten). Verifies the official metadata (online,
                       about 160 KB, or offline from a captured --metadata-dir)
                       against the pinned manifest, inspects the destination
                       and computes the space plan on the physical volume.
  --execute            Requires two independent records, both checked before
                       any network request and again under the lock before the
                       first payload request:
                       * --authorization FILE: the owner's explicit approval
                         naming model, revision, manifest digest, destination
                         and the maximum number of bytes that may be transferred
                         (human decision);
                       * --host-report/--host-baseline: a fresh hostmem10 GO for
                         the preregistered 'download' profile, validated by
                         tools/hostgate10.py against the pinned protocol, this
                         destination and this physical volume (technical
                         eligibility; protocol amendment 001).
                       Without both nothing is downloaded; neither implies the
                       other.

Safety properties (each covered by tests against a local fake HTTP server)
  * HTTPS only, host allowlist on every redirect hop, generic User-Agent, no
    credentials, Accept-Encoding: identity, bounded metadata responses.
  * Range responses are accepted only as 206 with an exact Content-Range,
    checked before the body is read; a server that ignores Range is never read.
  * Payload goes to <name>.part next to the destination file, with an ownership
    marker <name>.part.acquire10.json. A .part without a matching marker is
    foreign and blocks the run; it is never touched.
  * Resume re-hashes the existing .part, then asks for the exact remainder.
  * The final name is created with link()+unlink(): an existing file is never
    replaced. An existing final file is re-hashed, never trusted.
  * SHA256 is computed while streaming and recorded next to the official LFS
    digest; a mismatch marks the .part 'rejected' and keeps it for inspection.
  * Free space is checked on the physical backing volume before the run and
    every --check-interval bytes; ENOSPC and the stop threshold retain .part.
  * Cleanup (--discard-rejected / --clean-temporaries) removes only .part files
    whose ownership marker matches this manifest.
"""
import argparse
import errno
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import socket
import ssl
import struct
import sys
import time
import urllib.parse
from datetime import datetime, timezone

SCHEMA = 'inwime-acquire10/1'
AUTH_SCHEMA = 'inwime-acquire10-authorization/1'
AGENT = 'Python-urllib/3'          # generic: no project or personal identifier
ENDPOINT = 'https://huggingface.co'
ALLOWED_SUFFIXES = ('.huggingface.co', '.hf.co')
LOOPBACK = ('127.0.0.1', 'localhost', '::1')
REDIRECTS = (301, 302, 303, 307, 308)
GiB = 2**30
MiB = 2**20


class AcquireError(Exception):
    """Fail-closed refusal; nothing was renamed or overwritten."""


class TransferStopped(AcquireError):
    """Interrupted transfer; the owned .part (if any) is retained for resume."""


class SpaceError(TransferStopped):
    pass


class Manifest:
    """Immutable description of the files to acquire; digest pinned in protocol.json."""

    def __init__(self, model, revision, shards, small, license='apache-2.0'):
        self.model, self.revision, self.license = model, revision, license
        # shard: (nbytes, lfs_sha256, git_blob_id, header_length, header_sha256)
        self.shards = dict(shards)
        # small: (nbytes, git_blob_id)
        self.small = dict(small)
        if set(self.shards) & set(self.small):
            raise ValueError('file listed twice')

    def files(self):
        """Small files first (cheap, verify the index), then shards in order."""
        return [*sorted(self.small), *sorted(self.shards)]

    def size(self, name):
        return (self.shards.get(name) or self.small[name])[0]

    def total_bytes(self):
        return sum(v[0] for v in self.shards.values()) + sum(v[0] for v in self.small.values())

    def canonical(self):
        return dict(model=self.model, revision=self.revision, license=self.license,
                    shards={k: dict(nbytes=v[0], lfs_sha256=v[1], blob_id=v[2], header_length=v[3],
                                    header_sha256=v[4]) for k, v in sorted(self.shards.items())},
                    small={k: dict(nbytes=v[0], blob_id=v[1]) for k, v in sorted(self.small.items())})

    def digest(self):
        raw = json.dumps(self.canonical(), sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(raw).hexdigest()


QWEN25_14B = Manifest(
    'Qwen/Qwen2.5-14B', '97e1e76335b7017d8f67c08a19d103c0504298c9',
    shards={
        'model-00001-of-00008.safetensors': (3885134104, 'e015e2bc9a26b4e46d77913d8c667608ae7e48aa1eca04af5786c2408f4bc0fa',
                                             '0504bafae1ddcbc85d48285c87c8321cbaff57fc', 6416,
                                             'b8a9bb6fed05e00312ee49fcc294baac74ba7af94337be6a8978135c25c41be1'),
        'model-00002-of-00008.safetensors': (3995327992, 'e7efa1adc8257218813dcb494bb2a3d5775fa268735ab39e5b8119e233c21462',
                                             'c9a026a826f7bcb4b1206bf4953c9690e20c2875', 9712,
                                             '1eab343927d487c3d12856938b4582f4dd9a4a08f123fd105fc8a3568bffbf3f'),
        'model-00003-of-00008.safetensors': (3995328064, '68127c9bc4fab170f7aaf63d5c7ac9e182afd10b74a1c6bb8025afefc11447cb',
                                             '383698ee64c0b8e0ec66f9ffc0f75af212855bee', 9784,
                                             '6671aca470735a98bfc64b489b89641e9d4d2428e3f554a689b860ec0b8c78b8'),
        'model-00004-of-00008.safetensors': (3995348776, '7250708a789da850ff40a4a5be335971dfa0d2bd7cba2e9905916dab06744d75',
                                             'f52854c5a074379894d35c1a1ba7fc3c95b852e4', 10016,
                                             'c75f9185e6258940c5d7fa28297a1a92720b5f524651947c42e3edfcc73630b2'),
        'model-00005-of-00008.safetensors': (3979614448, 'c2754167c1cbaf94b9af9c7eb646a2286a596f9ded5e2e3c4c5e6a4464352c9e',
                                             '885c5514c7a9ebb84bfecbccb6f7281ef4592b6d', 10472,
                                             '0447cdb3a415069c428ed4394c75a6d94a069ff46a1d2175a1ff89a882a9887c'),
        'model-00006-of-00008.safetensors': (3995328064, '7045cf78c68178b626546982d12b6e9c8e289f1bf1e65c42225ed13e07847180',
                                             '03bbc4b3b2b35894d533efe6514b73480926ea27', 9784,
                                             '765cb81a9f2c825475b607c7b54719c131ead0232741c10ed2dd901ead77afae'),
        'model-00007-of-00008.safetensors': (3995328064, '4d870a72c656404ee7524163ba996bf55050fff252dfe639a90715a9e2c47dba',
                                             'e238a7a4dec4d38d336806d198ab64df5b042d76', 9784,
                                             '42cda7e1576abc459afdc432b8da4dd632c399bd83aa192a162c22a12b18bf24'),
        'model-00008-of-00008.safetensors': (1698724408, '9b693615965d8548eab4d1dc6bb578aa063e8656e2b9d25125ad5b8c7f59565f',
                                             'dd1309620ebee47f63ac40b9b235b7ae5f9711e6', 560,
                                             '423a48781d74000ab9b3e53999732f870f7aac17eadea70bfe0a2ad363405006'),
    },
    small={
        'LICENSE': (11343, '6634c8cc3133b3848ec74b9f275acaaa1ea618ab'),
        'README.md': (3795, '1fc8268ebebe27091caa6165adf1f5fd71bdee30'),
        'config.json': (664, '83bba6e26639fb152bd0077977cf6ea8312b42a9'),
        'generation_config.json': (138, 'cbbb3133034e192527e5321b4c679154e4819ab8'),
        'merges.txt': (1671839, '20024bfe7c83998e9aeaf98a0cd6a2ce6306c2f0'),
        'model.safetensors.index.json': (47472, '7c8139ab9a4e680ff0e9741c678e26c43788abf0'),
        'tokenizer.json': (7031645, '443909a61d429dff23010e5bddd28ff530edda00'),
        'tokenizer_config.json': (7228, 'ba7e4c5637b9732dadcd66286ce48334e8b31e9e'),
        'vocab.json': (2776833, '4783fe10ac3adce15ac8f358ef5462739852c569'),
    })


class Reserves:
    """Space policy (bytes). Defaults are the preregistered M10 values."""

    def __init__(self, fixed_margin=32*GiB, temporary=0, verification=0, rollback=0,
                 evidence=GiB, swap_growth=7_344_226_304, stop_threshold=40*GiB,
                 check_interval=256*MiB):
        values = dict(fixed_margin=fixed_margin, temporary=temporary, verification=verification,
                      rollback=rollback, evidence=evidence, swap_growth=swap_growth,
                      stop_threshold=stop_threshold, check_interval=check_interval)
        for k, v in values.items():
            if type(v) is not int or v < 0:
                raise ValueError(f'reserve {k} must be a non-negative integer')
        if check_interval <= 0:
            raise ValueError('check_interval must be positive')
        self.__dict__.update(values)

    def held(self):
        return (self.fixed_margin + self.temporary + self.verification + self.rollback
                + self.evidence + self.swap_growth)

    def as_dict(self):
        return dict(vars(self))


def blob_id(raw):
    return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# ---------------------------------------------------------------- transport

def host_allowed(url, allow_loopback_http=False):
    u = urllib.parse.urlsplit(url)
    host = (u.hostname or '').lower()
    if u.username or u.password:
        return False
    if u.scheme == 'https' and (host == 'huggingface.co' or host.endswith(ALLOWED_SUFFIXES)):
        return True
    return bool(allow_loopback_http and u.scheme == 'http' and host in LOOPBACK)


def _small_socket(rcvbuf):
    def create(address, timeout=None, source_address=None, *args, **kwargs):
        host, port = address
        error = None
        for family, kind, proto, _, sockaddr in socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM):
            s = socket.socket(family, kind, proto)
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
                s.settimeout(timeout)
                s.connect(sockaddr)
                return s
            except OSError as e:
                error = e
                s.close()
        raise error or OSError('no address')
    return create


class Response:
    def __init__(self, connection, response, url):
        self.connection, self.response, self.url = connection, response, url
        self.status = response.status
        self.headers = {k.lower(): v for k, v in response.getheaders()}

    def read(self, n):
        return self.response.read(n)

    def close(self):
        self.connection.close()


class Transport:
    """Minimal HTTP client: allowlisted hosts, bounded redirects, request log."""

    def __init__(self, *, allow_loopback_http=False, timeout=60, rcvbuf=1 << 16, max_redirects=5,
                 metadata_cap=4*MiB):
        self.allow_loopback_http, self.timeout, self.rcvbuf = allow_loopback_http, timeout, rcvbuf
        self.max_redirects, self.metadata_cap = max_redirects, metadata_cap
        self.metadata_bytes = 0
        self.payload_bytes = 0
        self.log = []

    def open(self, url, headers=None):
        hops = 0
        while True:
            if not host_allowed(url, self.allow_loopback_http):
                raise AcquireError('refused URL (scheme/host not allowlisted): '
                                   + (urllib.parse.urlsplit(url).hostname or '?'))
            u = urllib.parse.urlsplit(url)
            cls = http.client.HTTPSConnection if u.scheme == 'https' else http.client.HTTPConnection
            kwargs = dict(timeout=self.timeout)
            if u.scheme == 'https':
                kwargs['context'] = ssl.create_default_context()
            connection = cls(u.hostname, u.port, **kwargs)
            connection._create_connection = _small_socket(self.rcvbuf)
            path = (u.path or '/') + ('?'+u.query if u.query else '')
            h = {'User-Agent': AGENT, 'Accept-Encoding': 'identity'} | (headers or {})
            connection.request('GET', path, headers=h)
            raw = connection.getresponse()
            r = Response(connection, raw, url)
            self.log.append(dict(host=u.hostname, path=u.path, status=r.status, range=h.get('Range'),
                                 content_range=r.headers.get('content-range'),
                                 content_length=r.headers.get('content-length')))
            if r.status in REDIRECTS:
                location = r.headers.get('location')
                r.close()
                hops += 1
                if not location or hops > self.max_redirects:
                    raise AcquireError('redirect without location or too many redirects')
                url = urllib.parse.urljoin(url, location)
                continue
            return r

    def fetch(self, url, cap, headers=None, want=(200,)):
        """Bounded small response; the body is read only for an expected status."""
        r = self.open(url, headers)
        try:
            if r.status not in want:
                raise AcquireError(f'unexpected HTTP status {r.status} for {urllib.parse.urlsplit(url).path}')
            length = r.headers.get('content-length')
            if length is not None and int(length) > cap:
                raise AcquireError('response larger than its bound')
            if self.metadata_bytes >= self.metadata_cap:
                raise AcquireError('global metadata cap exceeded')
            body = r.read(cap+1)
            self.metadata_bytes += len(body)
            if len(body) > cap or self.metadata_bytes > self.metadata_cap:
                raise AcquireError('response exceeds bound')
            return r, body
        finally:
            r.close()


def exact_range(r, first, last, total):
    return r.status == 206 and r.headers.get('content-range') == f'bytes {first}-{last}/{total}'


# ---------------------------------------------------------------- metadata

def resolve(endpoint, manifest, name):
    return f'{endpoint}/{manifest.model}/resolve/{manifest.revision}/{name}'


def check_api(metadata, manifest):
    """Official revision metadata must equal the pinned manifest exactly."""
    problems = []
    if metadata.get('sha') != manifest.revision:
        problems.append('revision sha differs')
    if metadata.get('gated') not in (False, None) or metadata.get('private') not in (False, None):
        problems.append('repository gated or private')
    if (metadata.get('cardData') or {}).get('license') != manifest.license:
        problems.append('license differs')
    entries = {x.get('rfilename'): x for x in metadata.get('siblings', [])}
    for name in manifest.files():
        e = entries.get(name)
        if e is None:
            problems.append('missing file '+name)
            continue
        if e.get('size') != manifest.size(name):
            problems.append('size differs '+name)
        if name in manifest.shards:
            size, sha, blob, _, _ = manifest.shards[name]
            if (e.get('lfs') or {}).get('sha256') != sha or e.get('blobId') != blob:
                problems.append('LFS sha256/blob differs '+name)
        elif e.get('blobId') != manifest.small[name][1]:
            problems.append('blob differs '+name)
    if problems:
        raise AcquireError('official metadata differs from the pinned manifest: '+'; '.join(problems))
    return dict(revision=metadata['sha'], license=manifest.license, files_checked=len(manifest.files()))


def parse_header(raw, total_size, name):
    """Independent safetensors header check: BF16 only, contiguous extents, exact payload."""
    try:
        parsed = json.loads(raw)
    except ValueError as error:
        raise AcquireError('invalid safetensors header JSON: '+name) from error
    parsed.pop('__metadata__', None)
    extents, tensors = [], {}
    for tensor, d in parsed.items():
        begin, end = d['data_offsets']
        count = 1
        for n in d['shape']:
            count *= n
        if d['dtype'] != 'BF16' or end-begin != 2*count or begin < 0:
            raise AcquireError('non-BF16 or inconsistent tensor: '+tensor)
        extents.append((begin, end))
        tensors[tensor] = dict(shape=d['shape'], nbytes=end-begin)
    extents.sort()
    cursor = 0
    for begin, end in extents:
        if begin != cursor:
            raise AcquireError('gap or overlap in safetensors payload: '+name)
        cursor = end
    if cursor != total_size-8-len(raw):
        raise AcquireError('payload/header extent mismatch: '+name)
    return tensors


def fetch_header(transport, url, name, manifest):
    size, _, _, header_length, header_sha = manifest.shards[name]
    r = transport.open(url, {'Range': 'bytes=0-7'})
    try:
        if not exact_range(r, 0, 7, size):
            raise AcquireError('Range not honoured exactly for header prefix (body not read): '+name)
        prefix = r.read(9)
    finally:
        r.close()
    transport.metadata_bytes += len(prefix)
    if len(prefix) != 8:
        raise AcquireError('header prefix length differs: '+name)
    length, = struct.unpack('<Q', prefix)
    if length != header_length:
        raise AcquireError('header length differs from the pinned value: '+name)
    r = transport.open(url, {'Range': f'bytes=8-{7+length}'})
    try:
        if not exact_range(r, 8, 7+length, size):
            raise AcquireError('Range not honoured exactly for header (body not read): '+name)
        raw = r.read(length+1)
    finally:
        r.close()
    transport.metadata_bytes += len(raw)
    if transport.metadata_bytes > transport.metadata_cap:
        raise AcquireError('global metadata cap exceeded')
    if len(raw) != length or hashlib.sha256(raw).hexdigest() != header_sha:
        raise AcquireError('header bytes differ from the pinned SHA256: '+name)
    return raw


def verify_metadata(api, small, headers, manifest):
    """Common check for online and offline metadata. Returns facts, raises on any difference."""
    facts = dict(api=check_api(api, manifest))
    for name in ('config.json', 'model.safetensors.index.json'):
        raw = small[name]
        if len(raw) != manifest.small[name][0] or blob_id(raw) != manifest.small[name][1]:
            raise AcquireError('official Git blob mismatch: '+name)
    config = json.loads(small['config.json'])
    index = json.loads(small['model.safetensors.index.json'])
    tensors = {}
    for name, raw in headers.items():
        size, _, _, length, sha = manifest.shards[name]
        if len(raw) != length or hashlib.sha256(raw).hexdigest() != sha:
            raise AcquireError('header bytes differ from the pinned SHA256: '+name)
        for t, d in parse_header(raw, size, name).items():
            if t in tensors:
                raise AcquireError('tensor present in two shards: '+t)
            tensors[t] = d | dict(shard=name)
    if set(headers) != set(manifest.shards):
        raise AcquireError('missing shard headers')
    if {t: d['shard'] for t, d in tensors.items()} != index['weight_map']:
        raise AcquireError('index weight_map disagrees with shard headers')
    payload = sum(d['nbytes'] for d in tensors.values())
    if payload != index['metadata']['total_size']:
        raise AcquireError('index total_size disagrees with headers')
    facts.update(tensor_count=len(tensors), payload_bytes=payload, dtypes=['BF16'],
                 config={k: config.get(k) for k in (
                     'architectures', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                     'num_attention_heads', 'num_key_value_heads', 'rms_norm_eps', 'rope_theta',
                     'vocab_size', 'tie_word_embeddings', 'use_sliding_window')},
                 header_bytes_checked=sum(8+len(r) for r in headers.values()))
    return facts


def online_metadata(transport, endpoint, manifest):
    api_url = f'{endpoint}/api/models/{manifest.model}/revision/{manifest.revision}?blobs=true'
    _, raw = transport.fetch(api_url, MiB)
    api = json.loads(raw)
    check_api(api, manifest)
    small = {}
    for name in ('config.json', 'model.safetensors.index.json'):
        small[name] = transport.fetch(resolve(endpoint, manifest, name), 256*1024)[1]
    headers = {name: fetch_header(transport, resolve(endpoint, manifest, name), name, manifest)
               for name in sorted(manifest.shards)}
    return api, small, headers


def offline_metadata(directory):
    """Metadata captured by the preflight (file names as in results/increment10/preflight-metadata)."""
    d = Path(directory)
    api = json.loads((d/'api-revision-blobs.json').read_bytes())
    small = {n: (d/('file-'+n)).read_bytes() for n in ('config.json', 'model.safetensors.index.json')}
    headers = {p.name.removeprefix('header-').removesuffix('.json'): p.read_bytes()
               for p in sorted(d.glob('header-*.safetensors.json'))}
    return api, small, headers


# ---------------------------------------------------------------- destination state

def marker_path(path):
    return path.with_name(path.name+'.part.acquire10.json')


def part_path(path):
    return path.with_name(path.name+'.part')


def expected_marker(manifest, name):
    if name in manifest.shards:
        size, sha, blob = manifest.shards[name][:3]
    else:
        size, blob = manifest.small[name]
        sha = None
    return dict(tool='acquire10', schema=SCHEMA, model=manifest.model, revision=manifest.revision,
                file=name, nbytes=size, lfs_sha256=sha, blob_id=blob)


def read_marker(path, manifest, name):
    m = marker_path(path)
    if not m.exists():
        return None
    try:
        data = json.loads(m.read_text())
    except (OSError, ValueError):
        return None
    want = expected_marker(manifest, name)
    if any(data.get(k) != v for k, v in want.items()) or data.get('state') not in ('partial', 'rejected'):
        return None
    return data


def write_marker(path, manifest, name, state, **extra):
    data = expected_marker(manifest, name) | dict(state=state, updated_utc=now()) | extra
    m = marker_path(path)
    temp = m.with_name(m.name+'.tmp')
    temp.write_text(json.dumps(data, indent=1)+'\n')
    os.replace(temp, m)        # the marker itself is tool-owned; replacing it is not a payload overwrite


def inspect(destination, manifest):
    """Per-file state without reading payload bytes. Never modifies anything."""
    destination = Path(destination)
    states, blockers = {}, []
    for name in manifest.files():
        path = destination/name
        part, size = part_path(path), manifest.size(name)
        marker = read_marker(path, manifest, name) if part.exists() else None
        if path.exists() and part.exists():
            state, transfer = 'conflict-final-and-part', 0
            blockers.append('final file and .part both present: '+name)
        elif path.exists():
            actual = path.stat().st_size
            state, transfer = ('present-unverified' if actual == size else 'present-wrong-size'), 0
            if actual != size:
                blockers.append('existing file has the wrong size (never overwritten): '+name)
        elif part.exists():
            have = part.stat().st_size
            if marker is None:
                state, transfer = 'foreign-part', size
                blockers.append('.part without matching acquire10 marker (not owned, untouched): '+name)
            elif marker['state'] == 'rejected':
                state, transfer = 'rejected-part', size
                blockers.append('.part rejected by digest check; use --discard-rejected: '+name)
            elif have > size:
                state, transfer = 'part-too-long', size
                blockers.append('.part longer than the official size: '+name)
            else:
                state, transfer = 'partial', size-have
        else:
            state, transfer = 'absent', size
        states[name] = dict(state=state, nbytes=size, transfer_bytes=transfer)
    extra = sorted(p.name for p in destination.iterdir()
                   if p.name not in manifest.files() and not p.name.endswith(('.part', '.part.acquire10.json'))
                   and p.name not in ('safetensors.json', 'provenance.json')) if destination.is_dir() else []
    return dict(destination_exists=destination.is_dir(), files=states, blockers=blockers,
                unmanaged_files=extra)


def existing_ancestor(path):
    path = Path(path).absolute()
    while not path.exists():
        path = path.parent
    return path


def space_plan(transfer_bytes, reserves, free_destination, free_backing):
    effective = min(free_destination, free_backing)
    required = transfer_bytes + reserves.held()
    blockers = []
    if effective < required:
        blockers.append(f'insufficient physical free space: {effective} < {required}')
    if effective - transfer_bytes < reserves.stop_threshold:
        blockers.append('the transfer would cross the stop threshold before completion')
    return dict(transfer_bytes=transfer_bytes, reserves=reserves.as_dict(), reserves_held=reserves.held(),
                required_free_bytes=required, free_destination_filesystem=free_destination,
                free_backing_volume=free_backing, effective_free_bytes=effective,
                surplus_bytes=effective-required, expected_free_after=effective-transfer_bytes,
                blockers=blockers)


# ---------------------------------------------------------------- transfer

def hash_file(path, chunk=MiB):
    digest, done = hashlib.sha256(), 0
    with path.open('rb') as f:
        while data := f.read(chunk):
            digest.update(data)
            done += len(data)
    return digest, done


class Guard:
    """Free-space guard evaluated on the physical volume during the transfer."""

    def __init__(self, destination, backing, reserves, usage=shutil.disk_usage):
        self.destination, self.backing, self.reserves, self.usage = destination, backing, reserves, usage
        self.minimum_free = None
        self.checks = 0

    def free(self):
        return min(self.usage(existing_ancestor(self.destination)).free, self.usage(self.backing).free)

    def check(self):
        free = self.free()
        self.checks += 1
        self.minimum_free = free if self.minimum_free is None else min(self.minimum_free, free)
        if free < self.reserves.stop_threshold:
            raise SpaceError(f'free space {free} below stop threshold {self.reserves.stop_threshold}; .part retained')


def transfer(transport, url, destination, name, manifest, guard, budget, opener=open, chunk=MiB):
    """Fetch or resume one file into its owned .part; return (digest, resumed_from, transferred)."""
    path = Path(destination)/name
    part, size = part_path(path), manifest.size(name)
    if part.exists():
        if read_marker(path, manifest, name) is None:
            raise AcquireError('.part not owned by acquire10: '+name)
        digest, done = hash_file(part)
    else:
        write_marker(path, manifest, name, 'partial', bytes=0)
        with opener(part, 'xb'):
            pass
        digest, done = hashlib.sha256(), 0
    resumed = done
    if done > size:
        raise AcquireError('.part longer than the official size: '+name)
    transferred = 0
    if done < size:
        need = size-done
        if budget['transferred']+need > budget['max_bytes']:
            raise AcquireError('authorized transfer budget would be exceeded')
        headers = {'Range': f'bytes={done}-'} if done else {}
        r = transport.open(url, headers)
        try:
            if done:
                if not exact_range(r, done, size-1, size):
                    raise TransferStopped('server did not honour the resume Range exactly (body not read); '
                                          '.part retained: '+name)
            elif r.status != 200:
                raise TransferStopped(f'unexpected HTTP status {r.status}; .part retained: '+name)
            length = r.headers.get('content-length')
            if length is None or int(length) != need:
                raise TransferStopped('Content-Length differs from the expected remainder (body not read): '+name)
            since = 0
            with opener(part, 'ab') as out:
                while done < size:
                    try:
                        data = r.read(min(size-done, chunk))
                    except (http.client.IncompleteRead, ConnectionError, socket.timeout) as error:
                        data = getattr(error, 'partial', b'') or b''
                        if data:
                            out.write(data); digest.update(data); done += len(data); transferred += len(data)
                            budget['transferred'] += len(data); transport.payload_bytes += len(data)
                        raise TransferStopped('connection interrupted; .part retained: '+name) from error
                    if not data:
                        raise TransferStopped('incomplete response; .part retained: '+name)
                    try:
                        out.write(data)
                        out.flush()
                    except OSError as error:
                        if error.errno in (errno.ENOSPC, errno.EDQUOT):
                            raise SpaceError('no space left while writing; .part retained: '+name) from error
                        raise
                    digest.update(data)
                    done += len(data); transferred += len(data); since += len(data)
                    budget['transferred'] += len(data)
                    transport.payload_bytes += len(data)
                    if since >= guard.reserves.check_interval:
                        since = 0
                        guard.check()
                os.fsync(out.fileno())
        finally:
            r.close()
            if part.exists():
                write_marker(path, manifest, name, 'partial', bytes=part.stat().st_size)
    return digest, resumed, transferred


def commit_file(destination, name, manifest, digest, report):
    """Verify the complete .part and create the final name without replacing anything."""
    path = Path(destination)/name
    part, size = part_path(path), manifest.size(name)
    actual = part.stat().st_size
    computed = digest.hexdigest()
    if name in manifest.shards:
        ok = actual == size and computed == manifest.shards[name][1]
        official = manifest.shards[name][1]
    else:
        raw = part.read_bytes()
        ok = actual == size and blob_id(raw) == manifest.small[name][1]
        official = manifest.small[name][1]
    if not ok:
        write_marker(path, manifest, name, 'rejected', bytes=actual, computed_sha256=computed)
        raise AcquireError('size/digest mismatch; .part kept and marked rejected: '+name)
    try:
        os.link(part, path)      # fails with EEXIST: an existing final file is never replaced
    except FileExistsError as error:
        raise AcquireError('final file appeared concurrently; nothing replaced: '+name) from error
    os.unlink(part)
    marker_path(path).unlink()
    report[name] = dict(nbytes=size, computed_sha256=computed,
                        **({'official_lfs_sha256': official} if name in manifest.shards else {'blob_id': official}))


def discard_owned(destination, manifest, states=('rejected',)):
    """Remove only .part files whose marker matches this manifest and state."""
    removed = []
    for name in manifest.files():
        path = Path(destination)/name
        marker = read_marker(path, manifest, name)
        if marker is not None and marker['state'] in states:
            part = part_path(path)
            if part.exists():
                part.unlink()
            marker_path(path).unlink()
            removed.append(name)
    return removed


# ---------------------------------------------------------------- authorization

def check_authorization(path, manifest, destination, plan_bytes):
    try:
        auth = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise AcquireError('authorization file unreadable or invalid') from error
    want = dict(schema=AUTH_SCHEMA, model=manifest.model, revision=manifest.revision,
                manifest_sha256=manifest.digest())
    wrong = [k for k, v in want.items() if auth.get(k) != v]
    if wrong:
        raise AcquireError('authorization does not match the pinned manifest: '+', '.join(wrong))
    if Path(auth.get('destination', '')).absolute() != Path(destination).absolute():
        raise AcquireError('authorization names a different destination')
    limit = auth.get('max_transfer_bytes')
    if type(limit) is not int or limit <= 0:
        raise AcquireError('authorization lacks an integer max_transfer_bytes')
    if plan_bytes > limit:
        raise AcquireError(f'planned transfer {plan_bytes} exceeds the authorized {limit} bytes')
    for k in ('approved_by', 'approved_utc'):
        if not isinstance(auth.get(k), str) or not auth[k]:
            raise AcquireError('authorization lacks '+k)
    return auth


# ---------------------------------------------------------------- driver

def write_evidence(path, report):
    if path is None:
        return
    with Path(path).open('x') as f:
        json.dump(report, f, indent=2, sort_keys=True); f.write('\n')


def host_readiness(host_gate, destination, backing, manifest, usage):
    """Technical eligibility from the host gate (tools/hostgate10.py); only a well-formed GO passes."""
    from . import hostgate10
    if host_gate is None or not callable(getattr(host_gate, 'evaluate', None)):
        raise AcquireError('host readiness gate missing: a fresh hostmem10 GO (--host-report/--host-baseline) '
                           'is required before any transfer')
    result = host_gate.evaluate(destination, backing, manifest, usage)
    if not isinstance(result, dict) or result.get('schema') != hostgate10.SCHEMA or result.get('profile') != 'download' \
            or result.get('verdict') != 'GO' or result.get('reasons'):
        verdict = result.get('verdict') if isinstance(result, dict) else None
        reasons = result.get('reasons') if isinstance(result, dict) else None
        raise AcquireError(f'host readiness gate is not GO ({verdict}): '+'; '.join(map(str, reasons or ['invalid result'])))
    return result


def dry_run(destination, backing, *, manifest=QWEN25_14B, reserves=None, endpoint=ENDPOINT,
            metadata_dir=None, transport=None, usage=shutil.disk_usage, host_gate=None):
    reserves = reserves or Reserves()
    report = dict(schema=SCHEMA, mode='dry-run', utc=now(), model=manifest.model, revision=manifest.revision,
                  manifest_sha256=manifest.digest(), download_total_bytes=manifest.total_bytes(),
                  payload_bytes_transferred=0, user_agent=AGENT)
    if metadata_dir is not None:
        report['metadata_source'] = 'offline-captured'
        api, small, headers = offline_metadata(metadata_dir)
    else:
        transport = transport or Transport()
        report['metadata_source'] = 'online'
        api, small, headers = online_metadata(transport, endpoint, manifest)
        report['requests'] = transport.log
        report['metadata_bytes_read'] = transport.metadata_bytes
        if transport.payload_bytes:
            raise AcquireError('internal error: payload bytes read in dry-run')
    report['metadata'] = verify_metadata(api, small, headers, manifest)
    state = inspect(destination, manifest)
    plan = space_plan(sum(f['transfer_bytes'] for f in state['files'].values()), reserves,
                      usage(existing_ancestor(destination)).free, usage(backing).free)
    report.update(destination=dict(path_relative=os.path.relpath(Path(destination).absolute()),
                                   backing_volume=str(backing), **state),
                  space=plan)
    # Storage readiness alone is never permission to download (review of PR #24): the host gate
    # must be GO as well, and owner authorization stays a separate record that a dry-run never reads.
    storage_ready = not state['blockers'] and not plan['blockers']
    if host_gate is None:
        host = dict(verdict='NOT EVALUATED', reasons=['storage-only dry-run: no --host-report/--host-baseline'])
    else:
        host = host_gate.evaluate(destination, backing, manifest, usage)
    report.update(storage_ready=storage_ready, host_readiness=host,
                  owner_authorization='not evaluated by a dry-run; --execute requires a separate owner record',
                  would_proceed_if_authorized=storage_ready and host.get('verdict') == 'GO' and not host.get('reasons'))
    return report


def execute(destination, backing, authorization, host_gate, *, manifest=QWEN25_14B, reserves=None,
            endpoint=ENDPOINT, transport=None, usage=shutil.disk_usage, opener=open, discard_rejected=False):
    reserves = reserves or Reserves()
    transport = transport or Transport()
    destination = Path(destination)
    report = dict(schema=SCHEMA, mode='execute', utc_start=now(), model=manifest.model,
                  revision=manifest.revision, manifest_sha256=manifest.digest(), user_agent=AGENT)
    if not destination.parent.is_dir():
        raise AcquireError('destination parent does not exist (not created implicitly)')
    # Owner authorization (human decision) and host readiness (technical eligibility) are separate
    # checks; both run before any network request and again under the lock before any payload request.
    check_authorization(authorization, manifest, destination,
                        sum(f['transfer_bytes'] for f in inspect(destination, manifest)['files'].values()))
    host_before_network = host_readiness(host_gate, destination, backing, manifest, usage)
    api, small, headers = online_metadata(transport, endpoint, manifest)
    report['metadata'] = verify_metadata(api, small, headers, manifest)
    destination.mkdir(exist_ok=True)
    lock = destination/'.acquire10.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise AcquireError('another acquire10 run holds the lock (or a stale lock must be reviewed)') from error
    try:
        os.write(fd, f'{os.getpid()}\n'.encode()); os.close(fd)
        if discard_rejected:
            report['discarded_rejected_parts'] = discard_owned(destination, manifest)
        state = inspect(destination, manifest)
        planned = sum(f['transfer_bytes'] for f in state['files'].values())
        auth = check_authorization(authorization, manifest, destination, planned)
        report['authorization'] = {k: auth[k] for k in ('approved_by', 'approved_utc', 'max_transfer_bytes')}
        host_before_payload = host_readiness(host_gate, destination, backing, manifest, usage)
        report['host_readiness'] = {
            when: {k: r.get(k) for k in ('verdict', 'report_sha256', 'baseline_sha256', 'report_utc', 'report_age_s',
                                         'protocol_sha256', 'values')}
            for when, r in (('before_network', host_before_network), ('before_payload', host_before_payload))}
        if state['blockers']:
            raise AcquireError('destination state blocks the run: '+'; '.join(state['blockers']))
        guard = Guard(destination, backing, reserves, usage)
        plan = space_plan(planned, reserves, usage(existing_ancestor(destination)).free, usage(backing).free)
        report['space_before'] = plan
        if plan['blockers']:
            raise SpaceError('; '.join(plan['blockers']))
        budget = dict(max_bytes=auth['max_transfer_bytes'], transferred=0)
        committed, verification = {}, {}
        for name in manifest.files():
            path = destination/name
            if path.exists():
                digest, done = hash_file(path)
                ok = done == manifest.size(name) and (
                    digest.hexdigest() == manifest.shards[name][1] if name in manifest.shards
                    else blob_id(path.read_bytes()) == manifest.small[name][1])
                if not ok:
                    raise AcquireError('existing file differs from the pinned manifest (never overwritten): '+name)
                verification[name] = dict(method='existing file re-hashed', computed_sha256=digest.hexdigest())
                continue
            digest, resumed, moved = transfer(transport, resolve(endpoint, manifest, name), destination, name,
                                              manifest, guard, budget, opener=opener)
            commit_file(destination, name, manifest, digest, committed)
            verification[name] = dict(method='streamed' if not resumed else
                                      ('complete .part re-hashed' if not moved else 'resumed .part re-hashed then streamed'),
                                      resumed_from_bytes=resumed, transferred_bytes=moved,
                                      computed_sha256=digest.hexdigest())
        descriptor = dict(format='inwime-safetensors-bf16-v1', index='model.safetensors.index.json',
                          index_sha256=hashlib.sha256((destination/'model.safetensors.index.json').read_bytes()).hexdigest(),
                          files={n: dict(nbytes=manifest.shards[n][0], sha256=verification[n]['computed_sha256'])
                                 for n in sorted(manifest.shards)})
        provenance = dict(model=manifest.model, revision=manifest.revision, license=manifest.license,
                          manifest_sha256=manifest.digest(),
                          files={n: dict(nbytes=manifest.size(n), sha256=verification[n]['computed_sha256'])
                                 for n in manifest.files()})
        for fname, data in (('safetensors.json', descriptor), ('provenance.json', provenance)):
            path = destination/fname
            if path.exists():
                if json.loads(path.read_text()) != data:
                    raise AcquireError('existing descriptor differs (never overwritten): '+fname)
            else:
                with path.open('x') as f:
                    json.dump(data, f, indent=2); f.write('\n')
        report.update(verification=verification, committed=committed, payload_bytes_transferred=budget['transferred'],
                      minimum_free_during_transfer=guard.minimum_free, guard_checks=guard.checks,
                      free_after=guard.free(), utc_end=now(), requests=len(transport.log))
        return report
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('destination', type=Path)
    p.add_argument('--backing-volume', type=Path, required=True,
                   help='mount of the physical volume that stores the destination (e.g. /mnt/c for the WSL VHDX)')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='default; no payload transfer')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--clean-temporaries', action='store_true',
                      help='remove only acquire10-owned .part files (partial and rejected)')
    mode.add_argument('--print-manifest', action='store_true')
    p.add_argument('--authorization', type=Path, help='owner approval record (human decision)')
    p.add_argument('--host-report', type=Path, help='fresh hostmem10 report, profile download, taken with --baseline')
    p.add_argument('--host-baseline', type=Path, help='the baseline report named by --host-report')
    p.add_argument('--metadata-dir', type=Path, help='dry-run: verify captured metadata offline')
    p.add_argument('--evidence', type=Path)
    p.add_argument('--discard-rejected', action='store_true')
    for name, default in Reserves().as_dict().items():
        p.add_argument('--reserve-'+name.replace('_', '-'), dest='reserve_'+name, type=int, default=default)
    a = p.parse_args(argv)
    from .hostgate10 import Gate
    if a.print_manifest:
        print(json.dumps(QWEN25_14B.canonical() | dict(manifest_sha256=QWEN25_14B.digest(),
                                                       download_total_bytes=QWEN25_14B.total_bytes()), indent=1))
        return 0
    reserves = Reserves(**{k: getattr(a, 'reserve_'+k) for k in Reserves().as_dict()})
    if a.clean_temporaries:
        print(json.dumps(dict(removed=discard_owned(a.destination, QWEN25_14B, ('partial', 'rejected'))), indent=1))
        return 0
    if a.execute:
        if a.authorization is None:
            p.error('--execute requires --authorization (explicit owner approval record)')
        if a.host_report is None or a.host_baseline is None:
            p.error('--execute requires --host-report and --host-baseline (fresh hostmem10 GO, profile download)')
        if a.metadata_dir is not None:
            p.error('--execute always re-verifies metadata online')
        report = execute(a.destination, a.backing_volume, a.authorization, Gate(a.host_report, a.host_baseline),
                         reserves=reserves, discard_rejected=a.discard_rejected)
    else:
        if a.authorization is not None or a.discard_rejected:
            p.error('--authorization/--discard-rejected only apply to --execute')
        if (a.host_report is None) != (a.host_baseline is None):
            p.error('--host-report and --host-baseline go together')
        gate = Gate(a.host_report, a.host_baseline) if a.host_report is not None else None
        report = dry_run(a.destination, a.backing_volume, reserves=reserves, metadata_dir=a.metadata_dir,
                         host_gate=gate)
    write_evidence(a.evidence, report)
    summary = {k: report.get(k) for k in ('mode', 'manifest_sha256', 'download_total_bytes',
                                          'payload_bytes_transferred', 'storage_ready', 'would_proceed_if_authorized')}
    if 'host_readiness' in report:
        h = report['host_readiness']
        summary['host_readiness'] = {k: h.get(k) for k in ('verdict', 'reasons')} if 'verdict' in h else \
            {k: v.get('verdict') for k, v in h.items()}
    if 'space' in report:
        summary['space'] = {k: report['space'][k] for k in ('transfer_bytes', 'required_free_bytes',
                                                            'effective_free_bytes', 'surplus_bytes', 'blockers')}
        summary['destination_blockers'] = report['destination']['blockers']
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except AcquireError as error:
        print('acquire10: refused: '+str(error), file=sys.stderr)
        sys.exit(2)

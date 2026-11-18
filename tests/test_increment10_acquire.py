"""Increment 10, C1: tools/acquire10.py against a local fake HTTP server (no network, no weights).

Covers the pinned 14B manifest, the offline metadata check of the preflight capture,
host allowlist, strict Range handling, interrupted/ignored-Range resume, wrong
checksums, space exhaustion (plan, guard and ENOSPC), partial and foreign files,
no silent overwrite, owned-only cleanup and the authorization gate.
"""
from collections import namedtuple
import errno
import hashlib
import http.server
import json
from pathlib import Path
import struct
import threading

import pytest

from tools import acquire10 as acq

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT/'results'/'increment10'/'preflight-metadata'
Usage = namedtuple('Usage', 'total used free')
BIG = Usage(10**15, 0, 10**15)


# ---------------------------------------------------------------- synthetic repository

def safetensors(tensors):
    header, payload, offset = {'__metadata__': {'format': 'pt'}}, b'', 0
    for name, (shape, seed) in tensors.items():
        n = 1
        for d in shape:
            n *= d
        data = bytes((seed*7+i) % 251 for i in range(2*n))
        header[name] = dict(dtype='BF16', shape=shape, data_offsets=[offset, offset+len(data)])
        payload += data
        offset += len(data)
    raw = json.dumps(header).encode()
    raw += b' '*(-len(raw) % 8)
    return raw, struct.pack('<Q', len(raw))+raw+payload


def build_repo():
    h1, s1 = safetensors({'a.weight': ([64, 48], 1), 'b.weight': ([300], 2)})
    h2, s2 = safetensors({'c.weight': ([128, 40], 3)})
    shards = {'model-00001-of-00002.safetensors': (s1, h1), 'model-00002-of-00002.safetensors': (s2, h2)}
    index = json.dumps(dict(metadata=dict(total_size=sum(len(s)-8-len(h) for s, h in shards.values())),
                            weight_map={'a.weight': 'model-00001-of-00002.safetensors',
                                        'b.weight': 'model-00001-of-00002.safetensors',
                                        'c.weight': 'model-00002-of-00002.safetensors'})).encode()
    small = {'config.json': b'{"model_type": "qwen2"}', 'model.safetensors.index.json': index,
             'LICENSE': b'Apache License 2.0\n'*40}
    files = {n: s for n, (s, _) in shards.items()} | small
    manifest = acq.Manifest(
        'test/tiny', 'a'*40,
        shards={n: (len(s), hashlib.sha256(s).hexdigest(), acq.blob_id(s), len(h), hashlib.sha256(h).hexdigest())
                for n, (s, h) in shards.items()},
        small={n: (len(b), acq.blob_id(b)) for n, b in small.items()})
    return manifest, files


def api_json(manifest, files):
    siblings = []
    for name, data in files.items():
        e = dict(rfilename=name, size=len(data), blobId=acq.blob_id(data))
        if name in manifest.shards:
            e['lfs'] = dict(sha256=hashlib.sha256(data).hexdigest(), size=len(data))
        siblings.append(e)
    return dict(sha=manifest.revision, gated=False, private=False, cardData=dict(license='apache-2.0'),
                siblings=siblings)


class FakeHF:
    """Behaviour per file: ok, ignore_range, truncate:N, corrupt, bad_content_range, redirect_foreign,
    wrong_length, error500."""

    def __init__(self, manifest, files):
        self.manifest, self.files = manifest, dict(files)
        self.mode, self.requests = {}, []
        self.api = api_json(manifest, files)
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def do_GET(self):
                fake.requests.append(dict(path=self.path, range=self.headers.get('Range'),
                                          agent=self.headers.get('User-Agent'),
                                          authorization=self.headers.get('Authorization')))
                if self.path.startswith('/api/models/'):
                    return self.send(200, json.dumps(fake.api).encode())
                prefix = f'/{fake.manifest.model}/resolve/{fake.manifest.revision}/'
                if not self.path.startswith(prefix):
                    return self.send(404, b'')
                name = self.path[len(prefix):]
                data = fake.files.get(name)
                if data is None:
                    return self.send(404, b'')
                mode = fake.mode.get(name, 'ok')
                rng = self.headers.get('Range')
                header = name in fake.manifest.shards and rng in (
                    'bytes=0-7', f'bytes=8-{7+fake.manifest.shards[name][3]}')
                if header:      # payload modes never alter the metadata (header) requests
                    mode = mode.removeprefix('header_') if mode.startswith('header_') else 'ok'
                elif mode.startswith('header_'):
                    mode = 'ok'
                if mode == 'redirect_foreign':
                    self.send_response(302)
                    self.send_header('Location', 'https://example.com/x')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                if mode == 'error500':
                    return self.send(500, b'')
                if mode == 'corrupt':
                    data = bytes([data[0] ^ 1]) + data[1:-1] + bytes([data[-1] ^ 0x80])
                if rng and mode != 'ignore_range':
                    first, last = rng.removeprefix('bytes=').split('-')
                    first = int(first)
                    last = int(last) if last else len(data)-1
                    body = data[first:last+1]
                    cr = f'bytes {first}-{last}/{len(data)}'
                    if mode == 'bad_content_range':
                        cr = f'bytes 0-{len(data)-1}/{len(data)}'
                    return self.send(206, body, content_range=cr)
                if mode.startswith('truncate:'):
                    cut = int(mode.split(':')[1])
                    self.send_response(200)
                    self.send_header('Content-Length', str(len(data)))
                    self.end_headers()
                    self.wfile.write(data[:cut])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                if mode == 'wrong_length':
                    return self.send(200, data, length=len(data)+5)
                return self.send(200, data)

            def send(self, status, body, content_range=None, length=None):
                self.send_response(status)
                self.send_header('Content-Length', str(len(body) if length is None else length))
                if content_range:
                    self.send_header('Content-Range', content_range)
                self.end_headers()
                if length is None or length == len(body):
                    self.wfile.write(body)
                else:
                    self.wfile.write(body)
                    self.close_connection = True

        class Server(http.server.ThreadingHTTPServer):
            def handle_error(self, request, client_address):
                pass      # clients that refuse a body close the socket early by design

        self.server = Server(('127.0.0.1', 0), Handler)
        self.endpoint = f'http://127.0.0.1:{self.server.server_address[1]}'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def payload_gets(self):
        """Requests that could carry payload: no Range, or a Range beyond the header."""
        out = []
        for r in self.requests:
            if r['path'].startswith('/api/'):
                continue
            name = r['path'].rsplit('/', 1)[1]
            if name in self.manifest.shards:
                hl = self.manifest.shards[name][3]
                if r['range'] in ('bytes=0-7', f'bytes=8-{7+hl}'):
                    continue
            if name in self.manifest.small and name in ('config.json', 'model.safetensors.index.json') \
                    and r['range'] is None:
                continue
            out.append(r)
        return out

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def repo():
    manifest, files = build_repo()
    fake = FakeHF(manifest, files)
    yield manifest, files, fake
    fake.close()


def small_reserves(**kw):
    values = dict(fixed_margin=0, temporary=0, verification=0, rollback=0, evidence=0, swap_growth=0,
                  stop_threshold=0, check_interval=4096)
    values.update(kw)
    return acq.Reserves(**values)


def authorize(path, manifest, dest, max_bytes=None, **override):
    data = dict(schema=acq.AUTH_SCHEMA, model=manifest.model, revision=manifest.revision,
                manifest_sha256=manifest.digest(), destination=str(dest),
                max_transfer_bytes=manifest.total_bytes() if max_bytes is None else max_bytes,
                approved_by='test-owner', approved_utc='2026-11-18T00:00:00Z') | override
    path.write_text(json.dumps(data))
    return path


def run(fake, manifest, dest, auth, **kw):
    kw.setdefault('reserves', small_reserves())
    kw.setdefault('usage', lambda p: BIG)
    return acq.execute(dest, dest.parent, auth, manifest=manifest, endpoint=fake.endpoint,
                       transport=acq.Transport(allow_loopback_http=True), **kw)


# ---------------------------------------------------------------- pinned 14B manifest

def test_pinned_manifest_totals_match_preflight():
    m = acq.QWEN25_14B
    assert m.revision == '97e1e76335b7017d8f67c08a19d103c0504298c9'
    assert len(m.shards) == 8
    assert sum(v[0] for v in m.shards.values()) == 29_540_133_920
    assert m.total_bytes() == 29_551_684_877
    assert sum(v[3]+8 for v in m.shards.values()) == 66_592
    assert 'inwime' not in acq.AGENT.lower()


def test_pinned_manifest_digest_is_registered_in_protocol():
    protocol = json.loads((ROOT/'results'/'increment10'/'protocol.json').read_text())
    assert protocol['checkpoint']['acquisition_manifest_sha256'] == acq.QWEN25_14B.digest()


def test_offline_captured_metadata_matches_pins():
    api, small, headers = acq.offline_metadata(METADATA)
    facts = acq.verify_metadata(api, small, headers, acq.QWEN25_14B)
    assert facts['tensor_count'] == 579
    assert facts['payload_bytes'] == 29_540_067_328
    assert facts['config']['num_key_value_heads'] == 8 and facts['config']['rms_norm_eps'] == 1e-5


@pytest.mark.parametrize('mutation', ['size', 'lfs', 'revision', 'license', 'gated', 'header', 'index'])
def test_offline_metadata_mutations_rejected(mutation):
    api, small, headers = acq.offline_metadata(METADATA)
    api = json.loads(json.dumps(api))
    shard = 'model-00003-of-00008.safetensors'
    entry = next(e for e in api['siblings'] if e['rfilename'] == shard)
    if mutation == 'size': entry['size'] += 1
    if mutation == 'lfs': entry['lfs']['sha256'] = '0'*64
    if mutation == 'revision': api['sha'] = 'b'*40
    if mutation == 'license': api['cardData']['license'] = 'other'
    if mutation == 'gated': api['gated'] = 'auto'
    if mutation == 'header':
        raw = bytearray(headers[shard]); raw[-1] ^= 1; headers[shard] = bytes(raw)
    if mutation == 'index':
        small['model.safetensors.index.json'] = small['model.safetensors.index.json'].replace(b'00003', b'00004', 1)
    with pytest.raises(acq.AcquireError):
        acq.verify_metadata(api, small, headers, acq.QWEN25_14B)


@pytest.mark.parametrize('url,loopback,ok', [
    ('https://huggingface.co/x', False, True), ('https://us.aws.cdn.hf.co/x', False, True),
    ('https://cdn-lfs.huggingface.co/x', False, True), ('http://huggingface.co/x', False, False),
    ('https://huggingface.co.evil.com/x', False, False), ('https://evilhf.co/x', False, False),
    ('https://user:pw@huggingface.co/x', False, False), ('http://127.0.0.1:9/x', False, False),
    ('http://127.0.0.1:9/x', True, True), ('http://10.0.0.1/x', True, False)])
def test_host_allowlist(url, loopback, ok):
    assert acq.host_allowed(url, loopback) is ok


def test_reserves_validation_and_defaults():
    r = acq.Reserves()
    assert r.held() == 32*2**30 + 2**30 + 7_344_226_304
    assert acq.QWEN25_14B.total_bytes() + r.held() == 72_329_391_373
    with pytest.raises(ValueError):
        acq.Reserves(fixed_margin=-1)
    with pytest.raises(ValueError):
        acq.Reserves(check_interval=0)


def test_space_plan_blocks():
    r = small_reserves(fixed_margin=100, stop_threshold=50)
    assert acq.space_plan(1000, r, 2000, 2000)['blockers'] == []
    assert acq.space_plan(1000, r, 2000, 1050)['blockers']      # backing volume is the binding one
    assert acq.space_plan(1000, small_reserves(stop_threshold=1500), 2000, 2000)['blockers']


# ---------------------------------------------------------------- dry-run

def test_dry_run_online_reads_no_payload(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'
    report = acq.dry_run(dest, tmp_path, manifest=manifest, endpoint=fake.endpoint,
                         transport=acq.Transport(allow_loopback_http=True), reserves=small_reserves(),
                         usage=lambda p: BIG)
    assert report['payload_bytes_transferred'] == 0 and report['would_proceed_if_authorized'] is True
    assert not dest.exists()
    assert fake.payload_gets() == []
    assert all(r['authorization'] is None for r in fake.requests)
    assert all('inwime' not in (r['agent'] or '').lower() for r in fake.requests)
    assert report['space']['transfer_bytes'] == manifest.total_bytes()


def test_dry_run_header_range_ignored_refused(repo, tmp_path):
    manifest, files, fake = repo
    fake.mode['model-00002-of-00002.safetensors'] = 'header_ignore_range'
    with pytest.raises(acq.AcquireError, match='Range not honoured'):
        acq.dry_run(tmp_path/'d', tmp_path, manifest=manifest, endpoint=fake.endpoint,
                    transport=acq.Transport(allow_loopback_http=True), usage=lambda p: BIG)


def test_dry_run_header_wrong_content_range_refused(repo, tmp_path):
    manifest, files, fake = repo
    fake.mode['model-00001-of-00002.safetensors'] = 'header_bad_content_range'
    with pytest.raises(acq.AcquireError, match='Range not honoured'):
        acq.dry_run(tmp_path/'d', tmp_path, manifest=manifest, endpoint=fake.endpoint,
                    transport=acq.Transport(allow_loopback_http=True), usage=lambda p: BIG)


def test_redirect_to_foreign_host_refused(repo, tmp_path):
    manifest, files, fake = repo
    fake.mode['config.json'] = 'redirect_foreign'
    with pytest.raises(acq.AcquireError, match='refused URL'):
        acq.dry_run(tmp_path/'d', tmp_path, manifest=manifest, endpoint=fake.endpoint,
                    transport=acq.Transport(allow_loopback_http=True), usage=lambda p: BIG)


def test_dry_run_reports_space_shortage(repo, tmp_path):
    manifest, files, fake = repo
    report = acq.dry_run(tmp_path/'d', tmp_path, manifest=manifest, endpoint=fake.endpoint,
                         transport=acq.Transport(allow_loopback_http=True), reserves=small_reserves(),
                         usage=lambda p: Usage(10, 10, 100))
    assert report['would_proceed_if_authorized'] is False and report['space']['blockers']


def test_cli_defaults_to_dry_run_and_execute_needs_authorization(tmp_path, capsys):
    dest = tmp_path/'ckpt'
    assert acq.main([str(dest), '--backing-volume', str(tmp_path), '--metadata-dir', str(METADATA)]) == 0
    assert not dest.exists()
    out = json.loads(capsys.readouterr().out)
    assert out['mode'] == 'dry-run' and out['payload_bytes_transferred'] == 0
    with pytest.raises(SystemExit):
        acq.main([str(dest), '--backing-volume', str(tmp_path), '--execute'])
    with pytest.raises(SystemExit):
        acq.main([str(dest), '--backing-volume', str(tmp_path), '--authorization', 'x.json'])
    assert not dest.exists()


def test_cli_evidence_never_overwritten(tmp_path):
    evidence = tmp_path/'e.json'
    evidence.write_text('keep')
    with pytest.raises(FileExistsError):
        acq.main([str(tmp_path/'c'), '--backing-volume', str(tmp_path), '--metadata-dir', str(METADATA),
                  '--evidence', str(evidence)])
    assert evidence.read_text() == 'keep'


# ---------------------------------------------------------------- authorization gate

@pytest.mark.parametrize('field,value', [('manifest_sha256', '0'*64), ('revision', 'b'*40),
                                         ('schema', 'other'), ('max_transfer_bytes', 10),
                                         ('approved_by', ''), ('destination', '/elsewhere')])
def test_authorization_mismatch_refused_before_network(repo, tmp_path, field, value):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'
    auth = authorize(tmp_path/'auth.json', manifest, dest, **{field: value})
    with pytest.raises(acq.AcquireError):
        run(fake, manifest, dest, auth)
    assert fake.requests == [] and not dest.exists()


def test_unreadable_authorization_refused(repo, tmp_path):
    manifest, files, fake = repo
    (tmp_path/'auth.json').write_text('{not json')
    with pytest.raises(acq.AcquireError):
        run(fake, manifest, tmp_path/'ckpt', tmp_path/'auth.json')
    assert fake.requests == []


# ---------------------------------------------------------------- execute

def test_execute_happy_path_and_idempotent_rerun(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    report = run(fake, manifest, dest, auth)
    assert report['payload_bytes_transferred'] == manifest.total_bytes()
    for name, data in files.items():
        assert (dest/name).read_bytes() == data
        assert not (dest/(name+'.part')).exists() and not (dest/(name+'.part.acquire10.json')).exists()
    for name in manifest.shards:
        v = report['committed'][name]
        assert v['computed_sha256'] == v['official_lfs_sha256'] == hashlib.sha256(files[name]).hexdigest()
    provenance = json.loads((dest/'provenance.json').read_text())
    assert provenance['manifest_sha256'] == manifest.digest()
    assert not (dest/'.acquire10.lock').exists()
    again = run(fake, manifest, dest, auth)
    assert again['payload_bytes_transferred'] == 0
    assert {v['method'] for v in again['verification'].values()} == {'existing file re-hashed'}


def test_interrupted_transfer_then_resume(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    shard = 'model-00001-of-00002.safetensors'
    fake.mode[shard] = 'truncate:3000'
    with pytest.raises(acq.TransferStopped):
        run(fake, manifest, dest, auth)
    part = dest/(shard+'.part')
    assert part.exists() and 0 < part.stat().st_size < len(files[shard])
    marker = json.loads((dest/(shard+'.part.acquire10.json')).read_text())
    assert marker['state'] == 'partial' and marker['bytes'] == part.stat().st_size
    assert not (dest/shard).exists() and not (dest/'.acquire10.lock').exists()
    have = part.stat().st_size
    fake.mode.clear()
    report = run(fake, manifest, dest, auth)
    assert (dest/shard).read_bytes() == files[shard]
    v = report['verification'][shard]
    assert v['method'] == 'resumed .part re-hashed then streamed' and v['resumed_from_bytes'] == have
    assert any(r['range'] == f'bytes={have}-' for r in fake.requests)


def test_resume_with_ignored_range_keeps_part(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    shard = 'model-00002-of-00002.safetensors'
    fake.mode[shard] = 'truncate:1000'
    with pytest.raises(acq.TransferStopped):
        run(fake, manifest, dest, auth)
    part = dest/(shard+'.part')
    before = part.read_bytes()
    fake.mode[shard] = 'ignore_range'
    with pytest.raises(acq.TransferStopped, match='Range'):
        run(fake, manifest, dest, auth)
    assert part.read_bytes() == before and not (dest/shard).exists()


def test_wrong_checksum_rejected_then_discarded(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    shard = 'model-00001-of-00002.safetensors'
    fake.mode[shard] = 'corrupt'
    with pytest.raises(acq.AcquireError, match='mismatch'):
        run(fake, manifest, dest, auth)
    assert not (dest/shard).exists()
    marker = json.loads((dest/(shard+'.part.acquire10.json')).read_text())
    assert marker['state'] == 'rejected' and marker['computed_sha256'] != manifest.shards[shard][1]
    fake.mode.clear()
    with pytest.raises(acq.AcquireError, match='rejected'):
        run(fake, manifest, dest, auth)
    report = run(fake, manifest, dest, auth, discard_rejected=True)
    assert report['discarded_rejected_parts'] == [shard]
    assert (dest/shard).read_bytes() == files[shard]


def test_small_file_blob_mismatch_rejected(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'
    fake.mode['LICENSE'] = 'corrupt'
    with pytest.raises(acq.AcquireError, match='mismatch'):
        run(fake, manifest, dest, authorize(tmp_path/'auth.json', manifest, dest))
    assert not (dest/'LICENSE').exists()


def test_wrong_content_length_not_read(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'
    fake.mode['model-00002-of-00002.safetensors'] = 'wrong_length'
    with pytest.raises(acq.TransferStopped, match='Content-Length'):
        run(fake, manifest, dest, authorize(tmp_path/'auth.json', manifest, dest))
    assert (dest/'model-00002-of-00002.safetensors.part').stat().st_size == 0


def test_space_shortage_before_transfer(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    with pytest.raises(acq.SpaceError):
        run(fake, manifest, dest, auth, reserves=small_reserves(fixed_margin=10**9),
            usage=lambda p: Usage(0, 0, 10**6))
    assert fake.payload_gets() == []


def test_space_guard_stops_mid_transfer(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    calls = []

    def usage(path):
        calls.append(path)
        return BIG if len(calls) <= 2 else Usage(0, 0, 10)   # plan passes, first guard check fails
    with pytest.raises(acq.SpaceError, match='stop threshold'):
        run(fake, manifest, dest, auth, reserves=small_reserves(stop_threshold=1000, check_interval=1024),
            usage=usage)
    parts = list(dest.glob('*.part'))
    assert parts and all(json.loads(Path(str(p)+'.acquire10.json').read_text())['state'] == 'partial'
                         for p in parts)


def test_enospc_retains_part(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'
    auth = authorize(tmp_path/'auth.json', manifest, dest)

    class Full:
        def __init__(self, f):
            self.f = f

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.f.close()

        def write(self, data):
            if self.f.tell()+len(data) > 2000:
                raise OSError(errno.ENOSPC, 'No space left on device')
            return self.f.write(data)

        def flush(self):
            self.f.flush()

        def fileno(self):
            return self.f.fileno()

    def opener(path, mode):
        f = open(path, mode)
        return Full(f) if str(path).endswith('00001-of-00002.safetensors.part') else f
    with pytest.raises(acq.SpaceError, match='no space'):
        run(fake, manifest, dest, auth, opener=opener)
    assert (dest/'model-00001-of-00002.safetensors.part').exists()
    assert not (dest/'model-00001-of-00002.safetensors').exists()


def test_transfer_budget_enforced(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'
    with pytest.raises(acq.AcquireError, match='exceeds the authorized'):
        run(fake, manifest, dest, authorize(tmp_path/'auth.json', manifest, dest,
                                            max_bytes=manifest.total_bytes()-1))


# ---------------------------------------------------------------- partial / foreign / overwrite

def test_foreign_part_blocks_and_is_untouched(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'; dest.mkdir()
    foreign = dest/'model-00001-of-00002.safetensors.part'
    foreign.write_bytes(b'not mine')
    with pytest.raises(acq.AcquireError, match='not owned|blocks'):
        run(fake, manifest, dest, authorize(tmp_path/'auth.json', manifest, dest))
    assert foreign.read_bytes() == b'not mine'
    assert acq.discard_owned(dest, manifest, ('partial', 'rejected')) == []
    assert foreign.read_bytes() == b'not mine'


def test_part_longer_than_official_blocks(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'; dest.mkdir()
    shard = 'model-00002-of-00002.safetensors'
    (dest/(shard+'.part')).write_bytes(files[shard]+b'xx')
    acq.write_marker(dest/shard, manifest, shard, 'partial', bytes=len(files[shard])+2)
    state = acq.inspect(dest, manifest)
    assert state['files'][shard]['state'] == 'part-too-long' and state['blockers']


def test_existing_file_never_overwritten(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'; dest.mkdir()
    shard = 'model-00001-of-00002.safetensors'
    wrong = bytes(len(files[shard]))
    (dest/shard).write_bytes(wrong)
    with pytest.raises(acq.AcquireError, match='never overwritten'):
        run(fake, manifest, dest, authorize(tmp_path/'auth.json', manifest, dest))
    assert (dest/shard).read_bytes() == wrong
    (dest/shard).write_bytes(b'short')
    state = acq.inspect(dest, manifest)
    assert state['files'][shard]['state'] == 'present-wrong-size' and state['blockers']


def test_final_and_part_conflict_blocks(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'; dest.mkdir()
    (dest/'LICENSE').write_bytes(files['LICENSE'])
    (dest/'LICENSE.part').write_bytes(b'')
    assert acq.inspect(dest, manifest)['files']['LICENSE']['state'] == 'conflict-final-and-part'


def test_clean_temporaries_only_owned(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    fake.mode['model-00001-of-00002.safetensors'] = 'truncate:500'
    with pytest.raises(acq.TransferStopped):
        run(fake, manifest, dest, auth)
    (dest/'unrelated.bin').write_bytes(b'keep')
    (dest/'other.part').write_bytes(b'keep')
    removed = acq.discard_owned(dest, manifest, ('partial', 'rejected'))
    assert removed == ['model-00001-of-00002.safetensors']
    assert (dest/'unrelated.bin').read_bytes() == b'keep' and (dest/'other.part').read_bytes() == b'keep'
    assert (dest/'LICENSE').exists()    # committed files are never cleaned


def test_lock_prevents_concurrent_run(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'; dest.mkdir()
    (dest/'.acquire10.lock').write_text('123')
    with pytest.raises(acq.AcquireError, match='lock'):
        run(fake, manifest, dest, authorize(tmp_path/'auth.json', manifest, dest))
    assert (dest/'.acquire10.lock').read_text() == '123'


def test_server_error_status_stops(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'
    fake.mode['model-00002-of-00002.safetensors'] = 'error500'
    # header metadata step sees a non-206 answer first: refused without reading the body
    with pytest.raises(acq.AcquireError):
        run(fake, manifest, dest, authorize(tmp_path/'auth.json', manifest, dest))


def test_resume_with_wrong_content_range_keeps_part(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'
    auth = authorize(tmp_path/'auth.json', manifest, dest)
    shard = 'model-00002-of-00002.safetensors'
    fake.mode[shard] = 'truncate:1500'
    with pytest.raises(acq.TransferStopped):
        run(fake, manifest, dest, auth)
    part = dest/(shard+'.part')
    before = part.read_bytes()
    fake.mode[shard] = 'bad_content_range'
    with pytest.raises(acq.TransferStopped, match='Range'):
        run(fake, manifest, dest, auth)
    assert part.read_bytes() == before and not (dest/shard).exists()


def test_complete_part_verified_without_request(repo, tmp_path):
    manifest, files, fake = repo
    dest = tmp_path/'ckpt'; dest.mkdir()
    shard = 'model-00001-of-00002.safetensors'
    (dest/(shard+'.part')).write_bytes(files[shard])
    acq.write_marker(dest/shard, manifest, shard, 'partial', bytes=len(files[shard]))
    report = run(fake, manifest, dest, authorize(tmp_path/'auth.json', manifest, dest))
    assert report['verification'][shard]['method'] == 'complete .part re-hashed'
    assert not any(r['path'].endswith(shard) and r['range'] not in ('bytes=0-7', f'bytes=8-{7+manifest.shards[shard][3]}')
                   for r in fake.requests)

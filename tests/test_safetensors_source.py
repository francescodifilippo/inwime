"""Increment 6: original bytes, independent addressing, bounded I/O and failures."""
from array import array
from concurrent.futures import CancelledError
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import threading

import pytest

from inwime_runtime import CPURuntime, Budget, BudgetError, CheckpointError, ReadError
from inwime_runtime.core import Coordinator, FileBackend, open_catalog
from inwime_runtime.gemv import StreamedGEMV
from inwime_runtime.safetensors_source import SafetensorsSource
from inwime_reference.checkpoint import load_checkpoint
from inwime_reference.decoder import ReferenceDecoder
from tests.test_gemv_cpu import native
from tests.test_runtime_cpu import export_reference
from tools.checkpoint4 import generate
from tools.fixture_safetensors6 import write_fixture
from tools.verify_increment2 import TOKENS, compare, runtime_export

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=['dense', 'moe'])
def fixture(tmp_path, request):
    original = ROOT/'fixtures/reference_decoder'/request.param
    safe, split = tmp_path/'safe', tmp_path/'split'
    write_fixture(original, safe)
    generate(split, source=original, shard_bytes=510, segment_bytes=22, execute=True)
    return original, split, safe


def hashes(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}


@pytest.mark.parametrize('mode', ['sync', 'pipeline'])
@pytest.mark.parametrize('compiled', [False, True])
def test_decoder_original_v2_safetensors_and_independent_oracle(fixture, native, mode, compiled):
    original, split, safe = fixture
    before = [hashes(p) for p in fixture]
    oracle, state = ReferenceDecoder(load_checkpoint(original)), None
    math_backend = native if compiled else None
    with CPURuntime(original, tile_elements=73, read_elements=13, math_backend=math_backend) as a, \
            CPURuntime(split, tile_elements=73, read_elements=13, io_mode=mode,
                       math_backend=math_backend) as b, \
            CPURuntime(safe, source='safetensors', tile_elements=73, read_elements=13,
                       io_mode=mode, math_backend=math_backend) as c:
        for tokens in (TOKENS[:3], *([t] for t in TOKENS[3:])):
            expected = oracle.prefill(tokens, state)
            state = expected.state
            with a.prefill(tokens) as x, b.prefill(tokens) as y, c.prefill(tokens) as z:
                exports = [runtime_export(r) for r in (x, y, z)]
                assert exports[0] == exports[1] == exports[2]
                # Also compare raw buffers: equality of Python floats alone
                # would miss a sign-of-zero difference.
                for prop in ('logits',):
                    assert bytes(getattr(x, prop)) == bytes(getattr(y, prop)) == bytes(getattr(z, prop))
                for left, middle, right in zip(x.hidden, y.hidden, z.hidden):
                    assert bytes(left) == bytes(middle) == bytes(right)
                for left, middle, right in zip(x.kv, y.kv, z.kv):
                    for u, v, w in zip(left, middle, right):
                        assert bytes(u) == bytes(v) == bytes(w)
                compare([exports[2]], [export_reference(expected)])
        stats = c.stats()
        assert stats['payload_bytes'] == sum(v['nbytes'] for v in json.loads((original/'manifest.json').read_text())['files'].values())
        assert stats['integrity_read_bytes'] == stats['source_file_bytes']
        assert stats['source_header_bytes']+stats['payload_bytes'] == stats['source_file_bytes']
        assert stats['max_backend_read_bytes'] <= 26
        assert stats['audited_retained_metadata_bytes'] <= c.plan.required['metadata']
        assert stats['reserved_peak_bytes'] < 128*2**20
        assert stats['tensor_read_bytes'] == a.stats()['tensor_read_bytes'] == b.stats()['tensor_read_bytes']
        assert stats['requests'] == stats['completed'] == stats['released']
    assert before == [hashes(p) for p in fixture]
    assert not c.control.fds and not c.control.buffers


@pytest.mark.parametrize('mode', ['sync', 'pipeline'])
@pytest.mark.parametrize('tile,read', [(7, 3), (73, 13)])
def test_every_original_word_and_irregular_gemv(fixture, native, mode, tile, read):
    original, _, safe = fixture
    m = json.loads((original/'manifest.json').read_text())
    with CPURuntime(safe, source='safetensors', tile_elements=tile, read_elements=read,
                    io_mode=mode, math_backend=native) as engine:
        for name, d in m['tensors'].items():
            # Address using the ORIGINAL manifest, independently of the new parser.
            with (original/d['shard']).open('rb') as f:
                f.seek(d['offset'])
                raw = f.read(d['nbytes'])
            expected = b''.join(struct.pack('<d', struct.unpack('<f', struct.pack('<I', w << 16))[0])
                                for w, in struct.iter_unpack('<H', raw))
            actual = array('d')
            for start in range(0, len(expected)//8, tile):
                count = min(tile, len(expected)//8-start)
                with engine.control.acquire(name, start, count) as lease:
                    actual.extend(lease.value(i) for i in range(count))
            assert actual.tobytes() == expected
    for name in ('layers.0.q', 'layers.0.o', 'lm_head'):
        matrix = load_checkpoint(original).weights[name]
        x = array('d', [(i-7)/19 for i in range(matrix.shape[1])])
        expected = [math.fsum(float(w)*v for w, v in zip(row, x)) for row in matrix]
        with StreamedGEMV(safe, source='safetensors', tile_elements=tile, read_elements=read,
                          io_mode=mode, math_backend=native) as op:
            out = array('d', [123.])*matrix.shape[0]
            op.run('checkpoint.'+name, x, out)
            assert list(out) == pytest.approx(expected, abs=1e-14, rel=1e-14)
            assert op.stats()['inference_read_bytes'] == matrix.size*2


def save_descriptor(path, d):
    (path/'safetensors.json').write_text(json.dumps(d))


def rewrite_header(path, shard, change):
    p = path/shard
    data = p.read_bytes()
    length, = struct.unpack('<Q', data[:8])
    header = json.loads(data[8:8+length])
    change(header)
    raw = json.dumps(header, separators=(',', ':')).encode()
    # Intentionally no padding: absolute BF16 offsets may be odd.
    data = struct.pack('<Q', len(raw))+raw+data[8+length:]
    p.write_bytes(data)
    d = json.loads((path/'safetensors.json').read_text())
    d['files'][shard] = dict(nbytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    save_descriptor(path, d)


@pytest.mark.parametrize('defect', ['dtype', 'negative', 'bool', 'float', 'overflow', 'size', 'gap',
                                   'overlap', 'outside', 'metadata', 'shape-rank', 'offset-count'])
def test_bad_tensor_headers_fail_closed(fixture, defect):
    _, _, safe = fixture
    shard = next(iter(json.loads((safe/'safetensors.json').read_text())['files']))
    def damage(header):
        name = next(n for n in header if n != '__metadata__')
        d = header[name]
        if defect == 'dtype':
            d['dtype'] = 'F16'
        elif defect in ('negative', 'bool', 'float', 'overflow'):
            d['shape'][0] = {'negative': -1, 'bool': True, 'float': 1., 'overflow': 2**63}[defect]
        elif defect == 'size':
            d['shape'][0] += 1
        elif defect == 'shape-rank':
            d['shape'] = [1]*33
        elif defect == 'offset-count':
            d['data_offsets'].append(0)
        elif defect == 'metadata':
            header['__metadata__'] = {'bad': 1}
        elif defect == 'outside':
            d['data_offsets'] = [2**62, 2**62+2]
        elif defect == 'overlap':
            other = next(n for n in header if n not in ('__metadata__', name))
            header[other] = d.copy()
        else:
            d['data_offsets'] = [n+2 for n in d['data_offsets']]
    rewrite_header(safe, shard, damage)
    with pytest.raises(CheckpointError):
        SafetensorsSource(safe)


@pytest.mark.parametrize('defect', ['missing-shard', 'index-shard', 'index-tensor', 'index-hash', 'index-size',
                                   'traversal', 'absolute', 'backslash', 'missing-index',
                                   'duplicate-key', 'duplicate-tensor', 'tail', 'truncated',
                                   'header-length', 'binding', 'binding-shape', 'symlink'])
def test_bad_references_metadata_and_files(fixture, defect):
    _, _, safe = fixture
    d = json.loads((safe/'safetensors.json').read_text())
    shard = next(iter(d['files']))
    if defect == 'missing-shard':
        (safe/shard).unlink()
    elif defect in ('index-shard', 'index-tensor', 'index-hash', 'index-size'):
        p = safe/d['index']
        index = json.loads(p.read_text())
        if defect == 'index-shard':
            index['weight_map'][next(iter(index['weight_map']))] = 'unknown.safetensors'
        elif defect == 'index-tensor':
            index['weight_map']['absent'] = shard
        elif defect == 'index-size':
            index['metadata']['total_size'] += 2
        p.write_text(json.dumps(index))
        if defect != 'index-hash':
            d['index_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
    elif defect in ('traversal', 'absolute', 'backslash'):
        d['index'] = {'traversal': '../outside.json', 'absolute': '/tmp/outside.json',
                      'backslash': '..\\outside.json'}[defect]
    elif defect == 'missing-index':
        d['index'] = None
    elif defect == 'duplicate-key':
        (safe/'safetensors.json').write_bytes(b'{"files":{},"files":{}}')
    elif defect == 'duplicate-tensor':
        other = list(d['files'])[1]
        header = json.loads((safe/shard).read_bytes()[8:8+struct.unpack('<Q', (safe/shard).read_bytes()[:8])[0]])
        name = next(n for n in header if n != '__metadata__')
        rewrite_header(safe, other, lambda h: h.update({name: header[name]}))
    elif defect == 'binding':
        d['bindings']['embedding'] = d['bindings']['position']
    elif defect == 'binding-shape':
        d['bindings']['embedding'], d['bindings']['position'] = d['bindings']['position'], d['bindings']['embedding']
    elif defect == 'symlink':
        (safe/shard).rename(safe/'real')
        (safe/shard).symlink_to('real')
    else:
        p = safe/shard
        raw = p.read_bytes()
        if defect == 'tail':
            raw += b'xx'
        elif defect == 'truncated':
            raw = raw[:10]
        else:
            raw = struct.pack('<Q', 2**64-1)+raw[8:]
        p.write_bytes(raw)
        d['files'][shard]['nbytes'] = len(raw)
    if defect not in ('duplicate-key', 'duplicate-tensor'):
        save_descriptor(safe, d)
    with pytest.raises((CheckpointError, OSError)):
        CPURuntime(safe, source='safetensors')


@pytest.mark.parametrize('limit', ['max_input_bytes', 'max_metadata_bytes', 'max_metadata_heap_bytes',
                                  'max_shards', 'max_tensors'])
def test_metadata_limits_and_cleanup(fixture, limit):
    _, _, safe = fixture
    before = len(list(Path('/proc/self/fd').iterdir()))
    with pytest.raises(CheckpointError):
        SafetensorsSource(safe, **{limit: 1})
    assert len(list(Path('/proc/self/fd').iterdir())) == before


def test_budget_rejected_before_payload_and_hash_cleanup(fixture):
    _, _, safe = fixture
    plan = CPURuntime.minimum_plan(safe, source='safetensors')
    class Backend(FileBackend):
        calls = 0
        def readinto(self, *args):
            self.calls += 1
            return super().readinto(*args)
    backend = Backend()
    with pytest.raises(BudgetError):
        CPURuntime(safe, source='safetensors', budget=Budget(plan.total-1, plan.required), backend=backend)
    assert backend.calls == 0
    d = json.loads((safe/'safetensors.json').read_text())
    d['files'][next(iter(d['files']))]['sha256'] = '0'*64
    save_descriptor(safe, d)
    before = len(list(Path('/proc/self/fd').iterdir()))
    with pytest.raises(CheckpointError, match='checksum'):
        CPURuntime(safe, source='safetensors')
    assert len(list(Path('/proc/self/fd').iterdir())) == before
    with CPURuntime(safe, source='safetensors', integrity='metadata') as runtime:
        assert runtime.stats()['integrity_read_bytes'] == 0


@pytest.mark.parametrize('mode', ['sync', 'pipeline'])
@pytest.mark.parametrize('fault', ['short', 'error', 'cancel', 'mutate'])
def test_second_shard_failure_preserves_entire_published_bank(fixture, native, mode, fault):
    _, _, safe = fixture
    event = threading.Event()
    class Backend(FileBackend):
        armed = False
        first = None
        faulted = False
        reads_first = 0
        def readinto(self, fd, offset, destination):
            if self.armed and not self.faulted:
                if self.first is None:
                    self.first = fd
                if fd != self.first:
                    self.faulted = True
                    if fault == 'short':
                        return super().readinto(fd, offset, destination[:-1])
                    if fault == 'error':
                        raise OSError('injected second-shard failure')
                    if fault == 'cancel':
                        event.set()
                    else:
                        # Mutate the already-open inode via a separate writable FD.
                        path = Path(os.readlink(f'/proc/self/fd/{fd}'))
                        with path.open('r+b') as stream:
                            stream.seek(offset)
                            stream.write(b'xx')
                else:
                    self.reads_first += 1
            return super().readinto(fd, offset, destination)
    backend = Backend()
    with CPURuntime(safe, source='safetensors', io_mode=mode, math_backend=native,
                    tile_elements=73, read_elements=13, backend=backend) as runtime:
        with runtime.prefill([1, 7]):
            pass
        published = runtime._published
        before = runtime.control.buffers['kv'+str(published[0])].tobytes()
        backend.armed = True
        with pytest.raises((OSError, CancelledError, CheckpointError)):
            runtime.decode(4, cancel_event=event)
        assert backend.faulted and backend.reads_first > 0
        assert runtime._published == published
        assert runtime.control.buffers['kv'+str(published[0])].tobytes() == before
        assert runtime.failed and not runtime.control.active
        assert all(r is None for r in getattr(runtime.control, 'slots', []))
    assert not runtime.control.fds and not runtime.control.buffers
    assert not any(t.name.startswith('inwime-read-') for t in threading.enumerate())


@pytest.mark.parametrize('mode', ['sync', 'pipeline'])
def test_atomic_path_replacement_uses_verified_fd_or_fails_conservatively(fixture, mode):
    _, _, safe = fixture
    with CPURuntime(safe, source='safetensors', io_mode=mode) as runtime:
        tensor = runtime.catalog.tensors['embedding']
        old_inode = os.fstat(runtime.control.fds[tensor.shard]).st_ino
        replacement = safe/'replacement'
        replacement.write_bytes(b'x'*(safe/tensor.shard).stat().st_size)
        replacement.replace(safe/tensor.shard)
        assert os.fstat(runtime.control.fds[tensor.shard]).st_ino == old_inode
        try:
            with runtime.control.acquire('embedding', 0, 1) as lease:
                assert math.isfinite(lease.value(0))
        except CheckpointError as error:
            assert 'changed' in str(error)
        finally:
            runtime.control.abort()


def test_metadata_admission_race_and_explicit_selection(fixture):
    _, _, safe = fixture
    catalog = open_catalog(safe, source='safetensors')
    plan = CPURuntime.minimum_plan(safe, source='safetensors')
    name = next(iter(catalog.files))
    with (safe/name).open('r+b') as f:
        f.write(b'xxxxxxxx')
    with pytest.raises(CheckpointError, match='admission'):
        Coordinator(catalog, plan, integrity='metadata')
    with pytest.raises(ValueError, match='source must'):
        open_catalog(safe, source='auto')


def test_replacement_with_fifo_is_rejected_without_blocking(fixture):
    _, _, safe = fixture
    catalog = open_catalog(safe, source='safetensors')
    plan = CPURuntime.minimum_plan(safe, source='safetensors')
    shard = safe/next(iter(catalog.files))
    shard.unlink()
    os.mkfifo(shard)
    before = len(list(Path('/proc/self/fd').iterdir()))
    with pytest.raises(CheckpointError, match='regular file'):
        Coordinator(catalog, plan)
    assert len(list(Path('/proc/self/fd').iterdir())) == before


def test_single_shard_generic_scalar_empty_and_unaligned_absolute_offsets(tmp_path):
    raw = json.dumps({'zero': dict(dtype='BF16', shape=[0], data_offsets=[0, 0]),
                      'scalar': dict(dtype='BF16', shape=[], data_offsets=[0, 2]),
                      'matrix': dict(dtype='BF16', shape=[1, 1], data_offsets=[2, 4])},
                     separators=(',', ':')).encode()
    raw += b' '*(1-len(raw) % 2)  # odd header length, valid format
    payload = struct.pack('<Q', len(raw))+raw+b'\x00\x80\x80\x3f'
    (tmp_path/'model.safetensors').write_bytes(payload)
    save_descriptor(tmp_path, dict(format='inwime-safetensors-bf16-v1', index=None,
                    files={'model.safetensors': dict(nbytes=len(payload),
                           sha256=hashlib.sha256(payload).hexdigest())}))
    catalog = SafetensorsSource(tmp_path)
    assert catalog.tensors['scalar'].offset % 2 == 1
    assert catalog.tensors['scalar'].shape == () and catalog.tensors['zero'].elements == 0
    with StreamedGEMV(tmp_path, source='safetensors', tile_elements=1, read_elements=1) as op:
        result = array('d', [0.])
        op.run('matrix', array('d', [3.]), result)
        assert result[0] == 3.
        with op.control.acquire('scalar', 0, 1) as lease:
            assert struct.pack('<d', lease.value(0)) == struct.pack('<d', -0.)


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":'+b'['*17+b'0'+b']'*17+b'}',
                              b'{"x":['+b'0,'*65537+b'0]}', b'\xff', b'[]'])
def test_bounded_json_defects(tmp_path, raw):
    (tmp_path/'safetensors.json').write_bytes(raw)
    with pytest.raises(CheckpointError):
        SafetensorsSource(tmp_path)


@pytest.mark.parametrize('mode', ['sync', 'pipeline'])
@pytest.mark.parametrize('compiled', [False, True])
def test_original_nonfinite_payload_is_not_silently_converted(fixture, native, mode, compiled):
    _, _, safe = fixture
    from inwime_runtime.decoder import DecoderCatalog
    catalog = DecoderCatalog(SafetensorsSource(safe))
    tensor = catalog.tensors['embedding']
    shard = safe/tensor.shard
    with shard.open('r+b') as stream:
        stream.seek(tensor.offset)
        stream.write(b'\x80\x7f')  # +Inf BF16, valid container, rejected math contract
    d = json.loads((safe/'safetensors.json').read_text())
    d['files'][tensor.shard]['sha256'] = hashlib.sha256(shard.read_bytes()).hexdigest()
    save_descriptor(safe, d)
    with CPURuntime(safe, source='safetensors', io_mode=mode,
                    math_backend=native if compiled else None) as engine:
        previous = engine.state_digest()
        with pytest.raises(CheckpointError, match='nonfinite'):
            engine.prefill([0])
        assert engine.failed and not engine.control.active
        assert engine.state_digest() == previous


def test_payload_corruption_is_detected_by_external_hash(fixture):
    _, _, safe = fixture
    catalog = SafetensorsSource(safe)
    tensor = next(iter(catalog.tensors.values()))
    with (safe/tensor.shard).open('r+b') as stream:
        stream.seek(tensor.offset)
        stream.write(b'\x00\x00')
    with pytest.raises(CheckpointError, match='checksum'):
        CPURuntime(safe, source='safetensors')


def test_parser_reads_only_metadata_and_handles_growth(fixture, monkeypatch):
    _, _, safe = fixture
    original = os.pread
    reads = []
    def checked(fd, count, offset):
        filename = Path(os.readlink(f'/proc/self/fd/{fd}'))
        if filename.suffix == '.safetensors':
            # Read header size independently, outside the intercepted path.
            with filename.open('rb') as f:
                header_size, = struct.unpack('<Q', f.read(8))
            assert offset+count <= 8+header_size
        reads.append(count)
        return original(fd, count, offset)
    monkeypatch.setattr(os, 'pread', checked)
    source = SafetensorsSource(safe)
    assert sum(reads) == source.metadata_read_bytes
    def changed(fd, count, offset):
        data = checked(fd, count, offset)
        filename = Path(os.readlink(f'/proc/self/fd/{fd}'))
        if filename.name == 'safetensors.json':
            with filename.open('ab') as f:
                f.write(b' ')
        return data
    monkeypatch.setattr(os, 'pread', changed)
    with pytest.raises(CheckpointError, match='changed'):
        SafetensorsSource(safe)


@pytest.mark.parametrize('encoding', ['utf-16-le', 'utf-32-le'])
@pytest.mark.parametrize('location', ['descriptor', 'index', 'header'])
def test_only_utf8_metadata_is_accepted(fixture, encoding, location):
    _, _, safe = fixture
    d = json.loads((safe/'safetensors.json').read_text())
    if location == 'descriptor':
        (safe/'safetensors.json').write_bytes(json.dumps(d).encode(encoding))
    elif location == 'index':
        path = safe/d['index']
        path.write_bytes(path.read_text().encode(encoding))
        d['index_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        save_descriptor(safe, d)
    else:
        shard = next(iter(d['files']))
        data = (safe/shard).read_bytes()
        size, = struct.unpack('<Q', data[:8])
        header = json.loads(data[8:8+size])
        raw = json.dumps(header).encode(encoding)
        data = struct.pack('<Q', len(raw))+raw+data[8+size:]
        (safe/shard).write_bytes(data)
        d['files'][shard] = dict(nbytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        save_descriptor(safe, d)
    with pytest.raises(CheckpointError):
        SafetensorsSource(safe)


def test_physical_source_does_not_interpret_model_binding(fixture):
    from inwime_runtime.decoder import DecoderCatalog
    _, _, safe = fixture
    d = json.loads((safe/'safetensors.json').read_text())
    d.update(config='opaque-to-source', variant='future-model-family', bindings=[])
    save_descriptor(safe, d)
    physical = SafetensorsSource(safe)
    assert all(n.startswith('checkpoint.') for n in physical.tensors)
    assert not hasattr(physical, 'spec') and not hasattr(physical, 'variant')
    with pytest.raises(CheckpointError, match='bindings must be an object'):
        DecoderCatalog(physical)
    # A semantic view must never rewrite the physical source's names.
    write = json.loads((fixture[0]/'manifest.json').read_text())
    d.update(config=write['config'], variant=write['variant'],
             bindings={n: 'checkpoint.'+n for n in write['tensors']})
    save_descriptor(safe, d)
    physical = SafetensorsSource(safe)
    original_names = set(physical.tensors)
    view = DecoderCatalog(physical)
    assert set(view.tensors) == set(write['tensors'])
    assert set(physical.tensors) == original_names
    assert view.identity == physical.identity
    assert view.retained_bytes > physical.retained_bytes


@pytest.mark.parametrize('mode', ['sync', 'pipeline'])
@pytest.mark.parametrize('when', ['after-index', 'after-open'])
def test_truncation_cannot_return_a_validated_region(fixture, mode, when):
    from inwime_runtime.decoder import open_catalog as open_decoder_catalog
    from inwime_runtime.pipeline import PipelineCoordinator
    _, _, safe = fixture
    catalog = open_decoder_catalog(safe, source='safetensors')
    plan = CPURuntime.minimum_plan(safe, source='safetensors', io_mode=mode)
    tensor = catalog.tensors['embedding']
    coordinator = Coordinator if mode == 'sync' else PipelineCoordinator
    def truncate():
        with (safe/tensor.shard).open('r+b') as stream:
            stream.truncate(tensor.offset)
    if when == 'after-index':
        truncate()
        with pytest.raises(CheckpointError, match='expected size'):
            coordinator(catalog, plan)
    else:
        control = coordinator(catalog, plan)
        try:
            truncate()
            with pytest.raises(CheckpointError, match='changed'):
                control.acquire('embedding', 0, 1)
            control.abort()
            assert not control.active
        finally:
            control.close()

"""Physical segmentation through existing runtime and independent oracle seams."""
import json
import shutil
import threading
from concurrent.futures import CancelledError
from pathlib import Path

import pytest

from inwime_runtime import CPURuntime, CheckpointError, ReadError
from inwime_runtime.core import Catalog, FileBackend
from tools.checkpoint4 import generate
from tools.verify_increment2 import TOKENS, compare, runtime_export
from tests.test_runtime_cpu import export_reference
from inwime_reference.checkpoint import load_checkpoint
from inwime_reference.decoder import ReferenceDecoder

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=['dense', 'moe'])
def segmented(tmp_path, request):
    source = ROOT/'fixtures/reference_decoder'/request.param
    target = tmp_path/'segments'
    generate(target, source=source, shard_bytes=510, segment_bytes=14, execute=True)
    return source, target


@pytest.mark.parametrize('mode', ['sync', 'pipeline'])
@pytest.mark.parametrize('tile,read', [(3, 1), (19, 11)])
def test_full_decoder_matches_original_and_oracle(segmented, mode, tile, read):
    source, target = segmented
    oracle = ReferenceDecoder(load_checkpoint(source))
    state = oracle.prefill(TOKENS[:3])
    with CPURuntime(source, tile_elements=tile, read_elements=read) as original, CPURuntime(
            target, tile_elements=tile, read_elements=read, io_mode=mode) as split:
        with original.prefill(TOKENS[:3]) as a, split.prefill(TOKENS[:3]) as b:
            x, y = runtime_export(a), runtime_export(b)
            assert x == y
            compare([y], [export_reference(state)])
        for token in TOKENS[3:]:
            state = oracle.decode(token, state.state)
            with original.decode(token) as a, split.decode(token) as b:
                x, y = runtime_export(a), runtime_export(b)
                assert x == y
                compare([y], [export_reference(state)])


@pytest.mark.parametrize('defect', ['missing', 'overlap', 'incomplete', 'physical', 'unknown',
                                    'unaligned_offset', 'unaligned_length'])
def test_invalid_segments_rejected_before_payload(segmented, defect):
    _, path = segmented
    m = json.loads((path/'manifest.json').read_text())
    parts = m['tensors']['embedding']['segments']
    if defect == 'missing':
        del parts[1]
    elif defect == 'overlap':
        parts[1]['start'] -= 1
    elif defect == 'incomplete':
        parts.pop()
    elif defect == 'physical':
        parts[1]['offset'] = parts[0]['offset']
    elif defect == 'unaligned_offset':
        parts[1]['offset'] += 1
    elif defect == 'unaligned_length':
        parts[1]['nbytes'] -= 1
    else:
        parts[1]['shard'] = 'absent.bin'
    (path/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(CheckpointError):
        Catalog(path)


@pytest.mark.parametrize('mode', ['sync', 'pipeline'])
@pytest.mark.parametrize('fault', ['short', 'timeout', 'cancel'])
def test_cross_segment_failure_never_publishes_kv(segmented, mode, fault):
    _, path = segmented
    event = threading.Event()
    class Backend(FileBackend):
        armed = False
        calls = 0
        def readinto(self, fd, offset, destination):
            if self.armed:
                self.calls += 1
                if self.calls == 3:
                    if fault == 'short':
                        return super().readinto(fd, offset, destination[:-1])
                    if fault == 'timeout':
                        raise TimeoutError('injected backend timeout')
                    event.set()
            return super().readinto(fd, offset, destination)
    backend = Backend()
    with CPURuntime(path, tile_elements=19, read_elements=11, io_mode=mode,
                    backend=backend) as engine:
        with engine.prefill([1]) as output:
            old = [[list(v) for v in layer] for layer in output.kv]
        published = engine._published
        backend.armed = True
        with pytest.raises((ReadError, TimeoutError, CancelledError)):
            engine.decode(7, cancel_event=event)
        assert engine._published == published
        c = engine.spec
        bank = engine.control.buffers['kv'+str(published[0])]
        for layer in range(c.layers):
            for kind in range(2):
                start = (layer*2+kind)*engine.capacity*c.width
                assert list(bank[start:start+c.width]) == old[layer][kind]
        assert not engine.control.active


def test_payload_corruption_default_and_explicit_policy(segmented):
    _, path = segmented
    first = next(iter(Catalog(path).files))
    with (path/first).open('r+b') as f:
        f.write(b'\x00\x00')
    with pytest.raises(CheckpointError, match='checksum'):
        CPURuntime(path)
    with CPURuntime(path, integrity='metadata') as engine:
        assert engine.stats()['integrity_read_bytes'] == 0
        assert engine.stats()['integrity_policy'] == 'metadata'


def test_generation_preflight_prevents_descriptor_explosion(tmp_path):
    path = tmp_path/'too-many'
    with pytest.raises(ValueError, match='descriptor'):
        generate(path, vocab=100000, segment_bytes=2, execute=True)
    assert not path.exists()


def test_generator_determinism_and_selective_regions(tmp_path):
    from tools.checkpoint4 import pattern
    a, b = tmp_path/'a', tmp_path/'b'
    for target in (a, b):
        generate(target, variant='moe', experts=5, hidden=128,
                 shard_bytes=510, segment_bytes=14, execute=True)
    assert (a/'manifest.json').read_bytes() == (b/'manifest.json').read_bytes()
    for mode in ('sync', 'pipeline'):
        with CPURuntime(a, io_mode=mode, tile_elements=19, read_elements=11) as engine:
            for start, count in ((0, 19), (6, 19), (250, 19), (495, 1)):
                with engine.control.acquire('embedding', start, count) as lease:
                    expected = [(-1 if i%256 >= 128 else 1)*(1+(i%128)/128)/16
                                for i in range(start, start+count)]
                    assert [lease.value(i) for i in range(count)] == expected
            assert engine.stats()['audited_retained_metadata_bytes'] <= engine.plan.required['metadata']


@pytest.mark.parametrize('mode', ['sync', 'pipeline'])
def test_later_shard_identity_change_is_rejected(segmented, mode):
    _, path = segmented
    with CPURuntime(path, io_mode=mode, tile_elements=300, read_elements=19) as engine:
        tensor = engine.catalog.tensors['embedding']
        later = next(s for s in tensor.segments if s.shard != tensor.segments[0].shard)
        with (path/later.shard).open('ab') as f:
            f.write(b'xx')
        with pytest.raises(CheckpointError, match='changed'):
            engine.control.acquire('embedding', later.start-1, 2)
        engine.control.abort()
        assert not engine.control.active


def test_missing_later_shard_fails_open(segmented):
    _, path = segmented
    catalog = Catalog(path)
    name = list(catalog.files)[-1]
    (path/name).unlink()
    with pytest.raises(FileNotFoundError):
        CPURuntime(path)


def test_repacker_rejects_unindexed_source_tail(segmented, tmp_path):
    source, _ = segmented
    copy = tmp_path/'source-copy'
    shutil.copytree(source, copy)
    first = next(iter(Catalog(copy).files))
    with (copy/first).open('ab') as f:
        f.write(b'\x00\x00')
    target = tmp_path/'invalid-repack'
    with pytest.raises(ValueError, match='size'):
        generate(target, source=copy, execute=True)
    assert not (target/'manifest.json').exists()


def test_repacker_preserves_every_logical_byte_across_v1_v2(segmented, tmp_path):
    source, target = segmented
    again = tmp_path/'repacked-v2'
    generate(again, source=target, shard_bytes=254, segment_bytes=22, execute=True)
    original = json.loads((source/'manifest.json').read_text())
    # Reconstruct the small fixtures independently of Catalog.spans/runtime.
    for destination in (target, again):
        manifest = json.loads((destination/'manifest.json').read_text())
        assert manifest['config'] == original['config']
        for name, tensor in original['tensors'].items():
            with (source/tensor['shard']).open('rb') as f:
                f.seek(tensor['offset'])
                expected = f.read(tensor['nbytes'])
            actual = bytearray()
            for segment in manifest['tensors'][name]['segments']:
                assert segment['start']*2 == len(actual)
                with (destination/segment['shard']).open('rb') as f:
                    f.seek(segment['offset'])
                    actual.extend(f.read(segment['nbytes']))
            assert bytes(actual) == expected, name

"""Independent GEMV checks through the streamed operation and decoder interfaces."""
from array import array
from pathlib import Path
import math
import subprocess

import pytest

from inwime_runtime.gemv import StreamedGEMV
from inwime_runtime.math_cpu import NativeMath, ScalarMath
from inwime_reference.checkpoint import load_checkpoint
from tools.checkpoint4 import generate
from inwime_runtime import CPURuntime, CheckpointError
from inwime_reference.decoder import ReferenceDecoder
from tests.test_runtime_cpu import export_reference
from tools.verify_increment2 import TOKENS, compare, runtime_export
from tests.test_runtime_cpu import FaultBackend
from concurrent.futures import CancelledError
import threading

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='session')
def native():
    subprocess.run(['make', 'cpu-kernel'], cwd=ROOT, check=True)
    return NativeMath(ROOT/'build/libinwime_cpu.so')


@pytest.mark.parametrize('mode', ['sync', 'pipeline'])
@pytest.mark.parametrize('tile,read', [(7, 3), (73, 13)])
@pytest.mark.parametrize('compiled', [False, True])
def test_multishard_gemv_matches_independent_fsum(tmp_path, native, mode, tile, read, compiled):
    source = ROOT/'fixtures/reference_decoder/dense'
    path = tmp_path/'split'
    generate(path, source=source, shard_bytes=510, segment_bytes=22, execute=True)
    matrix = load_checkpoint(source).weights['layers.0.ffn.down']
    x = array('d', [(i-7)/19 for i in range(matrix.shape[1])])
    expected = [math.fsum(float(w)*v for w, v in zip(row, x)) for row in matrix]
    with StreamedGEMV(path, tile_elements=tile, read_elements=read, io_mode=mode,
                      math_backend=native if compiled else ScalarMath()) as operation:
        actual = array('d', [123.])*matrix.shape[0]
        operation.run('layers.0.ffn.down', x, actual)
        assert list(actual) == pytest.approx(expected, abs=1e-14, rel=1e-14)
        stats = operation.stats()
        assert stats['inference_read_bytes'] == matrix.size*2
        assert stats['gemv_weight_elements'] == matrix.size
        assert stats['gemv_flops'] == 2*matrix.size
        assert stats['requests'] == stats['completed'] == stats['released']
        assert stats['peak_region_elements'] <= tile
        assert not stats['active_region']
    assert not operation.control.fds and not operation.control.buffers


@pytest.mark.parametrize('variant', ['dense', 'moe'])
@pytest.mark.parametrize('mode', ['sync', 'pipeline'])
@pytest.mark.parametrize('segmented', [False, True])
def test_compiled_decoder_matches_existing_oracle_and_scalar(tmp_path, native, variant, mode, segmented):
    source = ROOT/'fixtures/reference_decoder'/variant
    path = source
    if segmented:
        path = tmp_path/'split'
        generate(path, source=source, shard_bytes=510, segment_bytes=22, execute=True)
    oracle = ReferenceDecoder(load_checkpoint(source))
    state = None
    with CPURuntime(path, tile_elements=73, read_elements=13, io_mode=mode,
                    math_backend=native) as compiled, CPURuntime(
                        source, tile_elements=7, read_elements=3) as scalar:
        for tokens in (TOKENS[:3], *([t] for t in TOKENS[3:])):
            expected = oracle.prefill(tokens, state)
            state = expected.state
            with compiled.prefill(tokens) as a, scalar.prefill(tokens) as b:
                x, y = runtime_export(a), runtime_export(b)
                assert x == y
                compare([x], [export_reference(expected)])


def test_native_expands_all_finite_bf16_codes_exactly(native):
    import struct
    words = [i for i in range(65536) if (i >> 7) & 255 != 255]
    raw = array('B', b''.join(struct.pack('<H', i) for i in words))
    values = array('d', [0.])*len(words)
    native.expand(raw, len(words), values)
    for word, actual in zip(words, values):
        expected = struct.unpack('<f', struct.pack('<I', word << 16))[0]
        assert actual == expected
        assert math.copysign(1., actual) == math.copysign(1., expected)
    with pytest.raises(CheckpointError, match='nonfinite'):
        native.expand(array('B', b'\x80\x7f'), 1, values)


@pytest.mark.parametrize('mode', ['sync', 'pipeline'])
@pytest.mark.parametrize('fault', ['short', 'error', 'cancel'])
def test_native_io_failure_or_cancellation_preserves_published_kv(native, mode, fault):
    backend = FaultBackend()
    event = threading.Event()
    with CPURuntime(ROOT/'fixtures/reference_decoder/dense', tile_elements=73,
                    read_elements=17, math_backend=native, io_mode=mode,
                    backend=backend) as engine:
        with engine.prefill([1, 7]):
            pass
        previous = engine.state_digest()
        if fault == 'cancel':
            original = backend.readinto
            def interrupted(*args):
                result = original(*args)
                event.set()
                return result
            backend.readinto = interrupted
        else:
            backend.mode, backend.remaining = fault, 8
        with pytest.raises(CancelledError if fault == 'cancel' else OSError):
            engine.decode(4, cancel_event=event)
        assert engine.state_digest() == previous
        assert engine.failed and not engine.control.active
    assert not engine.control.fds and not engine.control.buffers


@pytest.mark.parametrize('views', ['array', 'memoryview', 'distinct-exporters'])
def test_gemv_rejects_aliased_input_and_output_before_reads(native, views):
    with StreamedGEMV(ROOT/'fixtures/reference_decoder/dense', math_backend=native) as op:
        vector = array('d', [1.])*16
        if views == 'distinct-exporters':
            import numpy as np
            source = np.asarray(vector)
            destination = source.view()
        else:
            source, destination = ((memoryview(vector), memoryview(vector))
                                   if views == 'memoryview' else (vector, vector))
        before = op.stats()['inference_read_bytes']
        with pytest.raises(ValueError, match='alias'):
            op.run('layers.0.q', source, destination)
        assert op.stats()['inference_read_bytes'] == before


def test_native_keeps_caller_arenas_exported_until_kernel_return(native, monkeypatch):
    """Controlled kernel barrier: exported arenas must reject resize until completion."""
    weights, source, destination = array('d', [2., 3.]), array('d', [4., 5.]), array('d', [0.])
    original = native.library.inwime_linear
    def checked(*args):
        for arena in (weights, source, destination):
            with pytest.raises(BufferError):
                arena.append(0.)
        return original(*args)
    monkeypatch.setattr(native.library, 'inwime_linear', checked)
    native.linear(weights, 2, source, destination, 2, 0)
    assert destination[0] == 23.
    for arena in (weights, source, destination):
        arena.append(0.)  # exports have ended after return

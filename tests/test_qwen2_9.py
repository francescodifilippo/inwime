"""Qwen2 adapter: independent stdlib reference, mapping, negatives and transactions."""
from concurrent.futures import CancelledError
from dataclasses import replace
import json
import math
import struct
import threading

import pytest

from inwime_runtime.core import BudgetError, CheckpointError, FileBackend, ReadError
from inwime_runtime.math_cpu import NativeMath
from inwime_runtime.qwen2 import Qwen2Catalog, Qwen2Runtime
from inwime_runtime.safetensors_source import SafetensorsSource
from tools import fixture_qwen9
from tools.repack7 import repack


@pytest.fixture
def checkpoint(tmp_path):
    return fixture_qwen9.write(tmp_path/'qwen')


def tensors(directory):
    """Independent decode of the file: header JSON + struct BF16->FP32->float."""
    raw = (directory/'model.safetensors').read_bytes()
    length, = struct.unpack('<Q', raw[:8])
    header = json.loads(raw[8:8+length])
    base = 8+length
    out = {}
    for name, d in header.items():
        begin, end = d['data_offsets']
        words = struct.unpack(f'<{(end-begin)//2}H', raw[base+begin:base+end])
        values = [struct.unpack('<f', struct.pack('<I', w << 16))[0] for w in words]
        shape = d['shape']
        out[name] = values if len(shape) == 1 else [values[r*shape[1]:(r+1)*shape[1]] for r in range(shape[0])]
    return out


def reference(directory, tokens, transform=None):
    """Full-sequence causal Qwen2 forward written from the equations, not the runtime."""
    c = json.loads((directory/'config.json').read_text())
    w = tensors(directory)
    if transform:
        transform(w)
    width, heads, kvh = c['hidden_size'], c['num_attention_heads'], c['num_key_value_heads']
    dim, eps, theta = width//heads, c['rms_norm_eps'], c['rope_theta']

    def matvec(m, x, bias=None):
        return [math.fsum(a*b for a, b in zip(row, x)) + (bias[i] if bias else 0.) for i, row in enumerate(m)]

    def rms(x, g):
        scale = 1/math.sqrt(math.fsum(v*v for v in x)/len(x)+eps)
        return [v*scale*s for v, s in zip(x, g)]

    def rotate(v, pos):
        out = list(v)
        for h in range(len(v)//dim):
            for j in range(dim//2):
                inv = 1.0/(theta**(2*j/dim))
                cos, sin = math.cos(pos*inv), math.sin(pos*inv)
                a, b = h*dim+j, h*dim+j+dim//2
                out[a] = v[a]*cos - v[b]*sin
                out[b] = v[b]*cos + v[a]*sin
        return out

    xs = [list(w['model.embed_tokens.weight'][t]) for t in tokens]
    hidden, kv = [], []
    for layer in range(c['num_hidden_layers']):
        p = f'model.layers.{layer}.'
        normed = [rms(x, w[p+'input_layernorm.weight']) for x in xs]
        q = [rotate(matvec(w[p+'self_attn.q_proj.weight'], n, w[p+'self_attn.q_proj.bias']), i)
             for i, n in enumerate(normed)]
        k = [rotate(matvec(w[p+'self_attn.k_proj.weight'], n, w[p+'self_attn.k_proj.bias']), i)
             for i, n in enumerate(normed)]
        v = [matvec(w[p+'self_attn.v_proj.weight'], n, w[p+'self_attn.v_proj.bias']) for n in normed]
        kv.append((k, v))
        new = []
        for i, x in enumerate(xs):
            attended = []
            for h in range(heads):
                g = h // (heads//kvh)
                scores = [math.fsum(q[i][h*dim+d]*k[t][g*dim+d] for d in range(dim))/math.sqrt(dim)
                          for t in range(i+1)]
                top = max(scores)
                e = [math.exp(s-top) for s in scores]
                total = math.fsum(e)
                attended += [math.fsum(e[t]*v[t][g*dim+d] for t in range(i+1))/total for d in range(dim)]
            x = [a+b for a, b in zip(x, matvec(w[p+'self_attn.o_proj.weight'], attended))]
            n = rms(x, w[p+'post_attention_layernorm.weight'])
            gate = matvec(w[p+'mlp.gate_proj.weight'], n)
            up = matvec(w[p+'mlp.up_proj.weight'], n)
            act = [z/(1+math.exp(-z))*u for z, u in zip(gate, up)]
            x = [a+b for a, b in zip(x, matvec(w[p+'mlp.down_proj.weight'], act))]
            new.append(x)
        xs = new
        hidden.append([list(x) for x in xs])
    logits = [matvec(w['lm_head.weight'], rms(x, w['model.norm.weight'])) for x in xs]
    return logits, hidden, kv


def close(a, b, tol=1e-11):
    return all(abs(x-y) <= tol*(1+abs(y)) for x, y in zip(a, b))


@pytest.mark.parametrize('mode', ['sync', 'pipeline'])
@pytest.mark.parametrize('native', [False, True])
def test_reference_parity(checkpoint, mode, native):
    tokens = [3, 1, 7, 12, 5]
    logits, hidden, kv = reference(checkpoint, tokens)
    math_backend = NativeMath('build/libinwime_cpu.so') if native else None
    flat = lambda rows: [x for row in rows for x in row]
    with Qwen2Runtime(checkpoint, capacity=8, tile_elements=7, read_elements=3, io_mode=mode,
                      math_backend=math_backend) as r:
        c = r.spec
        with r.prefill(tokens[:2]) as result:
            assert close(result.logits, flat(logits[:2]))
        for i, t in enumerate(tokens[2:], 2):
            with r.decode(t) as result:  # incremental path against the full-sequence reference
                assert close(result.logits, logits[i])
                assert close(result.hidden[0], hidden[0][i]) and close(result.hidden[-1], hidden[-1][i])
                k, v = result.kv[-1]
                assert close(k, flat(kv[-1][0][:i+1])) and close(v, flat(kv[-1][1][:i+1]))
    # Recompute through one fresh full-prefill session and compare all outputs.
    with Qwen2Runtime(checkpoint, capacity=8, tile_elements=5, read_elements=5, io_mode=mode,
                      math_backend=math_backend) as r:
        with r.prefill(tokens) as result:
            assert close(result.logits, [x for row in logits for x in row])
            for layer in range(c.layers):
                assert close(result.hidden[layer], [x for row in hidden[layer] for x in row])
                k, v = result.kv[layer]
                assert close(k, [x for row in kv[layer][0] for x in row])
                assert close(v, [x for row in kv[layer][1] for x in row])


def test_bias_and_untied_head_matter(checkpoint):
    """Sensitivity: zero biases or a tied head change the reference, so parity is not vacuous."""
    base = reference(checkpoint, [3, 1])[0]

    def no_bias(w):
        for name in w:
            if name.endswith('.bias'):
                w[name] = [0.]*len(w[name])

    def tied(w):
        w['lm_head.weight'] = w['model.embed_tokens.weight']
    assert reference(checkpoint, [3, 1], no_bias)[0] != base
    assert reference(checkpoint, [3, 1], tied)[0] != base
    with Qwen2Runtime(checkpoint, capacity=4) as r:
        with r.prefill([3, 1]) as result:
            assert close(result.logits, [x for row in base for x in row])


def test_chunks_shards_modes_identical(checkpoint, tmp_path):
    dest = tmp_path/'sharded'
    repack(checkpoint, dest, 3)
    (dest/'config.json').write_text((checkpoint/'config.json').read_text())
    outputs = []
    for directory, mode, chunks in [(checkpoint, 'sync', [[1, 3, 2, 4]]), (checkpoint, 'sync', [[1], [3, 2], [4]]),
                                    (dest, 'pipeline', [[1, 3, 2, 4]]), (dest, 'sync', [[1, 3], [2], [4]])]:
        with Qwen2Runtime(directory, capacity=4, tile_elements=7, read_elements=3, io_mode=mode) as r:
            logits = []
            for chunk in chunks:
                with r.prefill(chunk) as result:
                    logits.append(bytes(result.logits))
                    kv = tuple(tuple(bytes(v) for v in pair) for pair in result.kv)
            outputs.append((b''.join(logits), kv))
    assert len(set(outputs)) == 1


@pytest.mark.parametrize('field,value', [
    ('model_type', 'llama'), ('architectures', ['LlamaForCausalLM']), ('hidden_act', 'gelu'),
    ('tie_word_embeddings', True), ('use_sliding_window', True), ('use_mrope', True),
    ('rope_scaling', {'type': 'yarn', 'factor': 4}), ('layer_types', ['sliding_attention']*2),
    ('rope_theta', 0), ('rms_norm_eps', float('inf')), ('num_key_value_heads', 3),
    ('head_dim', 8), ('num_hidden_layers', True), ('vocab_size', 0)])
def test_config_rejected(checkpoint, field, value):
    path = checkpoint/'config.json'
    d = json.loads(path.read_text()); d[field] = value
    path.write_text(json.dumps(d))
    with pytest.raises(CheckpointError):
        Qwen2Catalog(SafetensorsSource(checkpoint))


@pytest.mark.parametrize('case', ['missing_bias', 'missing_head', 'extra_o_bias', 'bias_shape',
                                  'weight_shape', 'transposed_k'])
def test_tensor_set_rejected(tmp_path, case):
    kwargs = {}
    if case == 'missing_bias': kwargs['drop'] = ['model.layers.1.self_attn.v_proj.bias']
    if case == 'missing_head': kwargs['drop'] = ['lm_head.weight']
    if case == 'extra_o_bias': kwargs['extra'] = {'model.layers.0.self_attn.o_proj.bias': [8]}
    if case == 'bias_shape': kwargs['extra'] = {'model.layers.0.self_attn.k_proj.bias': [8]}
    if case == 'weight_shape': kwargs['extra'] = {'model.layers.0.mlp.down_proj.weight': [12, 8]}
    if case == 'transposed_k': kwargs['extra'] = {'model.layers.0.self_attn.k_proj.weight': [8, 4]}
    directory = fixture_qwen9.write(tmp_path/case, **kwargs)
    with pytest.raises(CheckpointError):
        Qwen2Catalog(SafetensorsSource(directory))


def test_unsupported_dtype(tmp_path):
    directory = fixture_qwen9.write(tmp_path/'f16', dtype_override={'model.norm.weight': 'F16'})
    with pytest.raises(CheckpointError):
        SafetensorsSource(directory)


@pytest.mark.parametrize('fault', ['corrupt', 'truncate', 'descriptor_sha'])
def test_integrity(checkpoint, fault):
    p = checkpoint/'model.safetensors'
    if fault == 'descriptor_sha':
        d = json.loads((checkpoint/'safetensors.json').read_text())
        d['files']['model.safetensors']['sha256'] = '0'*64
        (checkpoint/'safetensors.json').write_text(json.dumps(d))
    else:
        raw = bytearray(p.read_bytes())
        if fault == 'truncate': raw = raw[:-2]
        else: raw[-1] ^= 1
        p.write_bytes(raw)
    with pytest.raises(CheckpointError):
        Qwen2Runtime(checkpoint, capacity=4)


@pytest.mark.parametrize('tokens', [[], [-1], [16], [True], [1.0], [1]*9])
def test_invalid_tokens(checkpoint, tokens):
    with Qwen2Runtime(checkpoint, capacity=8) as r:
        before = r.state_digest()
        with pytest.raises(ValueError):
            r.prefill(tokens)
        assert before == r.state_digest() and not r.failed


@pytest.mark.parametrize('mode', ['sync', 'pipeline'])
@pytest.mark.parametrize('phase', ['prefill', 'decode'])
@pytest.mark.parametrize('fault', ['cancel', 'short', 'io'])
def test_transaction(checkpoint, mode, phase, fault):
    event = threading.Event()

    class Backend(FileBackend):
        armed = False
        count = 0

        def readinto(self, fd, offset, view):
            n = super().readinto(fd, offset, view)
            if self.armed:
                self.count += 1
                if self.count == 25:  # after candidate KV has already been written
                    if fault == 'cancel': event.set()
                    if fault == 'short': return n-1
                    if fault == 'io': raise OSError('injected')
            return n
    backend = Backend()
    with Qwen2Runtime(checkpoint, capacity=8, tile_elements=7, read_elements=3, io_mode=mode,
                      backend=backend) as r:
        if phase == 'decode':
            with r.prefill([1, 2]):
                pass
        before, published = r.state_digest(), r._published
        backend.armed = True
        with pytest.raises((CancelledError, ReadError, OSError)):
            if phase == 'decode': r.decode(3, cancel_event=event)
            else: r.prefill([1, 2, 3], cancel_event=event)
        assert r.state_digest() == before and r._published == published
        assert not r.control.active and not r.result_open and r.failed
        with pytest.raises(RuntimeError):
            r.prefill([4])  # fail-stop: no resume after a failed call
    assert not r.control.fds and not r.control.buffers


def test_budget(checkpoint):
    plan = Qwen2Runtime.minimum_plan(checkpoint, capacity=4, tile_elements=7, read_elements=3)
    budget = plan.budget()
    with pytest.raises(BudgetError):
        Qwen2Runtime(checkpoint, capacity=4, tile_elements=7, read_elements=3,
                     budget=replace(budget, total=plan.total-1))
    short = dict(budget.categories); short['kv'] -= 1
    with pytest.raises(BudgetError):
        Qwen2Runtime(checkpoint, capacity=4, tile_elements=7, read_elements=3,
                     budget=replace(budget, categories=short))
    with Qwen2Runtime(checkpoint, capacity=4, tile_elements=7, read_elements=3, budget=budget) as r:
        assert r.plan.total == plan.total


def test_reads_cover_every_tensor(checkpoint):
    with Qwen2Runtime(checkpoint, capacity=4, tile_elements=7, read_elements=3) as r:
        with r.prefill([1, 2]):
            pass
        reads = r.stats()['tensor_read_bytes']
        catalog = r.catalog
    for name, tensor in catalog.tensors.items():
        expected = 2*2*(catalog.spec.width if name == 'embedding' else tensor.elements)
        assert reads[name] == expected, name


def test_import_cpu_only():
    import subprocess, sys
    subprocess.run([sys.executable, '-c', "from inwime_runtime.qwen2 import Qwen2Runtime; import sys; "
                    "assert not ({'torch','transformers','tokenizers','numpy'} & set(sys.modules))"], check=True)

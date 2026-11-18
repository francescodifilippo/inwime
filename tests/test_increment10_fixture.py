"""Increment 10, C4: Qwen2.5-14B-shaped synthetic fixture against an independent reference.

The production runtime is used unchanged. The reference below is written from the
equations (math.fsum, struct BF16 decode) and shares nothing with the runtime but
the checkpoint bytes. Sensitivity tests prove the parity is not vacuous: a wrong
epsilon, GQA grouping, RoPE convention or a dropped bias each move the reference
far beyond the thresholds.
"""
import json
import math
from pathlib import Path
import pytest

from inwime_runtime.math_cpu import NativeMath
from inwime_runtime.qwen2 import Qwen2Catalog, Qwen2Runtime
from inwime_runtime.safetensors_source import SafetensorsSource
from tools import fixture_qwen10 as fx

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT/'results'/'increment10'/'preflight-metadata'
TOKENS = [3, fx.TINY_TOKEN, 7, 30, 5, fx.TINY_TOKEN, 12, 2]     # prefill 5 + 3 decodes
NATIVE = ROOT/'build'/'libinwime_cpu.so'


@pytest.fixture(scope='module')
def checkpoint(tmp_path_factory):
    return fx.write(tmp_path_factory.mktemp('q10')/'qwen10')


@pytest.fixture(scope='module')
def weights(checkpoint):
    return fx.load(checkpoint)


def reference(c, w, tokens, *, eps=None, grouping='upstream', rope='split-half', bias=True):
    """Full-sequence causal forward recomputed from scratch at every position."""
    width, heads, kvh = c['hidden_size'], c['num_attention_heads'], c['num_key_value_heads']
    dim, theta = width//heads, c['rope_theta']
    eps = c['rms_norm_eps'] if eps is None else eps
    group = heads//kvh

    def kv_head(h):
        return h//group if grouping == 'upstream' else h % kvh

    def matvec(m, x, b=None):
        return [math.fsum(a*v for a, v in zip(row, x)) + (b[i] if b is not None and bias else 0.0)
                for i, row in enumerate(m)]

    def rms(x, g):
        s = 1/math.sqrt(math.fsum(v*v for v in x)/len(x)+eps)
        return [v*s*k for v, k in zip(x, g)]

    def rotate(v, pos):
        out = list(v)
        for h in range(len(v)//dim):
            for j in range(dim//2):
                inv = 1.0/(theta**(2*j/dim))
                cos, sin = math.cos(pos*inv), math.sin(pos*inv)
                if rope == 'split-half':
                    a, b = h*dim+j, h*dim+j+dim//2
                else:                                  # interleaved (GPT-J) convention: wrong for Qwen2
                    a, b = h*dim+2*j, h*dim+2*j+1
                out[a] = v[a]*cos - v[b]*sin
                out[b] = v[b]*cos + v[a]*sin
        return out

    xs = [list(w['model.embed_tokens.weight'][t]) for t in tokens]
    hidden, kv = [], []
    for layer in range(c['num_hidden_layers']):
        p = f'model.layers.{layer}.'
        normed = [rms(x, w[p+'input_layernorm.weight']) for x in xs]
        q = [rotate(matvec(w[p+'self_attn.q_proj.weight'], n, w[p+'self_attn.q_proj.bias']), i) for i, n in enumerate(normed)]
        k = [rotate(matvec(w[p+'self_attn.k_proj.weight'], n, w[p+'self_attn.k_proj.bias']), i) for i, n in enumerate(normed)]
        v = [matvec(w[p+'self_attn.v_proj.weight'], n, w[p+'self_attn.v_proj.bias']) for n in normed]
        kv.append((k, v))
        new = []
        for i, x in enumerate(xs):
            att = []
            for h in range(heads):
                g = kv_head(h)
                scores = [math.fsum(q[i][h*dim+d]*k[t][g*dim+d] for d in range(dim))/math.sqrt(dim)
                          for t in range(i+1)]
                top = max(scores)
                e = [math.exp(s-top) for s in scores]
                z = math.fsum(e)
                att += [math.fsum(e[t]*v[t][g*dim+d] for t in range(i+1))/z for d in range(dim)]
            x = [a+b for a, b in zip(x, matvec(w[p+'self_attn.o_proj.weight'], att))]
            n = rms(x, w[p+'post_attention_layernorm.weight'])
            gate, up = matvec(w[p+'mlp.gate_proj.weight'], n), matvec(w[p+'mlp.up_proj.weight'], n)
            x = [a+b for a, b in zip(x, matvec(w[p+'mlp.down_proj.weight'], [z/(1+math.exp(-z))*u for z, u in zip(gate, up)]))]
            new.append(x)
        xs = new
        hidden.append([list(x) for x in xs])
    logits = [matvec(w['lm_head.weight'], rms(x, w['model.norm.weight'])) for x in xs]
    return logits, hidden, kv


@pytest.fixture(scope='module')
def expected(checkpoint, weights):
    config = json.loads((checkpoint/'config.json').read_text())
    return config, reference(config, weights, TOKENS)


def flat(rows):
    return [x for row in rows for x in row]


def close(a, b, tol=1e-11):
    a, b = list(a), list(b)
    return len(a) == len(b) and all(abs(x-y) <= tol*(1+abs(y)) for x, y in zip(a, b))


def max_rel(a, b):
    return max(abs(x-y)/(1+abs(y)) for x, y in zip(flat(a), flat(b)))


# ---------------------------------------------------------------- structure

def test_fixture_preserves_14b_properties(checkpoint):
    catalog = Qwen2Catalog(SafetensorsSource(checkpoint))
    s = catalog.spec
    assert (s.heads, s.kv_heads, s.heads//s.kv_heads) == (10, 2, 5)
    assert s.rms_epsilon == 1e-5 and s.theta == 1e6 and s.dim == 8
    assert s.width % 7 and s.hidden % 7 and s.vocab % 7 and s.kv_width % 3       # tiles 7/5/3 never divide
    index = json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
    shards_of_layer1 = {f for n, f in index.items() if n.startswith('model.layers.1.')}
    assert len(set(index.values())) == 3 and len(shards_of_layer1) == 2
    assert index['lm_head.weight'] == 'model-00003-of-00003.safetensors'
    assert all(n.endswith('.bias') == ('_proj.bias' in n) for n in index)
    assert not any('o_proj.bias' in n or 'mlp' in n and n.endswith('.bias') for n in index)


def test_tiny_row_makes_epsilon_dominant(weights):
    row = weights['model.embed_tokens.weight'][fx.TINY_TOKEN]
    mean_square = math.fsum(v*v for v in row)/len(row)
    assert 1e-7 < mean_square < 1e-5                 # comparable to eps=1e-5, larger than 1e-6 is irrelevant


def test_fixture_is_deterministic(tmp_path, checkpoint):
    other = fx.write(tmp_path/'again')
    for name in ('model.safetensors.index.json', *json.loads((checkpoint/'safetensors.json').read_text())['files']):
        assert (other/name).read_bytes() == (checkpoint/name).read_bytes()


# ---------------------------------------------------------------- sensitivity (non-vacuous parity)

@pytest.mark.parametrize('variant', [dict(eps=1e-6), dict(grouping='modulo'), dict(rope='interleaved'),
                                     dict(bias=False)])
def test_reference_is_sensitive(checkpoint, weights, expected, variant):
    config, (logits, hidden, kv) = expected
    other = reference(config, weights, TOKENS[:6], **variant)[0]
    assert max_rel(other, logits[:6]) > 1e-3


def test_epsilon_effect_is_localised_on_tiny_token(checkpoint, weights, expected):
    config, (logits, _, _) = expected
    wrong = reference(config, weights, TOKENS[:2], eps=1e-6)[0]
    assert max_rel([wrong[1]], [logits[1]]) > 1e-2


# ---------------------------------------------------------------- runtime parity (runtime unchanged)

@pytest.mark.parametrize('mode', ['sync', 'pipeline'])
@pytest.mark.parametrize('native', [False, True])
def test_prefill_and_incremental_decode(checkpoint, expected, mode, native):
    config, (logits, hidden, kv) = expected
    math_backend = NativeMath(str(NATIVE)) if native else None
    with Qwen2Runtime(checkpoint, capacity=8, tile_elements=7, read_elements=3, io_mode=mode,
                      math_backend=math_backend) as r:
        layers = r.spec.layers
        with r.prefill(TOKENS[:5]) as out:
            assert close(out.logits, flat(logits[:5]))
            for layer in range(layers):
                assert close(out.hidden[layer], flat(row for row in hidden[layer][:5]))
        for i in range(5, len(TOKENS)):                 # three real decodes, KV grown incrementally
            with r.decode(TOKENS[i]) as out:
                assert close(out.logits, logits[i])
                for layer in range(layers):
                    assert close(out.hidden[layer], hidden[layer][i])
                    k, v = out.kv[layer]
                    assert close(k, flat(kv[layer][0][:i+1])) and close(v, flat(kv[layer][1][:i+1]))
                assert r.length == i+1


@pytest.mark.parametrize('tile,read', [(5, 5), (16, 16), (262144, 262144)])
def test_full_prefill_tilings_and_modes_identical(checkpoint, expected, tile, read):
    config, (logits, hidden, kv) = expected
    digests = set()
    for mode in ('sync', 'pipeline'):
        with Qwen2Runtime(checkpoint, capacity=8, tile_elements=tile, read_elements=read, io_mode=mode,
                          math_backend=NativeMath(str(NATIVE))) as r:
            with r.prefill(TOKENS) as out:
                assert close(out.logits, flat(logits))
                for layer in range(r.spec.layers):
                    k, v = out.kv[layer]
                    assert close(k, flat(kv[layer][0])) and close(v, flat(kv[layer][1]))
                digests.add((bytes(out.logits), tuple(bytes(h) for h in out.hidden),
                             tuple(bytes(x) for pair in out.kv for x in pair)))
    assert len(digests) == 1          # sync and pipeline byte-identical for a given tiling


def test_streaming_reads_every_tensor_exactly(checkpoint):
    """Each matrix/bias/norm streamed once per token across the three shards; embedding by rows only."""
    tokens = TOKENS[:3]
    with Qwen2Runtime(checkpoint, capacity=8, tile_elements=7, read_elements=3) as r:
        with r.prefill(tokens):
            pass
        stats = r.control.stats()
        catalog = r.catalog
    reads = stats['tensor_read_bytes']
    assert set(reads) == set(catalog.tensors)
    for name, tensor in catalog.tensors.items():
        want = len(tokens)*catalog.spec.width*2 if name == 'embedding' else len(tokens)*tensor.elements*2
        assert reads[name] == want, name
    assert {t.shard for t in catalog.tensors.values()} == {f'model-0000{i}-of-00003.safetensors' for i in (1, 2, 3)}


# ---------------------------------------------------------------- real 14B headers (structural only)

def test_real_14b_headers_accepted_by_unchanged_catalog(tmp_path):
    directory = fx.sparse_checkpoint(METADATA, tmp_path/'q14')
    allocated = sum(p.stat().st_blocks*512 for p in directory.iterdir())
    assert allocated < 64*2**20                 # holes: no payload materialised
    catalog = Qwen2Catalog(SafetensorsSource(directory))
    s = catalog.spec
    assert (s.width, s.hidden, s.layers, s.heads, s.kv_heads, s.vocab) == (5120, 13824, 48, 40, 8, 152064)
    assert s.rms_epsilon == 1e-5 and s.theta == 1e6 and len(catalog.tensors) == 579
    plan_sync = Qwen2Runtime.minimum_plan(directory)
    plan_pipe = Qwen2Runtime.minimum_plan(directory, io_mode='pipeline')
    protocol = json.loads((ROOT/'results'/'increment10'/'protocol.json').read_text())['memory_limits']
    # The metadata estimate contains the directory path string: tolerate the path length only.
    slack = 4*len(str(directory)) + 256
    assert abs(plan_sync.total - protocol['plan_bytes']['sync']) <= slack
    assert abs(plan_pipe.total - protocol['plan_bytes']['pipeline']) <= slack

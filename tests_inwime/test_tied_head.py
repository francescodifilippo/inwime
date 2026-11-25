"""Seams: the Qwen2 adapter with a tied head (#38) and the fixture generator that feeds it.

A tied Checkpoint (`tie_word_embeddings=true`, no `lm_head.weight`) is run by the additive
adapter; its twin with the same embedding values serialized a second time as an untied head is
run by the Percorso congelato, unchanged. Both must give the same bits.
"""
import hashlib

import pytest

import hubmodels as hm
import tinyqwen
from inwime.engine import CatalogQwen2Runtime
from inwime.qwen2_tied import TiedQwen2Catalog
from inwime_runtime.core import CheckpointError
from inwime_runtime.math_cpu import NativeMath, ScalarMath
from inwime_runtime.qwen2 import Qwen2Runtime
from inwime_runtime.safetensors_source import SafetensorsSource
from serving import Server

FX = tinyqwen.generator()
TINY = FX.TINY_TOKEN
PROMPT, CONTINUATION = [3, TINY, 7, 30, 5], [TINY, 12, 2]
# Tiles of 7 elements: no tensor is a multiple of 7, so the head streams the embedding through
# many partial leases; 262144 is the Motore's tile, larger than any tensor of the fixture.
TILES = (7, 262144)
# The Python reference and the native kernel the Motore certifies (built by `make cpu-kernel`).
KERNEL = tinyqwen.kernel_path()
MATHS = ['python'] + (['native'] if KERNEL else [])


def math(name):
    return NativeMath(str(KERNEL)) if name == 'native' else ScalarMath()


@pytest.fixture(scope='module')
def heads(tmp_path_factory):
    root = tmp_path_factory.mktemp('heads')
    return {head: tinyqwen.generator(head).write(root/head) for head in ('tied', 'duplicated')}


def tied_session(directory, tile, backend):
    catalog = TiedQwen2Catalog(SafetensorsSource(directory))
    return CatalogQwen2Runtime(catalog, len(PROMPT)+len(CONTINUATION), integrity='full',
                               math_backend=math(backend), tile_elements=tile, read_elements=tile)


def frozen_session(directory, tile, backend):
    return Qwen2Runtime(directory, capacity=len(PROMPT)+len(CONTINUATION), tile_elements=tile,
                        read_elements=tile, integrity='full', math_backend=math(backend))


def capture(runtime):
    """Bytes of logits, every residual and the whole KV after the prompt and after each decode."""
    steps = []
    with runtime as r:
        calls = [(r.prefill, PROMPT)] + [(r.decode, token) for token in CONTINUATION]
        for call, argument in calls:
            with call(argument) as out:
                steps.append((bytes(out.logits), tuple(bytes(h) for h in out.hidden),
                              tuple(bytes(v) for pair in out.kv for v in pair)))
        return steps, r.plan.required


@pytest.mark.parametrize('backend', MATHS)
@pytest.mark.parametrize('tile', TILES)
def test_tied_head_gives_the_bits_of_the_same_tensor_duplicated(heads, tile, backend):
    tied, tied_plan = capture(tied_session(heads['tied'], tile, backend))
    duplicated, duplicated_plan = capture(frozen_session(heads['duplicated'], tile, backend))
    assert tied == duplicated
    # The head never gets a buffer of its own: every category of the plan equals the untied
    # twin's, except metadata, smaller by the header of the tensor it does not have.
    metadata = tied_plan.pop('metadata'), duplicated_plan.pop('metadata')
    assert tied_plan == duplicated_plan
    assert metadata[0] < metadata[1]


def test_the_head_is_the_embedding_and_is_stored_once(heads):
    catalog = TiedQwen2Catalog(SafetensorsSource(heads['tied']))
    assert catalog.tensors['lm_head'] is catalog.tensors['embedding']
    # The twin serializes the head a second time: 37 x 80 BF16 values, 5920 bytes.
    payload = {head: SafetensorsSource(d).payload_bytes for head, d in heads.items()}
    assert payload['duplicated']-payload['tied'] == 5920


def test_the_frozen_path_still_refuses_a_tied_head(heads):
    with pytest.raises(CheckpointError, match='tying'):
        Qwen2Runtime(heads['tied'])


def test_a_tied_config_with_a_serialized_head_is_refused(tmp_path):
    # Bytes of the head are never assumed equal to the embedding without checking them.
    directory = FX.write(tmp_path/'both', config=dict(FX.CONFIG, tie_word_embeddings=True))
    with pytest.raises(CheckpointError, match='tying'):
        TiedQwen2Catalog(SafetensorsSource(directory))


# SHA256 of the files `tools/fixture_qwen10.write` produces with its defaults, taken from the
# pinned generator (protocol digest 49889f04...4c3b). `safetensors.json` lists the digest of
# every shard and of the index, so it covers all the weights.
HISTORICAL = {'config.json': '9708df3c515ebb62e2ec0e70b924c82ac10b41fba2002cbbedddd1521daa2e7c',
              'safetensors.json': '77feb7d1cd1e552d1459ae4837e7efc52b5ee23b7892e2d88997a416842d6ea9'}


def digests(directory):
    return {name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in HISTORICAL}


def test_the_head_variants_leave_the_historical_generator_output_unchanged(tmp_path):
    tinyqwen.generator('tied').write(tmp_path/'tied')
    tinyqwen.generator('duplicated').write(tmp_path/'duplicated')
    assert digests(tinyqwen.generator().write(tmp_path/'historical')) == HISTORICAL


# -- the Motore over HTTP -------------------------------------------------------------------------

MESSAGES = [{'role': 'user', 'content': 'weather in rome'}]


def activate(run, hub, directory, source):
    """Make the tied chat Checkpoint active, imported as a directory or installed from a Hub
    into the Hugging Face cache (the two catalogs the Motore builds)."""
    if source == 'imported':
        assert run('import', str(directory)).returncode == 0
        model_id = str(directory)
    else:
        files = {f.name: f.read_bytes() for f in directory.iterdir()
                 if f.name not in ('safetensors.json', 'provenance.json')}
        hub.add(hm.repo('acme/tiny-qwen2-tied', files, revision=hm.sha(38)))
        model_id = 'acme/tiny-qwen2-tied@'+hm.sha(38)
        installed = run('install', model_id, '--yes', free=500*1024**3)
        assert installed.returncode == 0, installed.stderr
    used = run('use', model_id)
    assert used.returncode == 0, used.stderr
    return model_id


@pytest.mark.parametrize('source', ['imported', 'installed'])
def test_the_motore_serves_a_tied_head_as_the_frozen_path_serves_its_duplicated_twin(
        run, hub, home, tmp_path, source):
    tied = tinyqwen.write_chat(tmp_path/'tied', head='tied')
    twin = tinyqwen.write_chat(tmp_path/'duplicated', head='duplicated')
    model_id = activate(run, hub, tied, source)
    server = Server(run, home, env=tinyqwen.kernel_env())
    try:
        status, body = server.request('POST', '/v1/chat/completions', {
            'model': model_id, 'messages': MESSAGES, 'temperature': 0, 'max_tokens': 6,
            'inwime_diagnostics': True})
    finally:
        server.stop()
    assert status == 200, body
    ids = body['inwime']['prompt_token_ids']
    assert ids == tinyqwen.auto_tokenizer(twin).apply_chat_template(
        MESSAGES, add_generation_prompt=True, return_dict=False)
    assert body['inwime']['completion_token_ids'] == tinyqwen.frozen_greedy_ids(twin, ids, 6)
    run.check_guard()

"""Seam: `inwime serve` as a client reaches it over HTTP, served by the real Motore.

The server runs in a guarded subprocess (conftest.py: private HOME, fresh registry, no request
may leave the machine) on a tiny Qwen2 Checkpoint made by the M10 fixture generator
(tinyqwen.py), imported and made active with the ordinary `inwime models` commands. Results
are compared with the Percorso congelato called directly in the test process.
"""
import json
import os
import stat
import subprocess
import sys

import pytest

import hubmodels as hm
import tinyqwen
from serving import Server, assert_openai_error, stream


@pytest.fixture(autouse=True)
def no_request_leaves_the_machine(run):
    yield
    run.check_guard()


@pytest.fixture
def checkpoint(tmp_path):
    return tinyqwen.write(tmp_path/'tiny-qwen2')


@pytest.fixture
def active(run, checkpoint):
    assert run('import', str(checkpoint)).returncode == 0
    used = run('use', str(checkpoint))
    assert used.returncode == 0, used.stderr
    return str(checkpoint)


@pytest.fixture
def server(run, home, active):
    s = Server(run, home)
    yield s
    s.stop()


def test_models_lists_the_active_model_to_an_authenticated_client(server, active, checkpoint):
    weights = sum(f.stat().st_size for f in checkpoint.glob('*.safetensors'))
    assert 'SHA256 check of the weights: %d bytes read' % weights in ''.join(server.banner)
    status, body = server.request('GET', '/v1/models')
    assert status == 200
    assert body['object'] == 'list'
    assert [m['id'] for m in body['data']] == [active]
    assert body['data'][0]['object'] == 'model'


def test_api_key_is_generated_once_private_and_never_printed(run, home, active):
    first = Server(run, home)
    key = first.key
    mode = stat.S_IMODE(first.key_file.stat().st_mode)
    output = first.stop()
    assert mode == 0o600
    assert len(key) >= 32
    assert key not in output
    second = Server(run, home)
    assert second.key == key
    assert second.request('GET', '/v1/models')[0] == 200
    assert key not in second.stop()


def test_requests_without_the_right_key_are_rejected_with_401(server):
    for kwargs in ({'auth': False}, {'key': 'wrong-key'}):
        status, body = server.request('GET', '/v1/models', **kwargs)
        assert status == 401
        assert_openai_error(body, 'invalid_api_key')
    assert 'wrong-key' not in server.stop()


def test_completion_is_the_greedy_continuation_of_the_frozen_path(server, active, checkpoint):
    prompt, max_tokens = 't3 t5 t7 t11', 5
    status, body = server.request('POST', '/v1/completions',
                                  {'model': active, 'prompt': prompt, 'max_tokens': max_tokens})
    assert status == 200, body
    reference, tokenizer = tinyqwen.frozen_greedy(checkpoint, prompt, max_tokens)
    generated = reference['generated_ids']
    assert not reference['stopped_on_eos'] and len(generated) == max_tokens
    assert generated == [21, 32, 28, 14, 24]       # known-good literal from the frozen path
    assert body['object'] == 'text_completion' and body['model'] == active
    [choice] = body['choices']
    assert choice['text'] == tokenizer.decode(generated)
    assert choice['finish_reason'] == 'length'
    assert body['usage'] == {'prompt_tokens': 4, 'completion_tokens': max_tokens,
                             'total_tokens': 4+max_tokens}


def test_streamed_completion_sends_the_greedy_tokens_as_they_come(server, active):
    s = stream(server, '/v1/completions', {'model': active, 'prompt': 't3 t5 t7 t11', 'max_tokens': 5,
                                           'stream': True, 'stream_options': {'include_usage': True}})
    assert s.status == 200 and s.events[0][:2] == ('comment', 'keepalive')
    *content, last = s.chunks
    assert all(c['object'] == 'text_completion' for c in s.chunks)
    # The frozen greedy tokens of test_completion_is_the_greedy_continuation_of_the_frozen_path,
    # one chunk per token, then the finish.
    assert [c['choices'][0]['text'] for c in content] == ['t21', ' t32', ' t28', ' t14', ' t24', '']
    assert content[-1]['choices'][0]['finish_reason'] == 'length'
    assert last['choices'] == [] and last['usage'] == {'prompt_tokens': 4, 'completion_tokens': 5,
                                                       'total_tokens': 9}
    assert s.events[-1][:2] == ('data', '[DONE]')


def test_a_stop_sequence_across_tokens_ends_the_completion_without_leaking_into_the_stream(server, active):
    # Greedy continuation of this prompt: t21 t32 t28 t14 t24. The stop spans tokens 2 and 3.
    request = {'model': active, 'prompt': 't3 t5 t7 t11', 'max_tokens': 5, 'stop': ['t32 t28', 'zz']}
    status, body = server.request('POST', '/v1/completions', request)
    assert status == 200, body
    assert body['choices'][0]['text'] == 't21 ' and body['choices'][0]['finish_reason'] == 'stop'
    assert body['usage']['completion_tokens'] == 3
    s = stream(server, '/v1/completions', dict(request, stream=True))
    texts = [c['choices'][0]['text'] for c in s.chunks]
    assert ''.join(texts) == 't21 '
    assert not any('t3' in t for t in texts)          # " t32" was held back, never sent
    assert s.chunks[-1]['choices'][0]['finish_reason'] == 'stop'


def test_any_eos_id_of_generation_config_ends_the_generation(server, active, checkpoint):
    # generation_config.json may list several EOS ids; t35 (greedy after "t2 t33") becomes one.
    (checkpoint/'generation_config.json').write_text(json.dumps({'bos_token_id': 0, 'eos_token_id': [1, 35]}))
    server.stop()
    restarted = Server(server.run, server.home)
    try:
        status, body = restarted.request('POST', '/v1/completions', {'model': active, 'prompt': 't2 t33', 'max_tokens': 6})
        assert status == 200, body
        assert body['choices'][0] == {'text': 't2', 'index': 0, 'logprobs': None, 'finish_reason': 'stop'}
        assert body['usage']['completion_tokens'] == 2
    finally:
        restarted.stop()


def test_completion_stops_at_eos_without_emitting_it(server, active):
    status, body = server.request('POST', '/v1/completions', {'model': active, 'prompt': 't2 t33', 'max_tokens': 6})
    assert status == 200, body
    # Frozen greedy on this fixture: t2 t35 t29 then <|endoftext|> (id 1), well before 6 tokens.
    [choice] = body['choices']
    assert choice['text'] == 't2 t35 t29'
    assert choice['finish_reason'] == 'stop'
    assert body['usage'] == {'prompt_tokens': 2, 'completion_tokens': 4, 'total_tokens': 6}


def test_requests_beyond_the_context_are_rejected_before_inference(server, active):
    # The fixture's max_position_embeddings is 64: 4 prompt tokens + 60 fit, + 61 do not.
    status, body = server.request('POST', '/v1/completions',
                                  {'model': active, 'prompt': 't3 t5 t7 t11', 'max_tokens': 61})
    assert status == 400
    assert_openai_error(body, 'context_length_exceeded')
    assert '64' in body['error']['message']
    status, body = server.request('POST', '/v1/completions',
                                  {'model': active, 'prompt': 't3 t5 t7 t11', 'max_tokens': 60})
    assert status == 200, body
    assert body['usage']['prompt_tokens']+body['usage']['completion_tokens'] <= 64


@pytest.mark.parametrize('body, param, code', [
    (b'{not json', None, None),
    ({'prompt': 't3'}, 'model', None),
    ({'prompt': ['t3', 't5']}, 'prompt', None),
    ({'prompt': 't3', 'max_tokens': 0}, 'max_tokens', None),
    ({'prompt': 't3', 'max_tokens': True}, 'max_tokens', None),
    ({'prompt': 't3', 'max_completion_tokens': -1}, 'max_completion_tokens', None),
    ({'prompt': 't3', 'max_tokens': 2, 'max_completion_tokens': 3}, 'max_completion_tokens', None),
    ({'prompt': 't3', 'logit_bias': {'5': 100}}, 'logit_bias', 'unsupported_parameter'),
    ({'prompt': 't3', 'stream': 'yes'}, 'stream', None),
    ({'prompt': 't3', 'temperature': 2.5}, 'temperature', None),
    ({'prompt': 't3', 'n': 2}, 'n', 'unsupported_parameter'),
])
def test_invalid_completion_requests_get_400(server, active, body, param, code):
    if isinstance(body, dict) and param != 'model':
        body = dict(body, model=active)
    status, reply = server.request('POST', '/v1/completions', body)
    assert status == 400
    assert_openai_error(reply, code)
    assert reply['error']['type'] == 'invalid_request_error'
    assert reply['error']['param'] == param


def test_max_completion_tokens_limits_the_completion(server, active):
    status, body = server.request('POST', '/v1/completions',
                                  {'model': active, 'prompt': 't3 t5 t7 t11', 'max_completion_tokens': 2})
    assert status == 200, body
    assert (body['choices'][0]['text'], body['choices'][0]['finish_reason']) == ('t21 t32', 'length')


def test_greedy_parameters_at_their_greedy_values_are_accepted(server, active):
    status, body = server.request('POST', '/v1/completions', {
        'model': active, 'prompt': 't2 t33', 'max_tokens': 6, 'temperature': 0, 'top_p': 1.0, 'n': 1,
        'stream': False})
    assert status == 200, body
    assert body['choices'][0]['text'] == 't2 t35 t29'


def test_unknown_model_and_endpoint_get_404(server, active):
    status, body = server.request('POST', '/v1/completions', {'model': 'acme/other', 'prompt': 't3'})
    assert status == 404
    assert_openai_error(body, 'model_not_found')
    status, body = server.request('GET', '/v1/nowhere')
    assert status == 404
    assert_openai_error(body)
    # Without the key, nothing about the routes is disclosed.
    assert server.request('GET', '/v1/nowhere', auth=False)[0] == 401
    assert server.request('GET', '/v1/completions', auth=False)[0] == 401
    status, body = server.request('GET', '/v1/completions')
    assert status == 405
    assert_openai_error(body, 'method_not_allowed')


def test_engine_failure_gets_500_and_the_server_keeps_serving(server, active, checkpoint):
    shard = next(checkpoint.glob('model-00001-*.safetensors'))
    st = shard.stat()
    os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns+10**9))   # same bytes, new identity
    status, body = server.request('POST', '/v1/completions', {'model': active, 'prompt': 't3', 'max_tokens': 2})
    assert status == 500
    assert_openai_error(body, 'checkpoint_changed')
    assert body['error']['type'] == 'server_error'
    assert str(checkpoint) not in body['error']['message']
    assert server.request('GET', '/v1/models')[0] == 200


def test_an_engine_failure_during_a_stream_ends_it_with_an_error_event(server, active, checkpoint):
    shard = next(checkpoint.glob('model-00001-*.safetensors'))
    st = shard.stat()
    os.utime(shard, ns=(st.st_atime_ns, st.st_mtime_ns+10**9))
    s = stream(server, '/v1/completions', {'model': active, 'prompt': 't3', 'max_tokens': 2, 'stream': True})
    assert s.status == 200 and s.events[0][:2] == ('comment', 'keepalive')
    [data] = s.chunks
    assert_openai_error(data, 'checkpoint_changed')
    assert str(checkpoint) not in data['error']['message']
    assert s.events[-1][:2] == ('data', '[DONE]')


def test_concurrent_requests_each_get_their_own_frozen_result(server, active, checkpoint):
    from concurrent.futures import ThreadPoolExecutor
    prompts = ['t3 t5 t7 t11', 't2 t33', 't9 t4', 't3 t5 t7 t11']
    with ThreadPoolExecutor(len(prompts)) as pool:
        replies = list(pool.map(lambda p: server.request(
            'POST', '/v1/completions', {'model': active, 'prompt': p, 'max_tokens': 5}), prompts))
    for prompt, (status, body) in zip(prompts, replies):
        assert status == 200, body
        reference, tokenizer = tinyqwen.frozen_greedy(checkpoint, prompt, 5)
        ids = reference['generated_ids']
        assert body['choices'][0]['text'] == tokenizer.decode(ids[:-1] if reference['stopped_on_eos'] else ids)


def serve_command(run, home, *args):
    return subprocess.run([sys.executable, '-m', 'inwime', 'serve', '--port', '0', *args], capture_output=True,
                          text=True, env=run.environment(), cwd=home, timeout=60)


def test_a_non_loopback_address_needs_an_explicit_choice(run, home, active):
    refused = serve_command(run, home, '--host', '0.0.0.0')
    assert refused.returncode == 2
    assert 'loopback' in refused.stderr and 'listening' not in refused.stdout


def test_serve_without_an_active_model_refuses_to_start(run, home):
    refused = serve_command(run, home)
    assert refused.returncode == 3
    assert 'inwime models use' in refused.stderr


def test_serve_refuses_a_key_file_other_users_can_read(run, home, active):
    Server(run, home).stop()
    key_file = home/'.local/share/inwime/server/api-key'
    key_file.chmod(0o644)
    refused = serve_command(run, home)
    assert refused.returncode != 0
    assert 'chmod 600' in refused.stderr
    assert key_file.read_text().strip() not in refused.stdout+refused.stderr


@pytest.mark.parametrize('config, reason', [({'tie_word_embeddings': True}, 'tying'), ({}, 'tokenizer')])
def test_serve_refuses_a_checkpoint_it_cannot_run(run, home, tmp_path, config, reason):
    # Tied head: rejected by the Percorso congelato (#38 adds it). No config change: the bare
    # fixture, which has no tokenizer files.
    fixture = tinyqwen.generator()
    directory = fixture.write(tmp_path/'unservable', config=dict(fixture.CONFIG, **config))
    assert run('import', str(directory)).returncode == 0
    assert run('use', str(directory)).returncode == 0
    refused = serve_command(run, home)
    assert refused.returncode == 13, refused.stderr
    assert 'cannot be served' in refused.stderr and reason in refused.stderr
    assert 'Traceback' not in refused.stderr and 'listening' not in refused.stdout


def test_one_coordinatore_per_process(checkpoint, monkeypatch):
    from inwime.engine import Coordinatore
    monkeypatch.setenv('INWIME_CPU_KERNEL', 'off')
    first = Coordinatore('tiny', checkpoint)
    try:
        with pytest.raises(RuntimeError, match='already exists'):
            Coordinatore('tiny', checkpoint)
        reply = first.generate(first.prepare_completion('t2 t33', 6, {})).result(timeout=60)
        assert (reply['text'], reply['finish_reason']) == ('t2 t35 t29', 'stop')
    finally:
        first.close()
    Coordinatore('tiny', checkpoint).close()      # free again once the first one is closed


def install_from_hub(run, hub, checkpoint, repo_id='acme/tiny-qwen2-m10'):
    """The same Checkpoint as published on a Hub: weights, index, config and tokenizer, without
    InWiMe's own descriptor. Installed into the Hugging Face cache, then made active."""
    files = {f.name: f.read_bytes() for f in checkpoint.iterdir()
             if f.name not in ('safetensors.json', 'provenance.json')}
    hub.add(hm.repo(repo_id, files, revision=hm.sha(37)))
    model_id = '%s@%s' % (repo_id, hm.sha(37))
    installed = run('install', model_id, '--yes', free=500*1024**3)
    assert installed.returncode == 0, installed.stderr
    used = run('use', model_id)
    assert used.returncode == 0, used.stderr
    return model_id


def test_an_installed_checkpoint_is_served_like_the_imported_one(run, hub, home, checkpoint):
    model_id = install_from_hub(run, hub, checkpoint)
    server = Server(run, home)
    try:
        weights = sum(f.stat().st_size for f in checkpoint.glob('*.safetensors'))
        assert 'SHA256 check of the weights: %d bytes read' % weights in ''.join(server.banner)
        assert [m['id'] for m in server.request('GET', '/v1/models')[1]['data']] == [model_id]
        for prompt in ('t3 t5 t7 t11', 't2 t33'):
            status, body = server.request('POST', '/v1/completions',
                                          {'model': model_id, 'prompt': prompt, 'max_tokens': 5})
            assert status == 200, body
            reference, tokenizer = tinyqwen.frozen_greedy(checkpoint, prompt, 5)
            ids = reference['generated_ids']
            assert body['choices'][0]['text'] == tokenizer.decode(ids[:-1] if reference['stopped_on_eos'] else ids)
            assert body['usage']['completion_tokens'] == len(ids)
    finally:
        server.stop()
    # Serving wrote nothing into the Hugging Face cache.
    snapshot = home/'.cache/huggingface/hub/models--acme--tiny-qwen2-m10/snapshots'/hm.sha(37)
    assert sorted(f.name for f in snapshot.iterdir()) == sorted(
        f.name for f in checkpoint.iterdir() if f.name not in ('safetensors.json', 'provenance.json'))


def test_a_changed_installed_blob_gets_500_until_restart(run, hub, home, checkpoint):
    model_id = install_from_hub(run, hub, checkpoint)
    server = Server(run, home)
    try:
        blob = next((home/'.cache/huggingface/hub/models--acme--tiny-qwen2-m10/snapshots'/hm.sha(37))
                    .glob('model-00002-*.safetensors')).resolve()
        st = blob.stat()
        os.utime(blob, ns=(st.st_atime_ns, st.st_mtime_ns+10**9))
        status, body = server.request('POST', '/v1/completions', {'model': model_id, 'prompt': 't3', 'max_tokens': 2})
        assert status == 500
        assert_openai_error(body, 'checkpoint_changed')
    finally:
        server.stop()


def test_an_installed_shard_relinked_to_another_blob_is_refused(run, hub, home, checkpoint):
    install_from_hub(run, hub, checkpoint)
    snapshot = home/'.cache/huggingface/hub/models--acme--tiny-qwen2-m10/snapshots'/hm.sha(37)
    first, second = sorted(snapshot.glob('model-*.safetensors'))[:2]
    target = os.readlink(second)
    first.unlink()
    first.symlink_to(target)
    refused = serve_command(run, home)
    assert refused.returncode == 13, refused.stderr
    assert 'listening' not in refused.stdout and 'Traceback' not in refused.stderr

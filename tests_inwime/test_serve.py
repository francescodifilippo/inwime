"""Seam: `inwime serve` as a client reaches it over HTTP, served by the real Motore.

The server runs in a guarded subprocess (conftest.py: private HOME, fresh registry, no request
may leave the machine) on a tiny Qwen2 Checkpoint made by the M10 fixture generator
(tinyqwen.py), imported and made active with the ordinary `inwime models` commands. Results
are compared with the Percorso congelato called directly in the test process.
"""
import json
import os
import re
import stat
import subprocess
import sys
import urllib.error
import urllib.request

import pytest

import tinyqwen


class Server:
    def __init__(self, run, home, args=()):
        self.run, self.home = run, home
        self.process = subprocess.Popen([sys.executable, '-m', 'inwime', 'serve', '--port', '0', *args],
                                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, env=run.environment(),
                                        cwd=home)
        self.banner = []
        for line in self.process.stdout:
            self.banner.append(line)
            found = re.search(r'listening on (http://\S+)', line)
            if found:
                self.url = found.group(1)
                break
        else:
            raise AssertionError('server did not start: %s%s' % (''.join(self.banner),
                                                                 self.process.stderr.read()))

    @property
    def key_file(self):
        return self.home/'.local/share/inwime/server/api-key'

    @property
    def key(self):
        return self.key_file.read_text().strip()

    def request(self, method, path, body=None, key=None, auth=True):
        headers = {'Content-Type': 'application/json'}
        if auth:
            headers['Authorization'] = 'Bearer %s' % (self.key if key is None else key)
        data = None if body is None else (body if isinstance(body, bytes) else json.dumps(body).encode())
        request = urllib.request.Request(self.url+path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def stop(self):
        if self.process.returncode is not None:
            return self.output
        self.process.terminate()
        try:
            out, err = self.process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            out, err = self.process.communicate()
        self.output = ''.join(self.banner)+out+err
        return self.output


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


def assert_openai_error(body, code=None):
    error = body['error']
    assert isinstance(error['message'], str) and error['message']
    assert set(error) >= {'message', 'type', 'param', 'code'}
    if code is not None:
        assert error['code'] == code


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
    ({'prompt': 't3', 'stream': True}, 'stream', 'unsupported_parameter'),
    ({'prompt': 't3', 'temperature': 0.7}, 'temperature', 'unsupported_parameter'),
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
        reply = first.complete('t2 t33', 6).result(timeout=60)
        assert (reply['text'], reply['finish_reason']) == ('t2 t35 t29', 'stop')
    finally:
        first.close()
    Coordinatore('tiny', checkpoint).close()      # free again once the first one is closed

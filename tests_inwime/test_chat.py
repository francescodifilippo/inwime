"""Seam: `/v1/chat/completions` (and streaming `/v1/completions`) as a client reaches them over HTTP.

The server runs the real Motore on the M10 fixture weights, served as a Qwen2.5 chat model
(tinyqwen.write_chat: byte-level BPE in Qwen's pipeline and the real Qwen2.5 chat template).
References are independent of the server: Transformers' own `apply_chat_template` for the
prompt, and the Percorso congelato called directly for the tokens.
"""
import json
import os
from pathlib import Path
import threading
import time

import pytest

import tinyqwen
from conftest import GUARD, PACKAGE_PARENT
from serving import Server, assert_openai_error, stream

TESTS = str(Path(__file__).resolve().parent)

CONVERSATION = [
    {'role': 'system', 'content': 'be brief.'},
    {'role': 'user', 'content': 'weather in rome'},
    {'role': 'assistant', 'content': 'let me look.'},
    {'role': 'tool', 'content': 'sunny'},
    {'role': 'user', 'content': 'thanks'},
]


@pytest.fixture(autouse=True)
def no_request_leaves_the_machine(run):
    yield
    run.check_guard()


def make_active(run, directory):
    assert run('import', str(directory)).returncode == 0
    used = run('use', str(directory))
    assert used.returncode == 0, used.stderr
    return str(directory)


@pytest.fixture
def chat_checkpoint(tmp_path):
    return tinyqwen.write_chat(tmp_path/'tiny-chat')


@pytest.fixture
def chat_server(run, home, chat_checkpoint):
    model = make_active(run, chat_checkpoint)
    s = Server(run, home, env=tinyqwen.kernel_env())
    s.model = model
    yield s
    s.stop()


def reference_ids(checkpoint, messages, **kwargs):
    return tinyqwen.auto_tokenizer(checkpoint).apply_chat_template(
        messages, add_generation_prompt=True, return_dict=False, **kwargs)


def test_chat_prompt_is_rendered_as_transformers_does_and_continued_greedily(chat_server, chat_checkpoint):
    status, body = chat_server.request('POST', '/v1/chat/completions', {
        'model': chat_server.model, 'messages': CONVERSATION, 'temperature': 0, 'max_tokens': 6,
        'inwime_diagnostics': True})
    assert status == 200, body
    ids = reference_ids(chat_checkpoint, CONVERSATION)
    assert body['inwime']['prompt_token_ids'] == ids
    generated = tinyqwen.frozen_greedy_ids(chat_checkpoint, ids, 6)
    assert body['object'] == 'chat.completion' and body['model'] == chat_server.model
    [choice] = body['choices']
    assert choice['message'] == {'role': 'assistant', 'content': tinyqwen.auto_tokenizer(chat_checkpoint).decode(
        [t for t in generated if t not in tinyqwen.CHAT_EOS], skip_special_tokens=True)}
    assert choice['finish_reason'] == ('stop' if generated[-1] in tinyqwen.CHAT_EOS else 'length')
    assert body['usage'] == {'prompt_tokens': len(ids), 'completion_tokens': len(generated),
                             'total_tokens': len(ids)+len(generated)}


TOOLS = [{'type': 'function', 'function': {
    'name': 'get_weather', 'description': 'the weather in a city',
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}]


def test_tools_are_rendered_into_the_prompt_as_transformers_does(chat_server, chat_checkpoint):
    status, body = chat_server.request('POST', '/v1/chat/completions', {
        'model': chat_server.model, 'messages': CONVERSATION[1:], 'tools': TOOLS, 'temperature': 0,
        'max_tokens': 1, 'inwime_diagnostics': True})
    assert status == 200, body
    with_tools = reference_ids(chat_checkpoint, CONVERSATION[1:], tools=TOOLS)
    assert body['inwime']['prompt_token_ids'] == with_tools
    assert with_tools != reference_ids(chat_checkpoint, CONVERSATION[1:])


def test_a_checkpoint_without_chat_template_refuses_chat_but_keeps_completions(run, home, tmp_path):
    model = make_active(run, tinyqwen.write_chat(tmp_path/'no-template', template=None))
    server = Server(run, home)
    try:
        status, body = server.request('POST', '/v1/chat/completions', {
            'model': model, 'messages': [{'role': 'user', 'content': 'hi'}]})
        assert status == 400
        assert_openai_error(body, 'chat_template_unavailable')
        assert 'chat template' in body['error']['message'] and '/v1/completions' in body['error']['message']
        status, body = server.request('POST', '/v1/completions', {'model': model, 'prompt': 'hi', 'max_tokens': 2})
        assert status == 200, body
    finally:
        server.stop()


def test_a_chat_prompt_beyond_the_context_is_refused_before_inference(chat_server):
    long = [{'role': 'user', 'content': 'weather ' * 1200}]
    for extra in ({}, {'max_tokens': 4}, {'stream': True}):
        s = stream(chat_server, '/v1/chat/completions', {'model': chat_server.model, 'messages': long, **extra})
        assert s.status == 400 and s.events == []
        assert_openai_error(s.body, 'context_length_exceeded')
        assert '1024' in s.body['error']['message']


@pytest.mark.parametrize('messages', [
    [], 'hi', [{'role': 'robot', 'content': 'hi'}], [{'role': 'user'}], [{'role': 'user', 'content': 5}],
    [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'x'}}]}],
])
def test_malformed_messages_get_400(chat_server, messages):
    status, body = chat_server.request('POST', '/v1/chat/completions',
                                       {'model': chat_server.model, 'messages': messages})
    assert status == 400
    assert_openai_error(body)
    assert body['error']['param'] == 'messages'


def chat_body(server, **extra):
    return {'model': server.model, 'messages': CONVERSATION, 'temperature': 0, 'max_tokens': 8, **extra}


def test_streamed_chat_sends_a_keepalive_first_then_the_same_text_as_non_stream(chat_server):
    status, plain = chat_server.request('POST', '/v1/chat/completions', chat_body(chat_server))
    assert status == 200, plain
    assert 'inwime' not in plain                 # the diagnostic extension only when asked for
    s = stream(chat_server, '/v1/chat/completions', chat_body(chat_server, stream=True))
    assert s.status == 200
    assert s.headers['content-type'].startswith('text/event-stream')
    assert s.headers['cache-control'] == 'no-cache'
    assert s.events[0][:2] == ('comment', 'keepalive')
    assert s.events[-1][:2] == ('data', '[DONE]')
    chunks = s.chunks
    assert chunks[0]['choices'][0]['delta']['role'] == 'assistant'
    assert s.text() == plain['choices'][0]['message']['content']
    finishes = [ch['finish_reason'] for c in chunks for ch in c['choices']]
    assert finishes[-1] == plain['choices'][0]['finish_reason'] and not any(finishes[:-1])
    assert all('usage' not in c or c['usage'] is None for c in chunks)


def test_include_usage_adds_a_final_usage_chunk(chat_server):
    status, plain = chat_server.request('POST', '/v1/chat/completions', chat_body(chat_server))
    s = stream(chat_server, '/v1/chat/completions',
               chat_body(chat_server, stream=True, stream_options={'include_usage': True}))
    *content, last = s.chunks
    assert last['choices'] == [] and last['usage'] == plain['usage']
    assert all(c['usage'] is None for c in content)


def test_keepalive_comments_repeat_while_the_request_waits_and_never_look_like_progress(
        run, home, chat_checkpoint, tmp_path):
    model = make_active(run, chat_checkpoint)
    # The pacer (pacer.py) stalls the worker inside the first request's prefill while this gate
    # file exists, so the streamed request waits in the queue for as long as the test decides.
    gates = tmp_path/'pacer'
    gates.mkdir()
    (gates/'stall-prefill').touch()
    server = Server(run, home, args=('--keepalive-seconds', '0.1'), module='pacer', env=dict(
        tinyqwen.kernel_env(), INWIME_PACER_DIR=str(gates),
        PYTHONPATH=os.pathsep.join((GUARD, TESTS, PACKAGE_PARENT))))
    server.model = model
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(1) as pool:
            busy = pool.submit(server.request, 'POST', '/v1/chat/completions', {
                'model': model, 'messages': CONVERSATION[1:], 'tools': TOOLS, 'max_tokens': 1})
            deadline = time.monotonic()+30
            while 'held prefill' not in ((gates/'events').read_text() if (gates/'events').exists() else ''):
                assert time.monotonic() < deadline, 'the first request never reached the worker'
                time.sleep(0.01)
            release = threading.Timer(1.0, (gates/'stall-prefill').unlink)
            release.start()
            s = stream(server, '/v1/chat/completions', chat_body(server, stream=True, max_tokens=2))
            release.join()
            assert busy.result()[0] == 200
        first_data = next(i for i, e in enumerate(s.events) if e[0] == 'data')
        waiting = s.events[:first_data]
        assert len(waiting) >= 5 and all(e[:2] == ('comment', 'keepalive') for e in waiting)
        gaps = [b[2]-a[2] for a, b in zip(waiting, waiting[1:])]
        assert max(gaps) < 1.0
        # Every data chunk carries model output or the end of the choice: none is a placeholder.
        for chunk in s.chunks:
            [choice] = chunk['choices']
            assert choice['delta'].get('content') or choice['finish_reason']
    finally:
        server.stop()


QWEN_INSTRUCT_SAMPLING = {'do_sample': True, 'temperature': 0.7, 'top_p': 0.8, 'top_k': 20,
                          'repetition_penalty': 1.1}       # generation_config.json of Qwen2.5-0.5B-Instruct


def sampling(body):
    return {name: (v['value'], v['source']) for name, v in body['inwime']['sampling'].items()}


def test_sampling_parameters_come_from_the_client_then_generation_config_then_defaults(run, home, tmp_path):
    model = make_active(run, tinyqwen.write_chat(tmp_path/'sampling', generation=QWEN_INSTRUCT_SAMPLING))
    server = Server(run, home, env=tinyqwen.kernel_env())
    try:
        status, body = server.request('POST', '/v1/chat/completions', {
            'model': model, 'messages': CONVERSATION, 'max_tokens': 2, 'temperature': 0.5, 'seed': 7,
            'stop': ['zz'], 'inwime_diagnostics': True})
        assert status == 200, body
        assert sampling(body) == {'temperature': (0.5, 'client'), 'top_p': (0.8, 'generation_config'),
                                  'top_k': (20, 'generation_config'), 'repetition_penalty': (1.1, 'generation_config'),
                                  'seed': (7, 'client'), 'stop': (['zz'], 'client')}
        assert body['inwime']['computation']['path']
    finally:
        server.stop()


def test_invalid_generation_config_values_are_ignored(run, home, tmp_path):
    model = make_active(run, tinyqwen.write_chat(tmp_path/'odd', generation={
        'do_sample': True, 'temperature': -1, 'top_k': 20.0, 'top_p': 0.8, 'stop_strings': [5]}))
    server = Server(run, home, env=tinyqwen.kernel_env())
    try:
        status, body = server.request('POST', '/v1/chat/completions', {
            'model': model, 'messages': CONVERSATION, 'max_tokens': 2, 'inwime_diagnostics': True})
        assert status == 200, body
        effective = sampling(body)
        assert effective['temperature'] == (1.0, 'default') and effective['top_k'] == (50, 'default')
        assert effective['top_p'] == (0.8, 'generation_config') and effective['stop'] == ([], 'default')
    finally:
        server.stop()


def test_without_do_sample_the_checkpoint_default_is_greedy(chat_server):
    # generation_config.json of the fixture has no do_sample: Transformers' default False, greedy.
    status, body = chat_server.request('POST', '/v1/chat/completions', {
        'model': chat_server.model, 'messages': CONVERSATION, 'max_tokens': 2, 'inwime_diagnostics': True})
    assert status == 200, body
    effective = sampling(body)
    assert effective['temperature'] == (0, 'default')
    assert effective['top_p'] == (1.0, 'default') and effective['top_k'] == (50, 'default')
    assert effective['repetition_penalty'] == (1.0, 'default') and effective['stop'] == ([], 'default')
    assert effective['seed'][1] == 'server' and type(effective['seed'][0]) is int


def sample(server, seed, **extra):
    status, body = server.request('POST', '/v1/chat/completions', {
        'model': server.model, 'messages': CONVERSATION, 'max_tokens': 6, 'temperature': 1.5, 'top_k': 0,
        'seed': seed, 'inwime_diagnostics': True, **extra})
    assert status == 200, body
    return body['inwime']['completion_token_ids']


def test_the_same_seed_gives_the_same_tokens_and_seeds_differ(chat_server):
    assert sample(chat_server, 11) == sample(chat_server, 11)
    assert len({tuple(sample(chat_server, seed)) for seed in range(5)}) > 1
    assert sample(chat_server, -3) != sample(chat_server, 3)


def test_sampled_tokens_stay_inside_top_k_and_top_p(chat_server, chat_checkpoint):
    prompt = reference_ids(chat_checkpoint, CONVERSATION)
    tokens = sample(chat_server, 3, top_k=3, temperature=2.0)
    for i, token in enumerate(tokens):
        logits = tinyqwen.frozen_logits(chat_checkpoint, prompt+tokens[:i])
        top3 = sorted(range(len(logits)), key=lambda t: (-logits[t], t))[:3]
        assert token in top3
    # A tiny nucleus keeps only the most probable token: greedy, whatever the temperature and seed.
    assert sample(chat_server, 5, top_p=1e-9) == tinyqwen.frozen_greedy_ids(chat_checkpoint, prompt, 6)


def test_repetition_penalty_follows_the_transformers_rule(chat_server, chat_checkpoint):
    prompt = reference_ids(chat_checkpoint, CONVERSATION)
    expected = tinyqwen.frozen_penalized_greedy(chat_checkpoint, prompt, 16, 1.5)
    # Plain greedy repeats a token within 16 steps on this fixture: the penalty changes the result.
    assert expected != tinyqwen.frozen_greedy_ids(chat_checkpoint, prompt, 16)
    assert sample(chat_server, 1, temperature=0, repetition_penalty=1.5, max_tokens=16) == expected


@pytest.mark.parametrize('extra, param', [
    ({'temperature': -0.1}, 'temperature'), ({'temperature': 2.5}, 'temperature'), ({'top_p': 0}, 'top_p'),
    ({'top_p': 1.5}, 'top_p'), ({'top_k': -1}, 'top_k'), ({'top_k': 2.5}, 'top_k'),
    ({'repetition_penalty': 0}, 'repetition_penalty'), ({'seed': 'x'}, 'seed'), ({'stop': ''}, 'stop'),
    ({'stop': ['a', 'b', 'c', 'd', 'e']}, 'stop'), ({'stop': [3]}, 'stop'),
    (b'NaN', 'temperature'),
])
def test_out_of_range_sampling_parameters_get_400(chat_server, extra, param):
    if extra == b'NaN':                         # Python's json accepts NaN; a client may send it
        request = ('{"model": %s, "messages": %s, "temperature": NaN}'
                   % (json.dumps(chat_server.model), json.dumps(CONVERSATION))).encode()
    else:
        request = {'model': chat_server.model, 'messages': CONVERSATION, **extra}
    status, body = chat_server.request('POST', '/v1/chat/completions', request)
    assert status == 400
    assert_openai_error(body)
    assert body['error']['param'] == param

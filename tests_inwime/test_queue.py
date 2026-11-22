"""Seam: the queue, the 429 refusal and cancellation of `inwime serve`, as clients reach them over HTTP.

The server runs the real Motore on the M10 fixture in a guarded subprocess, wrapped by
pacer.py: gate files hold the Percorso congelato at its own cancellation checks, in the prefill
or in the decode, and an event log tells what the worker did (runtime calls, CancelledError
raised by the runtime, sessions closed with their published KV length). Expected texts are the
frozen greedy continuations already pinned by test_serve.py.
"""
from concurrent.futures import ThreadPoolExecutor
import http.client
import json
import os
import time
import urllib.parse
from pathlib import Path

import pytest

import tinyqwen
from conftest import GUARD, PACKAGE_PARENT
from serving import Server, assert_openai_error

TESTS = str(Path(__file__).resolve().parent)

# Frozen greedy continuations of the fixture (test_serve.py).
LONG = ('t3 t5 t7 t11', 2, 't21 t32')          # prompt, max_tokens, text
SHORT = ('t2 t33', 6, 't2 t35 t29')


@pytest.fixture(autouse=True)
def no_request_leaves_the_machine(run):
    yield
    run.check_guard()


@pytest.fixture
def active(run, tmp_path):
    checkpoint = tinyqwen.write(tmp_path/'tiny-qwen2')
    assert run('import', str(checkpoint)).returncode == 0
    used = run('use', str(checkpoint))
    assert used.returncode == 0, used.stderr
    return str(checkpoint)


class Pacer:
    def __init__(self, directory):
        self.directory = directory
        directory.mkdir()

    def gate(self, name, closed=True):
        if closed:
            (self.directory/name).touch()
        else:
            (self.directory/name).unlink()

    def events(self):
        log = self.directory/'events'
        return log.read_text().splitlines() if log.exists() else []

    def wait(self, predicate, timeout=30):
        """Until `predicate(events)` holds; returns the events."""
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            events = self.events()
            if predicate(events):
                return events
            time.sleep(0.01)
        raise AssertionError('pacer condition not reached; events: %s' % self.events())

    def wait_for(self, line, count=1, **kwargs):
        return self.wait(lambda events: events.count(line) >= count, **kwargs)

    def admitted(self, count):
        return self.wait(lambda events: sum(e.startswith('admitted ') for e in events) >= count)


@pytest.fixture
def paced(run, home, active, tmp_path):
    """`paced(*serve args)` starts a paced server; returns (server, pacer)."""
    started = []

    def start(*args):
        pacer = Pacer(tmp_path/('pacer%d' % len(started)))
        server = Server(run, home, args=args, module='pacer', env={
            'INWIME_PACER_DIR': str(pacer.directory),
            'PYTHONPATH': os.pathsep.join((GUARD, TESTS, PACKAGE_PARENT))})
        server.model = active
        started.append(server)
        return server, pacer

    yield start
    for server in started:
        server.stop()


def body(server, prompt, max_tokens, **extra):
    return dict({'model': server.model, 'prompt': prompt, 'max_tokens': max_tokens}, **extra)


def post(server, payload, path='/v1/completions'):
    """One request on its own connection: (status, headers, parsed body)."""
    url = urllib.parse.urlsplit(server.url)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=60)
    try:
        connection.request('POST', path, json.dumps(payload).encode(), {
            'Content-Type': 'application/json', 'Authorization': 'Bearer %s' % server.key})
        response = connection.getresponse()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, json.loads(response.read())
    finally:
        connection.close()


def send(server, payload, path='/v1/completions'):
    """Send a request and leave its response unread: closing the connection disconnects the client."""
    url = urllib.parse.urlsplit(server.url)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=60)
    connection.request('POST', path, json.dumps(payload).encode(), {
        'Content-Type': 'application/json', 'Authorization': 'Bearer %s' % server.key})
    return connection


def admit_eventually(pool, server, pacer, payload, timeout=10):
    """Send `payload` until the server admits it (a disconnection is noticed asynchronously, so the
    first tries may still find the queue full); returns the future of its reply."""
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        before = sum(e.startswith('admitted ') for e in pacer.events())
        reply = pool.submit(post, server, payload)
        pacer.wait(lambda events: reply.done() or sum(e.startswith('admitted ') for e in events) > before)
        if not reply.done() or reply.result()[0] != 429:
            return reply
        time.sleep(0.05)
    raise AssertionError('the request was never admitted; events: %s' % pacer.events())


def text(reply):
    status, _, payload = reply
    assert status == 200, payload
    return payload['choices'][0]['text']


def test_a_full_queue_refuses_with_429_and_no_invented_retry_after(paced):
    server, pacer = paced('--queue-size', '1')
    pacer.gate('hold-prefill')
    with ThreadPoolExecutor(2) as pool:
        running = pool.submit(post, server, body(server, *LONG[:2]))
        pacer.wait_for('held prefill')
        waiting = pool.submit(post, server, body(server, *SHORT[:2]))
        pacer.admitted(2)
        status, headers, refused = post(server, body(server, *SHORT[:2]))
        assert status == 429
        assert_openai_error(refused, 'queue_full')
        assert 'retry-after' not in headers
        pacer.gate('hold-prefill', closed=False)
        assert text(running.result()) == LONG[2]
        assert text(waiting.result()) == SHORT[2]
    assert text(post(server, body(server, *SHORT[:2]))) == SHORT[2]     # room again once drained


def test_waiting_requests_run_in_arrival_order_behind_a_default_queue_of_four(paced, active):
    server, pacer = paced()
    pacer.gate('hold-prefill')
    prompts = ['t9 t4', 't2 t33', 't6 t6 t1', 't3 t5 t7 t11']
    with ThreadPoolExecutor(5) as pool:
        replies = [pool.submit(post, server, body(server, 't8', 3))]
        pacer.wait_for('held prefill')
        for count, prompt in enumerate(prompts, 2):
            replies.append(pool.submit(post, server, body(server, prompt, 3)))
            pacer.admitted(count)                # sent one after the other, admitted in that order
        assert post(server, body(server, 't7', 3))[0] == 429    # a fifth waiting request
        pacer.gate('hold-prefill', closed=False)
        texts = [text(r.result()) for r in replies]
    events = pacer.events()
    admitted = [e.split(' ', 1)[1] for e in events if e.startswith('admitted ')]
    prefilled = [e.split(' ', 1)[1] for e in events if e.startswith('prefill ')]
    assert prefilled == admitted
    for prompt, got in zip(['t8']+prompts, texts):
        reference, tokenizer = tinyqwen.frozen_greedy(Path(active), prompt, 3)
        ids = reference['generated_ids']
        assert got == tokenizer.decode(ids[:-1] if reference['stopped_on_eos'] else ids)


@pytest.mark.parametrize('stream', [False, True])
def test_a_client_that_disconnects_while_queued_leaves_the_queue_unserved(paced, stream):
    server, pacer = paced('--queue-size', '1')
    pacer.gate('hold-prefill')
    with ThreadPoolExecutor(3) as pool:
        running = pool.submit(post, server, body(server, *LONG[:2]))
        pacer.wait_for('held prefill')
        gone = send(server, body(server, 't9 t4', 3, stream=stream))
        gone_ids = pacer.admitted(2)[-1].split(' ', 1)[1]
        gone.close()
        # Its place frees while the request in progress still holds the worker.
        waiting = admit_eventually(pool, server, pacer, body(server, *SHORT[:2]))
        assert not running.done()
        pacer.gate('hold-prefill', closed=False)
        assert text(running.result()) == LONG[2]
        assert text(waiting.result()) == SHORT[2]
    prefilled = [e.split(' ', 1)[1] for e in pacer.events() if e.startswith('prefill ')]
    assert len(prefilled) == 2 and gone_ids not in prefilled


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('phase, published', [('prefill', 0), ('decode', 4)])
def test_a_disconnection_during_the_computation_stops_it_in_the_runtime(paced, phase, published, stream):
    server, pacer = paced()
    pacer.gate('hold-'+phase)
    gone = send(server, body(server, 't3 t5 t7 t11', 8, stream=stream))
    pacer.wait_for('held '+phase)
    gone.close()
    events = pacer.wait(lambda events: events[-1:] and events[-1].startswith('closed '))
    # The runtime itself raised CancelledError at its check; the session closed with the KV
    # published before the interrupted call (none for the prefill, the 4 prompt tokens in the
    # first decode): the candidate state of that call was never published.
    assert events[-2:] == ['cancelled '+phase, 'closed length=%d failed=True' % published]
    assert sum(e.startswith('decode ') for e in events) == (phase == 'decode')
    pacer.gate('hold-'+phase, closed=False)
    assert text(post(server, body(server, *SHORT[:2]))) == SHORT[2]


def test_a_cancelled_request_keeps_its_slot_until_the_worker_has_unwound_it(paced):
    server, pacer = paced('--queue-size', '0')
    pacer.gate('stall-decode')               # between two cancellation checks of the runtime
    gone = send(server, body(server, *LONG[:2]))
    pacer.wait_for('held decode')
    gone.close()
    pacer.wait_for('cancel seen decode')
    status, _, refused = post(server, body(server, *SHORT[:2]))
    assert status == 429, refused            # still computing: nothing released yet
    pacer.gate('stall-decode', closed=False)
    pacer.wait(lambda events: events[-1:] == ['closed length=4 failed=True'])
    assert text(post(server, body(server, *SHORT[:2]))) == SHORT[2]


@pytest.mark.parametrize('stream', [False, True])
def test_an_identical_retry_after_a_client_timeout_never_computes_beside_the_abandoned_request(
        paced, active, stream):
    server, pacer = paced()
    request = body(server, 't3 t5 t7 t11', 8, stream=stream)
    pacer.gate('stall-decode')
    abandoned = send(server, request)
    pacer.wait_for('held decode')
    abandoned.close()                        # the harness gives up on its own timeout ...
    with ThreadPoolExecutor(1) as pool:
        retry = pool.submit(post, server, dict(request, stream=False))   # ... and sends it again
        pacer.admitted(2)
        pacer.wait_for('cancel seen decode')
        time.sleep(0.2)
        assert sum(e.startswith('prefill ') for e in pacer.events()) == 1
        pacer.gate('stall-decode', closed=False)
        retried = text(retry.result())
    events = pacer.events()
    closed = events.index('closed length=4 failed=True')
    assert events[closed-1] == 'cancelled decode'
    [first, second] = [i for i, e in enumerate(events) if e.startswith('prefill ')]
    assert first < closed < second            # one computation at a time, the abandoned one stopped
    reference, tokenizer = tinyqwen.frozen_greedy(Path(active), 't3 t5 t7 t11', 8)
    ids = reference['generated_ids']
    assert retried == tokenizer.decode(ids[:-1] if reference['stopped_on_eos'] else ids)


def stream_events(connection):
    """The data events of an SSE response on `connection`, until the stream ends."""
    response = connection.getresponse()
    assert response.status == 200
    data = [line[5:].strip().decode() for line in response.read().splitlines() if line.startswith(b'data:')]
    connection.close()
    return [d if d == '[DONE]' else json.loads(d) for d in data]


def test_max_request_seconds_ends_requests_waiting_in_the_queue_and_in_progress(paced):
    server, pacer = paced('--max-request-seconds', '1.5')
    pacer.gate('stall-prefill')              # the worker cannot notice the cancellation yet
    with ThreadPoolExecutor(2) as pool:
        started = time.monotonic()
        running = pool.submit(post, server, body(server, *LONG[:2]))
        pacer.wait_for('held prefill')
        waiting = pool.submit(post, server, body(server, *SHORT[:2]))
        streamed = send(server, body(server, 't9 t4', 3, stream=True))
        pacer.admitted(3)
        for status, _, error in (running.result(), waiting.result()):
            assert status == 503
            assert_openai_error(error, 'request_timeout')
            assert error['error']['type'] == 'server_error'
        events = stream_events(streamed)
        assert time.monotonic()-started < 10
    assert events[-1] == '[DONE]'
    assert_openai_error(events[-2], 'request_timeout')
    pacer.gate('stall-prefill', closed=False)
    pacer.wait(lambda events: events[-2:] == ['cancelled prefill', 'closed length=0 failed=True'])
    assert sum(e.startswith('prefill ') for e in pacer.events()) == 1   # the queued ones never ran
    assert text(post(server, body(server, *SHORT[:2]))) == SHORT[2]


@pytest.mark.parametrize('args', [('--queue-size', '-1'), ('--max-request-seconds', '0'),
                                  ('--max-request-seconds', 'nan')])
def test_serve_refuses_invalid_queue_and_time_limits(run, home, active, args):
    import subprocess
    import sys
    refused = subprocess.run([sys.executable, '-m', 'inwime', 'serve', '--port', '0', *args],
                             capture_output=True, text=True, env=run.environment(), cwd=home, timeout=60)
    assert refused.returncode == 2
    assert args[0] in refused.stderr and 'listening' not in refused.stdout


def test_keepalives_and_other_requests_go_on_while_the_worker_computes(paced):
    from serving import stream
    server, pacer = paced('--keepalive-seconds', '0.1', '--queue-size', '0')
    pacer.gate('hold-decode')
    with ThreadPoolExecutor(1) as pool:
        streamed = pool.submit(stream, server, '/v1/completions', body(server, *LONG[:2], stream=True))
        pacer.wait_for('held decode')
        time.sleep(0.5)
        started = time.monotonic()
        assert server.request('GET', '/v1/models')[0] == 200
        assert post(server, body(server, *SHORT[:2]))[0] == 429
        answered = time.monotonic()-started
        pacer.gate('hold-decode', closed=False)
        s = streamed.result()
    assert answered < 1.0
    assert s.text() == LONG[2]
    first, second = [i for i, e in enumerate(s.events) if e[0] == 'data'][:2]
    during = s.events[first+1:second]         # after the first token, while the decode was held
    assert len(during) >= 4 and all(e[:2] == ('comment', 'keepalive') for e in during)

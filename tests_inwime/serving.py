"""`inwime serve` in a guarded subprocess, as the HTTP tests reach it (see conftest.py)."""
import http.client
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


class Server:
    def __init__(self, run, home, args=(), env=None, module='inwime'):
        """`module`: run as `python -m <module> serve`; `pacer` wraps the server for the queue tests."""
        self.run, self.home = run, home
        self.process = subprocess.Popen([sys.executable, '-m', module, 'serve', '--port', '0', *args],
                                        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, env=run.environment(env=env),
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


def assert_openai_error(body, code=None):
    error = body['error']
    assert isinstance(error['message'], str) and error['message']
    assert set(error) >= {'message', 'type', 'param', 'code'}
    if code is not None:
        assert error['code'] == code


class Stream:
    """One SSE response, read as it arrives: status, headers and events in order. Each event is
    (kind, value, seconds since the request was sent): ('comment', text) for `: text`, ('data',
    parsed JSON or '[DONE]')."""

    def __init__(self, status, headers, events, body=None):
        self.status, self.headers, self.events, self.body = status, headers, events, body

    @property
    def chunks(self):
        return [value for kind, value, _ in self.events if kind == 'data' and value != '[DONE]']

    def text(self, field='content'):
        """The streamed text: chat deltas (`field`) or completion text."""
        parts = []
        for chunk in self.chunks:
            for choice in chunk['choices']:
                parts.append(choice.get('text') or (choice.get('delta') or {}).get(field) or '')
        return ''.join(parts)


def stream(server, path, body, key=None, timeout=120):
    url = urllib.parse.urlsplit(server.url)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=timeout)
    started = time.monotonic()
    connection.request('POST', path, json.dumps(body).encode(), {
        'Content-Type': 'application/json', 'Authorization': 'Bearer %s' % (key or server.key)})
    response = connection.getresponse()
    headers = {k.lower(): v for k, v in response.getheaders()}
    if not headers.get('content-type', '').startswith('text/event-stream'):
        result = Stream(response.status, headers, [], json.loads(response.read()))
        connection.close()
        return result
    events, data = [], []
    while True:
        line = response.readline()
        if not line:
            break
        line = line.decode().rstrip('\r\n')
        now = time.monotonic()-started
        if line.startswith(':'):
            events.append(('comment', line[1:].strip(), now))
        elif line.startswith('data:'):
            data.append(line[5:].lstrip())
        elif line == '' and data:
            value = '\n'.join(data)
            events.append(('data', value if value == '[DONE]' else json.loads(value), now))
            data = []
    connection.close()
    return Stream(response.status, headers, events)

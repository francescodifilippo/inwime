"""`inwime serve`: an OpenAI-compatible HTTP API over the Motore, on loopback by default.

Every request needs the bearer API key. The key is generated with `secrets` at the first
start, stored in `<INWIME_HOME>/server/api-key` with mode 0600, and never printed or logged;
the server refuses to start if that file is readable or writable by anyone else. Access logs
are off.

Endpoints: `GET /v1/models`, `POST /v1/completions` and `POST /v1/chat/completions`, whole or
streamed as SSE. A request is validated and prepared (tokenized, or rendered with the chat
template) before it waits for the worker, so a refused request gets its HTTP error at once.

The computation runs on the Coordinatore's worker thread, never on the event loop, so
keepalives and other requests go on during an inference. One inference at a time; up to
`--queue-size` requests (default 4) wait in arrival order and the next one gets 429
`queue_full`. That 429 has no `Retry-After`: the server has no founded estimate of when a
place frees (the token limits of the queued requests only bound their length). A request is
withdrawn when its client disconnects, when `--max-request-seconds` passes since its arrival
(503 `request_timeout`; no limit by default) or when the server stops: a waiting one leaves the
queue at once, a running one is cancelled inside the runtime and keeps its place until the
worker has unwound it. An identical retry after a client timeout therefore waits for the
abandoned computation to stop instead of running beside it.
"""
import asyncio
import hmac
import ipaddress
import json
import os
import secrets
import socket
import stat
import sys
import time
import uuid

from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import catalog, chat, registry
from .engine import DEFAULT_QUEUE_SIZE, Coordinatore, EngineError, NotServable, cancelled
from .generation import SamplingError, client_parameters

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8642
DEFAULT_MAX_TOKENS = 16           # OpenAI's documented default for /v1/completions
DEFAULT_KEEPALIVE_SECONDS = 15.0  # SSE comment interval while no model output is ready
KEY_BYTES = 32


class ServeError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.exit_status = status


# -- network and key ---------------------------------------------------------------------------

def is_loopback(host):
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def key_path():
    return registry.home()/'server'/'api-key'


def create_key(path):
    """Write a new key to a private temporary file, then link it into place: the key file appears
    complete or not at all, and of two servers starting together only one key wins."""
    key = 'inwime-' + secrets.token_urlsafe(KEY_BYTES)
    tmp = path.with_name('.%s.%d.tmp' % (path.name, os.getpid()))
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(key+'\n')
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)
    except FileExistsError:
        pass                                   # another server created it first: read that one
    finally:
        tmp.unlink(missing_ok=True)


def api_key(path=None):
    """The server key: read from a private file, or generated into one at the first start."""
    path = key_path() if path is None else path
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not path.exists() and not path.is_symlink():
            create_key(path)
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    except OSError as error:
        raise ServeError(1, 'cannot create or open the API key file %s: %s'
                         % (path, error.strerror or error)) from error
    with os.fdopen(fd) as f:
        st = os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid():
            raise ServeError(1, 'the API key file %s must be a regular file owned by you' % path)
        if st.st_mode & 0o077:
            raise ServeError(1, 'the API key file %s is accessible to other users (mode %o): '
                             'run `chmod 600` on it, or delete it to generate a new key'
                             % (path, stat.S_IMODE(st.st_mode)))
        key = f.read(4096).strip()
    if len(key) < KEY_BYTES:
        raise ServeError(1, 'the API key file %s does not hold a valid key: delete it to generate a new one'
                         % path)
    return key


# -- OpenAI-style errors -----------------------------------------------------------------------

def error_response(status, message, *, type='invalid_request_error', code=None, param=None, headers=None):
    return JSONResponse({'error': {'message': message, 'type': type, 'param': param, 'code': code}},
                        status_code=status, headers=headers)


def invalid(message, param=None, code=None):
    return EngineError(400, message, param=param, code=code)


# Parameters this version accepts only at their neutral values: one choice, no logprobs, no logit
# bias, no presence or frequency penalty.
FIXED = {'n': (None, 1), 'best_of': (None, 1), 'echo': (None, False), 'logprobs': (None,),
         'suffix': (None,), 'presence_penalty': (None, 0), 'frequency_penalty': (None, 0),
         'logit_bias': (None,)}


def token_limit(body, default=DEFAULT_MAX_TOKENS):
    """`max_tokens` or its newer name `max_completion_tokens`; `default` if neither."""
    given = {name: body[name] for name in ('max_tokens', 'max_completion_tokens') if body.get(name) is not None}
    if len(set(map(repr, given.values()))) > 1:
        raise invalid('max_tokens and max_completion_tokens differ', 'max_completion_tokens')
    if not given:
        return default
    name, value = next(iter(given.items()))
    if type(value) is not int or value < 1:
        raise invalid('%s must be a positive integer' % name, name)
    return value


def check_model(body, model_id):
    if not isinstance(body, dict):
        raise invalid('the request body must be a JSON object')
    model = body.get('model')
    if not isinstance(model, str) or not model:
        raise invalid('model is required', 'model')
    if model != model_id:
        raise EngineError(404, 'the model %r does not exist: the active model is %r' % (model, model_id),
                          code='model_not_found', param='model')


def check_fixed(body, fixed):
    for name, allowed in fixed.items():
        value = body.get(name)
        if not any(value is a or (type(value) in (int, float) and type(a) is int and value == a)
                   for a in allowed):
            raise invalid('%s=%s is not supported yet: this version serves one choice, without logprobs, '
                          'logit bias or presence and frequency penalties' % (name, json.dumps(value)), name,
                          'unsupported_parameter')


def sampling_parameters(body):
    try:
        return client_parameters(body)
    except SamplingError as error:
        raise invalid(str(error), error.param) from error


class Request:
    """A validated request: the arguments of its preparation and how to answer it."""

    def __init__(self, body, arguments):
        self.arguments = arguments
        if not (body.get('stream') is None or type(body.get('stream')) is bool):
            raise invalid('stream must be a boolean', 'stream')
        self.stream = body.get('stream') is True
        options = body.get('stream_options')
        if options is not None and not isinstance(options, dict):
            raise invalid('stream_options must be an object', 'stream_options')
        self.include_usage = self.stream and (options or {}).get('include_usage') is True
        self.diagnostics = body.get('inwime_diagnostics') is True


def completion_request(body, model_id):
    check_model(body, model_id)
    prompt = body.get('prompt')
    if not isinstance(prompt, str):
        raise invalid('prompt must be a single string in this version', 'prompt')
    max_tokens = token_limit(body)
    check_fixed(body, FIXED)
    return Request(body, (prompt, max_tokens, sampling_parameters(body)))


CHAT_FIXED = {k: v for k, v in FIXED.items() if k not in ('best_of', 'echo', 'suffix', 'logprobs')}
CHAT_FIXED.update({'logprobs': (None, False), 'top_logprobs': (None,), 'tool_choice': (None, 'auto')})


def chat_request(body, model_id):
    check_model(body, model_id)
    try:
        messages = chat.messages(body.get('messages'))
    except ValueError as error:
        raise invalid(str(error), 'messages') from error
    tools = body.get('tools')
    if tools is not None and not (isinstance(tools, list) and all(isinstance(t, dict) for t in tools)):
        raise invalid('tools must be an array of objects', 'tools')
    max_tokens = token_limit(body, None)
    check_fixed(body, CHAT_FIXED)
    return Request(body, (messages, tools or None, max_tokens, sampling_parameters(body)))


class JobStream(StreamingResponse):
    """A streamed response that withdraws its job however the response ends: finished, client
    disconnected (Starlette then cancels the stream, even before it starts) or server stopping."""

    def __init__(self, job, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.job = job

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            self.job.cancel()


async def disconnected(request):
    """Return when the client of a request whose body has been read closes its connection."""
    while (await request.receive())['type'] != 'http.disconnect':
        pass


# -- application -------------------------------------------------------------------------------

def timed_out(limit):
    return EngineError(503, 'the request took longer than the server limit of %g seconds '
                       '(--max-request-seconds), waiting and computing included' % limit,
                       type='server_error', code='request_timeout')


def build_app(coordinatore, key, keepalive_seconds=DEFAULT_KEEPALIVE_SECONDS, max_request_seconds=None):
    """`max_request_seconds`: None for no time limit; else each request, from its arrival, is
    cancelled when the limit passes, whether it still waits in the queue or is computing."""
    expected = ('Bearer '+key).encode()

    def remaining(deadline):
        return None if deadline is None else max(0., deadline-time.monotonic())

    def authorized(request):
        given = request.headers.get('authorization', '').encode('latin-1', 'replace')
        return hmac.compare_digest(given, expected)

    def unauthorized():
        return error_response(401, 'missing or invalid API key: send it as "Authorization: Bearer <key>"',
                              code='invalid_api_key', headers={'WWW-Authenticate': 'Bearer'})

    async def models(request):
        if not authorized(request):
            return unauthorized()
        return JSONResponse({'object': 'list', 'data': [
            {'id': coordinatore.model_id, 'object': 'model', 'created': started, 'owned_by': 'inwime'}]})

    async def read_json(request):
        try:
            return json.loads(await request.body())
        except (ValueError, UnicodeDecodeError):
            raise EngineError(400, 'the request body is not valid JSON')

    def failure(error):
        """An inference failure as an OpenAI error object and HTTP status."""
        if isinstance(error, EngineError):
            return error.status, {'message': error.message, 'type': error.type, 'param': error.param,
                                  'code': error.code}
        # The exception text may hold paths: only its type reaches the client and the log.
        name = catalog.printable(type(error).__name__)
        sys.stderr.write('inwime serve: inference failed: %s\n' % name)
        return 500, {'message': 'inference failed (%s); see the server log' % name, 'type': 'server_error',
                     'param': None, 'code': 'inference_failed'}

    def failure_response(error):
        status, error = failure(error)
        return error_response(status, error['message'], type=error['type'], code=error['code'],
                              param=error['param'])

    def usage(result):
        return {'prompt_tokens': result['prompt_tokens'], 'completion_tokens': result['completion_tokens'],
                'total_tokens': result['prompt_tokens']+result['completion_tokens']}

    class Reply:
        """The OpenAI objects of one response, chat or text completion, whole or in chunks."""

        def __init__(self, is_chat, request):
            self.chat, self.request = is_chat, request
            self.id = ('chatcmpl-' if is_chat else 'cmpl-')+uuid.uuid4().hex
            self.created, self.first = int(time.time()), True

        def whole(self, result):
            if self.chat:
                choice = {'index': 0, 'message': {'role': 'assistant', 'content': result['text']},
                          'logprobs': None, 'finish_reason': result['finish_reason']}
            else:
                choice = {'text': result['text'], 'index': 0, 'logprobs': None,
                          'finish_reason': result['finish_reason']}
            reply = {'id': self.id, 'object': 'chat.completion' if self.chat else 'text_completion',
                     'created': self.created, 'model': coordinatore.model_id, 'choices': [choice],
                     'usage': usage(result)}
            if self.request.diagnostics:
                reply['inwime'] = result['diagnostics']
            return reply

        def chunk(self, text, finish_reason=None, choices=True):
            if self.chat:
                delta = {'content': text} if text else {}
                if self.first:
                    delta = {'role': 'assistant', **delta}
                choice = {'index': 0, 'delta': delta, 'logprobs': None, 'finish_reason': finish_reason}
            else:
                choice = {'text': text, 'index': 0, 'logprobs': None, 'finish_reason': finish_reason}
            self.first = False
            chunk = {'id': self.id, 'object': 'chat.completion.chunk' if self.chat else 'text_completion',
                     'created': self.created, 'model': coordinatore.model_id,
                     'choices': [choice] if choices else []}
            if self.request.include_usage:
                chunk['usage'] = None
            return chunk

    class Output:
        """Text the worker emits, handed to the event loop of the stream that waits for it."""

        def __init__(self):
            self.loop, self.queue = asyncio.get_running_loop(), asyncio.Queue()

        def __call__(self, item):
            try:
                self.loop.call_soon_threadsafe(self.queue.put_nowait, item)
            except RuntimeError:        # the event loop is gone: the server is shutting down
                pass

    def event(value):
        return 'data: %s\n\n' % json.dumps(value, separators=(',', ':'))

    async def sse(reply, job, output, deadline):
        """Headers and a first keepalive go out at once, before the request reaches the worker;
        then one `: keepalive` comment per interval without model output. Comments carry no data:
        a client never mistakes them for progress of the model."""
        done = object()
        yield ': keepalive\n\n'
        job.future.add_done_callback(lambda f: output(done))
        while True:
            left = remaining(deadline)
            try:
                item = await asyncio.wait_for(output.queue.get(), min(keepalive_seconds, left)
                                              if left is not None else keepalive_seconds)
            except TimeoutError:
                if remaining(deadline) == 0:      # JobStream withdraws the job when the stream ends
                    yield event({'error': failure(timed_out(max_request_seconds))[1]})
                    yield 'data: [DONE]\n\n'
                    return
                yield ': keepalive\n\n'
                continue
            if item is done:
                break
            yield event(reply.chunk(item))
        try:
            if job.future.cancelled():                # withdrawn from the queue by a server stop
                raise cancelled()
            result = job.future.result()
        except Exception as error:
            yield event({'error': failure(error)[1]})
            yield 'data: [DONE]\n\n'
            return
        last = reply.chunk('', result['finish_reason'])
        if reply.request.diagnostics:
            last['inwime'] = result['diagnostics']
        yield event(last)
        if reply.request.include_usage:
            yield event(dict(reply.chunk('', choices=False), usage=usage(result)))
        yield 'data: [DONE]\n\n'

    async def serve_request(request, is_chat):
        if not authorized(request):
            return unauthorized()
        deadline = None if max_request_seconds is None else time.monotonic()+max_request_seconds
        try:
            body = await read_json(request)
            parsed = (chat_request if is_chat else completion_request)(body, coordinatore.model_id)
            prepare = coordinatore.prepare_chat if is_chat else coordinatore.prepare_completion
            prepared = await asyncio.to_thread(prepare, *parsed.arguments)
            output = Output() if parsed.stream else None
            job = coordinatore.generate(prepared, output)
        except Exception as error:
            return failure_response(error)
        reply = Reply(is_chat, parsed)
        if parsed.stream:
            return JobStream(job, sse(reply, job, output, deadline), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
        computed = asyncio.wrap_future(job.future)
        gone = asyncio.ensure_future(disconnected(request))
        finished = left = False
        try:
            await asyncio.wait((computed, gone), timeout=remaining(deadline),
                               return_when=asyncio.FIRST_COMPLETED)
            finished, left = computed.done(), gone.done()
        finally:
            gone.cancel()
            if not finished:
                # The time limit passed, the client left or the server is stopping. Decided before
                # the cancellation, which may complete `computed` at once.
                job.cancel()
        if not finished:
            if left:
                return Response(status_code=499)      # nobody reads it: the client is gone
            return failure_response(timed_out(max_request_seconds))
        if computed.cancelled():                      # withdrawn from the queue by a server stop
            return failure_response(cancelled())
        try:
            result = computed.result()
        except Exception as error:
            return failure_response(error)
        return JSONResponse(reply.whole(result))

    async def completions(request):
        return await serve_request(request, is_chat=False)

    async def chat_completions(request):
        return await serve_request(request, is_chat=True)

    async def http_error(request, exc):
        if not authorized(request):            # nothing about the routes is disclosed without the key
            return unauthorized()
        if exc.status_code == 404:
            return error_response(404, 'unknown endpoint %s' % request.url.path, code='not_found')
        if exc.status_code == 405:
            return error_response(405, 'method %s not allowed on %s' % (request.method, request.url.path),
                                  code='method_not_allowed')
        return error_response(exc.status_code, exc.detail)

    async def server_error(request, exc):
        sys.stderr.write('inwime serve: internal error: %s\n' % catalog.printable(type(exc).__name__))
        return error_response(500, 'internal server error', type='server_error')

    started = int(time.time())
    return Starlette(routes=[Route('/v1/models', models, methods=['GET']),
                             Route('/v1/completions', completions, methods=['POST']),
                             Route('/v1/chat/completions', chat_completions, methods=['POST'])],
                     exception_handlers={HTTPException: http_error, Exception: server_error})


def bind(host, port):
    host = '127.0.0.1' if host == 'localhost' else host
    sock = None
    try:
        family, _, _, _, address = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM,
                                                      flags=socket.AI_PASSIVE)[0]
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(address)
    except OSError as error:
        if sock is not None:
            sock.close()
        raise ServeError(1, 'cannot listen on %s port %d: %s' % (host, port, error.strerror or error)) from error
    sock.listen(64)
    sock.set_inheritable(False)
    return sock


def serve(host=DEFAULT_HOST, port=DEFAULT_PORT, allow_non_loopback=False, out=sys.stdout,
          keepalive_seconds=DEFAULT_KEEPALIVE_SECONDS, queue_size=DEFAULT_QUEUE_SIZE, max_request_seconds=None):
    import uvicorn
    if not is_loopback(host) and not allow_non_loopback:
        raise ServeError(2, '%s is not a loopback address: the server listens on loopback only unless '
                         '--allow-non-loopback is given' % catalog.printable(host))
    record = registry.active_record()
    if record is None:
        raise ServeError(3, 'no active model: choose one with `inwime models use ID` (see `inwime models list`)')
    manifest = None
    if record.get('source') == 'installed':
        manifest = registry.read_json(registry.manifest_path(*record['id'].rsplit('@', 1)))
        if not registry.intact(manifest):
            raise ServeError(13, 'the active model %s is no longer a verified, complete install: re-run '
                             '`inwime models install %s`' % (record['id'], record['id']))
    key = api_key()
    try:
        coordinatore = Coordinatore(record['id'], record['path'], manifest, queue_size)
    except NotServable as error:
        raise ServeError(13, 'the active model %s cannot be served: %s' % (record['id'], error)) from error
    try:
        sock = bind(host, port)
        address = sock.getsockname()
        shown = '[%s]' % address[0] if sock.family == socket.AF_INET6 else address[0]
        out.write(catalog.printable('inwime serve: model %s, kernel %s' % (coordinatore.model_id,
                                                                         coordinatore.math.name)) + '\n')
        out.write('start-up SHA256 check of the weights: %d bytes read in %.1f s\n'
                  % (coordinatore.integrity_read_bytes, coordinatore.integrity_read_ns/1e9))
        out.write(catalog.printable('API key file: %s (send it as "Authorization: Bearer <key>")'
                                    % key_path()) + '\n')
        if not is_loopback(host):
            out.write('warning: listening on a non-loopback address; anyone who can reach it and holds the '
                      'key can use the server\n')
        out.write('listening on http://%s:%d\n' % (shown, address[1]))
        out.flush()
        config = uvicorn.Config(build_app(coordinatore, key, keepalive_seconds, max_request_seconds),
                                log_level='warning', access_log=False,
                                lifespan='off')
        uvicorn.Server(config).run(sockets=[sock])
    finally:
        coordinatore.close()
    return 0

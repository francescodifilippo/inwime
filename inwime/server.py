"""`inwime serve`: an OpenAI-compatible HTTP API over the Motore, on loopback by default.

Every request needs the bearer API key. The key is generated with `secrets` at the first
start, stored in `<INWIME_HOME>/server/api-key` with mode 0600, and never printed or logged;
the server refuses to start if that file is readable or writable by anyone else. Access logs
are off.

Tracer bullet (#37): `GET /v1/models` and non-stream greedy `POST /v1/completions`. Sampling,
streaming and chat come with #39; the queue limit and cancellation with #40.
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
from starlette.responses import JSONResponse
from starlette.routing import Route

from . import catalog, registry
from .engine import Coordinatore, EngineError, NotServable

DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8642
DEFAULT_MAX_TOKENS = 16           # OpenAI's documented default for /v1/completions
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


# Parameters this version accepts only at their greedy, single-choice, non-stream values.
FIXED = {'stream': (None, False), 'temperature': (None, 0), 'top_p': (None, 1), 'n': (None, 1),
         'best_of': (None, 1), 'echo': (None, False), 'logprobs': (None,), 'stop': (None,),
         'suffix': (None,), 'presence_penalty': (None, 0), 'frequency_penalty': (None, 0),
         'logit_bias': (None,), 'top_k': (None,), 'repetition_penalty': (None, 1)}


def token_limit(body):
    """`max_tokens` or its newer name `max_completion_tokens`; OpenAI's default 16 if neither."""
    given = {name: body[name] for name in ('max_tokens', 'max_completion_tokens') if body.get(name) is not None}
    if len(set(map(repr, given.values()))) > 1:
        raise invalid('max_tokens and max_completion_tokens differ', 'max_completion_tokens')
    name, value = next(iter(given.items()), ('max_tokens', DEFAULT_MAX_TOKENS))
    if type(value) is not int or value < 1:
        raise invalid('%s must be a positive integer' % name, name)
    return value


def completion_request(body, model_id):
    if not isinstance(body, dict):
        raise invalid('the request body must be a JSON object')
    model = body.get('model')
    if not isinstance(model, str) or not model:
        raise invalid('model is required', 'model')
    if model != model_id:
        raise EngineError(404, 'the model %r does not exist: the active model is %r' % (model, model_id),
                          code='model_not_found', param='model')
    prompt = body.get('prompt')
    if not isinstance(prompt, str):
        raise invalid('prompt must be a single string in this version', 'prompt')
    max_tokens = token_limit(body)
    for name, allowed in FIXED.items():
        value = body.get(name)
        if not any(value is a or (type(value) in (int, float) and type(a) is int and value == a)
                   for a in allowed):
            raise invalid('%s=%s is not supported yet: this version serves greedy, single-choice, non-stream '
                          'completions only' % (name, json.dumps(value)), name, 'unsupported_parameter')
    return prompt, max_tokens


# -- application -------------------------------------------------------------------------------

def build_app(coordinatore, key):
    expected = ('Bearer '+key).encode()

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

    async def completions(request):
        if not authorized(request):
            return unauthorized()
        try:
            body = json.loads(await request.body())
        except (ValueError, UnicodeDecodeError):
            return error_response(400, 'the request body is not valid JSON')
        try:
            prompt, max_tokens = completion_request(body, coordinatore.model_id)
            result = await asyncio.wrap_future(coordinatore.complete(prompt, max_tokens))
        except EngineError as error:
            return error_response(error.status, error.message, type=error.type, code=error.code,
                                  param=error.param)
        except Exception as error:
            # The exception text may hold paths: only its type reaches the client and the log.
            name = catalog.printable(type(error).__name__)
            sys.stderr.write('inwime serve: inference failed: %s\n' % name)
            return error_response(500, 'inference failed (%s); see the server log' % name,
                                  type='server_error', code='inference_failed')
        return JSONResponse({
            'id': 'cmpl-'+uuid.uuid4().hex, 'object': 'text_completion', 'created': int(time.time()),
            'model': coordinatore.model_id,
            'choices': [{'text': result['text'], 'index': 0, 'logprobs': None,
                         'finish_reason': result['finish_reason']}],
            'usage': {'prompt_tokens': result['prompt_tokens'], 'completion_tokens': result['completion_tokens'],
                      'total_tokens': result['prompt_tokens']+result['completion_tokens']}})

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
                             Route('/v1/completions', completions, methods=['POST'])],
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


def serve(host=DEFAULT_HOST, port=DEFAULT_PORT, allow_non_loopback=False, out=sys.stdout):
    import uvicorn
    if not is_loopback(host) and not allow_non_loopback:
        raise ServeError(2, '%s is not a loopback address: the server listens on loopback only unless '
                         '--allow-non-loopback is given' % catalog.printable(host))
    record = registry.active_record()
    if record is None:
        raise ServeError(3, 'no active model: choose one with `inwime models use ID` (see `inwime models list`)')
    key = api_key()
    try:
        coordinatore = Coordinatore(record['id'], record['path'])
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
        config = uvicorn.Config(build_app(coordinatore, key), log_level='warning', access_log=False,
                                lifespan='off')
        uvicorn.Server(config).run(sockets=[sock])
    finally:
        coordinatore.close()
    return 0

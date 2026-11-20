"""A local, in-process stand-in for the Hugging Face Hub HTTP API (loopback only).

Implements the routes `huggingface_hub` uses for the catalog: model listing with Link-header
pagination, revision info with blobs, and file resolution with byte ranges. Access control
mirrors the real Hub's X-Error-Code responses. Every request is recorded; requests to routes
the fake does not implement are recorded as unexpected and answered with 418.
"""
import hashlib
import json
import math
import struct
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlencode, urlsplit


DTYPE_BYTES = {'BF16': 2, 'F16': 2, 'F32': 4, 'I8': 1}


def safetensors_bytes(tensors, dtype='BF16'):
    """Real safetensors serialization of zero-filled tensors: {name: shape}."""
    header, offset = {}, 0
    for name, shape in tensors.items():
        size = math.prod(shape)*DTYPE_BYTES[dtype]
        header[name] = {'dtype': dtype, 'shape': list(shape), 'data_offsets': [offset, offset+size]}
        offset += size
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' '*(-len(raw) % 8)
    return struct.pack('<Q', len(raw)) + raw + bytes(offset)


@dataclass
class Repo:
    id: str
    revisions: dict                       # sha -> {filename: bytes}
    branches: dict                        # name -> sha
    gated: object = False                 # False, 'auto' or 'manual'
    granted: set = field(default_factory=set)   # tokens that accepted the licence
    license: str | None = 'apache-2.0'
    tags: list = field(default_factory=list)
    pipeline_tag: str | None = 'text-generation'
    library_name: str | None = 'transformers'
    downloads: int = 0
    likes: int = 0
    created_at: str = '2026-01-01T00:00:00.000Z'
    parameters: dict | None = None        # safetensors summary {dtype: count}
    reported_sizes: dict = field(default_factory=dict)   # filename -> size the API claims

    def resolve(self, revision):
        if revision in self.revisions:
            return revision
        return self.branches.get(revision)


class FakeHub:
    def __init__(self, page_size=2):
        self.repos = {}
        self.page_size = page_size
        self.requests = []
        self.unexpected = []
        self.throttle = {}                # path prefix -> remaining 429 answers
        self.lock = threading.Lock()
        hub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                hub.handle(self, head=False)

            def do_HEAD(self):
                hub.handle(self, head=True)

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.endpoint = 'http://127.0.0.1:%d' % self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    def add(self, repo):
        self.repos[repo.id] = repo
        return repo

    def rate_limit(self, prefix, times, retry_after='0'):
        self.throttle[prefix] = (times, retry_after)

    # -- request handling ------------------------------------------------------------------

    def handle(self, request, head):
        url = urlsplit(request.path)
        path, query = unquote(url.path), parse_qs(url.query)
        auth = request.headers.get('Authorization')
        token = auth[len('Bearer '):] if auth and auth.startswith('Bearer ') else None
        with self.lock:
            self.requests.append({'method': request.command, 'path': path, 'query': query,
                                  'range': request.headers.get('Range'), 'token': token})
            for prefix, (times, retry_after) in list(self.throttle.items()):
                if path.startswith(prefix) and times > 0:
                    self.throttle[prefix] = (times-1, retry_after)
                    return self.send(request, 429, {'error': 'rate limited'}, head,
                                     headers={'Retry-After': retry_after})
        try:
            if path == '/api/models':
                return self.list_models(request, query, head)
            if path.startswith('/api/models/') and '/revision/' in path:
                repo_id, revision = path[len('/api/models/'):].split('/revision/', 1)
                return self.model_info(request, repo_id, revision, token, head)
            parts = path.lstrip('/').split('/')
            if len(parts) >= 5 and parts[2] == 'resolve':
                return self.resolve(request, '/'.join(parts[:2]), parts[3], '/'.join(parts[4:]),
                                    token, head)
        except Exception as error:      # a bug in the fake must not look like a Hub answer
            self.unexpected.append((request.command, request.path, repr(error)))
            return self.send(request, 418, {'error': 'fake hub failure'}, head)
        self.unexpected.append((request.command, request.path, 'unimplemented route'))
        return self.send(request, 418, {'error': 'unimplemented route'}, head)

    def send(self, request, status, body, head, headers=(), error_code=None):
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()
        request.send_response(status)
        request.send_header('Content-Length', str(len(raw)))
        request.send_header('Content-Type', 'application/octet-stream' if isinstance(body, bytes)
                            else 'application/json')
        if error_code:
            request.send_header('X-Error-Code', error_code)
            request.send_header('X-Error-Message', error_code)
        for name, value in dict(headers).items():
            request.send_header(name, value)
        request.end_headers()
        if not head:
            request.wfile.write(raw)

    def not_found(self, request, token, head):
        # The real Hub answers 401 without a token, so as not to reveal private repositories.
        return self.send(request, 404 if token else 401, {'error': 'Repository not found'}, head,
                         error_code='RepoNotFound')

    def summary(self, repo):
        sha = repo.branches['main']
        files = repo.revisions[sha]
        item = {'id': repo.id, '_id': repo.id, 'modelId': repo.id, 'author': repo.id.split('/')[0],
                'sha': sha, 'private': False, 'disabled': False, 'gated': repo.gated,
                'downloads': repo.downloads, 'likes': repo.likes, 'tags': list(repo.tags),
                'pipeline_tag': repo.pipeline_tag, 'library_name': repo.library_name,
                'createdAt': repo.created_at, 'lastModified': repo.created_at,
                'cardData': {'license': repo.license} if repo.license else {}}
        if repo.parameters:
            item['safetensors'] = {'parameters': dict(repo.parameters),
                                   'total': sum(repo.parameters.values())}
        item['siblings'] = [{'rfilename': name} for name in sorted(files)]
        return item

    def list_models(self, request, query, head):
        found = list(self.repos.values())
        if 'search' in query:
            found = [r for r in found if query['search'][0].lower() in r.id.lower()]
        if 'author' in query:
            found = [r for r in found if r.id.split('/')[0] == query['author'][0]]
        for tag in query.get('filter', []):
            found = [r for r in found if tag in r.tags]
        if 'pipeline_tag' in query:
            found = [r for r in found if r.pipeline_tag == query['pipeline_tag'][0]]
        if 'num_parameters' in query:
            bounds = dict(part.split(':') for part in query['num_parameters'][0].split(','))
            scale = {'K': 1e3, 'M': 1e6, 'B': 1e9, 'T': 1e12}

            def count(text):
                return float(text[:-1])*scale[text[-1]] if text[-1] in scale else float(text)
            low, high = count(bounds.get('min', '0')), count(bounds.get('max', '1e30'))
            found = [r for r in found if r.parameters and low <= sum(r.parameters.values()) <= high]
        key = {'downloads': lambda r: r.downloads, 'likes': lambda r: r.likes,
               'trendingScore': lambda r: r.likes, 'createdAt': lambda r: r.created_at,
               'lastModified': lambda r: r.created_at}.get(query.get('sort', ['downloads'])[0])
        found.sort(key=lambda r: (key(r), r.id), reverse=True)
        limit = int(query.get('limit', ['1000'])[0])
        cursor = int(query.get('cursor', ['0'])[0])
        size = min(limit, self.page_size)
        page = found[cursor:cursor+size]
        headers = {}
        if cursor+size < len(found):
            following = {k: v for k, v in query.items() if k != 'cursor'}
            following['cursor'] = [str(cursor+size)]
            headers['Link'] = '<%s/api/models?%s>; rel="next"' % (self.endpoint,
                                                                 urlencode(following, doseq=True))
        return self.send(request, 200, [self.summary(r) for r in page], head, headers=headers)

    def model_info(self, request, repo_id, revision, token, head):
        repo = self.repos.get(repo_id)
        if repo is None:
            return self.not_found(request, token, head)
        sha = repo.resolve(revision)
        if sha is None:
            return self.send(request, 404, {'error': 'Invalid rev id'}, head, error_code='RevisionNotFound')
        info = self.summary(repo)
        info['sha'] = sha
        siblings = []
        for name, data in sorted(repo.revisions[sha].items()):
            entry = {'rfilename': name, 'size': repo.reported_sizes.get(name, len(data)),
                     'blobId': hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()}
            if name.endswith(('.safetensors', '.bin', '.gguf', '.pt')):
                entry['lfs'] = {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                                'pointerSize': 134}
            siblings.append(entry)
        info['siblings'] = siblings
        return self.send(request, 200, info, head)

    def resolve(self, request, repo_id, revision, filename, token, head):
        repo = self.repos.get(repo_id)
        if repo is None:
            return self.not_found(request, token, head)
        sha = repo.resolve(revision)
        if sha is None:
            return self.send(request, 404, {'error': 'Invalid rev id'}, head, error_code='RevisionNotFound')
        if repo.gated:
            if token is None:
                return self.send(request, 401, {'error': 'gated, authenticate'}, head, error_code='GatedRepo')
            if token not in repo.granted:
                return self.send(request, 403, {'error': 'not in authorized list'}, head,
                                 error_code='GatedRepo')
        files = repo.revisions[sha]
        if filename not in files:
            return self.send(request, 404, {'error': 'Entry not found'}, head, error_code='EntryNotFound')
        data = files[filename]
        headers = {'X-Repo-Commit': sha, 'ETag': '"%s"' % hashlib.sha256(data).hexdigest()}
        span = request.headers.get('Range')
        if span:
            first, _, last = span[len('bytes='):].partition('-')
            first, last = int(first), min(int(last), len(data)-1)
            headers['Content-Range'] = 'bytes %d-%d/%d' % (first, last, len(data))
            return self.send(request, 206, data[first:last+1], head, headers=headers)
        return self.send(request, 200, data, head, headers=headers)

    def served_bytes(self, suffix):
        """Upper bound of payload bytes requested for files ending with `suffix`."""
        total = 0
        for r in self.requests:
            if '/resolve/' not in r['path'] or not r['path'].endswith(suffix):
                continue
            if r['range'] is None:
                return math.inf
            first, _, last = r['range'][len('bytes='):].partition('-')
            total += int(last)-int(first)+1
        return total

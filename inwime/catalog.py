"""Hugging Face catalog: search the Hub and describe a Revisione without downloading weights.

Reads only through `huggingface_hub` (pinned in pyproject), at the endpoint it is configured
for (`HF_ENDPOINT`), with its standard authentication (`hf auth login`, `HF_TOKEN`). The catalog
keeps nothing on disk: small metadata files are fetched into memory, safetensors headers through
byte ranges, and the Hugging Face cache is never used (the adapter check reads a copy of
config.json from a temporary directory removed at once). (`huggingface_hub` itself, unless
HF_HUB_DISABLE_TELEMETRY=1, caches its agent-harness registry in HF_HOME.) InWiMe only checks
whether a token is configured, to word access errors; it never prints, logs or stores one.
"""
import contextlib
import json
import math
import os
import re
import time

from . import capability, evidence

SCHEMA_INFO = 'inwime-model-info/1'
SCHEMA_SEARCH = 'inwime-model-search/1'
SORTS = {'downloads': 'downloads', 'likes': 'likes', 'trending': 'trending_score',
         'created': 'created_at', 'modified': 'last_modified'}
MAX_PAGE_SIZE = 100
PARAMETER_COUNT = re.compile(r'^\d+(\.\d+)?[KMBT]?$')
METADATA_FILE_LIMIT = 16*2**20            # bound for any metadata file read into memory
RATE_LIMIT_RETRIES = 3                    # extra attempts after an HTTP 429
RATE_LIMIT_MAX_WAIT = 60                  # seconds; a longer announced wait is reported, not slept


class CatalogError(Exception):
    """A catalog failure the user can act on; `kind` selects the exit status."""

    EXIT = {'usage': 2, 'not_found': 3, 'revision_not_found': 3, 'auth_required': 4,
            'access_not_granted': 5, 'rate_limited': 6, 'offline': 7, 'hub_error': 7,
            'metadata_bound': 8}

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind

    @property
    def exit_status(self):
        return self.EXIT[self.kind]


def endpoint():
    """The Hub endpoint `huggingface_hub` is configured for, read at call time."""
    from huggingface_hub import constants
    return os.environ.get('HF_ENDPOINT', constants.ENDPOINT).rstrip('/') or constants.ENDPOINT


def public_endpoint():
    """The endpoint as shown to the user: any user:password@ part is removed."""
    from urllib.parse import urlsplit, urlunsplit
    parts = urlsplit(endpoint())
    host = parts.hostname or ''
    if ':' in host:
        host = '[%s]' % host
    if parts.port is not None:
        host += ':%d' % parts.port
    return urlunsplit(parts._replace(netloc=host))


def retry_after(headers):
    """Seconds the Hub asks to wait (Retry-After or the IETF RateLimit header), or None."""
    from huggingface_hub.utils import parse_ratelimit_headers
    limit = parse_ratelimit_headers(headers)
    if limit is not None and limit.remaining == 0:
        return limit.reset_in_seconds
    value = headers.get('Retry-After', '').strip()
    return int(value) if value.isdigit() else None


def call(function, *args, retries=RATE_LIMIT_RETRIES, **kwargs):
    """One Hub request; an HTTP 429 is retried after the announced, bounded wait."""
    from huggingface_hub.errors import HfHubHTTPError
    for attempt in range(retries+1):
        try:
            return function(*args, **kwargs)
        except HfHubHTTPError as error:
            if error.response is None or error.response.status_code != 429:
                raise
            wait = retry_after(error.response.headers)
            if attempt == retries or (wait or 0) > RATE_LIMIT_MAX_WAIT:
                raise CatalogError('rate_limited', 'the Hub rate limit was reached (HTTP 429)%s; retry later'
                                   % ('' if wait is None else ', reset in %d s' % wait)) from None
            time.sleep(2**attempt if wait is None else wait)


@contextlib.contextmanager
def translated(repo_id=None):
    """Hub failures as CatalogError, with what to do next. Messages never carry a token."""
    import httpx2
    from huggingface_hub import errors, get_token
    page = '%s/%s' % (public_endpoint(), repo_id) if repo_id else public_endpoint()
    repo_id = repo_id or 'the catalog'
    try:
        yield
    except errors.GatedRepoError:
        if get_token() is None:
            raise CatalogError('auth_required', '%s is gated: authenticate with `hf auth login` or the '
                               'HF_TOKEN environment variable, then accept its licence at %s'
                               % (repo_id, page)) from None
        raise CatalogError('access_not_granted', 'your Hugging Face account has no access to gated %s: '
                           'accept its licence at %s (some owners approve requests manually)'
                           % (repo_id, page)) from None
    except errors.RevisionNotFoundError:
        raise CatalogError('revision_not_found', 'revision not found in %s' % repo_id) from None
    except errors.RepositoryNotFoundError:
        raise CatalogError('not_found', 'repository %s not found (private repositories need an '
                           'authenticated account with access)' % repo_id) from None
    except errors.DisabledRepoError:
        raise CatalogError('not_found', 'repository %s is disabled on the Hub' % repo_id) from None
    except errors.OfflineModeIsEnabled:
        raise CatalogError('offline', 'HF_HUB_OFFLINE is set: the catalog needs the Hub at %s'
                           % public_endpoint()) from None
    except errors.HfHubHTTPError as error:
        status = error.response.status_code if error.response is not None else None
        raise CatalogError('hub_error', 'the Hub answered HTTP %s for %s' % (status, repo_id)) from None
    except httpx2.TransportError as error:     # its text may quote URLs: only the type is shown
        raise CatalogError('hub_error', 'cannot reach the Hub at %s (%s)'
                           % (public_endpoint(), type(error).__name__)) from None


def api():
    from huggingface_hub import HfApi
    return HfApi(endpoint=endpoint())


def parse_spec(spec):
    """`repo[@revision]` -> (repo_id, revision); revision defaults to `main`."""
    from huggingface_hub.utils import HFValidationError, validate_repo_id
    repo_id, _, revision = spec.partition('@')
    try:
        validate_repo_id(repo_id)
    except HFValidationError as error:
        raise CatalogError('usage', 'invalid repository id %r: %s' % (repo_id, error)) from None
    if '@' in spec and not revision:
        raise CatalogError('usage', 'empty revision after @')
    return repo_id, revision or 'main'


def fetch(repo_id, filename, revision, size):
    """One small file of the Revisione, into memory (never into the cache)."""
    from huggingface_hub import hf_hub_url
    if size is not None and size > METADATA_FILE_LIMIT:
        raise CatalogError('metadata_bound', '%s is larger than the %d-byte metadata bound'
                           % (filename, METADATA_FILE_LIMIT))
    url = hf_hub_url(repo_id, filename, revision=revision, endpoint=endpoint())
    response = call(lambda: _get(url))
    if len(response.content) > METADATA_FILE_LIMIT:
        raise CatalogError('metadata_bound', '%s exceeds the metadata bound' % filename)
    return response.content


WEIGHT_SUFFIXES = ('.safetensors', '.bin', '.pt', '.pth', '.ckpt', '.gguf', '.h5', '.msgpack', '.onnx')
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer.model', 'tiktoken.model', 'vocab.json', 'vocab.txt',
                   'merges.txt', 'spiece.model', 'tokenizer_config.json', 'special_tokens_map.json',
                   'added_tokens.json')
SAFETENSORS_INDEX = 'model.safetensors.index.json'


def file_size(sibling):
    """Size of a file of the Revisione as the Hub records it (LFS size for LFS files)."""
    if sibling.lfs is not None:
        return sibling.lfs.size
    return sibling.size


def read_json(repo_id, revision, siblings, filename, raw=False):
    """(parsed, bytes) of a JSON file of the Revisione, or (None, None) if absent."""
    if filename not in siblings:
        return (None, None) if raw else None
    data = fetch(repo_id, filename, revision, file_size(siblings[filename]))
    try:
        parsed = json.loads(data)
    except ValueError:
        parsed = {capability.INVALID_JSON: True}
    return (parsed, data) if raw else parsed


class HeaderError(Exception):
    """A safetensors header that cannot be read."""


def summarize_weights(sizes, index, read_header):
    """The weight files a loader would read, their sizes and their tensor headers.

    `sizes` maps every file of the Checkpoint to its size; `read_header(name)` returns
    {tensor: (dtype, shape)} of one safetensors file or raises HeaderError; None skips the
    headers. Shared by the catalog (headers through byte ranges), the installer (file selection
    only) and the local registry (headers read from disk).
    """
    indexed = isinstance(index, dict) and isinstance(index.get('weight_map'), dict)
    if indexed:
        files = sorted(set(index['weight_map'].values()))
    elif 'model.safetensors' in sizes:
        files = ['model.safetensors']
    else:
        files = []
    missing = [f for f in files if f not in sizes]
    present = [sizes[f] for f in files if f in sizes]
    others = sorted(n for n in sizes if n.endswith(WEIGHT_SUFFIXES) and n not in files)
    result = {'format': 'safetensors' if files else None, 'files': files, 'sharded': indexed,
              'missing_files': missing, 'bytes': None if missing or None in present else sum(present),
              'other_weight_files': others, 'parameters': None, 'dtypes': None, 'headers_read': False,
              'header_error': None}
    tensors = {}
    if read_header is not None and files and not missing and len(files) <= capability.MAX_HEADER_FILES:
        dtypes = {}
        for name in files:
            try:
                header = read_header(name)
            except HeaderError as error:
                result['header_error'] = 'unreadable safetensors header in %s: %s' % (name, error)
                return result, {}
            for tensor, (dtype, shape) in header.items():
                tensors[tensor] = (dtype, tuple(shape))
                dtypes[dtype] = dtypes.get(dtype, 0) + math.prod(shape)
            if len(tensors) > capability.MAX_CATALOG_TENSORS:
                result['header_error'] = 'more than %d tensors: the catalog stops reading headers' \
                    % capability.MAX_CATALOG_TENSORS
                return result, {}
        result.update(parameters=sum(dtypes.values()), dtypes=dict(sorted(dtypes.items())),
                      headers_read=True)
    return result, tensors


def weights(repo_id, revision, siblings, index):
    """Weight summary of a Revisione on the Hub; headers read through byte ranges."""
    from huggingface_hub.errors import SafetensorsParsingError
    hub = api()

    def read_header(name):
        try:
            header = call(hub.parse_safetensors_file_metadata, repo_id, name, revision=revision)
        except SafetensorsParsingError as error:
            raise HeaderError(str(error)) from None
        return {tensor: (meta.dtype, meta.shape) for tensor, meta in header.tensors.items()}

    return summarize_weights({name: file_size(s) for name, s in siblings.items()}, index, read_header)


def templates(repo_id, revision, siblings, tokenizer_config):
    """Chat templates of the Revisione: tokenizer_config.json, chat_template.jinja/json."""
    found = []
    if isinstance(tokenizer_config, dict):
        value = tokenizer_config.get('chat_template')
        if isinstance(value, str):
            found.append(('default', value))
        elif isinstance(value, list):
            found += [(t.get('name'), t.get('template')) for t in value if isinstance(t, dict)]
    if 'chat_template.jinja' in siblings:
        raw = fetch(repo_id, 'chat_template.jinja', revision, file_size(siblings['chat_template.jinja']))
        found.append(('default', raw.decode('utf-8', 'replace')))
    loaded = read_json(repo_id, revision, siblings, 'chat_template.json')
    if isinstance(loaded, dict) and isinstance(loaded.get('chat_template'), str):
        found.append(('default', loaded['chat_template']))
    texts = [t for _, t in found if isinstance(t, str) and t]
    tools = any(re.search(r'\btools\b', t) for t in texts) or any(n == 'tool_use' for n, _ in found)
    return {'present': bool(texts), 'tools': bool(tools)}


def licence_of(model):
    card = model.card_data.to_dict() if model.card_data is not None else {}
    value = card.get('license')
    if value == 'other' and card.get('license_name'):
        return 'other: %s' % card['license_name']
    if value:
        return value if isinstance(value, str) else ', '.join(map(str, value))
    tags = [t[len('license:'):] for t in model.tags or () if t.startswith('license:')]
    return tags[0] if tags else None


def context_length(config):
    for key in ('max_position_embeddings', 'n_positions', 'max_sequence_length', 'seq_length'):
        if type(config.get(key)) is int:
            return config[key]
    return None


def search(text=None, *, author=None, task=None, family=None, licence=None, min_params=None,
           max_params=None, sort='downloads', limit=20, page=1):
    """One page of the Hub catalog, most popular first by default. No whitelist."""
    from itertools import islice
    if type(limit) is not int or not 1 <= limit <= MAX_PAGE_SIZE:
        raise CatalogError('usage', 'the page size must be between 1 and %d' % MAX_PAGE_SIZE)
    if type(page) is not int or page < 1:
        raise CatalogError('usage', 'pages are numbered from 1')
    if sort not in SORTS:
        raise CatalogError('usage', 'sort must be one of %s' % ', '.join(SORTS))
    bounds = []
    for name, value in (('min', min_params), ('max', max_params)):
        if value is not None:
            if not PARAMETER_COUNT.match(value):
                raise CatalogError('usage', 'parameter counts look like 500M, 7B or 1.5B, not %r' % value)
            bounds.append('%s:%s' % (name, value))
    tags = [t for t in (family, licence and 'license:'+licence) if t]
    query = {'text': text, 'author': author, 'task': task, 'family': family, 'license': licence,
             'min_params': min_params, 'max_params': max_params, 'sort': sort}
    expand = ['sha', 'downloads', 'likes', 'pipeline_tag', 'tags', 'gated', 'safetensors',
              'lastModified', 'library_name']
    skip, wanted = (page-1)*limit, limit+1          # one more than the page tells whether more exist
    with translated():
        listing = api().list_models(search=text or None, author=author, pipeline_tag=task,
                                    filter=tags or None, num_parameters=','.join(bounds) or None,
                                    sort=SORTS[sort], limit=skip+wanted, expand=expand)
        models = call(lambda: list(islice(listing, skip, skip+wanted)), retries=0)
    results = []
    for model in models[:limit]:
        results.append(dict({'id': model.id, 'revision': model.sha, 'task': model.pipeline_tag,
                             'library': model.library_name, 'downloads': model.downloads,
                             'likes': model.likes,
                             'parameters': model.safetensors.total if model.safetensors else None,
                             'license': licence_of(model), 'gated': model.gated or False,
                             'last_modified': model.last_modified.isoformat() if model.last_modified else None},
                            **evidence.highlights(model.id, model.sha)))
    return {'schema': SCHEMA_SEARCH, 'client': client(), 'endpoint': public_endpoint(), 'query': query,
            'page': page, 'limit': limit, 'has_more': len(models) > limit, 'results': results}


def client():
    import huggingface_hub
    return 'huggingface_hub ' + huggingface_hub.__version__


def _get(url):
    from huggingface_hub.utils import build_hf_headers, get_session, hf_raise_for_status
    response = get_session().get(url, headers=build_hf_headers(), follow_redirects=True)
    hf_raise_for_status(response)
    return response


def info(spec):
    repo_id, requested = parse_spec(spec)
    with translated(repo_id):
        return _info(repo_id, requested)


def _info(repo_id, requested):
    model = call(api().model_info, repo_id, revision=requested, files_metadata=True)
    revision = model.sha
    siblings = {s.rfilename: s for s in model.siblings or ()}
    config, config_raw = read_json(repo_id, revision, siblings, 'config.json', raw=True)
    config = config if isinstance(config, dict) else {}
    tokenizer_config = read_json(repo_id, revision, siblings, 'tokenizer_config.json')
    index = read_json(repo_id, revision, siblings, SAFETENSORS_INDEX)
    weight_info, tensors = weights(repo_id, revision, siblings, index)
    assessment = capability.assess(capability.RevisionMetadata(
        repo_id=repo_id, revision=revision, files=frozenset(siblings), config=config,
        config_raw=config_raw, tokenizer_config=tokenizer_config, index=index, weights=weight_info,
        tensors=tensors))
    return {
        'schema': SCHEMA_INFO,
        'client': client(),
        'endpoint': public_endpoint(),
        'repo_id': repo_id,
        'requested_ref': requested,
        'revision': revision,
        'hub_url': '%s/%s/tree/%s' % (public_endpoint(), repo_id, revision),
        'pipeline_tag': model.pipeline_tag,
        'library_name': model.library_name,
        'license': licence_of(model),
        'gated': model.gated if model.gated in ('auto', 'manual') else False,
        'architectures': config.get('architectures'),
        'model_type': config.get('model_type'),
        'declared_dtype': config.get('torch_dtype', config.get('dtype')),
        'context_length': context_length(config),
        'weights': weight_info,
        'tokenizer': {'files': sorted(n for n in siblings if n in TOKENIZER_FILES)},
        'chat_template': templates(repo_id, revision, siblings, tokenizer_config),
        'certification': evidence.certification(repo_id, revision),
        'capability': assessment,
    }


# -- text rendering ----------------------------------------------------------------------------

def printable(text):
    """Hub-sourced text made inert for a terminal: control characters become \\xNN escapes."""
    return ''.join(c if c.isprintable() else '\\x%02x' % ord(c) if ord(c) < 256 else '\\u%04x' % ord(c)
                   for c in str(text))


def _lines(lines):
    return '\n'.join(printable(line) for line in lines) + '\n'

def _count(n):
    if n is None:
        return '?'
    for scale, suffix in ((1e12, 'T'), (1e9, 'B'), (1e6, 'M'), (1e3, 'K')):
        if n >= scale:
            return '%.2f%s' % (n/scale, suffix)
    return str(n)


def _bytes(n):
    if n is None:
        return 'unknown'
    return '%.2f GiB (%d bytes)' % (n/2**30, n) if n >= 2**30 else '%.2f MiB (%d bytes)' % (n/2**20, n)


def _marks(entry):
    certification = entry.get('certification')
    return ('*' if certification else ' ') + ('v' if entry.get('verified_revisions') else ' ')


def render_search(result):
    rows = [('', 'MODEL', 'TASK', 'PARAMS', 'DOWNLOADS', 'LIKES', 'LICENSE', 'GATED')]
    for m in result['results']:
        rows.append((_marks(m), m['id'], m['task'] or '-', _count(m['parameters']), str(m['downloads'] or 0),
                     str(m['likes'] or 0), m['license'] or '-', m['gated'] or 'no'))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines = ['  '.join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
    if not result['results']:
        lines.append('no models on this page')
    lines.append('')
    lines.append('* v0.1 certification target   v verified revision (see `inwime models info`)')
    lines.append('page %d, %d per page%s' % (result['page'], result['limit'],
                 '; more results: repeat with --page %d' % (result['page']+1) if result['has_more'] else ''))
    return _lines(lines)


def render_info(info):
    w = info['weights']
    cap = info['capability']
    out = ['%s @ %s' % (info['repo_id'], info['revision']),
           '  page:            %s' % info['hub_url'],
           '  requested:       %s' % info['requested_ref'],
           '  task/library:    %s / %s' % (info['pipeline_tag'] or '-', info['library_name'] or '-'),
           '  license:         %s' % (info['license'] or 'unknown'),
           '  gated:           %s' % (info['gated'] or 'no'),
           '  architecture:    %s (model_type %s)' % (', '.join(info['architectures'] or ['unknown']),
                                                     info['model_type'] or 'unknown'),
           '  context length:  %s' % (info['context_length'] or 'unknown'),
           '  declared dtype:  %s' % (info['declared_dtype'] or 'unknown'),
           '  weights:         %s, %s' % (w['format'] or 'no safetensors', _bytes(w['bytes'])),
           '  files:           %d%s' % (len(w['files']), ' (sharded, with index)' if w['sharded'] else ''),
           '  parameters:      %s' % ('%s by dtype %s' % (w['parameters'], w['dtypes'])
                                     if w['headers_read'] else 'unknown (headers not read)'),
           '  tokenizer:       %s' % (', '.join(info['tokenizer']['files']) or 'none'),
           '  chat template:   %s' % ('yes, with tools' if info['chat_template']['tools'] else 'yes'
                                     if info['chat_template']['present'] else 'no')]
    if w['other_weight_files']:
        out.append('  other weights:   %s' % ', '.join(w['other_weight_files']))
    c = info['certification']
    if c:
        out.append('  certification:   %s: %s at %s%s' % (c['status'], c['role'], c['revision'],
                                                         '' if c['this_revision'] else ' (not this revision)'))
    out += ['', 'InWiMe support: %s%s' % (cap['support'], ' (adapter %s)' % cap['adapter'] if cap['adapter'] else '')]
    for state in cap['states']:
        proof = ' [%s; %s]' % (state['path'], state['evidence']) if 'evidence' in state else ''
        proof += ' (on %s)' % state['scope'] if 'scope' in state else ''
        out.append('  %-24s %s%s' % (state['state'], state['status'].replace('_', ' '), proof))
    if cap['diagnosis']:
        out += ['', 'Diagnosis:']
        out += ['  - [%s] %s' % (d['category'], d['message']) for d in cap['diagnosis']]
    return _lines(out)

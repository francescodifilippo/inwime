"""Motore and its single Coordinatore: the one authority over the active model and inference.

One `Coordinatore` per process. It owns the active model, its tokenizer, the verified identity
of its files and the only worker thread that computes; the HTTP layer never touches the
runtime. Under it, each request gets its own session of the Percorso congelato (the
`inwime_runtime` Qwen2 adapter), whose `core.Coordinator` is the Gestore delle regioni:
memory reservations, leases and weight I/O stay there. An imported Checkpoint is opened by the
frozen loader as is; one installed in the Hugging Face cache goes through `snapshot.py`, which
hands the same session a catalog of the verified blobs.

The frozen runtime is bound to the thread that creates it, so every runtime is created, used
and closed on the worker thread. Requests run one at a time in arrival order.

Tokens are chosen by `generation.Sampling` on the frozen FP64 logits; at temperature 0 it is the
M10 greedy rule of `inwime_runtime.tokenizer.greedy` (the lowest token id wins a tie). Chat
prompts are rendered by Transformers (`chat.py`). The context capacity is the Checkpoint's
`max_position_embeddings`; the memory-based capacity plan comes with #42.
"""
from concurrent.futures import Future
import queue
import secrets
import threading
from pathlib import Path

from inwime_runtime.core import CheckpointError, Coordinator
from inwime_runtime.llama import llama_plan, read_config
from inwime_runtime.math_cpu import NativeMath, ScalarMath
from inwime_runtime.qwen2 import Qwen2Catalog, Qwen2Runtime
from inwime_runtime.tokenizer import LlamaTokenizer

from . import catalog, chat, kernel, snapshot
from .generation import Sampling, TextStream


# Chat token limit when the client sets none. OpenAI's is the rest of the context, but the
# Percorso congelato holds FP64 logits for its whole capacity (prompt + limit): for a 32k-context
# Qwen2.5 that would be tens of GB. Until the capacity plan (#42), a fixed limit.
DEFAULT_CHAT_MAX_TOKENS = 512


class EngineError(Exception):
    """A failure with an OpenAI-style meaning: HTTP status, error type, code and parameter."""

    def __init__(self, status, message, *, type='invalid_request_error', code=None, param=None):
        super().__init__(message)
        self.status, self.message, self.type, self.code, self.param = status, message, type, code, param


class NotServable(Exception):
    """The active model cannot be served by this version; the message says why."""


def math_backend():
    """The native kernel when present and bit-identical to the Python reference, else Python."""
    described = kernel.describe()
    if described['native']:
        return NativeMath(described['path'])
    return ScalarMath()


class CatalogQwen2Runtime(Qwen2Runtime):
    """`Qwen2Runtime` over a catalog given by the caller instead of one read from a directory.

    The body is `Qwen2Runtime.__init__` with its defaults (tiles of 262144 elements, synchronous
    I/O, default budget) minus the construction of the catalog: same plan, same Gestore delle
    regioni, same computation, all inherited unchanged."""

    def __init__(self, catalog, capacity, *, integrity, math_backend):
        self.catalog = catalog
        self.spec, self.capacity = catalog.spec, capacity
        self.plan = llama_plan(catalog, capacity, 262144, 262144, 'sync')
        self.math_backend = math_backend
        self.control = Coordinator(catalog, self.plan, None, None, integrity=integrity,
                                   math_backend=math_backend)
        self._published = (0, 0, 0)
        self.running = self.failed = self.result_open = self.closed = False


class Coordinatore:
    _instance = None
    _guard = threading.Lock()

    def __init__(self, model_id, directory, manifest=None):
        """`manifest`: the install manifest of a Checkpoint in the Hugging Face cache; None for an
        imported directory in the frozen loader's format."""
        with Coordinatore._guard:
            if Coordinatore._instance is not None:
                raise RuntimeError('a Coordinatore already exists in this process')
            Coordinatore._instance = self
        self.model_id, self.directory, self.manifest = model_id, Path(directory), manifest
        self.catalog = None
        self._jobs = queue.SimpleQueue()
        self._worker = threading.Thread(target=self._work, name='inwime-worker', daemon=True)
        self._worker.start()
        try:
            self.submit(self._load).result()
        except BaseException:
            self.close()
            raise

    # -- worker -----------------------------------------------------------------------------

    def _work(self):
        while True:
            job = self._jobs.get()
            if job is None:
                return
            future, function, args = job
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(function(*args))
            except BaseException as error:
                future.set_exception(error)

    def submit(self, function, *args):
        future = Future()
        self._jobs.put((future, function, args))
        return future

    def close(self, timeout=60):
        """Stop the worker after the job in progress. The process stays bound to this Coordinatore
        while that job still runs: a second one must never compute next to it."""
        self._jobs.put(None)
        self._worker.join(timeout=timeout)
        if self._worker.is_alive():
            return False
        with Coordinatore._guard:
            if Coordinatore._instance is self:
                Coordinatore._instance = None
        return True

    # -- model --------------------------------------------------------------------------------

    def _load(self):
        """Admit the active model once: adapter, tokenizer and a full SHA256 check of the weights.
        Later sessions reopen the same files and must find them unchanged."""
        try:
            config = read_config(self.directory/'config.json')
        except (OSError, ValueError) as error:
            raise NotServable('cannot read config.json: %s' % error) from error
        if config.get('model_type') != 'qwen2':
            raise NotServable('model_type %r is not served by this version: the HTTP tracer serves '
                              'Qwen2 Checkpoint accepted by the Percorso congelato'
                              % (config.get('model_type'),))
        self.math = math_backend()
        try:
            if self.manifest is not None:
                self.catalog = Qwen2Catalog(snapshot.snapshot_source(self.manifest))
            with self._session(1, 'full') as runtime:
                self.verified = dict(runtime.control.stamps)
                self.context = runtime.spec.context
                stats = runtime.control.stats()
                self.integrity_read_bytes = stats['integrity_read_bytes']
                self.integrity_read_ns = stats['integrity_read_ns']
        except (OSError, ValueError, KeyError) as error:
            # CheckpointError is a ValueError: tied head, missing descriptor, checksum mismatch.
            raise NotServable('%s: %s' % (type(error).__name__, error)) from error
        try:
            self.tokenizer = LlamaTokenizer(self.directory)
        except Exception as error:      # the tokenizers library raises a bare Exception
            raise NotServable('tokenizer: %s: %s' % (type(error).__name__, error)) from error
        self.chat, self.chat_unavailable = chat.load(self.directory)
        try:
            self.generation_config = read_config(self.directory/'generation_config.json')
        except (OSError, ValueError):
            self.generation_config = {}
        if not isinstance(self.generation_config, dict):
            self.generation_config = {}

    def _session(self, capacity, integrity):
        if self.catalog is not None:
            return CatalogQwen2Runtime(self.catalog, capacity, integrity=integrity, math_backend=self.math)
        return Qwen2Runtime(self.directory, capacity=capacity, integrity=integrity, math_backend=self.math)

    # -- requests -----------------------------------------------------------------------------
    #
    # A request is prepared off the worker (tokenization and chat rendering only read the
    # tokenizers), so a refused request gets its HTTP error at once, even behind a long inference.

    def _check_context(self, prompt_tokens, max_tokens, param):
        if prompt_tokens+max_tokens > self.context:
            raise EngineError(400, "this model's maximum context length is %d tokens; the request has %d "
                              'prompt tokens and max_tokens %d' % (self.context, prompt_tokens, max_tokens),
                              code='context_length_exceeded', param=param)

    def _sampling(self, client):
        return Sampling(client, self.generation_config, secrets.randbits(63))

    def prepare_completion(self, prompt, max_tokens, client):
        """`client`: the sampling parameters the request sets (generation.client_parameters)."""
        ids = self.tokenizer.encode(prompt)
        if not ids:
            raise EngineError(400, 'the prompt has no tokens', param='prompt')
        self._check_context(len(ids), max_tokens, 'prompt')
        return Prepared(ids, max_tokens, self.tokenizer.decode, self._sampling(client))

    def prepare_chat(self, messages, tools, max_tokens, client):
        """`max_tokens` None: DEFAULT_CHAT_MAX_TOKENS, or the rest of the context if smaller."""
        if self.chat is None:
            raise EngineError(400, 'the active model %s cannot chat: %s; /v1/completions remains available'
                              % (self.model_id, self.chat_unavailable), code='chat_template_unavailable',
                              param='messages')
        try:
            _, ids = self.chat.render(messages, tools)
        except Exception as error:      # jinja2 errors, or the template's own raise_exception()
            raise EngineError(400, 'the chat template rejected the messages: %s'
                              % catalog.printable(str(error))[:500], code='chat_template_error',
                              param='messages') from error
        if max_tokens is None:
            max_tokens = max(1, min(DEFAULT_CHAT_MAX_TOKENS, self.context-len(ids)))
        self._check_context(len(ids), max_tokens, 'messages')
        return Prepared(ids, max_tokens, self.chat.decode, self._sampling(client))

    def generate(self, prepared, emit=None):
        """Compute a prepared request on the worker; returns a concurrent Future of its result.
        `emit(text)` is called on the worker with each piece of text as it becomes final."""
        return self.submit(self._generate, prepared, emit)

    def _generate(self, prepared, emit):
        """Prefill the prompt ids, then choose and decode tokens until `max_tokens` tokens, an EOS
        token (counted, without text) or a stop sequence (its token counted, its text dropped)."""
        ids, max_tokens, sampling = prepared.ids, prepared.max_tokens, prepared.sampling
        changed = EngineError(500, 'the Checkpoint files changed after their verification at start-up; '
                              'restart the server to verify them again', type='server_error',
                              code='checkpoint_changed')
        try:
            session = self._session(len(ids)+max_tokens-1, 'metadata')
        except CheckpointError as error:
            # Admitted at start-up, so any rejection now means its files changed since then.
            raise changed from error
        text = TextStream(prepared.decode, sampling.stop)
        emit = emit or (lambda delta: None)
        generated = []
        with session as runtime:
            if runtime.control.stamps != self.verified:
                # Same device, inode, size and times as the files hashed at start-up, or no inference.
                raise changed
            vocab = runtime.spec.vocab
            with runtime.prefill(ids) as result:
                chosen = sampling.choose(result.logits[(len(ids)-1)*vocab:len(ids)*vocab], ids)
            while True:
                generated.append(chosen)
                if chosen in self.tokenizer.eos:
                    break
                delta = text.push(chosen)
                if delta:
                    emit(delta)
                if text.stopped or len(generated) == max_tokens:
                    break
                with runtime.decode(chosen) as result:
                    chosen = sampling.choose(result.logits[:vocab], ids+generated)
        delta = text.finish()
        if delta:
            emit(delta)
        stopped = text.stopped or generated[-1] in self.tokenizer.eos
        return {'text': text.released, 'finish_reason': 'stop' if stopped else 'length',
                'prompt_tokens': len(ids), 'completion_tokens': len(generated),
                'diagnostics': {'prompt_token_ids': ids, 'completion_token_ids': generated,
                                'sampling': sampling.describe(),
                                'computation': {'path': 'percorso congelato', 'kernel': self.math.name}}}


class Prepared:
    """A request ready for the worker: prompt ids, token limit, decoder of its text, sampling."""

    def __init__(self, ids, max_tokens, decode, sampling):
        self.ids, self.max_tokens, self.decode, self.sampling = ids, max_tokens, decode, sampling

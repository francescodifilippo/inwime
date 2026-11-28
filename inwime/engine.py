"""Motore and its single Coordinatore: the one authority over the active model and inference.

One `Coordinatore` per process. It owns the active model, its tokenizer, the verified identity
of its files and the only worker thread that computes; the HTTP layer never touches the
runtime. Under it, each request gets its own session of the Percorso additivo (`additive.py`,
#41): block prefill and the operators of `ops.py`, bit-identical to the Percorso congelato on the
certified configurations, over the frozen Qwen2 catalog for an untied head or the additive one of
`qwen2_tied` for a tied head. `INWIME_COMPUTE_PATH=frozen` rolls the Motore back to the previous
path (ADR 0001): the frozen `Qwen2Runtime` session, one position at a time, unchanged.
Its `core.Coordinator` is the Gestore delle regioni: memory reservations, leases and weight I/O
stay there. Each operator is native only if it passed the gate of `gate.py` when the model was
loaded; otherwise it is the Python reference. An imported Checkpoint is opened by the
frozen loader as is; one installed in the Hugging Face cache goes through `snapshot.py`, which
hands the same session a catalog of the verified blobs.

The frozen runtime is bound to the thread that creates it, so every runtime is created, used
and closed on the worker thread. Requests run one at a time in arrival order, behind a bounded
queue; a cancelled request stops at the next cancellation check of the runtime (before each
layer, each weight read and each operator call, and inside the native linear and attention
loops), and publishes nothing.

Tokens are chosen by `generation.Sampling` on the FP64 logits of the last position; at temperature 0 it is the
M10 greedy rule of `inwime_runtime.tokenizer.greedy` (the lowest token id wins a tie). Chat
prompts are rendered by Transformers (`chat.py`). The context capacity is the Checkpoint's
`max_position_embeddings`; the memory-based capacity plan comes with #42.
"""
import collections
from concurrent.futures import CancelledError, Future
import os
import secrets
import threading
from pathlib import Path

from inwime_runtime.core import CheckpointError, Coordinator
from inwime_runtime.llama import llama_plan, read_config
from inwime_runtime.math_cpu import NativeMath, ScalarMath
from inwime_runtime.qwen2 import Qwen2Runtime
from inwime_runtime.safetensors_source import SafetensorsSource
from inwime_runtime.tokenizer import LlamaTokenizer

from . import catalog, chat, gate, kernel, snapshot
from .additive import AdditiveQwen2Runtime, CancelSignal
from .qwen2_tied import qwen2_catalog
from .generation import Sampling, TextStream


# Chat token limit when the client sets none. OpenAI's is the rest of the context, but a session
# reserves FP64 KV for its whole capacity (prompt + limit): for a 32k-context Qwen2.5-7B that
# would be gigabytes. Until the capacity plan (#42), a fixed limit.
DEFAULT_CHAT_MAX_TOKENS = 512

DEFAULT_QUEUE_SIZE = 4            # requests waiting behind the one in progress

PATH_ENV = 'INWIME_COMPUTE_PATH'  # 'additive' (default) or 'frozen': the rollback of ADR 0001
PATHS = {'additive': 'percorso additivo', 'frozen': 'percorso congelato'}


class EngineError(Exception):
    """A failure with an OpenAI-style meaning: HTTP status, error type, code and parameter."""

    def __init__(self, status, message, *, type='invalid_request_error', code=None, param=None):
        super().__init__(message)
        self.status, self.message, self.type, self.code, self.param = status, message, type, code, param


def shutting_down():
    """The answer to a request the server will not complete because it is stopping: refused,
    withdrawn from the queue or cancelled in progress. A cancellation for any other reason (client
    gone, time limit) reaches no client as such."""
    return EngineError(503, 'the server is shutting down; the request was not completed',
                       type='server_error', code='shutting_down')


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
    regioni, same computation, all inherited unchanged. The catalog is the frozen one or the
    tied-head one of `qwen2_tied`; the tile sizes are parameters only so that tests can stream
    the head through partial leases."""

    def __init__(self, catalog, capacity, *, integrity, math_backend, tile_elements=262144,
                 read_elements=262144):
        self.catalog = catalog
        self.spec, self.capacity = catalog.spec, capacity
        self.plan = llama_plan(catalog, capacity, tile_elements, read_elements, 'sync')
        self.math_backend = math_backend
        self.control = Coordinator(catalog, self.plan, None, None, integrity=integrity,
                                   math_backend=math_backend)
        self._published = (0, 0, 0)
        self.running = self.failed = self.result_open = self.closed = False


class Coordinatore:
    _instance = None
    _guard = threading.Lock()

    def __init__(self, model_id, directory, manifest=None, queue_size=DEFAULT_QUEUE_SIZE):
        """`manifest`: the install manifest of a Checkpoint in the Hugging Face cache; None for an
        imported directory in the frozen loader's format. `queue_size`: requests that may wait
        behind the one in progress."""
        with Coordinatore._guard:
            if Coordinatore._instance is not None:
                raise RuntimeError('a Coordinatore already exists in this process')
            Coordinatore._instance = self
        self.model_id, self.directory, self.manifest = model_id, Path(directory), manifest
        self.queue_size = queue_size
        self.catalog = None
        self._queue_lock = threading.Condition()
        self._waiting = collections.deque()
        self._running = None
        self._closing = False
        self._worker = threading.Thread(target=self._work, name='inwime-worker', daemon=True)
        self._worker.start()
        try:
            self._admit(Job(self._load)).future.result()
        except BaseException:
            self.close()
            raise

    # -- worker and queue ---------------------------------------------------------------------
    #
    # One job computes at a time, on the worker; up to `queue_size` more wait in arrival order.
    # The slot of the job in progress frees only when the worker is done with it: its runtime
    # session is closed, so nothing it reserved is still held.

    def _work(self):
        while True:
            with self._queue_lock:
                while not self._waiting and not self._closing:
                    self._queue_lock.wait()
                if not self._waiting:
                    return
                job = self._running = self._waiting.popleft()
            try:
                if job.future.set_running_or_notify_cancel():
                    try:
                        job.future.set_result(job.function(*job.args))
                    except BaseException as error:
                        job.future.set_exception(error)
            finally:
                with self._queue_lock:
                    self._running = None
                    self._queue_lock.notify_all()

    def _refuse(self, limit):
        """EngineError 503 when the Coordinatore is closing; with `limit`, 429 when the job in
        progress and `limit` waiting ones fill every place. Counted with the waiting ones: the
        worker may not have taken the next job yet."""
        if self._closing:
            raise shutting_down()
        if limit is not None and len(self._waiting)+(self._running is not None) > limit:
            raise EngineError(429, 'the server is busy: one inference is in progress and %d more '
                              'wait in the queue, which is full; retry later' % limit,
                              type='requests', code='queue_full')

    def check_room(self):
        """Refuse early, before a request is prepared, when the queue is already full or the
        server is stopping. Advisory: `generate` checks again when it admits the prepared request."""
        with self._queue_lock:
            self._refuse(self.queue_size)

    def _admit(self, job, limit=None):
        """Queue `job`; with `limit`, refuse it when that many jobs already wait behind a running one."""
        with self._queue_lock:
            self._refuse(limit)
            job.coordinatore = self
            self._waiting.append(job)
            self._queue_lock.notify_all()
        return job

    def _cancel(self, job):
        with self._queue_lock:
            if job in self._waiting:
                # Never started: nothing to undo, its place frees at once.
                self._waiting.remove(job)
                job.future.cancel()
                self._queue_lock.notify_all()
                return
        job.cancel_event.set()

    def drain(self):
        """Start stopping: refuse new jobs and withdraw the waiting ones (both 503 `shutting_down`).
        The job in progress goes on; the worker stops once it is done."""
        with self._queue_lock:
            self._closing = True
            waiting, self._waiting = list(self._waiting), collections.deque()
            self._queue_lock.notify_all()
        for job in waiting:
            job.future.cancel()

    def cancel_running(self):
        """Cancel the job in progress, if any: it stops at the next cancellation check."""
        with self._queue_lock:
            running = self._running
        if running is not None:
            running.cancel_event.set()

    def close(self, timeout=60):
        """Drain, cancel the job in progress, then stop the worker once that job is unwound. The
        process stays bound to this Coordinatore while a job still runs: a second one must never
        compute next to it."""
        self.drain()
        self.cancel_running()
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
                              'Qwen2 Checkpoint, with a tied or untied head'
                              % (config.get('model_type'),))
        self.path = os.environ.get(PATH_ENV) or 'additive'
        if self.path not in PATHS:
            raise NotServable('%s=%r: expected one of %s' % (PATH_ENV, self.path, ', '.join(PATHS)))
        self.math = math_backend()
        self.ops, self.operators = gate.evaluate()
        self.tied = config.get('tie_word_embeddings') is True
        try:
            if self.manifest is not None:
                self.catalog = qwen2_catalog(snapshot.snapshot_source(self.manifest))
            with self._session(1, 'full') as runtime:
                self.verified = dict(runtime.control.stamps)
                self.context = runtime.spec.context
                stats = runtime.control.stats()
                self.integrity_read_bytes = stats['integrity_read_bytes']
                self.integrity_read_ns = stats['integrity_read_ns']
        except (OSError, ValueError, KeyError) as error:
            # CheckpointError is a ValueError: unsupported config, missing descriptor, checksum mismatch.
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
        # An imported directory is read again for each session, as `Qwen2Runtime` does.
        adapter = (self.catalog if self.catalog is not None
                   else qwen2_catalog(SafetensorsSource(self.directory)))
        if self.path == 'frozen':
            return CatalogQwen2Runtime(adapter, capacity, integrity=integrity, math_backend=self.math)
        return AdditiveQwen2Runtime(adapter, capacity, integrity=integrity, math_backend=self.math,
                                    ops=self.ops)

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
        """Queue a prepared request for the worker; returns its `Job`, whose `future` gives the
        result. `emit(text)` is called on the worker with each piece of text as it becomes final.
        Raises EngineError 429 when the queue is full."""
        job = Job(self._generate, prepared, emit)
        job.args += (job.cancel_event,)
        return self._admit(job, self.queue_size)

    def _generate(self, prepared, emit, cancel_event):
        """Prefill the prompt ids, then choose and decode tokens until `max_tokens` tokens, an EOS
        token (counted, without text) or a stop sequence (its token counted, its text dropped).

        `cancel_event` goes to every runtime call, which checks it before each layer and each weight
        read and rolls back its candidate KV when it raises: a cancelled request publishes nothing
        and its session is closed before the worker takes the next job."""
        try:
            return self._compute(prepared, emit, cancel_event)
        except CancelledError as error:
            raise shutting_down() from error

    def _compute(self, prepared, emit, cancel_event):
        if cancel_event.is_set():
            raise CancelledError('cancelled before it started')
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
            with runtime.prefill(ids, cancel_event=cancel_event) as result:
                # The last position's logits: the only ones of the additive path, the last row of the frozen one.
                logits = result.logits
                chosen = sampling.choose(logits[len(logits)-vocab:], ids)
            while True:
                generated.append(chosen)
                if chosen in self.tokenizer.eos:
                    break
                delta = text.push(chosen)
                if delta:
                    emit(delta)
                if text.stopped or len(generated) == max_tokens:
                    break
                with runtime.decode(chosen, cancel_event=cancel_event) as result:
                    chosen = sampling.choose(result.logits[:vocab], ids+generated)
        delta = text.finish()
        if delta:
            emit(delta)
        stopped = text.stopped or generated[-1] in self.tokenizer.eos
        return {'text': text.released, 'finish_reason': 'stop' if stopped else 'length',
                'prompt_tokens': len(ids), 'completion_tokens': len(generated),
                'diagnostics': {'prompt_token_ids': ids, 'completion_token_ids': generated,
                                'sampling': sampling.describe(),
                                'computation': self._computation(runtime)}}

    def _computation(self, runtime):
        out = {'path': PATHS[self.path], 'kernel': self.math.name,
               'adapter': 'TiedQwen2Catalog (additive, tied head)' if self.tied else 'Qwen2Catalog (frozen)'}
        if self.path == 'additive':
            out.update(block_positions=runtime.block,
                       operators={name: impl.name for name, impl in self.ops.items()})
        return out


class Job:
    """One unit of work for the worker: queued, then computed, cancellable at any point."""

    def __init__(self, function, *args):
        self.function, self.args = function, args
        self.future = Future()
        self.cancel_event = CancelSignal()      # also seen inside the native operators
        self.coordinatore = None

    def result(self):
        """The result of a done job; EngineError 503 `shutting_down` if it was withdrawn from the
        queue before it started."""
        if self.future.cancelled():
            raise shutting_down()
        return self.future.result()

    def cancel(self):
        """Withdraw the job: a waiting one leaves the queue at once; a running one gets its
        cancellation event, checked by the runtime, and keeps its slot until the worker has
        unwound it. Harmless once the job is done."""
        if self.coordinatore is not None:
            self.coordinatore._cancel(self)


class Prepared:
    """A request ready for the worker: prompt ids, token limit, decoder of its text, sampling."""

    def __init__(self, ids, max_tokens, decode, sampling):
        self.ids, self.max_tokens, self.decode, self.sampling = ids, max_tokens, decode, sampling

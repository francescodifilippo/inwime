"""Motore and its single Coordinatore: the one authority over the active model and inference.

One `Coordinatore` per process. It owns the active model, its tokenizer, the verified identity
of its files and the only worker thread that computes; the HTTP layer never touches the
runtime. Under it, each request gets its own session of the Percorso congelato (the
`inwime_runtime` Qwen2 adapter), whose `core.Coordinator` is the Gestore delle regioni:
memory reservations, leases and weight I/O stay there.

The frozen runtime is bound to the thread that creates it, so every runtime is created, used
and closed on the worker thread. Requests run one at a time in arrival order.

Tracer bullet (#37): greedy completions only, through `inwime_runtime.tokenizer.greedy`
unchanged (the M10 decoding rule: the lowest token id wins a tie). The context capacity is the
Checkpoint's `max_position_embeddings`; the memory-based capacity plan comes with #42.
"""
from concurrent.futures import Future
import queue
import threading
from pathlib import Path

from inwime_runtime.llama import read_config
from inwime_runtime.math_cpu import NativeMath, ScalarMath
from inwime_runtime.qwen2 import Qwen2Runtime
from inwime_runtime.tokenizer import LlamaTokenizer, greedy

from . import kernel


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


class Coordinatore:
    _instance = None
    _guard = threading.Lock()

    def __init__(self, model_id, directory):
        with Coordinatore._guard:
            if Coordinatore._instance is not None:
                raise RuntimeError('a Coordinatore already exists in this process')
            Coordinatore._instance = self
        self.model_id, self.directory = model_id, Path(directory)
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
            with Qwen2Runtime(self.directory, capacity=1, integrity='full', math_backend=self.math) as runtime:
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

    def _complete(self, prompt, max_tokens):
        ids = self.tokenizer.encode(prompt)
        if not ids:
            raise EngineError(400, 'the prompt has no tokens', param='prompt')
        if len(ids)+max_tokens > self.context:
            raise EngineError(400, "this model's maximum context length is %d tokens; the request has %d "
                              'prompt tokens and max_tokens %d' % (self.context, len(ids), max_tokens),
                              code='context_length_exceeded', param='prompt')
        with Qwen2Runtime(self.directory, capacity=len(ids)+max_tokens-1, integrity='metadata',
                          math_backend=self.math) as runtime:
            if runtime.control.stamps != self.verified:
                # Same device, inode, size and times as the files hashed at start-up, or no inference.
                raise EngineError(500, 'the Checkpoint files changed after their verification at start-up; '
                                  'restart the server to verify them again', type='server_error',
                                  code='checkpoint_changed')
            result = greedy(runtime, self.tokenizer, prompt, max_tokens)
        generated = result['generated_ids']
        text_ids = generated[:-1] if result['stopped_on_eos'] else generated
        return {'text': self.tokenizer.decode(text_ids),
                'finish_reason': 'stop' if result['stopped_on_eos'] else 'length',
                'prompt_tokens': len(result['input_ids']), 'completion_tokens': len(generated)}

    def complete(self, prompt, max_tokens):
        """A greedy completion, computed on the worker; returns a concurrent Future."""
        return self.submit(self._complete, prompt, max_tokens)

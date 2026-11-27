"""Test-only pacing of the runtime inside an `inwime serve` subprocess.

Run as `python -m pacer serve ...` with INWIME_PACER_DIR set. The tiny fixture computes a
request in milliseconds; to observe a request while it waits or computes, this wrapper holds
the runtime at its own cancellation checks and logs what the worker does. The product code is
unchanged: only the runtimes (the frozen and the additive one, which share the Gestore delle
regioni and its `check_cancelled`) and the public `Coordinatore.generate` are wrapped.

Gate files in INWIME_PACER_DIR, checked at every `check_cancelled` of the runtime:
- `hold-prefill`, `hold-decode`: the worker waits there while the file exists, unless the
  request is cancelled (a long computation that checks cancellation);
- `stall-prefill`, `stall-decode`: the worker waits there while the file exists, cancelled or
  not (a computation between two cancellation checks);
- `fail-prefill`, `fail-decode`: the runtime raises ReadError there (a failed weight read).

`events`, one line each, in order:
- `prepared`: `Coordinatore.prepare_completion` or `prepare_chat` tokenized a request;
- `admitted <prompt ids>` / `refused <prompt ids>`: `Coordinatore.generate` accepted or refused;
- `prefill <prompt ids>` / `decode <token>`: the worker starts a runtime call;
- `held prefill|decode`: the worker reached a gate (once per runtime call);
- `cancel seen prefill|decode`: the cancellation event was set while the worker waited at a gate;
- `cancelled prefill|decode`: the runtime raised CancelledError;
- `failed prefill|decode`: the runtime raised another exception;
- `closed length=<published KV length> failed=<bool>`: a runtime session was closed.
"""
from concurrent.futures import CancelledError
import os
from pathlib import Path
import sys
import threading
import time

from inwime_runtime import core, decoder, llama
from inwime import additive, engine

DIRECTORY = Path(os.environ['INWIME_PACER_DIR'])
_lock = threading.Lock()
_state = threading.local()


def record(line):
    with _lock, open(DIRECTORY/'events', 'a') as f:
        f.write(line+'\n')


def ids(tokens):
    return ' '.join(map(str, tokens))


def paced_prefill(original):
    def prefill(self, tokens, **kwargs):
        kind = 'decode' if self.length else 'prefill'
        _state.phase, _state.held, _state.seen = kind, False, False
        record('%s %s' % (kind, ids(tokens)))
        try:
            return original(self, tokens, **kwargs)
        except CancelledError:
            record('cancelled '+kind)
            raise
        except Exception:
            record('failed '+kind)
            raise
        finally:
            _state.phase = None
    return prefill


_check_cancelled = core.Coordinator.check_cancelled


def check_cancelled(self):
    kind = getattr(_state, 'phase', None)
    if kind is not None and (DIRECTORY/('fail-'+kind)).exists():
        raise core.ReadError('injected by the pacer')
    while kind is not None:
        cancelled = self.cancel_event is not None and self.cancel_event.is_set()
        if not ((DIRECTORY/('stall-'+kind)).exists()
                or ((DIRECTORY/('hold-'+kind)).exists() and not cancelled)):
            break
        if not _state.held:
            record('held '+kind)
            _state.held = True
        if cancelled and not _state.seen:
            record('cancel seen '+kind)
            _state.seen = True
        time.sleep(0.005)
    return _check_cancelled(self)


def recorded_close(original):
    def close(self):
        original(self)
        record('closed length=%d failed=%s' % (self.length, self.failed))
    return close


_generate = engine.Coordinatore.generate


def generate(self, prepared, *args, **kwargs):
    try:
        job = _generate(self, prepared, *args, **kwargs)
    except engine.EngineError:
        record('refused '+ids(prepared.ids))
        raise
    record('admitted '+ids(prepared.ids))
    return job


def prepared(original):
    def prepare(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        record('prepared')
        return result
    return prepare


engine.Coordinatore.prepare_completion = prepared(engine.Coordinatore.prepare_completion)
engine.Coordinatore.prepare_chat = prepared(engine.Coordinatore.prepare_chat)
llama.LlamaRuntime.prefill = paced_prefill(llama.LlamaRuntime.prefill)
additive.AdditiveQwen2Runtime.prefill = paced_prefill(additive.AdditiveQwen2Runtime.prefill)
core.Coordinator.check_cancelled = check_cancelled
decoder.CPURuntime.close = recorded_close(decoder.CPURuntime.close)
additive.AdditiveQwen2Runtime.close = recorded_close(additive.AdditiveQwen2Runtime.close)
engine.Coordinatore.generate = generate

if __name__ == '__main__':
    from inwime.cli import main
    sys.exit(main(sys.argv[1:]))

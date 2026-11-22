"""Test-only pacing of the Percorso congelato inside an `inwime serve` subprocess.

Run as `python -m pacer serve ...` with INWIME_PACER_DIR set. The tiny fixture computes a
request in milliseconds; to observe a request while it waits or computes, this wrapper holds
the frozen runtime at its own cancellation checks and logs what the worker does. The product
code is unchanged: only the frozen runtime and the public `Coordinatore.generate` are wrapped.

Gate files in INWIME_PACER_DIR, checked at every `check_cancelled` of the runtime:
- `hold-prefill`, `hold-decode`: the worker waits there while the file exists, unless the
  request is cancelled (a long computation that checks cancellation);
- `stall-prefill`, `stall-decode`: the worker waits there while the file exists, cancelled or
  not (a computation between two cancellation checks).

`events`, one line each, in order:
- `admitted <prompt ids>` / `refused <prompt ids>`: `Coordinatore.generate` accepted or refused;
- `prefill <prompt ids>` / `decode <token>`: the worker starts a runtime call;
- `held prefill|decode`: the worker reached a gate (once per runtime call);
- `cancel seen prefill|decode`: the cancellation event was set while the worker waited at a gate;
- `cancelled prefill|decode`: the runtime raised CancelledError;
- `closed length=<published KV length> failed=<bool>`: a runtime session was closed.
"""
from concurrent.futures import CancelledError
import os
from pathlib import Path
import sys
import threading
import time

from inwime_runtime import core, decoder, llama
from inwime import engine

DIRECTORY = Path(os.environ['INWIME_PACER_DIR'])
_lock = threading.Lock()
_state = threading.local()


def record(line):
    with _lock, open(DIRECTORY/'events', 'a') as f:
        f.write(line+'\n')


def ids(tokens):
    return ' '.join(map(str, tokens))


_prefill = llama.LlamaRuntime.prefill


def prefill(self, tokens, *, cancel_event=None):
    kind = 'decode' if self.length else 'prefill'
    _state.phase, _state.held, _state.seen = kind, False, False
    record('%s %s' % (kind, ids(tokens)))
    try:
        return _prefill(self, tokens, cancel_event=cancel_event)
    except CancelledError:
        record('cancelled '+kind)
        raise
    finally:
        _state.phase = None


_check_cancelled = core.Coordinator.check_cancelled


def check_cancelled(self):
    kind = getattr(_state, 'phase', None)
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


_close = decoder.CPURuntime.close


def close(self):
    _close(self)
    record('closed length=%d failed=%s' % (self.length, self.failed))


_generate = engine.Coordinatore.generate


def generate(self, prepared, *args, **kwargs):
    try:
        job = _generate(self, prepared, *args, **kwargs)
    except engine.EngineError:
        record('refused '+ids(prepared.ids))
        raise
    record('admitted '+ids(prepared.ids))
    return job


llama.LlamaRuntime.prefill = prefill
core.Coordinator.check_cancelled = check_cancelled
decoder.CPURuntime.close = close
engine.Coordinatore.generate = generate

if __name__ == '__main__':
    from inwime.cli import main
    sys.exit(main(sys.argv[1:]))

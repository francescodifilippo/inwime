"""Read-only in-run host monitor for the increment-10 acquisition (protocol amendment 002).

Review of PR #24 at e8c588f: the start gate (tools/hostgate10.py) only proves the host
state at the start; during the transfer acquire10 watched free space only. This monitor
applies, while bytes flow, the abort criteria preregistered in protocol.json
(host_memory_preflight.abort_criteria, frozen and unchanged):

  host AvailableBytes < 300 MB sustained 10 s ........ pause-and-annotate
  guest MemAvailable < 768 MiB ....................... kill-scope-invalidate-run
  host pagefile growth since baseline > 4 GiB ........ pause-and-annotate
  host commit headroom < 1 GiB ....................... kill-scope-invalidate-run
  host PagesInput > 2048/s sustained 30 s ............ annotate-io-attribution-invalid
  backing volume free < 40 GiB ....................... stop-acquisition-retain-part

For an acquisition (no cgroup scope, no I/O attribution) amendment 002 maps every
pause/kill/stop action to a cooperative stop of the transfer: the owned .part and its
marker are kept and the run can be resumed later, only through a new fresh hostgate10
GO. The I/O-attribution criterion stays an annotation, as preregistered.

Properties
  read only     one powershell.exe child runs a loop of read-only CIM queries
                (Get-CimInstance, ConvertTo-Json, Start-Sleep); guest values come from
                /proc and statvfs. Nothing on Windows or WSL is changed; the only process
                ever terminated is that child, at the end of the acquisition.
  bounded       fixed sampling interval; line length, sample buffer, kept samples and
                error log are capped; poll() does work only when a sample arrives.
  fail closed   a sample is valid only if every abort quantity is readable (the paging
                rate may be missing only on the first sample). No valid sample for
                max_unmonitored_s, a dead sampler or an unreadable stream stop the
                transfer; no first valid sample within startup_timeout_s refuses it.
  verifiable    summary with counters, extremes, annotations, stop reasons and the last
                samples, recorded in the acquisition report (also when it stops).

Limits (stated, not hidden): reaction latency is up to one sampling interval plus
the preregistered sustain time; the monitor stops this acquisition only and cannot
relieve host pressure caused by other processes; the sampler itself costs one
powershell.exe (about 100 MB, paid once at start).
"""
from collections import deque
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone

from . import hostmem10

SCHEMA = 'inwime-hostmonitor10/1'
ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT/'results'/'increment10'/'protocol.json'
PS_LOOP = ("$f={" + hostmem10.PS_SAMPLE + "}; while($true){ (& $f) | ConvertTo-Json -Compress -Depth 4;"
           " [Console]::Out.Flush(); Start-Sleep -Milliseconds %d }")


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


class StreamSource:
    """Child process emitting one JSON sample per line; bounded reader thread."""

    def __init__(self, command, *, max_line_bytes=65536, buffer=256, clock=time.monotonic):
        self.command, self.max_line_bytes, self.clock = command, max_line_bytes, clock
        self.buffer = deque(maxlen=buffer)
        self.lock = threading.Lock()
        self.proc = self.thread = None
        self.dropped = 0

    @classmethod
    def powershell(cls, interval_s, **kw):
        return cls([hostmem10.POWERSHELL, '-NoProfile', '-NonInteractive', '-Command',
                    PS_LOOP % int(interval_s*1000)], **kw)

    def _push(self, item):
        with self.lock:
            if len(self.buffer) == self.buffer.maxlen:
                self.dropped += 1
            self.buffer.append(item)

    def start(self):
        self.proc = subprocess.Popen(self.command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL)
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        skipping = False
        stream = self.proc.stdout
        while True:
            raw = stream.readline(self.max_line_bytes + 1)
            if not raw:
                self._push(('eof', self.clock(), 'sampler exited (code %s)' % self.proc.poll()))
                return
            complete = raw.endswith(b'\n')
            if skipping:
                skipping = not complete
                continue
            if not complete and len(raw) > self.max_line_bytes:
                skipping = True
                self._push(('error', self.clock(), 'sample line longer than %d bytes' % self.max_line_bytes))
                continue
            try:
                data = json.loads(raw)
            except ValueError:
                self._push(('error', self.clock(), 'unparsable sample line'))
                continue
            self._push(('sample', self.clock(), data) if isinstance(data, dict)
                       else ('error', self.clock(), 'sample is not a JSON object'))

    def drain(self):
        with self.lock:
            items = list(self.buffer)
            self.buffer.clear()
        return items

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()                 # our own sampler child, never another process
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        if self.proc is not None and self.proc.stdout:
            self.proc.stdout.close()
        if self.thread is not None:
            self.thread.join(timeout=5)


class HostMonitor:
    """Applies the preregistered abort criteria to a stream of host samples."""

    def __init__(self, source, abort_criteria, baseline_after, policy, destination, backing, *,
                 usage=shutil.disk_usage, clock=time.monotonic, guest=hostmem10.guest, sleep=time.sleep):
        self.source, self.criteria, self.baseline, self.policy = source, abort_criteria, baseline_after, policy
        self.destination, self.backing, self.usage, self.clock, self.guest = destination, backing, usage, clock, guest
        self.sleep = sleep
        mapping = policy['action_mapping']
        missing = sorted({c['action'] for c in abort_criteria} - set(mapping))
        if missing:
            raise ValueError('no preregistered acquisition mapping for actions: '+', '.join(missing))
        self.previous = None              # last valid Windows sample
        self.first_violation = {}
        self.started = self.last_valid = None
        self.stop_reasons = []
        self.annotations = deque(maxlen=policy['max_annotations'])
        self.annotated_episodes = set()
        self.errors = deque(maxlen=policy['max_errors_kept'])
        self.samples = deque(maxlen=policy['max_samples_kept'])
        self.counts = dict(valid=0, invalid=0, errors=0, transient_violations=0)
        self.extremes = {}
        self.closed = False

    # ------------------------------------------------------------ lifecycle

    def start(self):
        """Start the sampler and wait (bounded) for the first valid sample; return poll() of it."""
        self.started = self.clock()
        try:
            self.source.start()
        except OSError as error:
            return self._stop('host sampler could not start: '+type(error).__name__)
        while True:
            decision = self._process()
            if decision['stop'] or self.last_valid is not None:
                return decision
            if self.clock() - self.started > self.policy['startup_timeout_s']:
                return self._stop(f"no valid host sample within {self.policy['startup_timeout_s']} s of start")
            self.sleep(min(0.25, self.policy['sample_interval_s']))

    def poll(self):
        """Cheap; called for every chunk. Processes new samples and checks staleness."""
        if self.stop_reasons:
            return self.decision()
        decision = self._process()
        if decision['stop']:
            return decision
        reference = self.last_valid if self.last_valid is not None else self.started
        if reference is not None and self.clock() - reference > self.policy['max_unmonitored_s']:
            return self._stop(f"host not monitored: no valid sample for more than {self.policy['max_unmonitored_s']} s")
        return decision

    def close(self):
        if not self.closed:
            self.closed = True
            self.source.close()

    # ------------------------------------------------------------ evaluation

    def decision(self):
        return dict(stop=bool(self.stop_reasons), reasons=list(self.stop_reasons))

    def _stop(self, reason):
        self.stop_reasons.append(reason)
        return self.decision()

    def _error(self, t, message):
        self.counts['errors'] += 1
        self.errors.append(dict(t=round(t - (self.started or t), 3), error=message))

    def _process(self):
        for kind, t, data in self.source.drain():
            if kind == 'eof':
                self._error(t, data)
                return self._stop('host sampler stopped: '+str(data))
            if kind == 'error':
                self._error(t, data)
                continue
            self._sample(t, data)
            if self.stop_reasons:
                break
        return self.decision()

    def _sample(self, t, windows):
        snap = dict(windows=windows, guest=self.guest(),
                    checkpoint_volume=hostmem10.volume(self.destination, self.usage),
                    backing_volume=hostmem10.volume(self.backing, self.usage))
        q = hostmem10.measurements(snap, snap, self.baseline)       # point values of this sample
        rate = hostmem10.rate(self.previous, windows, 'PagesInput') if self.previous is not None else None
        q['host_pages_input_per_s'] = dict(value=rate, kind='estimate', unit='pages/s')
        values = {c['quantity']: q.get(c['quantity'], {}).get('value') for c in self.criteria}
        unreadable = [k for k, v in values.items() if not isinstance(v, (int, float)) or isinstance(v, bool)
                      if not (k == 'host_pages_input_per_s' and self.previous is None)]
        if unreadable:
            self.counts['invalid'] += 1
            self._error(t, 'unreadable: '+', '.join(sorted(unreadable)))
            return
        self.counts['valid'] += 1
        self.last_valid, self.previous = t, windows
        history = {}
        for c in self.criteria:
            v = values[c['quantity']]
            if isinstance(v, (int, float)) and not hostmem10.OPS[c['op']](v, c['threshold']):
                history[c['quantity']] = t - self.first_violation.setdefault(c['quantity'], t)
            else:
                self.first_violation.pop(c['quantity'], None)
        for k, v in values.items():
            if isinstance(v, (int, float)):
                lo, hi = self.extremes.get(k, (v, v))
                self.extremes[k] = (min(lo, v), max(hi, v))
        triggered = hostmem10.abort_reasons(q, self.criteria, history)
        fired = {r['quantity'] for r in triggered}
        self.counts['transient_violations'] += sum(1 for k in history if k not in fired)
        self.samples.append(dict(t=round(t - self.started, 3), **{k: v for k, v in values.items()}))
        for r in triggered:
            what = self.policy['action_mapping'][r['action']]
            record = dict(t=round(t - self.started, 3), utc=utc(), quantity=r['quantity'], value=r['value'],
                          op=r['op'], threshold=r['threshold'], preregistered_action=r['action'], applied=what)
            if what == 'stop':
                self.annotations.append(record)
                self.stop_reasons.append(f"{r['quantity']}={r['value']} violates {r['op']} {r['threshold']}"
                                         f" ({r['action']})")
            elif r['quantity'] not in self.annotated_episodes:
                self.annotations.append(record)
            if what != 'stop':
                self.annotated_episodes.add(r['quantity'])
        for k in list(self.annotated_episodes):
            if k not in history:
                self.annotated_episodes.discard(k)      # episode over; a new one is annotated again

    def summary(self):
        return dict(schema=SCHEMA, read_only=True, policy=self.policy, criteria_source=
                    'protocol.json host_memory_preflight.abort_criteria (frozen)',
                    counts=dict(self.counts), dropped_lines=getattr(self.source, 'dropped', 0),
                    extremes={k: dict(min=a, max=b) for k, (a, b) in sorted(self.extremes.items())},
                    annotations=list(self.annotations), stopped=bool(self.stop_reasons),
                    stop_reasons=list(self.stop_reasons), errors=list(self.errors), last_samples=list(self.samples))

    # ------------------------------------------------------------ repository wiring

    @classmethod
    def for_acquisition(cls, baseline_path, destination, backing, *, protocol_path=PROTOCOL,
                        usage=shutil.disk_usage):
        """Production monitor: frozen abort criteria, amendment-002 policy, powershell sampler."""
        from . import protocol10
        protocol = json.loads(Path(protocol_path).read_text())
        policy = protocol10.effective(protocol_path).get('acquisition_host_monitor')
        if policy is None:
            raise ValueError('no preregistered acquisition_host_monitor policy (amendment 002)')
        baseline = json.loads(Path(baseline_path).read_text())['after']
        source = StreamSource.powershell(policy['sample_interval_s'], max_line_bytes=policy['max_line_bytes'])
        return cls(source, protocol['host_memory_preflight']['abort_criteria'], baseline, policy,
                   destination, backing, usage=usage)

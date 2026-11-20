"""Deterministic host-sample sources for the increment-10 in-run monitor tests (amendment 002).

Not a test module. A FakeClock advances by `step` seconds at every drain of the
ScriptedSource, which releases one scripted item per drain: a Windows sample dict,
'error', 'eof' or None (nothing arrived). Scripts may look at the destination on disk,
so a host transition can be tied to "the first shard is being written" instead of to
fragile chunk counts.
"""
from collections import namedtuple
import json
from pathlib import Path

from tools import hostmonitor10 as hmon
from tools import protocol10

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT/'results'/'increment10'/'protocol.json').read_text())
ABORT = PROTOCOL['host_memory_preflight']['abort_criteria']
POLICY = protocol10.effective()['acquisition_host_monitor']
GiB, MiB = 2**30, 2**20
Usage = namedtuple('Usage', 'total used free')
ROOMY = Usage(10**15, 0, 10**15)


class FakeClock:
    def __init__(self, step=5.0):
        self.t, self.step = 0.0, step

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def win(t, available=4*GiB, pf_mb=7000, pages_per_s=0, committed=40*GiB, limit=50*GiB, pages_total=None):
    """One raw Windows sample at fake time t; PagesInput is a cumulative counter."""
    return dict(TotalPhysicalMemory=16_900_739_072, DimmBytes=17_179_869_184, AvailableBytes=available,
                CommittedBytes=committed, CommitLimit=limit, PagesInput=int(pages_per_s*t) if pages_total is None else pages_total, PagesOutput=0,
                PageReads=0, Timestamp100ns=int(t*10**7), Frequency100ns=10_000_000,
                PageFiles=[dict(Name='C:\\pagefile.sys', AllocatedBaseSize=34816, CurrentUsage=pf_mb,
                                PeakUsage=13799)], CFreeBytes=150*10**9)


def guest(mem_available=7*GiB):
    return dict(meminfo=dict(MemTotal=11_483_848_704, MemAvailable=mem_available, SwapTotal=8*GiB,
                             SwapFree=8*GiB-500*MiB), psi=dict(some=dict(avg10=0.0)), vmstat={}, user_cgroup={})


BASELINE = dict(utc='2026-11-18T21:00:00Z', windows=win(0), guest=guest())


class ScriptedSource:
    def __init__(self, script, clock):
        self.script, self.clock = script, clock
        self.index, self.started, self.closed = 0, False, False
        self.dropped = 0

    def start(self):
        self.started = True

    def drain(self):
        self.clock.t += self.clock.step
        item = self.script(self.index, self.clock.t)
        self.index += 1
        if item is None:
            return []
        if item in ('error', 'eof'):
            return [(item, self.clock.t, 'scripted '+item)]
        return [('sample', self.clock.t, item)]

    def close(self):
        self.closed = True


def monitor(script, destination, backing, *, guest_reader=None, usage=None, clock=None, policy=None):
    clock = clock or FakeClock()
    source = ScriptedSource(script, clock)
    m = hmon.HostMonitor(source, ABORT, BASELINE, policy or POLICY, destination, backing,
                         usage=usage or (lambda p: ROOMY), clock=clock, guest=guest_reader or (lambda: guest()),
                         sleep=clock.sleep)
    return m


def healthy(destination, backing):
    """A monitor whose host stays well inside every abort criterion."""
    return monitor(lambda i, t: win(t), destination, backing)

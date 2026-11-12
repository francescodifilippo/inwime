"""Bounded observation only: never owns requests, leases or KV transitions."""

import threading
import time


# Trace columns: monotonic_ns, kind, request_id, slot, value.
EVENTS = {'read_begin': 1, 'read_end': 2, 'compute_begin': 3, 'compute_end': 4,
          'reserve': 5, 'free': 6, 'ready': 7, 'cancel': 8,
          'wait_begin': 9, 'wait_end': 10}


class Timeline:
    def __init__(self, trace=None):
        self.lock = threading.Lock()
        self.trace = trace
        self.used = self.dropped = 0
        self.origin = self.last = time.monotonic_ns()
        self.readers = self.computers = self.occupied = self.peak = 0
        self.read_ns = self.read_union_ns = self.compute_ns = self.overlap_ns = 0
        self.wait_ns = self.occupancy_slot_ns = 0

    def _advance(self, now):
        elapsed = now-self.last
        self.read_ns += elapsed*self.readers
        self.read_union_ns += elapsed*bool(self.readers)
        self.compute_ns += elapsed*bool(self.computers)
        self.overlap_ns += elapsed*bool(self.readers and self.computers)
        self.occupancy_slot_ns += elapsed*self.occupied
        self.last = now

    def event(self, kind, request=0, slot=-1, value=0):
        with self.lock:
            now = time.monotonic_ns()
            self._advance(now)
            if kind == 'read_begin':
                self.readers += 1
            elif kind == 'read_end':
                self.readers -= 1
            elif kind == 'compute_begin':
                self.computers += 1
            elif kind == 'compute_end':
                self.computers -= 1
            elif kind == 'reserve':
                self.occupied += 1
                self.peak = max(self.peak, self.occupied)
            elif kind == 'free':
                self.occupied -= 1
            elif kind == 'wait_end':
                self.wait_ns += value
            if self.trace is not None and self.used+5 <= len(self.trace):
                for j, v in enumerate((now, EVENTS[kind], request, slot, value)):
                    self.trace[self.used+j] = v
                self.used += 5
            elif self.trace is not None:
                self.dropped += 1

    def stats(self):
        with self.lock:
            self._advance(time.monotonic_ns())
            return {'clock': 'time.monotonic_ns', 'origin_ns': self.origin,
                    'read_service_ns': self.read_ns, 'read_union_ns': self.read_union_ns,
                    'compute_ns': self.compute_ns, 'overlap_ns': self.overlap_ns,
                    'data_wait_ns': self.wait_ns, 'occupancy_slot_ns': self.occupancy_slot_ns,
                    'peak_occupied_slots': self.peak, 'occupied_slots': self.occupied,
                    'trace_events': self.used//5, 'trace_dropped': self.dropped}

    def export(self):
        """Explicit diagnostic copy, outside engine arenas; bounded by trace capacity."""
        with self.lock:
            return [list(self.trace[i:i+5]) for i in range(0, self.used, 5)]

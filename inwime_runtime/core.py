"""One authority for buffer reservations, region requests and file completions.

No dependency on the resident reference. Numeric payloads use preallocated
stdlib arrays; interpreter/allocator overhead is observed separately as RSS.
"""

from array import array
from dataclasses import dataclass, fields, is_dataclass
from contextlib import contextmanager
from bisect import bisect_right
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import threading
import time
from concurrent.futures import CancelledError

from .telemetry import Timeline


class BudgetError(MemoryError):
    pass


class CheckpointError(ValueError):
    pass


class ReadError(OSError):
    pass


def expand_bf16(raw, count, destination, offset=0):
    """Exact BF16 -> FP64 expansion; rejects nonfinite checkpoint values."""
    for i in range(count):
        word = struct.unpack_from('<H', raw, i*2)[0]
        exponent, fraction = (word >> 7) & 255, word & 127
        if exponent == 255:
            raise CheckpointError('nonfinite BF16 weight')
        value = (math.ldexp(fraction, -133) if exponent == 0
                 else math.ldexp(128+fraction, exponent-134))
        destination[offset+i] = -value if word & 0x8000 else value


def deep_size(obj, seen=None):
    """Retained Python metadata size; ndarray/arena payloads are not passed here."""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    total = sys.getsizeof(obj)
    if isinstance(obj, dict):
        total += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in obj.items())
    elif isinstance(obj, (tuple, list)):
        total += sum(deep_size(v, seen) for v in obj)
    elif is_dataclass(obj):
        total += sum(deep_size(getattr(obj, f.name), seen) for f in fields(obj))
    return total


@dataclass(frozen=True, slots=True)
class Spec:
    layers: int
    width: int
    heads: int
    hidden: int
    vocab: int
    context: int
    experts: int
    top_k: int
    rms_epsilon: float


@dataclass(frozen=True, slots=True)
class Segment:
    start: int
    shard: str
    offset: int
    elements: int


@dataclass(frozen=True, slots=True)
class Tensor:
    shape: tuple
    shard: str
    offset: int
    elements: int
    segments: tuple
    starts: tuple


def decoder_schema(config, variant, tensor_count):
    """Validate the existing synthetic model separately from its container."""
    c = Spec(**config)
    if variant not in ("dense", "moe"):
        raise CheckpointError("unsupported decoder variant")
    sizes = [getattr(c, f.name) for f in fields(c) if f.name != "rms_epsilon"]
    if (any(type(n) is not int or n <= 0 for n in sizes)
            or c.width % c.heads or c.top_k > c.experts
            or not math.isfinite(c.rms_epsilon) or c.rms_epsilon <= 0):
        raise CheckpointError("invalid decoder dimensions")
    count = 4 + c.layers * (9 if variant == "dense" else 7 + 3*c.experts)
    if tensor_count != count:
        raise CheckpointError("unexpected tensor/shard count")
    expected = {"embedding": (c.vocab, c.width), "position": (c.context, c.width),
                "final_norm": (c.width,), "lm_head": (c.vocab, c.width)}
    for layer in range(c.layers):
        p = f"layers.{layer}."
        for n in ("norm1", "norm2"):
            expected[p+n] = (c.width,)
        for n in ("q", "k", "v", "o"):
            expected[p+n] = (c.width, c.width)
        prefixes = [p+"ffn."] if variant == "dense" else [
            p+f"experts.{e}." for e in range(c.experts)]
        if variant == "moe":
            expected[p+"router"] = (c.experts, c.width)
        for f in prefixes:
            expected[f+"gate"] = expected[f+"up"] = (c.hidden, c.width)
            expected[f+"down"] = (c.width, c.hidden)
    return c, expected


class IndexedSource:
    """Normalized physical tensor index consumed by both I/O coordinators."""

    def region(self, name, start, count):
        if name not in self.tensors:
            raise CheckpointError("unknown tensor: " + name)
        tensor = self.tensors[name]
        if (type(start) is not int or type(count) is not int or start < 0
                or count <= 0 or start + count > tensor.elements):
            raise CheckpointError("region outside tensor")
        return tensor, None if tensor.offset is None else tensor.offset + start*2


    def spans(self, name, start, count, max_elements):
        """Lazy physical pieces (shard, byte offset, elements, destination index)."""
        if type(max_elements) is not int or max_elements <= 0:
            raise ValueError("positive maximum span required")
        tensor, _ = self.region(name, start, count)
        index = bisect_right(tensor.starts, start)-1
        done = 0
        while done < count:
            part = tensor.segments[index]
            relative = start+done-part.start
            number = min(count-done, part.elements-relative, max_elements)
            yield part.shard, part.offset+relative*2, number, done
            done += number
            if relative+number == part.elements:
                index += 1


class Catalog(IndexedSource):
    """Index only. The mathematical adapter supports the increment-1 schema."""

    def __init__(self, directory, max_manifest_bytes=16*1024*1024):
        self.directory = Path(directory)
        path = self.directory / "manifest.json"
        if path.stat().st_size > max_manifest_bytes:
            raise CheckpointError("manifest exceeds configured input bound")
        with path.open('rb') as source:
            raw = source.read(max_manifest_bytes+1)
        if len(raw) > max_manifest_bytes:
            raise CheckpointError("fixture manifest grew beyond its input bound")
        def unique_pairs(pairs):
            d = {}
            for k, v in pairs:
                if k in d:
                    raise CheckpointError("duplicate manifest key")
                d[k] = v
            return d
        try:
            m = json.loads(raw, object_pairs_hook=unique_pairs)
            if (m["format"] not in ("inwime-reference-bf16-v1", "inwime-segmented-bf16-v2") or m["byte_order"] != "little"
                    or m["layout"] != "C" or m["variant"] not in ("dense", "moe")):
                raise CheckpointError("unsupported manifest format")
            self.variant = m["variant"]
            self.spec, expected = decoder_schema(m["config"], self.variant, len(m["tensors"]))
            if not m["files"] or set(expected) != set(m["tensors"]):
                raise CheckpointError("unexpected tensor names/shards")
            self.files = {}
            for name, entry in m["files"].items():
                if (Path(name).name != name or name in (".", "..") or "\\" in name
                        or type(entry["nbytes"]) is not int or not 0 < entry["nbytes"] < 2**63
                        or len(entry["sha256"]) != 64
                        or any(x not in "0123456789abcdef" for x in entry["sha256"])):
                    raise CheckpointError("invalid source descriptor")
                self.files[name] = (entry["nbytes"], entry["sha256"])
            self.tensors = {}
            intervals = {name: [] for name in self.files}
            for name, shape in expected.items():
                d = m["tensors"][name]
                elements = math.prod(shape)
                if (d["dtype"] != "BF16" or tuple(d["shape"]) != shape
                        or type(d["nbytes"]) is not int or d["nbytes"] != elements*2):
                    raise CheckpointError("invalid tensor region: " + name)
                descriptors = (d["segments"] if m["format"] == "inwime-segmented-bf16-v2"
                               else [dict(start=0, shard=d["shard"], offset=d["offset"],
                                          nbytes=d["nbytes"])])
                segments, end = [], 0
                for part in descriptors:
                    start, offset, size = part["start"], part["offset"], part["nbytes"]
                    shard = part["shard"]
                    if (any(type(x) is not int for x in (start, offset, size))
                            or start != end or offset < 0 or offset % 2 or size <= 0
                            or size % 2 or shard not in self.files
                            or offset + size > self.files[shard][0]):
                        raise CheckpointError("missing, overlapping or invalid segment: " + name)
                    segments.append(Segment(start, shard, offset, size//2))
                    end += size//2
                    intervals[shard].append((offset, offset+size))
                if end != elements:
                    raise CheckpointError("incomplete tensor: " + name)
                first = segments[0]
                self.tensors[name] = Tensor(shape, first.shard if len(segments) == 1 else None,
                                            first.offset if len(segments) == 1 else None, elements,
                                            tuple(segments), tuple(x.start for x in segments))
            for name, spans in intervals.items():
                end = 0
                for start, stop in sorted(spans):
                    if start != end:
                        raise CheckpointError("overlap or gap in fixture shard")
                    end = stop
                if end != self.files[name][0]:
                    raise CheckpointError("unindexed shard data")
        except (KeyError, TypeError, OverflowError, json.JSONDecodeError) as e:
            raise CheckpointError("malformed fixture manifest") from e
        self.identity = hashlib.sha256(raw).hexdigest()
        self.payload_bytes = sum(size for size, _ in self.files.values())
        self.retained_bytes = deep_size((self.spec, self.variant, self.files, self.tensors,
                                         self.identity, str(self.directory)))
        # Account the overlapping raw/decoded/compact metadata during bootstrap.
        # JSON parser's transient internal allocator remains part of process RSS.
        self.bootstrap_bytes = (deep_size(raw) + deep_size(m) + self.retained_bytes
                                + deep_size(expected) + deep_size(intervals))
CATEGORIES = ("weights", "staging", "activations", "kv", "accumulators", "workspace", "metadata",
              "output", "requests")


@dataclass(frozen=True, slots=True)
class Budget:
    total: int
    categories: dict


@dataclass(frozen=True, slots=True)
class Plan:
    arrays: dict
    required: dict
    capacity: int
    tile_elements: int
    read_elements: int
    io_slots: int = 0
    io_workers: int = 0

    @property
    def total(self):
        return sum(self.required.values())

    def budget(self):
        return Budget(self.total, dict(self.required))


def memory_plan(catalog, capacity, tile_elements, read_elements, *, io_mode='sync',
                in_flight=2, io_workers=1, trace_capacity=0):
    c = catalog.spec
    if (type(capacity) is not int or not 1 <= capacity <= c.context
            or type(tile_elements) is not int or tile_elements <= 0
            or type(read_elements) is not int or not 1 <= read_elements <= tile_elements):
        raise ValueError("invalid capacity/tile/read dimensions")
    if (io_mode not in ('sync', 'pipeline') or type(in_flight) is not int or in_flight < 1
            or type(io_workers) is not int or not 1 <= io_workers <= in_flight
            or type(trace_capacity) is not int or trace_capacity < 0):
        raise ValueError("invalid I/O mode, request/worker capacity or trace capacity")
    slots = in_flight if io_mode == 'pipeline' else 0
    arrays = {}
    def add(name, category, count, code="d"):
        arrays[name] = (category, count, code)
    add("weights", "weights", tile_elements)
    add("staging", "staging", (tile_elements if slots else read_elements)*2, "B")
    for slot in range(1, slots):
        add(f"weights{slot}", "weights", tile_elements)
        add(f"staging{slot}", "staging", tile_elements*2, "B")
    if trace_capacity:
        add("trace", "metadata", trace_capacity*5, "q")
    for n in ("x", "norm", "q", "attention", "delta"):
        add(n, "activations", c.width)
    for n in ("gate", "up"):
        add(n, "activations", c.hidden)
    add("logits", "output", capacity*c.vocab)
    add("hidden", "output", capacity*c.layers*c.width)
    if catalog.variant == "moe":
        add("expert", "activations", c.width)
        add("scores", "activations", c.experts)
        add("route_ids", "output", capacity*c.layers*c.top_k, "q")
        add("route_probs", "output", capacity*c.layers*c.top_k)
        add("selected", "workspace", c.top_k, "q")
        add("probabilities", "workspace", c.top_k)
    for n in ("kv0", "kv1"):
        add(n, "kv", c.layers*2*capacity*c.width)
    add("accumulator", "accumulators", c.width//c.heads+1)
    required = dict.fromkeys(CATEGORIES, 0)
    headers = 0
    for _, (category, count, code) in arrays.items():
        required[category] += count*array(code).itemsize
        headers += sys.getsizeof(array(code))
    counters = {name: 0 for name in catalog.tensors}
    # Fixed control reservation: one request/lease/result, handles, counters,
    # bounded token arguments, scalar Python temporaries and arena descriptors.
    control = 16384 + capacity*128 + len(arrays)*256 + deep_size(counters) + deep_size(arrays)
    required["metadata"] += max(catalog.bootstrap_bytes,
                                catalog.retained_bytes + control + headers)
    # Bounded descriptors, futures, completions and worker controls. Python's
    # allocator and OS thread stacks are observed separately as process RSS.
    required["requests"] = slots*8192 + (io_workers*8192 if slots else 0)
    return Plan(arrays, required, capacity, tile_elements, read_elements, slots,
                io_workers if slots else 0)


class FileBackend:
    """Read-only local regular files; explicit offset, caller-owned destination."""

    _seek_lock = threading.Lock()

    def readinto(self, fd, offset, destination):
        if hasattr(os, "preadv"):
            return os.preadv(fd, [destination], offset)
        # Portable seek/readv fallback is serialized; preadv remains concurrent.
        with self._seek_lock:
            os.lseek(fd, offset, os.SEEK_SET)
            return os.readv(fd, [destination])


def signature(fd):
    s = os.fstat(fd)
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


class RegionLease:
    __slots__ = ("owner", "generation", "count", "released")

    def __init__(self, owner, generation, count):
        self.owner, self.generation, self.count = owner, generation, count
        self.released = False

    def value(self, index):
        self.owner.check_thread()
        if (self.released or self.owner.closed or self.generation != self.owner.generation
                or not self.owner.active or not 0 <= index < self.count):
            raise RuntimeError("expired or invalid region reference")
        self.owner.request_state = "IN_USE"
        return self.owner.buffers["weights"][index]

    def release(self):
        self.owner.check_thread()
        if not self.released:
            if self.generation != self.owner.generation or not self.owner.active:
                raise RuntimeError("invalid release")
            self.owner.active = False
            self.owner.request_state = "RELEASED"
            self.owner.released += 1
            self.owner.timeline.event('compute_end', self.generation)
            self.owner.timeline.event('free', self.generation)
            self.released = True

    def linear(self, backend, source, destination, columns, start, offset=0):
        """Synchronous trusted math adapter; the arena remains leased until return."""
        self.value(0)  # validate ownership, generation and liveness
        backend.linear(self.owner.buffers['weights'], self.count, source, destination,
                       columns, start, offset)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class Coordinator:
    """Single reservation/state authority, one synchronous region in flight."""

    def __init__(self, catalog, plan, budget=None, backend=None, *, integrity="full",
                 math_backend=None):
        if integrity not in ("full", "metadata"):
            raise ValueError("integrity must be full or metadata (explicitly unverified payload)")
        self.integrity_policy = integrity
        self.catalog, self.plan = catalog, plan
        budget = plan.budget() if budget is None else budget
        if (type(budget.total) is not int or budget.total < plan.total
                or set(budget.categories) != set(CATEGORIES)
                or any(type(budget.categories[k]) is not int
                       or budget.categories[k] < plan.required[k] for k in CATEGORIES)):
            raise BudgetError("budget below the calculated feasible plan")
        self.budget = Budget(budget.total, dict(budget.categories))
        self.owner_thread = threading.get_ident()
        self.buffers, self.fds, self.stamps = {}, {}, {}
        self.backend = FileBackend() if backend is None else backend
        self.expand = expand_bf16 if math_backend is None else math_backend.expand
        self.generation = self.issued = self.completed = self.failed = self.released = 0
        self.read_calls = self.integrity_bytes = self.inference_bytes = self.max_read = 0
        self.logical_requested_bytes = 0
        self.expand_ns = 0
        self.gemv_ns = self.gemv_elements = 0
        self.weight_peak = self.region_peak = 0
        self.active, self.closed, self.request_state = False, False, "IDLE"
        self.tensor_bytes = {name: 0 for name in catalog.tensors}
        self.cancel_event = None
        self.timeline = Timeline()
        try:
            for name, (_, count, code) in plan.arrays.items():
                self.buffers[name] = array(code, [0])*count
            self.timeline = Timeline(self.buffers.get('trace'))
            if array("d").itemsize != 8 or array("q").itemsize != 8:
                raise RuntimeError("this CPU adapter requires 64-bit doubles and indices")
            self.weight_peak = len(self.buffers["weights"])*8
            for name, (size, _) in catalog.files.items():
                fd = os.open(catalog.directory / name, os.O_RDONLY
                             | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
                self.fds[name] = fd
                s = os.fstat(fd)
                if not stat.S_ISREG(s.st_mode) or s.st_size != size:
                    raise CheckpointError("source is not a regular file of the expected size")
                self.stamps[name] = signature(fd)
                if (hasattr(catalog, 'source_stamps')
                        and self.stamps[name] != catalog.source_stamps[name]):
                    raise CheckpointError("source changed since metadata admission")
            for name, (size, expected) in (catalog.files.items() if integrity == "full" else ()):
                digest = hashlib.sha256()
                for offset in range(0, size, plan.read_elements*2):
                    count = min(plan.read_elements*2, size-offset)
                    view = self._read(name, offset, count, integrity=True)
                    digest.update(view)
                if digest.hexdigest() != expected:
                    raise CheckpointError("checkpoint checksum mismatch")
            self.integrity_read_ns = self.timeline.stats()['read_service_ns']
            # Startup integrity traffic is reported separately from inference.
            self.timeline = Timeline(self.buffers.get('trace'))
        except BaseException:
            self.close()
            raise

    def _read(self, name, offset, size, integrity=False):
        fd = self.fds[name]
        if signature(fd) != self.stamps[name]:
            raise CheckpointError("source changed after opening")
        view = memoryview(self.buffers["staging"])[:size]
        self.read_calls += 1
        self.timeline.event('read_begin', self.generation)
        try:
            n = self.backend.readinto(fd, offset, view)
        finally:
            self.timeline.event('read_end', self.generation)
        if type(n) is not int or not 0 <= n <= size:
            raise ReadError("invalid backend completion")
        self.max_read = max(self.max_read, size)
        if integrity:
            self.integrity_bytes += n
        else:
            self.inference_bytes += n
        if n != size:
            raise ReadError("short read: region is incomplete")
        if signature(fd) != self.stamps[name]:
            raise CheckpointError("source changed during reading")
        return view

    def acquire(self, name, start, count):
        self.check_thread()
        self.check_cancelled()
        if self.closed or self.active:
            raise RuntimeError("closed coordinator or a region is still in use")
        tensor, offset = self.catalog.region(name, start, count)
        if count > self.plan.tile_elements:
            raise BudgetError("region exceeds the reserved weight slot")
        self.logical_requested_bytes += count*2
        self.generation += 1
        self.issued += 1
        self.request_state, self.active = "READING", True
        self.timeline.event('reserve', self.generation)
        self.timeline.event('wait_begin', self.generation)
        waited = 0
        try:
            for shard, offset, number, done in self.catalog.spans(
                    name, start, count, self.plan.read_elements):
                self.check_cancelled()
                began = time.monotonic_ns()
                raw = self._read(shard, offset, number*2)
                waited += time.monotonic_ns()-began
                self.tensor_bytes[name] += len(raw)
                expand_began = time.monotonic_ns()
                self.expand(raw, number, self.buffers['weights'], done)
                self.expand_ns += time.monotonic_ns()-expand_began
            self.completed += 1
            self.region_peak = max(self.region_peak, count)
            self.request_state = "READY"
            self.timeline.event('wait_end', self.generation, value=waited)
            self.timeline.event('compute_begin', self.generation)
            return RegionLease(self, self.generation, count)
        except BaseException:
            self.active = False
            self.request_state = "FAILED"
            self.failed += 1
            self.timeline.event('free', self.generation)
            raise

    def check_cancelled(self):
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise CancelledError('inference cancelled')

    def stream(self, regions):
        """Consume a lazy, deterministic stream without retaining an operation graph."""
        for name, start, count in regions:
            with self.acquire(name, start, count) as lease:
                yield start, count, lease

    def abort(self):
        self.check_thread()

    def stats(self):
        self.check_thread()
        metadata = self.catalog.retained_bytes + deep_size((
            self.plan, self.budget, self.fds, self.stamps, self.tensor_bytes,
            self.generation, self.issued, self.completed, self.failed, self.released,
            self.read_calls, self.integrity_bytes, self.inference_bytes, self.max_read,
            self.owner_thread, self.request_state))
        metadata += sum(sys.getsizeof(b)-len(b)*b.itemsize for b in self.buffers.values())
        metadata += deep_size({k: v for k, v in vars(self.timeline).items() if k != 'trace'})
        if 'trace' in self.buffers:
            metadata += len(self.buffers['trace'])*self.buffers['trace'].itemsize
        return {"integrity_policy": self.integrity_policy,
                "required_by_category": dict(self.plan.required),
                "audited_retained_metadata_bytes": metadata,
                "reserved_peak_bytes": self.plan.total,
                "weight_slot_bytes": self.weight_peak,
                "peak_region_elements": self.region_peak,
                "peak_region_bf16_equivalent_bytes": self.region_peak*2,
                "payload_bytes": self.catalog.payload_bytes,
                "source_file_bytes": sum(size for size, _ in self.catalog.files.values()),
                "source_metadata_read_bytes": getattr(self.catalog, 'metadata_read_bytes', 0),
                "source_header_bytes": getattr(self.catalog, 'header_bytes', 0),
                "integrity_read_bytes": self.integrity_bytes,
                "integrity_read_ns": self.integrity_read_ns,
                "bf16_expand_ns": self.expand_ns,
                "gemv_ns": self.gemv_ns,
                "gemv_weight_elements": self.gemv_elements,
                "gemv_flops": 2*self.gemv_elements,
                "inference_read_bytes": self.inference_bytes,
                "logical_requested_bytes": self.logical_requested_bytes,
                "backend_read_calls": self.read_calls, "max_backend_read_bytes": self.max_read,
                "requests": self.issued, "completed": self.completed, "failed": self.failed,
                "released": self.released, "request_state": self.request_state,
                "active_region": self.active, "tensor_read_bytes": dict(self.tensor_bytes),
                "numeric_buffers_live": len(self.buffers), **self.timeline.stats()}

    def trace(self):
        self.check_thread()
        return self.timeline.export()

    @contextmanager
    def compute(self):
        """Numeric CPU phases outside a weight lease, on the owning thread."""
        self.check_thread()
        self.timeline.event('compute_begin')
        try:
            yield
        finally:
            self.timeline.event('compute_end')

    def close(self):
        self.check_thread()
        if self.active:
            raise RuntimeError("cannot close while a region is in use")
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()
        with self.timeline.lock:
            self.timeline.trace = None
        self.buffers.clear()
        self.closed = True

    def check_thread(self):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("synchronous coordinator must be used by its owning thread")


def open_catalog(directory, *, source='fixture'):
    """Explicit container selection; no probing or payload conversion."""
    if source == 'fixture':
        return Catalog(directory)
    if source == 'safetensors':
        from .safetensors_source import SafetensorsSource
        return SafetensorsSource(directory)
    raise ValueError("source must be fixture or safetensors")

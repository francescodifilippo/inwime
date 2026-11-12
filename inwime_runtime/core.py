"""One authority for buffer reservations, region requests and file completions.

No dependency on the resident reference. Numeric payloads use preallocated
stdlib arrays; interpreter/allocator overhead is observed separately as RSS.
"""

from array import array
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import threading


class BudgetError(MemoryError):
    pass


class CheckpointError(ValueError):
    pass


class ReadError(OSError):
    pass


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
class Tensor:
    shape: tuple
    shard: str
    offset: int
    elements: int


class Catalog:
    """Index only. The mathematical adapter supports the increment-1 schema."""

    def __init__(self, directory):
        self.directory = Path(directory)
        path = self.directory / "manifest.json"
        if path.stat().st_size > 65536:
            raise CheckpointError("fixture manifest exceeds 64 KiB input bound")
        with path.open('rb') as source:
            raw = source.read(65537)
        if len(raw) > 65536:
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
            if (m["format"] != "inwime-reference-bf16-v1" or m["byte_order"] != "little"
                    or m["layout"] != "C" or m["variant"] not in ("dense", "moe")):
                raise CheckpointError("unsupported manifest format")
            self.spec = c = Spec(**m["config"])
            sizes = [getattr(c, f.name) for f in fields(c) if f.name != "rms_epsilon"]
            if (any(type(n) is not int or n <= 0 for n in sizes)
                    or c.width % c.heads or c.top_k > c.experts
                    or not math.isfinite(c.rms_epsilon) or c.rms_epsilon <= 0):
                raise CheckpointError("invalid decoder dimensions")
            self.variant = m["variant"]
            count = 4 + c.layers * (9 if self.variant == "dense" else 7 + 3*c.experts)
            if len(m["tensors"]) != count or len(m["files"]) != 2:
                raise CheckpointError("unexpected tensor/shard count")
            expected = {"embedding": (c.vocab, c.width), "position": (c.context, c.width),
                        "final_norm": (c.width,), "lm_head": (c.vocab, c.width)}
            for layer in range(c.layers):
                p = f"layers.{layer}."
                for n in ("norm1", "norm2"):
                    expected[p+n] = (c.width,)
                for n in ("q", "k", "v", "o"):
                    expected[p+n] = (c.width, c.width)
                prefixes = [p+"ffn."] if self.variant == "dense" else [
                    p+f"experts.{e}." for e in range(c.experts)]
                if self.variant == "moe":
                    expected[p+"router"] = (c.experts, c.width)
                for f in prefixes:
                    expected[f+"gate"] = expected[f+"up"] = (c.hidden, c.width)
                    expected[f+"down"] = (c.width, c.hidden)
            if set(expected) != set(m["tensors"]):
                raise CheckpointError("unexpected tensor names")
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
                        or d["shard"] not in self.files or type(d["offset"]) is not int
                        or d["offset"] < 0 or d["offset"] % 2
                        or type(d["nbytes"]) is not int or d["nbytes"] != elements*2
                        or d["offset"] + d["nbytes"] > self.files[d["shard"]][0]):
                    raise CheckpointError("invalid tensor region: " + name)
                self.tensors[name] = Tensor(shape, d["shard"], d["offset"], elements)
                intervals[d["shard"]].append((d["offset"], d["offset"]+d["nbytes"]))
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

    def region(self, name, start, count):
        if name not in self.tensors:
            raise CheckpointError("unknown tensor: " + name)
        tensor = self.tensors[name]
        if (type(start) is not int or type(count) is not int or start < 0
                or count <= 0 or start + count > tensor.elements):
            raise CheckpointError("region outside tensor")
        return tensor, tensor.offset + start*2


CATEGORIES = ("weights", "staging", "activations", "kv", "accumulators", "workspace", "metadata")


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

    @property
    def total(self):
        return sum(self.required.values())

    def budget(self):
        return Budget(self.total, dict(self.required))


def memory_plan(catalog, capacity, tile_elements, read_elements):
    c = catalog.spec
    if (type(capacity) is not int or not 1 <= capacity <= c.context
            or type(tile_elements) is not int or tile_elements <= 0
            or type(read_elements) is not int or not 1 <= read_elements <= tile_elements):
        raise ValueError("invalid capacity/tile/read dimensions")
    arrays = {}
    def add(name, category, count, code="d"):
        arrays[name] = (category, count, code)
    add("weights", "weights", tile_elements)
    add("staging", "staging", read_elements*2, "B")
    for n in ("x", "norm", "q", "attention", "delta"):
        add(n, "activations", c.width)
    for n in ("gate", "up"):
        add(n, "activations", c.hidden)
    add("logits", "activations", capacity*c.vocab)
    add("hidden", "activations", capacity*c.layers*c.width)
    if catalog.variant == "moe":
        add("expert", "activations", c.width)
        add("scores", "activations", c.experts)
        add("route_ids", "activations", capacity*c.layers*c.top_k, "q")
        add("route_probs", "activations", capacity*c.layers*c.top_k)
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
    required["metadata"] = max(catalog.bootstrap_bytes,
                               catalog.retained_bytes + control + headers)
    return Plan(arrays, required, capacity, tile_elements, read_elements)


class FileBackend:
    """Read-only local regular files; explicit offset, caller-owned destination."""

    def readinto(self, fd, offset, destination):
        if hasattr(os, "preadv"):
            return os.preadv(fd, [destination], offset)
        os.lseek(fd, offset, os.SEEK_SET)
        # This synchronous fallback has no shared concurrent file position.
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
            self.released = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


class Coordinator:
    """Single reservation/state authority, one synchronous region in flight."""

    def __init__(self, catalog, plan, budget=None, backend=None):
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
        self.generation = self.issued = self.completed = self.failed = self.released = 0
        self.read_calls = self.integrity_bytes = self.inference_bytes = self.max_read = 0
        self.weight_peak = self.region_peak = 0
        self.active, self.closed, self.request_state = False, False, "IDLE"
        self.tensor_bytes = {name: 0 for name in catalog.tensors}
        try:
            for name, (_, count, code) in plan.arrays.items():
                self.buffers[name] = array(code, [0])*count
            if array("d").itemsize != 8 or array("q").itemsize != 8:
                raise RuntimeError("this CPU adapter requires 64-bit doubles and indices")
            self.weight_peak = len(self.buffers["weights"])*8
            for name, (size, _) in catalog.files.items():
                fd = os.open(catalog.directory / name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                self.fds[name] = fd
                s = os.fstat(fd)
                if not stat.S_ISREG(s.st_mode) or s.st_size != size:
                    raise CheckpointError("source is not a regular file of the expected size")
                self.stamps[name] = signature(fd)
            for name, (size, expected) in catalog.files.items():
                digest = hashlib.sha256()
                for offset in range(0, size, len(self.buffers["staging"])):
                    count = min(len(self.buffers["staging"]), size-offset)
                    view = self._read(name, offset, count, integrity=True)
                    digest.update(view)
                if digest.hexdigest() != expected:
                    raise CheckpointError("checkpoint checksum mismatch")
        except BaseException:
            self.close()
            raise

    def _read(self, name, offset, size, integrity=False):
        fd = self.fds[name]
        if signature(fd) != self.stamps[name]:
            raise CheckpointError("source changed after opening")
        view = memoryview(self.buffers["staging"])[:size]
        self.read_calls += 1
        n = self.backend.readinto(fd, offset, view)
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
        if self.closed or self.active:
            raise RuntimeError("closed coordinator or a region is still in use")
        tensor, offset = self.catalog.region(name, start, count)
        if count > self.plan.tile_elements:
            raise BudgetError("region exceeds the reserved weight slot")
        self.generation += 1
        self.issued += 1
        self.request_state, self.active = "READING", True
        try:
            for done in range(0, count, self.plan.read_elements):
                number = min(self.plan.read_elements, count-done)
                raw = self._read(tensor.shard, offset+done*2, number*2)
                self.tensor_bytes[name] += len(raw)
                for i in range(number):
                    word = struct.unpack_from("<H", raw, i*2)[0]
                    exponent, fraction = (word >> 7) & 255, word & 127
                    if exponent == 255:
                        raise CheckpointError("nonfinite BF16 weight")
                    value = (math.ldexp(fraction, -133) if exponent == 0
                             else math.ldexp(128+fraction, exponent-134))
                    self.buffers["weights"][done+i] = -value if word & 0x8000 else value
            self.completed += 1
            self.region_peak = max(self.region_peak, count)
            self.request_state = "READY"
            return RegionLease(self, self.generation, count)
        except BaseException:
            self.active = False
            self.request_state = "FAILED"
            self.failed += 1
            raise

    def stats(self):
        self.check_thread()
        metadata = self.catalog.retained_bytes + deep_size((
            self.plan, self.budget, self.fds, self.stamps, self.tensor_bytes,
            self.generation, self.issued, self.completed, self.failed, self.released,
            self.read_calls, self.integrity_bytes, self.inference_bytes, self.max_read,
            self.owner_thread, self.request_state))
        metadata += sum(sys.getsizeof(b)-len(b)*b.itemsize for b in self.buffers.values())
        return {"required_by_category": dict(self.plan.required),
                "audited_retained_metadata_bytes": metadata,
                "reserved_peak_bytes": self.plan.total,
                "weight_slot_bytes": self.weight_peak,
                "peak_region_elements": self.region_peak,
                "peak_region_bf16_equivalent_bytes": self.region_peak*2,
                "payload_bytes": self.catalog.payload_bytes,
                "integrity_read_bytes": self.integrity_bytes,
                "inference_read_bytes": self.inference_bytes,
                "backend_read_calls": self.read_calls, "max_backend_read_bytes": self.max_read,
                "requests": self.issued, "completed": self.completed, "failed": self.failed,
                "released": self.released, "request_state": self.request_state,
                "active_region": self.active, "tensor_read_bytes": dict(self.tensor_bytes),
                "numeric_buffers_live": len(self.buffers)}

    def close(self):
        self.check_thread()
        if self.active:
            raise RuntimeError("cannot close while a region is in use")
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()
        self.buffers.clear()
        self.closed = True

    def check_thread(self):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("synchronous coordinator must be used by its owning thread")

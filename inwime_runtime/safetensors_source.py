"""Read-only BF16 safetensors index; payloads stay in their original files.

The external safetensors.json descriptor supplies expected whole-file SHA256s.
It is a trust input, not an authenticity claim. Decoder adapters interpret any
semantic descriptor fields separately; this source preserves physical names.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct

from .core import (CheckpointError, IndexedSource, Segment, Tensor,
                   deep_size, signature)


def local_name(name):
    if (not isinstance(name, str) or not name or Path(name).name != name
            or name in ('.', '..') or '\\' in name or '\x00' in name):
        raise CheckpointError('invalid local source filename')
    return name


def digest_string(value):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise CheckpointError('expected external SHA256 required')
    return value


def bounded_json(raw):
    # Bound nesting and structural complexity BEFORE Python allocates a parsed
    # object tree. String contents/escapes must not count as JSON structure.
    quoted = escaped = False
    depth = nodes = 0
    for byte in raw:
        if quoted:
            if escaped:
                escaped = False
            elif byte == 92:
                escaped = True
            elif byte == 34:
                quoted = False
        elif byte == 34:
            quoted = True
        elif byte in (91, 123):
            depth += 1
            nodes += 1
        elif byte in (93, 125):
            depth -= 1
        elif byte in (44, 58):
            nodes += 1
        if depth > 16 or nodes > 65536:
            raise CheckpointError('metadata JSON complexity bound exceeded')

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CheckpointError('duplicate metadata key')
            result[key] = value
        return result

    def invalid_constant(value):
        raise CheckpointError('nonfinite JSON constant: '+value)

    result = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                        parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise CheckpointError('metadata root must be an object')
    return result


class SafetensorsSource(IndexedSource):
    def __init__(self, directory, *, max_input_bytes=8*2**20,
                 max_metadata_bytes=16*2**20, max_metadata_heap_bytes=64*2**20,
                 max_shards=64, max_tensors=4096):
        limits = (max_input_bytes, max_metadata_bytes, max_metadata_heap_bytes,
                  max_shards, max_tensors)
        if any(type(v) is not int or v <= 0 for v in limits):
            raise ValueError('positive metadata limits required')
        self.directory = Path(directory)
        self.files, self.tensors, self.source_stamps = {}, {}, {}
        self.metadata_read_bytes = self.header_bytes = self.payload_bytes = 0
        self.bootstrap_bytes = 0
        self.identity = ''
        identity = hashlib.sha256()
        descriptor = None

        def account(raw, parsed):
            self.metadata_read_bytes += len(raw)
            if self.metadata_read_bytes > max_metadata_bytes:
                raise CheckpointError('aggregate metadata input bound exceeded')
            identity.update(raw)
            # Includes a conservative allowance for UTF-8 decoder text, parsed
            # objects, normalization overlap, and the retained descriptor/index.
            heap = (5*len(raw) + 2*deep_size(parsed) + deep_size(descriptor)
                    + deep_size((self.files, self.tensors, self.source_stamps)))
            self.bootstrap_bytes = max(self.bootstrap_bytes, heap)
            if heap > max_metadata_heap_bytes:
                raise CheckpointError('metadata heap bound exceeded')

        def read_exact(fd, offset, count):
            if (count > max_input_bytes
                    or self.metadata_read_bytes+count > max_metadata_bytes):
                raise CheckpointError('metadata input bound exceeded')
            raw = os.pread(fd, count, offset)
            if len(raw) != count:
                raise CheckpointError('truncated metadata')
            return raw

        def open_file(name):
            fd = os.open(self.directory/local_name(name), os.O_RDONLY
                         | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                os.close(fd)
                raise CheckpointError('source must be a regular file')
            return fd

        def read_json_file(name):
            fd = open_file(name)
            try:
                stamp = signature(fd)
                raw = read_exact(fd, 0, stamp[2])
                parsed = bounded_json(raw)
                if signature(fd) != stamp:
                    raise CheckpointError('metadata changed during admission')
                account(raw, parsed)
                return raw, parsed
            finally:
                os.close(fd)

        try:
            _, descriptor = read_json_file('safetensors.json')
            if descriptor['format'] != 'inwime-safetensors-bf16-v1':
                raise CheckpointError('unsupported source descriptor')
            entries = descriptor['files']
            if not isinstance(entries, dict) or not 1 <= len(entries) <= max_shards:
                raise CheckpointError('shard count bound exceeded')
            for name, entry in entries.items():
                local_name(name)
                size, sha = entry['nbytes'], digest_string(entry['sha256'])
                if (not name.endswith('.safetensors') or type(size) is not int
                        or not 8 < size < 2**63):
                    raise CheckpointError('invalid shard size/name')
                self.files[name] = size, sha
            index_name = descriptor['index']
            weight_map = None
            declared_payload_bytes = None
            if index_name is not None:
                raw, index = read_json_file(index_name)
                if hashlib.sha256(raw).hexdigest() != digest_string(descriptor['index_sha256']):
                    raise CheckpointError('index checksum mismatch')
                weight_map = index['weight_map']
                if (not isinstance(weight_map, dict) or not 1 <= len(weight_map) <= max_tensors
                        or any(not isinstance(n, str) or not n or n == '__metadata__'
                               for n in weight_map)):
                    raise CheckpointError('invalid index weight_map/tensor count')
                for name in weight_map.values():
                    if local_name(name) not in self.files:
                        raise CheckpointError('index references an unknown shard')
                if set(weight_map.values()) != set(self.files):
                    raise CheckpointError('unreferenced shard in descriptor')
                metadata = index.get('metadata', {})
                if not isinstance(metadata, dict):
                    raise CheckpointError('invalid index metadata')
                if 'total_size' in metadata:
                    declared_payload_bytes = metadata['total_size']
                    if type(declared_payload_bytes) is not int or declared_payload_bytes < 0:
                        raise CheckpointError('invalid index total_size')
            elif len(self.files) != 1:
                raise CheckpointError('multi-shard source requires an index')

            for shard, (size, _) in self.files.items():
                fd = open_file(shard)
                try:
                    stamp = signature(fd)
                    if stamp[2] != size:
                        raise CheckpointError('shard size mismatch')
                    prefix = read_exact(fd, 0, 8)
                    # Count the prefix as metadata as well as the JSON header.
                    account(prefix, None)
                    length, = struct.unpack('<Q', prefix)
                    if length < 2 or length > size-8:
                        raise CheckpointError('invalid safetensors header length')
                    raw = read_exact(fd, 8, length)
                    if not raw.startswith(b'{'):
                        raise CheckpointError('safetensors header must start with an object')
                    header = bounded_json(raw)
                    account(raw, header)
                    base = 8+length
                    self.header_bytes += base
                    metadata = header.pop('__metadata__', {})
                    if (not isinstance(metadata, dict)
                            or any(not isinstance(v, str) for v in metadata.values())):
                        raise CheckpointError('invalid safetensors __metadata__')
                    if len(self.tensors)+len(header) > max_tensors:
                        raise CheckpointError('tensor count bound exceeded')
                    intervals = []
                    for name, d in header.items():
                        if not name or name in self.tensors:
                            raise CheckpointError('duplicate/empty tensor name across shards')
                        if weight_map is not None and weight_map.get(name) != shard:
                            raise CheckpointError('index/header tensor mismatch')
                        if d['dtype'] != 'BF16':
                            raise CheckpointError('only original BF16 tensors are supported')
                        shape, offsets = d['shape'], d['data_offsets']
                        if (not isinstance(shape, list) or len(shape) > 32
                                or any(type(n) is not int or not 0 <= n < 2**63 for n in shape)
                                or not isinstance(offsets, list) or len(offsets) != 2
                                or any(type(n) is not int or not 0 <= n < 2**63 for n in offsets)):
                            raise CheckpointError('invalid tensor shape/offsets')
                        elements = math.prod(shape)
                        begin, end = offsets
                        if (elements >= 2**62 or begin > end or end > size-base
                                or end-begin != elements*2 or begin % 2):
                            raise CheckpointError('invalid BF16 tensor extent/size')
                        # Absolute file offsets need not be aligned; only the
                        # BF16 payload-relative offset must be element-aligned.
                        part = Segment(0, shard, base+begin, elements)
                        self.tensors[name] = Tensor(tuple(shape), shard, base+begin,
                                                    elements, (part,), (0,))
                        intervals.append((begin, end))
                    end = 0
                    for begin, stop in sorted(intervals):
                        if begin != end:
                            raise CheckpointError('overlap or gap in safetensors payload')
                        end = stop
                    if end != size-base:
                        raise CheckpointError('unindexed safetensors payload tail')
                    self.payload_bytes += end
                    if signature(fd) != stamp:
                        raise CheckpointError('source changed during metadata admission')
                    self.source_stamps[shard] = stamp
                    # Account the normalized index overlapping the parsed header.
                    account_heap = (5*len(raw)+2*deep_size(header)+deep_size(descriptor)
                                    + deep_size((self.files, self.tensors, self.source_stamps,
                                                 weight_map, intervals)))
                    self.bootstrap_bytes = max(self.bootstrap_bytes, account_heap)
                    if account_heap > max_metadata_heap_bytes:
                        raise CheckpointError('metadata heap bound exceeded')
                finally:
                    os.close(fd)
            if weight_map is not None and set(weight_map) != set(self.tensors):
                raise CheckpointError('index contains missing tensors')
            if declared_payload_bytes is not None and declared_payload_bytes != self.payload_bytes:
                raise CheckpointError('index total_size disagrees with payload')
            self.identity = identity.hexdigest()
            self.descriptor = descriptor  # opaque application fields; never bind here
            self.retained_bytes = deep_size((self.descriptor, self.files, self.tensors,
                                            self.identity, str(self.directory), self.source_stamps,
                                            self.metadata_read_bytes, self.header_bytes))
            self.bootstrap_bytes += self.retained_bytes
            if self.bootstrap_bytes > max_metadata_heap_bytes:
                raise CheckpointError('metadata heap bound exceeded')
        except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as error:
            if isinstance(error, CheckpointError):
                raise
            raise CheckpointError('malformed safetensors source metadata') from error

"""A Checkpoint installed in the Hugging Face cache, seen as the frozen runtime's tensor source.

The frozen safetensors loader reads `safetensors.json` from the Checkpoint directory and opens
every shard with O_NOFOLLOW, while a Hugging Face snapshot has no such descriptor and links each
file to a content-addressed blob (`../../blobs/<sha256>` for weights). Nothing is written into
the cache. Instead:

1. every shard link is checked to name the blob whose SHA256 the install manifest verified;
2. the 8-byte prefix and JSON header of each blob are copied into a temporary *sparse* view
   (header bytes allocated, payload left as a hole, apparent size equal to the blob), next to
   a descriptor built from the manifest and a copy of the index;
3. the unchanged `SafetensorsSource` parses and validates that view, with all its bounds;
4. the parsed index is rebound to the real blobs: each shard name becomes the relative link
   target, so the Gestore delle regioni opens `<snapshot>/../../blobs/<sha256>`, a regular
   file, and checks its identity against the one recorded while reading the header.

The view is deleted before this function returns. Payload reads, the SHA256 integrity pass and
the computation are those of the Percorso congelato.
"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import tempfile

from inwime_runtime.core import CheckpointError, IndexedSource, signature
from inwime_runtime.safetensors_source import SafetensorsSource

BLOB_LINK = re.compile(r'\.\./\.\./blobs/([0-9a-f]{64})\Z')
INDEX = 'model.safetensors.index.json'
MAX_HEADER_BYTES = 16*2**20          # the frozen loader's aggregate metadata bound
MAX_INDEX_BYTES = 8*2**20            # the frozen loader's per-file metadata bound


class SnapshotSource(IndexedSource):
    """A parsed `SafetensorsSource` whose shard names point at verified blobs."""

    def __init__(self, parsed, snapshot, links, stamps):
        self.directory = Path(snapshot)
        self.files = {links[name]: entry for name, entry in parsed.files.items()}
        self.tensors = {}
        for name, tensor in parsed.tensors.items():
            self.tensors[name] = replace(tensor, shard=links[tensor.shard], segments=tuple(
                replace(segment, shard=links[segment.shard]) for segment in tensor.segments))
        self.source_stamps = {links[name]: stamps[name] for name in parsed.files}
        for attribute in ('metadata_read_bytes', 'header_bytes', 'payload_bytes', 'bootstrap_bytes',
                          'retained_bytes', 'identity', 'descriptor'):
            setattr(self, attribute, getattr(parsed, attribute))


def blob_link(snapshot, record):
    """The relative link of one weight file, which must name the blob the manifest verified."""
    try:
        target = os.readlink(snapshot/record['name'])
    except OSError as error:
        raise CheckpointError('%s is not a link into the Hugging Face blob store' % record['name']) from error
    found = BLOB_LINK.match(target)
    if not found or found.group(1) != record.get('sha256'):
        raise CheckpointError('%s does not link to the blob verified at install' % record['name'])
    return target


def copy_header(source, destination, expected_size):
    """Prefix and header of `source` into a sparse file of the same size; returns the identity
    of `source` while it was read."""
    fd = os.open(source, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        stamp = signature(fd)
        if stamp[2] != expected_size:
            raise CheckpointError('blob size differs from the install manifest')
        prefix = os.pread(fd, 8, 0)
        if len(prefix) != 8:
            raise CheckpointError('truncated safetensors prefix')
        length, = struct.unpack('<Q', prefix)
        if length > min(MAX_HEADER_BYTES, expected_size-8):
            raise CheckpointError('invalid safetensors header length')
        header = os.pread(fd, length, 8)
        if len(header) != length or signature(fd) != stamp:
            raise CheckpointError('blob changed while its header was read')
    finally:
        os.close(fd)
    with open(destination, 'xb') as f:
        f.write(prefix+header)
        f.truncate(expected_size)
    return stamp


def snapshot_source(manifest):
    """A frozen-loader source over a verified installed Checkpoint (see the module docstring)."""
    snapshot = Path(manifest['snapshot_dir'])
    records = {r['name']: r for r in manifest['files'] if r.get('outcome') == 'verified'}
    shards = sorted(n for n, r in records.items()
                    if n.endswith('.safetensors') and '/' not in n and r.get('kind') == 'lfs')
    if not shards:
        raise CheckpointError('the installed Checkpoint has no verified top-level safetensors weights')
    links = {name: blob_link(snapshot, records[name]) for name in shards}
    descriptor = {'format': 'inwime-safetensors-bf16-v1', 'index': None,
                  'files': {name: {'nbytes': records[name]['size'], 'sha256': records[name]['sha256']}
                            for name in shards}}
    stamps = {}
    with tempfile.TemporaryDirectory(prefix='inwime-snapshot-view-') as view:
        view = Path(view)
        for name in shards:
            stamps[name] = copy_header(snapshot/links[name], view/name, records[name]['size'])
        if INDEX in records:
            with open(snapshot/INDEX, 'rb') as f:
                raw = f.read(MAX_INDEX_BYTES+1)
            if len(raw) > MAX_INDEX_BYTES or hashlib.sha256(raw).hexdigest() != records[INDEX]['sha256']:
                raise CheckpointError('the safetensors index differs from the one verified at install')
            (view/INDEX).write_bytes(raw)
            descriptor.update(index=INDEX, index_sha256=records[INDEX]['sha256'])
        (view/'safetensors.json').write_text(json.dumps(descriptor))
        parsed = SafetensorsSource(view)
    return SnapshotSource(parsed, snapshot, links, stamps)

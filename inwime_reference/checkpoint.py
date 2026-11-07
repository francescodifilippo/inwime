"""Small synthetic BF16 checkpoint, independent of benchmark data and devices."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    layers: int = 2
    width: int = 16
    heads: int = 2
    hidden: int = 24
    vocab: int = 31
    context: int = 32
    experts: int = 3
    top_k: int = 2
    rms_epsilon: float = 1e-5


CONFIG = Config()
SHARDS = ("weights-00001-of-00002.bf16.bin", "weights-00002-of-00002.bf16.bin")


def decode_bf16_le(raw):
    """Exact finite BF16 -> FP64 expansion; no arithmetic rounding to BF16.

    Sign: bit 15; exponent: bits 14..7, bias 127; fraction: bits 6..0.
    The synthetic generator supplies native words, not floating-point inputs.
    """
    if len(raw) % 2:
        raise ValueError("BF16 payload must contain whole 16-bit words")
    bits = np.frombuffer(raw, dtype="<u2")
    fp32_bits = bits.astype(np.uint32) << 16
    values = fp32_bits.view(np.float32).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite BF16 payload")
    return values


def tensor_shapes(variant):
    if variant not in ("dense", "moe"):
        raise ValueError("variant must be dense or moe")
    c = CONFIG
    shapes = {"embedding": (c.vocab, c.width), "position": (c.context, c.width),
              "final_norm": (c.width,), "lm_head": (c.vocab, c.width)}
    for layer in range(c.layers):
        p = f"layers.{layer}."
        for name in ("norm1", "norm2"):
            shapes[p + name] = (c.width,)
        for name in ("q", "k", "v", "o"):
            shapes[p + name] = (c.width, c.width)
        prefixes = [p + "ffn."] if variant == "dense" else [
            p + f"experts.{e}." for e in range(c.experts)]
        if variant == "moe":
            shapes[p + "router"] = (c.experts, c.width)
        for f in prefixes:
            shapes[f + "gate"] = shapes[f + "up"] = (c.hidden, c.width)
            shapes[f + "down"] = (c.width, c.hidden)
    return dict(sorted(shapes.items()))


def generate_checkpoint(output, variant, seed=20261107):
    """Create a new directory exclusively. Generate finite BF16 bits directly.

    No FP32 source checkpoint is rounded or quantized. The integer PRNG and
    byte order are explicit so identical inputs produce identical files.
    """
    if type(seed) is not int or not 1 <= seed <= 0xFFFFFFFF:
        raise ValueError("seed must be a nonzero uint32")
    shapes = tensor_shapes(variant)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    state = seed
    payloads = [bytearray(), bytearray()]
    tensors = {}
    for index, (name, shape) in enumerate(shapes.items()):
        words = []
        for _ in range(int(np.prod(shape))):
            state ^= (state << 13) & 0xFFFFFFFF
            state ^= state >> 17
            state ^= (state << 5) & 0xFFFFFFFF
            # Magnitudes [1/16, 1/2), signs balanced; norm weights are 1.
            word = ((state >> 31) << 15) | ((123 + state % 3) << 7) | ((state >> 8) & 127)
            words.append(0x3F80 if "norm" in name else word)
        raw = np.asarray(words, dtype="<u2").tobytes()
        shard = index % 2
        tensors[name] = {"dtype": "BF16", "shape": list(shape), "shard": SHARDS[shard],
                         "offset": len(payloads[shard]), "nbytes": len(raw)}
        payloads[shard].extend(raw)
    files = {}
    for name, payload in zip(SHARDS, payloads):
        with (output / name).open("xb") as f:
            f.write(payload)
        files[name] = {"nbytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    manifest = {"format": "inwime-reference-bf16-v1", "byte_order": "little",
                "layout": "C", "variant": variant, "seed": seed,
                "generator": "xorshift32-native-bf16-v1", "config": asdict(CONFIG),
                "files": files, "tensors": tensors}
    with (output / "manifest.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    return manifest


@dataclass(frozen=True)
class Checkpoint:
    variant: str
    identity: str
    weights: dict
    payload_bytes: int
    config: Config = CONFIG


def load_checkpoint(directory):
    """Validate this fixed tiny fixture format and load all weights into FP64.

    This is deliberately a resident reference loader, not a general checkpoint
    index or an out-of-core reader. Shard hashes are checked before conversion.
    """
    directory = Path(directory)
    manifest_raw = (directory / "manifest.json").read_bytes()
    m = json.loads(manifest_raw)
    if (m.get("format") != "inwime-reference-bf16-v1" or m.get("byte_order") != "little"
            or m.get("layout") != "C" or m.get("config") != asdict(CONFIG)):
        raise ValueError("unsupported fixture format/config")
    shapes = tensor_shapes(m.get("variant"))
    if set(m.get("files", {})) != set(SHARDS) or set(m.get("tensors", {})) != set(shapes):
        raise ValueError("unexpected shards or tensor names")
    # Validate the exact generated layout before reading payloads or allocating.
    lengths = dict.fromkeys(SHARDS, 0)
    for index, (name, shape) in enumerate(shapes.items()):
        shard = SHARDS[index % 2]
        size = int(np.prod(shape)) * 2
        expected = {"dtype": "BF16", "shape": list(shape), "shard": shard,
                    "offset": lengths[shard], "nbytes": size}
        if m["tensors"][name] != expected:
            raise ValueError(f"invalid tensor descriptor: {name}")
        lengths[shard] += size
    payloads = {}
    for name in SHARDS:
        path = directory / name
        if m["files"][name]["nbytes"] != lengths[name] or path.stat().st_size != lengths[name]:
            raise ValueError(f"shard size mismatch: {name}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != m["files"][name]["sha256"]:
            raise ValueError(f"shard checksum mismatch: {name}")
        payloads[name] = raw
    weights = {}
    for name, shape in shapes.items():
        d = m["tensors"][name]
        raw = payloads[d["shard"]][d["offset"]:d["offset"] + d["nbytes"]]
        value = decode_bf16_le(raw).reshape(shape)
        value.setflags(write=False)
        weights[name] = value
    return Checkpoint(m["variant"], hashlib.sha256(manifest_raw).hexdigest(),
                      weights, sum(lengths.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=("dense", "moe"))
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    generate_checkpoint(args.output, args.variant, args.seed)
    print(f"Created new {args.variant} BF16 fixture: {args.output}")

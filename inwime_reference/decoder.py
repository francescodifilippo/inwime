"""Untrained two-block causal decoder, resident FP64 NumPy execution only.

No runtime/tile/cache/I/O scheduler dependency. Future execution must not import
the forward, attention, FFN, routing or state-update code from this module.
"""

from dataclasses import dataclass

import numpy as np

from .checkpoint import Checkpoint


def _readonly(value):
    value.setflags(write=False)
    return value


def _softmax(x):
    ex = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)


def _rms(x, scale, epsilon):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon) * scale


def _ffn(x, weights, prefix):
    gate = x @ weights[prefix + "gate"].T
    up = x @ weights[prefix + "up"].T
    return (gate / (1.0 + np.exp(-gate)) * up) @ weights[prefix + "down"].T


def route(scores, top_k):
    """Top-k logits, ties by ascending expert index; softmax over selected only."""
    scores = np.asarray(scores, dtype=np.float64)
    if (scores.ndim != 2 or not np.isfinite(scores).all()
            or type(top_k) is not int or not 1 <= top_k <= scores.shape[1]):
        raise ValueError("invalid router scores/top_k")
    indices = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    probabilities = _softmax(np.take_along_axis(scores, indices, axis=1))
    return indices, probabilities


@dataclass(frozen=True)
class LayerKV:
    keys: np.ndarray  # [heads, tokens, head_width]
    values: np.ndarray


@dataclass(frozen=True)
class KVState:
    checkpoint_identity: str
    length: int
    layers: tuple[LayerKV, ...]


@dataclass(frozen=True)
class Routing:
    indices: np.ndarray
    probabilities: np.ndarray


@dataclass(frozen=True)
class Result:
    logits: np.ndarray
    state: KVState
    hidden: tuple[np.ndarray, ...]  # post-residual output of each block
    routing: tuple[Routing, ...]  # empty for dense


class ReferenceDecoder:
    def __init__(self, checkpoint: Checkpoint):
        self.checkpoint = checkpoint

    def prefill(self, tokens, state=None):
        """Fresh or chunked prefill; previous state is never modified."""
        c, w = self.checkpoint.config, self.checkpoint.weights
        ids = np.asarray(tokens)
        if (ids.ndim != 1 or ids.size == 0 or ids.dtype.kind not in "iu"
                or np.any(ids < 0) or np.any(ids >= c.vocab)):
            raise ValueError("tokens must be a nonempty integer vector within vocabulary")
        start = 0
        if state is not None:
            if (not isinstance(state, KVState)
                    or state.checkpoint_identity != self.checkpoint.identity
                    or type(state.length) is not int or not 0 < state.length <= c.context
                    or len(state.layers) != c.layers):
                raise ValueError("incompatible KV state")
            start = state.length
            for kv in state.layers:
                shape = (c.heads, start, c.width // c.heads)
                if (kv.keys.shape != shape or kv.values.shape != shape
                        or kv.keys.dtype != np.float64 or kv.values.dtype != np.float64
                        or not np.isfinite(kv.keys).all() or not np.isfinite(kv.values).all()):
                    raise ValueError("invalid KV shape/dtype/content")
        end = start + len(ids)
        if end > c.context:
            raise ValueError("context limit exceeded")
        x = w["embedding"][ids] + w["position"][start:end]
        layers, hidden, routing = [], [], []
        head_width = c.width // c.heads
        for layer in range(c.layers):
            p = f"layers.{layer}."
            a = _rms(x, w[p + "norm1"], c.rms_epsilon)
            projections = [(a @ w[p + name].T).reshape(len(ids), c.heads, head_width)
                           .transpose(1, 0, 2) for name in ("q", "k", "v")]
            q, k, v = projections
            if state is not None:
                k = np.concatenate((state.layers[layer].keys, k), axis=1)
                v = np.concatenate((state.layers[layer].values, v), axis=1)
            scores = q @ k.transpose(0, 2, 1) / np.sqrt(float(head_width))
            allowed = np.arange(end)[None, :] <= np.arange(start, end)[:, None]
            scores = np.where(allowed[None, :, :], scores, -np.inf)
            attended = (_softmax(scores) @ v).transpose(1, 0, 2).reshape(len(ids), c.width)
            x = x + attended @ w[p + "o"].T
            f = _rms(x, w[p + "norm2"], c.rms_epsilon)
            if self.checkpoint.variant == "dense":
                delta = _ffn(f, w, p + "ffn.")
            else:
                indices, probabilities = route(f @ w[p + "router"].T, c.top_k)
                delta = np.zeros_like(f)
                for token in range(len(ids)):
                    for slot, expert in enumerate(indices[token]):
                        delta[token] += probabilities[token, slot] * _ffn(
                            f[token], w, p + f"experts.{expert}.")
                routing.append(Routing(_readonly(indices), _readonly(probabilities)))
            x = x + delta
            layers.append(LayerKV(_readonly(k), _readonly(v)))
            hidden.append(_readonly(x.copy()))
        logits = _rms(x, w["final_norm"], c.rms_epsilon) @ w["lm_head"].T
        return Result(_readonly(logits), KVState(self.checkpoint.identity, end, tuple(layers)),
                      tuple(hidden), tuple(routing))

    def decode(self, token, state):
        if type(token) is not int or state is None:
            raise ValueError("decode requires one Python integer token and an existing KV state")
        return self.prefill([token], state)

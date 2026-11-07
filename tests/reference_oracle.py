"""Scalar full-prefix oracle: stdlib math, no reference execution imports.

Recomputes every position at every layer; never uses the decoder KV update.
Shared inputs are only configuration and the exact loaded checkpoint values.
"""

import math


def full_prefix(tokens, config, weights, variant):
    w = {name: value.tolist() for name, value in weights.items()}
    c = config

    def linear(vector, matrix):
        return [math.fsum(float(a) * float(b) for a, b in zip(vector, row)) for row in matrix]

    def norm(vector, scale):
        inv = 1 / math.sqrt(math.fsum(v * v for v in vector) / c.width + c.rms_epsilon)
        return [v * inv * s for v, s in zip(vector, scale)]

    def dense(vector, prefix):
        gate = linear(vector, w[prefix + "gate"])
        up = linear(vector, w[prefix + "up"])
        product = [g / (1 + math.exp(-g)) * u for g, u in zip(gate, up)]
        return linear(product, w[prefix + "down"])

    def softmax(vector):
        largest = max(vector)
        e = [math.exp(v - largest) for v in vector]
        denominator = math.fsum(e)
        return [v / denominator for v in e]

    x = [[a + b for a, b in zip(w["embedding"][token], w["position"][i])]
         for i, token in enumerate(tokens)]
    hidden, all_kv, routing = [], [], []
    dim = c.width // c.heads
    for layer in range(c.layers):
        p = f"layers.{layer}."
        normalized = [norm(row, w[p + "norm1"]) for row in x]
        q = [linear(row, w[p + "q"]) for row in normalized]
        k = [linear(row, w[p + "k"]) for row in normalized]
        v = [linear(row, w[p + "v"]) for row in normalized]
        keys = [[[row[h * dim + d] for d in range(dim)] for row in k] for h in range(c.heads)]
        values = [[[row[h * dim + d] for d in range(dim)] for row in v] for h in range(c.heads)]
        all_kv.append((keys, values))
        attention = []
        for i in range(len(tokens)):
            joined = []
            for head in range(c.heads):
                query = q[i][head * dim:(head + 1) * dim]
                scores = [math.fsum(a * b for a, b in zip(query, keys[head][j])) / math.sqrt(dim)
                          for j in range(i + 1)]
                probs = softmax(scores)
                joined.extend(math.fsum(probs[j] * values[head][j][d] for j in range(i + 1))
                              for d in range(dim))
            attention.append(linear(joined, w[p + "o"]))
        residual = [[a + b for a, b in zip(row, att)] for row, att in zip(x, attention)]
        outputs, layer_indices, layer_probs = [], [], []
        for row in residual:
            f = norm(row, w[p + "norm2"])
            if variant == "dense":
                delta = dense(f, p + "ffn.")
            else:
                scores = linear(f, w[p + "router"])
                indices = sorted(range(c.experts), key=lambda e: (-scores[e], e))[:c.top_k]
                probs = softmax([scores[e] for e in indices])
                experts = [dense(f, p + f"experts.{e}.") for e in indices]
                delta = [math.fsum(prob * expert[d] for prob, expert in zip(probs, experts))
                         for d in range(c.width)]
                layer_indices.append(indices)
                layer_probs.append(probs)
            outputs.append([a + b for a, b in zip(row, delta)])
        x = outputs
        hidden.append([row[:] for row in x])
        if variant == "moe":
            routing.append((layer_indices, layer_probs))
    logits = [linear(norm(row, w["final_norm"]), w["lm_head"]) for row in x]
    return {"logits": logits, "hidden": hidden, "kv": all_kv, "routing": routing}

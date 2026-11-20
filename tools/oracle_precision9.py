"""Qwen2 precision profiles for the verification oracle only (increment-7 policy).

Same RMSNorm/RoPE functions as tools.oracle_precision7; eager attention is the
Qwen2 module's own function with the softmax kept in FP64. No InWiMe import,
no persistent library change; every replaced attribute is restored on exit.
"""
from contextlib import contextmanager
import types

import torch
from transformers.models.qwen2 import modeling_qwen2 as qwen2

from .oracle_precision7 import rms64, rope64, FloatingDtypeAudit  # noqa: F401  (re-export)

PROFILES = ('stock', 'full64')


def attention64(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    assert query.dtype == key.dtype == value.dtype == torch.float64
    keys = qwen2.repeat_kv(key, module.num_key_value_groups)
    values = qwen2.repeat_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query, keys.transpose(2, 3))*scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, :keys.shape[-2]]
    probabilities = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float64)
    probabilities = torch.nn.functional.dropout(probabilities, p=dropout, training=module.training)
    output = torch.matmul(probabilities, values).transpose(1, 2).contiguous()
    return output, probabilities


@contextmanager
def precision_profile(model, profile):
    if profile not in PROFILES:
        raise ValueError('unknown Qwen2 oracle precision profile')
    if profile == 'full64' and any(p.dtype != torch.float64 for p in model.parameters()):
        raise ValueError('FP64 interventions require double parameters')
    changed = []
    old_attention = qwen2.eager_attention_forward
    try:
        if profile == 'full64':
            for module in model.modules():
                if isinstance(module, qwen2.Qwen2RMSNorm):
                    changed.append((module, 'forward', module.forward, 'forward' in vars(module)))
                    module.forward = types.MethodType(rms64, module)
            rotary = model.model.rotary_emb
            if rotary.rope_type != 'default':
                raise ValueError('FP64 RoPE intervention supports default RoPE only')
            changed.extend([(rotary, 'forward', rotary.forward, 'forward' in vars(rotary)),
                            (rotary, 'inv_freq', rotary.inv_freq, True)])
            dim = model.config.hidden_size // model.config.num_attention_heads
            rotary.inv_freq = 1.0/(model.config.rope_theta **
                (torch.arange(0, dim, 2, dtype=torch.float64, device=rotary.inv_freq.device)/dim))
            rotary.forward = types.MethodType(rope64, rotary)
            qwen2.eager_attention_forward = attention64
        yield
    finally:
        qwen2.eager_attention_forward = old_attention
        for module, name, value, was_instance_attribute in reversed(changed):
            if was_instance_attribute:
                setattr(module, name, value)
            else:
                delattr(module, name)

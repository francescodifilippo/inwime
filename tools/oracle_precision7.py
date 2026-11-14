"""Process-local precision interventions for the verification oracle only.

No InWiMe imports, no persistent library/weight changes. Same Llama equations,
layouts, masks, grouping and operations; only precision policies are replaced.
"""
from contextlib import contextmanager
import types
import torch
from transformers.models.llama import modeling_llama as llama

PROFILES = ('stock', 'rms64', 'rms_rope64', 'full64')


def rms64(self, x):
    assert x.dtype == self.weight.dtype == torch.float64
    return self.weight * (x * torch.rsqrt(x.square().mean(-1, keepdim=True)+self.variance_epsilon))


def rope64(self, x, position_ids):
    assert x.dtype == self.inv_freq.dtype == torch.float64
    frequencies = (self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
                   @ position_ids[:, None, :].to(torch.float64)).transpose(1, 2)
    emb = torch.cat((frequencies, frequencies), dim=-1)
    return emb.cos()*self.attention_scaling, emb.sin()*self.attention_scaling


def attention64(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    assert query.dtype == key.dtype == value.dtype == torch.float64
    keys = llama.repeat_kv(key, module.num_key_value_groups)
    values = llama.repeat_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query, keys.transpose(2, 3))*scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, :keys.shape[-2]]
    probabilities = torch.nn.functional.softmax(scores, dim=-1, dtype=torch.float64)
    probabilities = torch.nn.functional.dropout(probabilities, p=dropout, training=module.training)
    output = torch.matmul(probabilities, values).transpose(1, 2).contiguous()
    return output, probabilities


@contextmanager
def precision_profile(model, profile):
    """Restore every changed attribute on exit, including exceptions."""
    if profile not in PROFILES:
        raise ValueError('unknown oracle precision profile')
    if profile != 'stock' and any(p.dtype != torch.float64 for p in model.parameters()):
        raise ValueError('FP64 interventions require double parameters')
    changed = []
    old_attention = llama.eager_attention_forward
    try:
        if profile != 'stock':
            for module in model.modules():
                if isinstance(module, llama.LlamaRMSNorm):
                    changed.append((module, 'forward', module.forward, 'forward' in vars(module)))
                    module.forward = types.MethodType(rms64, module)
        if profile in ('rms_rope64', 'full64'):
            rotary = model.model.rotary_emb
            if rotary.rope_type != 'default':
                raise ValueError('oracle FP64 RoPE intervention supports default RoPE only')
            changed.extend([(rotary, 'forward', rotary.forward, 'forward' in vars(rotary)),
                            (rotary, 'inv_freq', rotary.inv_freq, True)])
            # Recompute from configuration in FP64; promoting an FP32 buffer
            # would preserve its earlier rounding and would not be fully FP64.
            dim = model.config.hidden_size // model.config.num_attention_heads
            rotary.inv_freq = 1.0/(model.config.rope_theta **
                (torch.arange(0, dim, 2, dtype=torch.float64, device=rotary.inv_freq.device)/dim))
            rotary.forward = types.MethodType(rope64, rotary)
        if profile == 'full64':
            llama.eager_attention_forward = attention64
        yield
    finally:
        llama.eager_attention_forward = old_attention
        for module, name, value, was_instance_attribute in reversed(changed):
            if was_instance_attribute:
                setattr(module, name, value)
            else:
                delattr(module, name)


class FloatingDtypeAudit(torch.utils._python_dispatch.TorchDispatchMode):
    """Record actual dispatcher outputs, including internal casts and scalar tensors."""
    def __init__(self):
        super().__init__()
        self.operations = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        from torch.utils._pytree import tree_flatten
        result = func(*args, **(kwargs or {}))
        inputs = sorted({str(t.dtype) for t in tree_flatten((args, kwargs))[0]
                         if isinstance(t, torch.Tensor) and t.is_floating_point()})
        outputs = sorted({str(t.dtype) for t in tree_flatten(result)[0]
                          if isinstance(t, torch.Tensor) and t.is_floating_point()})
        if inputs or outputs:
            key = str(func)+' '+','.join(inputs)+' -> '+','.join(outputs)
            self.operations[key] = self.operations.get(key, 0)+1
        return result

    def summary(self):
        return dict(operations=self.operations,
                    non_fp64_output_operations={k:v for k,v in self.operations.items()
                                               if any(d in k.split(' -> ')[1]
                                                      for d in ('float32','bfloat16','float16'))},
                    cpu_autocast_enabled=torch.is_autocast_enabled('cpu'))

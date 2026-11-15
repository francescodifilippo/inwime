"""Small-case checks in the optional verification venv; never skipped.

1. The bounded streamed oracle equals stock from_pretrained Transformers bit for bit.
2. The Qwen2 full64 profile emits no lower-precision floating outputs and restores stock.
3. InWiMe Qwen2Runtime on the tiny Qwen2 fixture matches upstream: full64 at the FP64
   thresholds and stock FP32 at the FP32 thresholds (increment-7 contracts).
"""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from . import fixture_qwen9
from .oracle9 import run_oracle, build
from .oracle_precision9 import precision_profile, FloatingDtypeAudit

SEQUENCE, CHUNKS = [3, 1, 7, 12, 5, 9, 2], [4, 1, 2]
TOL = {'stock_fp32': (2e-3, 2e-4), 'full64': (2e-5, 2e-6)}


def failures(actual, expected, atol, rtol):
    return int(np.count_nonzero(np.abs(actual-expected) > atol+rtol*np.abs(expected)))


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = fixture_qwen9.write(Path(cls.temp.name)/'qwen')
        (cls.directory/'provenance.json').write_text('{"revision": "fixture"}')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def stock(self, dtype):
        model = AutoModelForCausalLM.from_pretrained(self.directory, local_files_only=True, dtype=dtype,
                                                     attn_implementation='eager').eval()
        residuals = []
        for layer in model.model.layers:
            layer.register_forward_hook(lambda m, i, o: residuals.append(o[0].clone()))
        logits, hidden, cache, start = [], [], None, 0
        with torch.inference_mode():
            for size in CHUNKS:
                residuals.clear()
                out = model(torch.tensor([SEQUENCE[start:start+size]]), past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                logits.append(out.logits[0].double().numpy()); hidden.append(np.stack([r.double().numpy() for r in residuals]))
                start += size
        width = model.config.num_key_value_heads*model.config.hidden_size//model.config.num_attention_heads
        kv = np.stack([np.stack([z[0].transpose(0, 1).contiguous().double().numpy().reshape(-1, width)
                                 for z in (l.keys, l.values)]) for l in cache.layers])
        return dict(logits=np.concatenate(logits), hidden=np.concatenate(hidden, axis=1), kv=kv)

    def test_streamed_equals_stock_from_pretrained(self):
        for dtype, contract in ((torch.float32, 'stock_fp32'), (torch.float64, 'stock_fp64')):
            streamed, meta = run_oracle(self.directory, contract, SEQUENCE, CHUNKS)
            reference = self.stock(dtype)
            for kind in ('logits', 'hidden', 'kv'):
                self.assertEqual(streamed[kind].tobytes(), reference[kind].tobytes(), (contract, kind))
            self.assertEqual(meta['loader']['loads_per_tensor'], len(CHUNKS))

    def test_full64_profile_dtypes_and_restore(self):
        model, _ = build(self.directory, torch.float64)
        loaded = AutoModelForCausalLM.from_pretrained(self.directory, local_files_only=True,
                                                      dtype=torch.float64, attn_implementation='eager').eval()
        audit = FloatingDtypeAudit()
        with torch.inference_mode():
            before = loaded(torch.tensor([SEQUENCE])).logits.clone()
            with precision_profile(loaded, 'full64'), audit:
                full = loaded(torch.tensor([SEQUENCE])).logits.clone()
            after = loaded(torch.tensor([SEQUENCE])).logits.clone()
        summary = audit.summary()
        self.assertEqual(summary['non_fp64_output_operations'], {})
        self.assertTrue(torch.equal(before, after))
        self.assertFalse(torch.equal(before, full))
        with self.assertRaises(ValueError):
            with precision_profile(AutoModelForCausalLM.from_pretrained(
                    self.directory, local_files_only=True, dtype=torch.float32), 'full64'):
                pass
        del model

    def test_inwime_matches_upstream_small(self):
        from inwime_runtime.qwen2 import Qwen2Runtime
        from inwime_runtime.math_cpu import NativeMath
        for contract in ('full64', 'stock_fp32'):
            upstream, _ = run_oracle(self.directory, contract, SEQUENCE, CHUNKS)
            atol, rtol = TOL[contract]
            with Qwen2Runtime(self.directory, capacity=8, tile_elements=7, read_elements=7,
                              math_backend=NativeMath('build/libinwime_cpu.so')) as r:
                logits, hidden, start = [], [], 0
                for size in CHUNKS:
                    with r.prefill(SEQUENCE[start:start+size]) as out:
                        logits.append(np.frombuffer(out.logits, dtype='<f8').reshape(size, -1).copy())
                        hidden.append(np.stack([np.frombuffer(h, dtype='<f8').reshape(size, -1) for h in out.hidden]))  # stack copies
                        # Views alias runtime arenas reused by the next call: copy explicitly.
                        kv = np.stack([np.stack([np.frombuffer(v, dtype='<f8').reshape(r.length, -1) for v in pair])
                                       for pair in out.kv]).copy()
                    start += size
            actual = dict(logits=np.concatenate(logits), hidden=np.concatenate(hidden, axis=1), kv=kv)
            for kind in ('logits', 'hidden', 'kv'):
                self.assertEqual(failures(actual[kind], upstream[kind], atol, rtol), 0, (contract, kind))
            self.assertEqual(actual['logits'].argmax(-1).tolist(), upstream['logits'].argmax(-1).tolist())


if __name__ == '__main__':
    unittest.main()

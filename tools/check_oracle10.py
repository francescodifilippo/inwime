"""Small-case checks of tools/oracle10.py in the optional verification venv; never skipped.

1. oracle10 (row-gather embedding, blocked or resident head, streamed layers)
   equals stock from_pretrained Transformers bit for bit, for the increment-10
   and increment-9 fixtures, several block sizes and all contracts (full64 is
   compared with from_pretrained under the same precision profile).
2. The memory budget is enforced before any tensor is read; no fallback.
3. The unchanged InWiMe runtime on the GQA 5:1 / eps 1e-5 fixture passes
   full64 at the FP64 thresholds and stock_fp32 at the FP32 thresholds on FULL
   coverage (every logit, residual and K/V entry), via tools.compare10.
4. Sensitivity: one perturbed BF16 word in the head or in a K projection makes
   the comparison FAIL; a wrong rms_norm_eps does too.
"""
import json
import shutil
import struct
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from . import compare10, fixture_qwen10, fixture_qwen9
from .oracle10 import run_oracle
from .oracle_precision9 import precision_profile

SEQUENCE10, CHUNKS10 = [3, fixture_qwen10.TINY_TOKEN, 7, 30, 5, fixture_qwen10.TINY_TOKEN, 12, 2], [5, 1, 1, 1]
SEQUENCE9, CHUNKS9 = [3, 1, 7, 12, 5, 9, 2], [4, 1, 2]
BUDGET = 4*2**30


def stock(directory, dtype, profile, sequence, chunks):
    model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True, dtype=dtype,
                                                 attn_implementation='eager').eval()
    residuals = []
    for layer in model.model.layers:
        layer.register_forward_hook(lambda m, i, o: residuals.append((o[0] if isinstance(o, tuple) else o)[0].clone()))
    logits, hidden, cache, start = [], [], None, 0
    with torch.inference_mode(), precision_profile(model, profile):
        for size in chunks:
            residuals.clear()
            out = model(torch.tensor([sequence[start:start+size]]), past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            logits.append(out.logits[0].double().numpy())
            hidden.append(np.stack([r.double().numpy() for r in residuals]))
            start += size
    width = model.config.num_key_value_heads*model.config.hidden_size//model.config.num_attention_heads
    kv = np.stack([np.stack([z[0].transpose(0, 1).contiguous().double().numpy().reshape(-1, width)
                             for z in (l.keys, l.values)]) for l in cache.layers])
    return dict(logits=np.concatenate(logits), hidden=np.concatenate(hidden, axis=1), kv=kv)


def runtime_arrays(directory, sequence, chunks):
    from inwime_runtime.qwen2 import Qwen2Runtime
    from inwime_runtime.math_cpu import NativeMath
    with Qwen2Runtime(directory, capacity=16, tile_elements=7, read_elements=7,
                      math_backend=NativeMath('build/libinwime_cpu.so')) as r:
        logits, hidden, start = [], [], 0
        for size in chunks:
            with r.prefill(sequence[start:start+size]) as out:
                logits.append(np.frombuffer(out.logits, dtype='<f8').reshape(size, -1).copy())
                hidden.append(np.stack([np.frombuffer(h, dtype='<f8').reshape(size, -1) for h in out.hidden]))
                kv = np.stack([np.stack([np.frombuffer(v, dtype='<f8').reshape(r.length, -1) for v in pair])
                               for pair in out.kv]).copy()
            start += size
        spec = r.spec
    return dict(logits=np.concatenate(logits), hidden=np.concatenate(hidden, axis=1), kv=kv), spec


def flip_word(directory, tensor, element):
    """Copy-on-write perturbation of one BF16 word (sign flip) in a fixture copy."""
    index = json.loads((directory/'model.safetensors.index.json').read_text())
    shard = directory/index['weight_map'][tensor]
    raw = bytearray(shard.read_bytes())
    length, = struct.unpack('<Q', raw[:8])
    header = json.loads(raw[8:8+length])
    begin = 8+length+header[tensor]['data_offsets'][0]+2*element
    word, = struct.unpack('<H', raw[begin:begin+2])
    raw[begin:begin+2] = struct.pack('<H', word ^ 0x8000)
    shard.write_bytes(bytes(raw))
    descriptor = json.loads((directory/'safetensors.json').read_text())
    import hashlib
    descriptor['files'][shard.name] = dict(nbytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    (directory/'safetensors.json').write_text(json.dumps(descriptor))


class Oracle10Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.q10 = fixture_qwen10.write(root/'q10')
        cls.q9 = fixture_qwen9.write(root/'q9')
        (cls.q9/'provenance.json').write_text('{"revision": "fixture"}')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def assertBitIdentical(self, a, b, label):
        for kind in compare10.KINDS:
            self.assertTrue(compare10.identical(a[kind], b[kind]), (label, kind))

    def test_matches_stock_from_pretrained(self):
        """Residuals and K/V always bit-identical; logits per the preregistered head policy.

        stock_fp32 (resident FP32 head) is bit-identical by construction. Blocked FP64 heads
        follow the oracle10 equivalence rule: max |delta| <= 1e-9 and identical argmax. A
        blocked FP32 head is not bit-identical (BLAS tail kernels) and is not used by any contract.
        """
        cases = [(self.q10, SEQUENCE10, CHUNKS10), (self.q9, SEQUENCE9, CHUNKS9)]
        for directory, sequence, chunks in cases:
            for contract, dtype, profile in (('stock_fp32', torch.float32, 'stock'),
                                             ('stock_fp64', torch.float64, 'stock'),
                                             ('full64', torch.float64, 'full64')):
                reference = stock(directory, dtype, profile, sequence, chunks)
                if dtype == torch.float32:
                    for block in (1, 7, 16384):
                        arrays, meta = run_oracle(directory, contract, sequence, chunks, block_rows=block,
                                                  memory_budget=BUDGET)
                        self.assertEqual(meta['head_mode'], 'resident')
                        self.assertBitIdentical(arrays, reference, (directory.name, contract, block))
                    blocked, _ = run_oracle(directory, contract, sequence, chunks, head_mode='blocked',
                                            block_rows=5, memory_budget=BUDGET)
                    for kind in ('hidden', 'kv'):
                        self.assertTrue(compare10.identical(blocked[kind], reference[kind]))
                    self.assertLess(np.abs(blocked['logits']-reference['logits']).max(), 1e-4)
                    continue
                for block in (1, 5, 8, 16, 36, 16384):
                    arrays, meta = run_oracle(directory, contract, sequence, chunks, block_rows=block,
                                              memory_budget=BUDGET)
                    self.assertEqual(meta['head_mode'], 'blocked')
                    rule = compare10.equivalence(arrays, reference)
                    self.assertTrue(rule['passed'], (directory.name, contract, block, rule))
                    self.assertFalse(meta['loader']['embedding_materialized'])
                    self.assertEqual(meta['loader']['embedding_rows_read'], len(sequence))
                    self.assertEqual(meta['loader']['head_rows_read'], len(chunks)*json.loads(
                        (directory/'config.json').read_text())['vocab_size'])

    def test_budget_refused_before_reading(self):
        with self.assertRaises(MemoryError):
            run_oracle(self.q10, 'full64', SEQUENCE10, CHUNKS10, memory_budget=10*2**20)
        with self.assertRaises(ValueError):
            run_oracle(self.q10, 'full64', SEQUENCE10, CHUNKS10, memory_budget=None)
        with self.assertRaises(ValueError):
            run_oracle(self.q10, 'full64', SEQUENCE10, CHUNKS10, head_mode='resident', memory_budget=BUDGET)

    def test_runtime_full_coverage_parity(self):
        actual, spec = runtime_arrays(self.q10, SEQUENCE10, CHUNKS10)
        shapes = compare10.required_shapes(spec.layers, len(SEQUENCE10), spec.width, spec.vocab, spec.kv_width)
        for contract, precision in (('full64', 'fp64'), ('stock_fp32', 'fp32')):
            upstream, _ = run_oracle(self.q10, contract, SEQUENCE10, CHUNKS10, memory_budget=BUDGET)
            result = compare10.compare(actual, upstream, shapes, precision)
            self.assertEqual((result['coverage'], result['verdict']), ('FULL', 'PASS'), (contract, result))
            self.assertTrue(result['kinds']['logits']['argmax_equal'])
            partial = compare10.compare(dict(logits=actual['logits']), upstream, shapes, precision)
            self.assertEqual((partial['coverage'], partial['verdict']), ('PARTIAL', 'NOT PROVEN'))

    def test_sensitivity_to_perturbed_weights_and_epsilon(self):
        actual, spec = runtime_arrays(self.q10, SEQUENCE10, CHUNKS10)
        shapes = compare10.required_shapes(spec.layers, len(SEQUENCE10), spec.width, spec.vocab, spec.kv_width)
        for tensor, element in (('lm_head.weight', 11), ('model.layers.0.self_attn.k_proj.weight', 3)):
            with tempfile.TemporaryDirectory() as d:
                copy = Path(d)/'q'
                shutil.copytree(self.q10, copy)
                flip_word(copy, tensor, element)
                upstream, _ = run_oracle(copy, 'full64', SEQUENCE10, CHUNKS10, memory_budget=BUDGET)
                self.assertEqual(compare10.compare(actual, upstream, shapes, 'fp64')['verdict'], 'FAIL', tensor)
        with tempfile.TemporaryDirectory() as d:
            copy = Path(d)/'q'
            shutil.copytree(self.q10, copy)
            config = json.loads((copy/'config.json').read_text())
            config['rms_norm_eps'] = 1e-6
            (copy/'config.json').write_text(json.dumps(config))
            upstream, _ = run_oracle(copy, 'full64', SEQUENCE10, CHUNKS10, memory_budget=BUDGET)
            self.assertEqual(compare10.compare(actual, upstream, shapes, 'fp64')['verdict'], 'FAIL')


if __name__ == '__main__':
    unittest.main()

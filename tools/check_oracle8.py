"""Small loader regression in the optional verification environment, never skipped."""
import tempfile
from pathlib import Path
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM, AutoModelForCausalLM
from .oracle8 import load_fp32


class LoaderTests(unittest.TestCase):
    def test_original_values_buffers_tying_and_forward(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        config = LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                             num_attention_heads=2, num_key_value_heads=2, vocab_size=32,
                             max_position_embeddings=16, tie_word_embeddings=True,
                             rope_theta=130000., attn_implementation='eager')
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            original = LlamaForCausalLM(config).eval().bfloat16()
            original.save_pretrained(directory)
            loaded, report = load_fp32(directory)
            reference = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True,
                dtype=torch.float32, attn_implementation='eager').eval()
            for name, value in loaded.state_dict().items():
                self.assertTrue(torch.equal(value, reference.state_dict()[name]), name)
            self.assertFalse(any(b.is_meta for b in loaded.buffers()))
            self.assertIs(loaded.lm_head.weight, loaded.model.embed_tokens.weight)
            self.assertEqual(report['resident_fp32_parameter_bytes'], sum(p.numel()*4 for p in loaded.parameters()))
            with torch.inference_mode():
                for model in (loaded, reference):
                    first = model(torch.tensor([[1, 2]]), use_cache=True)
                    last = model(torch.tensor([[3]]), past_key_values=first.past_key_values, use_cache=True)
                    if model is loaded:
                        logits = first.logits.clone(), last.logits.clone()
                        kv = [(l.keys.clone(), l.values.clone()) for l in last.past_key_values.layers]
                    else:
                        self.assertTrue(torch.equal(logits[0], first.logits))
                        self.assertTrue(torch.equal(logits[1], last.logits))
                        for pair, layer in zip(kv, last.past_key_values.layers):
                            self.assertTrue(torch.equal(pair[0], layer.keys))
                            self.assertTrue(torch.equal(pair[1], layer.values))


if __name__ == '__main__':
    unittest.main()

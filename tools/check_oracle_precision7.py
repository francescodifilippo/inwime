"""Oracle-only regression tests, run with the isolated torch environment.

Kept outside pytest discovery so the inference project's CPU-only test suite
never silently skips or acquires a mandatory torch dependency.
"""
import unittest
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama import modeling_llama as llama
from .oracle_precision7 import precision_profile, FloatingDtypeAudit, rms64, attention64


class PrecisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)

    def model(self):
        return LlamaForCausalLM(LlamaConfig(hidden_size=8,intermediate_size=16,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,
            max_position_embeddings=16,vocab_size=16,rope_theta=100000.,
            tie_word_embeddings=True,attn_implementation='eager')).eval().double()

    def test_rms64_avoids_internal_fp32(self):
        norm=llama.LlamaRMSNorm(4,1e-5).double()
        x=torch.tensor([[1.+2**-30,2.,3.,4.]],dtype=torch.float64)
        with torch.inference_mode():
            stock=norm(x);pure=rms64(norm,x)
            independent=x/torch.sqrt(x.square().mean(-1,keepdim=True)+1e-5)
        self.assertFalse(torch.equal(stock,pure))
        torch.testing.assert_close(pure,independent,atol=1e-15,rtol=1e-15)

    def test_softmax_is_an_independent_fp32_source(self):
        from types import SimpleNamespace
        module=SimpleNamespace(num_key_value_groups=2,training=False)
        q=torch.arange(8,dtype=torch.float64).reshape(1,2,1,4)/7
        k=torch.arange(12,dtype=torch.float64).reshape(1,1,3,4)/13
        v=k/3
        _,stock=llama.eager_attention_forward(module,q,k,v,None,.5)
        _,pure=attention64(module,q,k,v,None,.5)
        self.assertFalse(torch.equal(stock,pure))
        expected=(q@llama.repeat_kv(k,2).transpose(2,3)*.5).softmax(-1)
        torch.testing.assert_close(pure,expected,atol=0,rtol=0)

    def test_full64_has_no_lower_precision_outputs(self):
        model=self.model();audit=FloatingDtypeAudit()
        with precision_profile(model,'full64'),torch.inference_mode(),audit:
            result=model(torch.tensor([[1,2,3]]),use_cache=True)
            model(torch.tensor([[4]]),past_key_values=result.past_key_values,use_cache=True)
        self.assertEqual(audit.summary()['non_fp64_output_operations'],{})

    def test_restore_on_error_preserves_model_and_stock_forward(self):
        model=self.model();rotary=model.model.rotary_emb
        old_freq=rotary.inv_freq
        old_attention=llama.eager_attention_forward
        weights={k:v.clone() for k,v in model.state_dict().items()}
        with torch.inference_mode():before=model(torch.tensor([[1,2,3]])).logits.clone()
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with precision_profile(model,'full64'):
                self.assertIsNot(rotary.inv_freq,old_freq)
                raise RuntimeError('injected')
        self.assertIs(rotary.inv_freq,old_freq)
        self.assertIs(llama.eager_attention_forward,old_attention)
        self.assertNotIn('forward',vars(model.model.norm))
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,weights[k]))
        with torch.inference_mode():after=model(torch.tensor([[1,2,3]])).logits
        self.assertTrue(torch.equal(before,after))

    def test_refuses_false_fp64_profile(self):
        model=self.model().float()
        with self.assertRaises(ValueError):
            with precision_profile(model,'full64'):pass


if __name__=='__main__': unittest.main()

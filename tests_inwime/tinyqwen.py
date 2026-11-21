"""A tiny Qwen2 Checkpoint the real Motore can serve: the M10 fixture plus a word tokenizer.

The weights, index and descriptor come from `tools/fixture_qwen10.write`, called unchanged, so
the historical generator keeps its behaviour. This helper only adds the tokenizer files the
fixture never needed: a WordLevel vocabulary of the fixture's 37 ids, `<|bos|>` = 0 and
`<|endoftext|>` = 1 (the config's bos/eos), and `t2` ... `t36` for the others.
"""
import importlib.util
import json
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]/'tools'


def generator():
    spec = importlib.util.spec_from_file_location('fixture_qwen10', TOOLS/'fixture_qwen10.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(directory):
    from tokenizers import Tokenizer, models, pre_tokenizers
    fixture = generator()
    directory = fixture.write(directory)
    vocab = {'<|bos|>': 0, '<|endoftext|>': 1, **{'t%d' % i: i for i in range(2, fixture.CONFIG['vocab_size'])}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<|endoftext|>'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens(['<|bos|>', '<|endoftext|>'])
    tokenizer.save(str(directory/'tokenizer.json'))
    (directory/'tokenizer_config.json').write_text(json.dumps(
        {'bos_token': '<|bos|>', 'eos_token': '<|endoftext|>', 'add_bos_token': False}))
    (directory/'generation_config.json').write_text(json.dumps({'bos_token_id': 0, 'eos_token_id': 1}))
    return directory


def frozen_greedy(directory, prompt, max_tokens):
    """The reference: the Percorso congelato called directly, as M10 did, in this process."""
    from inwime_runtime.qwen2 import Qwen2Runtime
    from inwime_runtime.tokenizer import LlamaTokenizer, greedy
    tokenizer = LlamaTokenizer(directory)
    ids = tokenizer.encode(prompt)
    with Qwen2Runtime(directory, capacity=len(ids)+max_tokens-1) as runtime:
        return greedy(runtime, tokenizer, prompt, max_tokens), tokenizer

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


EMBEDDING, HEAD = 'model.embed_tokens.weight', 'lm_head.weight'
HEADS = ('untied', 'tied', 'duplicated')


def generator(head='untied'):
    """A private instance of the M10 generator (`tools/fixture_qwen10.py`, pinned, never edited).

    `head` extends it additively for #38, by rebinding the module globals that `write` and
    `default_split` look up at call time in this instance only; the file and every other
    instance keep the historical output.
    - 'untied': the historical generator, unchanged.
    - 'tied': `tie_word_embeddings` true and no `lm_head.weight`, as Qwen2.5-0.5B serializes it.
    - 'duplicated': the untied twin of 'tied': `lm_head.weight` holds the embedding's values.
    """
    if head not in HEADS:
        raise ValueError('head must be one of %s' % (HEADS,))
    spec = importlib.util.spec_from_file_location('fixture_qwen10', TOOLS/'fixture_qwen10.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shapes, value = module.shapes, module.value
    if head == 'tied':
        module.CONFIG = dict(module.CONFIG, tie_word_embeddings=True)
        module.shapes = lambda config=module.CONFIG: {
            name: shape for name, shape in shapes(config).items() if name != HEAD}
    elif head == 'duplicated':
        module.value = lambda name, index, width: value(EMBEDDING if name == HEAD else name, index, width)
    return module


def write(directory, head='untied'):
    from tokenizers import Tokenizer, models, pre_tokenizers
    fixture = generator(head)
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


# The Qwen2.5 chat template, verbatim from tokenizer_config.json of
# Qwen/Qwen2.5-7B@d149729398750b98c0af14eb82c78cfe92750796 (SHA256 of the file: 44d5f08f...7fa5).
CHAT_TEMPLATE = (Path(__file__).resolve().parent/'qwen25_chat_template.jinja').read_text()
# Qwen's special tokens take the first ids of the chat fixture: <|im_end|> = 2 is its EOS, with
# <|endoftext|> = 0 as the second EOS of generation_config.json, as in Qwen2.5-Instruct.
CHAT_SPECIALS = ('<|endoftext|>', '<|im_start|>', '<|im_end|>')
CHAT_EOS = (2, 0)
CHAT_VOCAB_SIZE = 512
QWEN_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
              r"|\s*[\r\n]+|\s+(?!\S)|\s+")


def write_chat(directory, generation=None, template=CHAT_TEMPLATE, head='untied'):
    """The fixture weights, with a vocabulary of 512, served as a Qwen2.5 chat model: a byte-level
    BPE tokenizer with the pipeline of Qwen2.5's tokenizer.json (NFC, Qwen split, ByteLevel), its
    merges trained on the chat template's own text so prompts stay short, the real Qwen2.5 chat
    template, two EOS ids, and a context of 1024 positions. `generation`: extra
    generation_config.json fields; `template` None: a Checkpoint without chat template; `head`:
    see `generator`."""
    from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers, trainers
    from tokenizers.models import BPE
    fixture = generator(head)
    directory = fixture.write(directory, config=dict(fixture.CONFIG, vocab_size=CHAT_VOCAB_SIZE,
                                                     max_position_embeddings=1024, bos_token_id=0,
                                                     eos_token_id=CHAT_EOS[0]))
    tokenizer = Tokenizer(BPE(unk_token=None, continuing_subword_prefix='', end_of_word_suffix='',
                              fuse_unk=False, byte_fallback=False))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_SPLIT), behavior='isolated', invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tokenizer.decoder = decoders.ByteLevel()
    corpus = [CHAT_TEMPLATE, 'You are a helpful assistant. The weather in Rome is sunny, thanks.']
    tokenizer.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=CHAT_VOCAB_SIZE, special_tokens=list(CHAT_SPECIALS), show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    assert tokenizer.get_vocab_size() == CHAT_VOCAB_SIZE
    assert [tokenizer.token_to_id(t) for t in CHAT_SPECIALS] == [0, 1, 2]
    tokenizer.save(str(directory/'tokenizer.json'))
    config = {'tokenizer_class': 'Qwen2Tokenizer', 'bos_token': None, 'eos_token': '<|im_end|>',
              'pad_token': '<|endoftext|>', 'add_bos_token': False, 'clean_up_tokenization_spaces': False,
              'model_max_length': 1024, 'additional_special_tokens': ['<|im_start|>', '<|im_end|>']}
    if template is not None:
        config['chat_template'] = template
    (directory/'tokenizer_config.json').write_text(json.dumps(config))
    (directory/'generation_config.json').write_text(json.dumps(
        {'bos_token_id': 0, 'eos_token_id': list(CHAT_EOS), **(generation or {})}))
    return directory


def auto_tokenizer(directory):
    """Transformers' own tokenizer for the Checkpoint: the reference for chat rendering."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(directory), trust_remote_code=False, local_files_only=True)


class IdsTokenizer:
    """Hands token ids to the frozen `greedy` unchanged, as if its prompt text encoded to them."""

    def __init__(self, ids, eos, bos):
        self.ids, self.eos, self.bos = list(ids), frozenset(eos), bos

    def encode(self, text):
        return list(self.ids)

    def decode(self, ids):
        return ''


def kernel_path():
    """The native kernel: packaged with an installed InWiMe, or built in the source tree by
    `make cpu-kernel`. None: the Python fallback only."""
    from inwime import kernel
    found = kernel.locate({})
    built = TOOLS.parent/'build'/'libinwime_cpu.so'
    return found or (built if built.is_file() else None)


def kernel_env():
    """Server environment using the native kernel when there is one (bit-identical to the Python
    fallback by its self-test; four times faster on long fixture prompts)."""
    path = kernel_path()
    return {'INWIME_CPU_KERNEL': str(path)} if path else {}


def runtime(directory, capacity):
    from inwime_runtime.math_cpu import NativeMath
    from inwime_runtime.qwen2 import Qwen2Runtime
    path = kernel_path()
    if path is None:
        return Qwen2Runtime(directory, capacity=capacity)
    return Qwen2Runtime(directory, capacity=capacity, math_backend=NativeMath(str(path)))


def frozen_greedy_ids(directory, ids, max_tokens, eos=CHAT_EOS):
    """The Percorso congelato's greedy continuation of given prompt ids."""
    from inwime_runtime.tokenizer import greedy
    with runtime(directory, len(ids)+max_tokens-1) as session:
        return greedy(session, IdsTokenizer(ids, eos, None), '', max_tokens)['generated_ids']


def frozen_penalized_greedy(directory, ids, max_tokens, penalty):
    """Greedy over the Percorso congelato's logits after Transformers' repetition penalty
    (RepetitionPenaltyLogitsProcessor): the logit of a token already in the sequence, prompt
    included, is divided by the penalty when positive and multiplied by it when negative."""
    def pick(logits, seen):
        penalized = [(l/penalty if l > 0 else l*penalty) if t in seen else l for t, l in enumerate(logits)]
        return max(range(len(penalized)), key=penalized.__getitem__)
    generated = []
    with runtime(directory, len(ids)+max_tokens-1) as session:
        vocab = session.spec.vocab
        with session.prefill(ids) as result:
            generated.append(pick(result.logits[(len(ids)-1)*vocab:len(ids)*vocab], set(ids)))
        while len(generated) < max_tokens:
            with session.decode(generated[-1]) as result:
                generated.append(pick(result.logits[:vocab], set(ids+generated)))
    return generated


def frozen_logits(directory, ids):
    """The Percorso congelato's FP64 logits after the given ids."""
    with runtime(directory, len(ids)) as session:
        with session.prefill(ids) as result:
            vocab = session.spec.vocab
            return list(result.logits[(len(ids)-1)*vocab:len(ids)*vocab])


def frozen_greedy(directory, prompt, max_tokens):
    """The reference: the Percorso congelato called directly, as M10 did, in this process."""
    from inwime_runtime.qwen2 import Qwen2Runtime
    from inwime_runtime.tokenizer import LlamaTokenizer, greedy
    tokenizer = LlamaTokenizer(directory)
    ids = tokenizer.encode(prompt)
    with Qwen2Runtime(directory, capacity=len(ids)+max_tokens-1) as runtime:
        return greedy(runtime, tokenizer, prompt, max_tokens), tokenizer

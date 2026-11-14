"""Official tokenizers JSON backend; optional dependency, never torch/Transformers."""
from pathlib import Path
from .llama import read_config


class LlamaTokenizer:
    def __init__(self, directory):
        from tokenizers import Tokenizer
        directory = Path(directory)
        self.backend = Tokenizer.from_file(str(directory/'tokenizer.json'))
        self.config = read_config(directory/'tokenizer_config.json')
        model = read_config(directory/'config.json')
        generation = read_config(directory/'generation_config.json')
        self.bos = generation.get('bos_token_id', model.get('bos_token_id'))
        eos = generation.get('eos_token_id', model.get('eos_token_id'))
        self.eos = frozenset(eos if isinstance(eos, list) else [eos])
        self.add_bos = self.config.get('add_bos_token', False)
        self.add_eos = self.config.get('add_eos_token', False)
        for key, value in [('bos_token', self.bos), ('eos_token', model.get('eos_token_id'))]:
            text = self.config.get(key)
            if isinstance(text, dict):
                text = text['content']
            if text is not None and self.backend.token_to_id(text) != value:
                raise ValueError('tokenizer/model special token mismatch')

    def encode(self, text, *, add_special_tokens=True):
        ids = self.backend.encode(text, add_special_tokens=False).ids
        if add_special_tokens:
            if self.add_bos:
                ids.insert(0, self.bos)
            if self.add_eos:
                ids.append(next(iter(self.eos)))
        return ids

    def decode(self, ids, *, skip_special_tokens=True):
        return self.backend.decode(list(ids), skip_special_tokens=skip_special_tokens)


def greedy(runtime, tokenizer, text, max_new_tokens=3):
    """Fresh session, lower ID wins ties. EOS is returned and stops generation."""
    if type(max_new_tokens) is not int or max_new_tokens <= 0 or runtime.length:
        raise ValueError('greedy requires a fresh session and positive token count')
    prompt = tokenizer.encode(text)
    if not prompt:
        prompt = [tokenizer.bos]
    if len(prompt)+max_new_tokens-1 > runtime.capacity:
        raise ValueError('generation exceeds session capacity')
    generated = []
    with runtime.prefill(prompt) as result:
        logits = result.logits[(len(prompt)-1)*runtime.spec.vocab:]
        chosen = max(range(runtime.spec.vocab), key=logits.__getitem__)
    for step in range(max_new_tokens):
        generated.append(chosen)
        if chosen in tokenizer.eos or step+1 == max_new_tokens:
            break
        with runtime.decode(chosen) as result:
            chosen = max(range(runtime.spec.vocab), key=result.logits.__getitem__)
    return dict(text=text, input_ids=prompt, generated_ids=generated,
                output=tokenizer.decode(prompt+generated),
                stopped_on_eos=generated[-1] in tokenizer.eos)

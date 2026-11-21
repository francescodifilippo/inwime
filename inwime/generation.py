"""What a generation emits: text released only once no later token can change it.

`TextStream` turns generated token ids into text deltas for streaming. It decodes the whole
generated text again at each token (the only decoding every tokenizer gets right), and holds
back two kinds of text at the end: an incomplete UTF-8 sequence, and the longest tail that
could still grow into a stop sequence. A stop sequence may span several tokens; when it
appears, the text before it is the final text and nothing of the stop sequence was released.
"""
import heapq
import math
import random


class TextStream:
    def __init__(self, decode, stop=()):
        self.decode, self.stop = decode, tuple(stop)
        self.ids, self.released, self.stopped = [], '', False

    def push(self, token):
        """Add a generated text token; returns the text it releases ('' if none yet). After a
        stop sequence appears, `stopped` is true and no more tokens may be pushed."""
        self.ids.append(token)
        text = self.decode(self.ids).rstrip('\ufffd')
        found = [i for i in (text.find(s, max(0, len(self.released)-len(s)+1)) for s in self.stop) if i >= 0]
        if found:
            self.stopped = True
            return self._release(text[:min(found)])
        held = max((n for s in self.stop for n in range(min(len(s)-1, len(text)), 0, -1)
                    if text.endswith(s[:n])), default=0)
        return self._release(text[:len(text)-held])

    def finish(self):
        """The text still held back once generation is over."""
        if self.stopped:
            return ''
        return self._release(self.decode(self.ids))

    def _release(self, text):
        if not text.startswith(self.released) or len(text) == len(self.released):
            return ''
        delta, self.released = text[len(self.released):], text
        return delta


# -- sampling --------------------------------------------------------------------------------
#
# Each parameter comes from the client, else from the Checkpoint's generation_config.json, else
# from the defaults of Transformers' GenerationConfig, the file's own format. As in Transformers,
# generation_config.json samples only with `do_sample: true`: otherwise, unless the client sets a
# temperature, decoding is greedy. Temperature 0 is greedy: the highest logit wins and the lowest
# token id wins a tie, the rule of the Percorso congelato. A seed reproduces the same tokens on
# the same Checkpoint, Percorso computazionale and platform.

DEFAULTS = {'temperature': 1.0, 'top_p': 1.0, 'top_k': 50, 'repetition_penalty': 1.0}
MAX_STOP = 4


def _number(value, name, low, high, *, low_open=False, integer=False):
    ok = type(value) is int if integer else type(value) in (int, float) and math.isfinite(value)
    if not ok or value < low or value > high or (low_open and value == low):
        raise SamplingError(name, '%s must be %s %s %g and at most %g' % (
            name, 'an integer' if integer else 'a number', 'above' if low_open else 'from', low, high))
    return value


class SamplingError(ValueError):
    def __init__(self, param, message):
        super().__init__(message)
        self.param = param


def stops(value, name='stop'):
    if value is None:
        return None
    values = [value] if isinstance(value, str) else value
    if (not isinstance(values, list) or len(values) > MAX_STOP
            or not all(isinstance(v, str) and v for v in values)):
        raise SamplingError(name, '%s must be a non-empty string or an array of at most %d non-empty strings'
                            % (name, MAX_STOP))
    return values


def valid_parameters(source):
    """The sampling parameters `source` sets, validated; absent ones are left out. Raises
    SamplingError for the first invalid one."""
    checks = {'temperature': lambda v: _number(v, 'temperature', 0, 2),
              'top_p': lambda v: _number(v, 'top_p', 0, 1, low_open=True),
              'top_k': lambda v: _number(v, 'top_k', 0, 2**31-1, integer=True),
              'repetition_penalty': lambda v: _number(v, 'repetition_penalty', 0, 100, low_open=True),
              'seed': lambda v: _number(v, 'seed', -2**63, 2**63-1, integer=True),
              'stop': stops}
    return {name: check(source[name]) for name, check in checks.items() if source.get(name) is not None}


def client_parameters(body):
    return valid_parameters(body)


def configured_parameters(generation):
    """The valid sampling parameters of generation_config.json: a value the client could not send
    either (a float top_k, a negative temperature, NaN) is ignored, as if absent."""
    result = {}
    for name in ('temperature', 'top_p', 'top_k', 'repetition_penalty'):
        try:
            result.update(valid_parameters({name: generation.get(name)}))
        except SamplingError:
            pass
    try:
        result.update(valid_parameters({'stop': generation.get('stop_strings')}))
    except SamplingError:
        pass
    return result


class Sampling:
    """The effective parameters of one request, each with its source: 'client',
    'generation_config', 'default' or, for a seed nobody set, 'server'."""

    def __init__(self, client, generation, seed):
        generation = generation or {}
        configured = configured_parameters(generation)
        effective = {}
        sample = generation.get('do_sample') is True
        for name, default in DEFAULTS.items():
            if name in client:
                effective[name] = (client[name], 'client')
            elif name == 'temperature' and not sample:
                effective[name] = (0, 'generation_config' if 'do_sample' in generation else 'default')
            elif name in configured:
                effective[name] = (configured[name], 'generation_config')
            else:
                effective[name] = (default, 'default')
        effective['seed'] = (client['seed'], 'client') if 'seed' in client else (seed, 'server')
        if 'stop' in client:
            effective['stop'] = (client['stop'], 'client')
        else:
            effective['stop'] = (configured['stop'], 'generation_config') if 'stop' in configured else ([], 'default')
        self.effective = effective
        value = {name: v for name, (v, _) in effective.items()}
        self.temperature, self.top_p, self.top_k = value['temperature'], value['top_p'], value['top_k']
        self.penalty, self.stop = value['repetition_penalty'], value['stop']
        # random.Random seeds with abs(): the 64-bit pattern keeps -5 and 5 apart.
        self.random = random.Random(value['seed'] % 2**64)

    def describe(self):
        return {name: {'value': v, 'source': source} for name, (v, source) in self.effective.items()}

    def choose(self, logits, history):
        """The next token after `logits`, given every token id so far (prompt and generated)."""
        vocab = len(logits)
        if self.penalty != 1:
            logits = list(logits)
            for token in set(history):
                value = logits[token]
                logits[token] = value/self.penalty if value > 0 else value*self.penalty
        if self.temperature == 0:
            return max(range(vocab), key=logits.__getitem__)
        rank = lambda t: (-logits[t], t)
        if 0 < self.top_k < vocab:
            order = heapq.nsmallest(self.top_k, range(vocab), key=rank)
        else:
            order = sorted(range(vocab), key=rank)
        top = logits[order[0]]
        weights = [math.exp((logits[t]-top)/self.temperature) for t in order]
        total = math.fsum(weights)
        if self.top_p < 1:
            # The smallest prefix of the most probable tokens holding at least top_p of the mass.
            kept, mass = 0, 0.0
            while kept < len(order) and (kept == 0 or mass < self.top_p*total):
                mass += weights[kept]
                kept += 1
            order, weights, total = order[:kept], weights[:kept], mass
        point, mass = self.random.random()*total, 0.0
        for token, weight in zip(order, weights):
            mass += weight
            if point < mass:
                return token
        return order[-1]

"""Chat prompts rendered with the Checkpoint's own chat template, by Transformers.

The prompt of `/v1/chat/completions` is what `AutoTokenizer.apply_chat_template()` of the pinned
Transformers version produces for the Checkpoint, text and token ids alike, with
`trust_remote_code=False` always: no code shipped with a Checkpoint ever runs. Transformers
needs no PyTorch for tokenizers and templates. A Checkpoint whose tokenizer or template
Transformers cannot load keeps `/v1/completions`; only chat is refused, with the reason.
"""
import os
import threading

ROLES = ('system', 'user', 'assistant', 'tool')
# Message fields the OpenAI API defines and chat templates read; others are not passed on.
FIELDS = ('role', 'content', 'name', 'tool_calls', 'tool_call_id')


class ChatTemplate:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Requests render off the worker while the worker decodes: Transformers may reconfigure
        # the shared Rust tokenizer inside a call, so calls never overlap.
        self.lock = threading.Lock()

    def render(self, messages, tools=None):
        """The rendered prompt and its token ids, exactly as `apply_chat_template(messages,
        tools=tools, add_generation_prompt=True)` gives them: the rendered text tokenized without
        added special tokens, which is how Transformers itself tokenizes it."""
        with self.lock:
            text = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                                      tokenize=False)
            return text, self.tokenizer(text, add_special_tokens=False)['input_ids']

    def decode(self, ids):
        with self.lock:
            return self.tokenizer.decode(list(ids), skip_special_tokens=True)


def load(directory):
    """(ChatTemplate, None) when the Checkpoint can chat, else (None, the reason)."""
    os.environ.setdefault('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')   # "PyTorch was not found"
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(directory), trust_remote_code=False,
                                                  local_files_only=True)
    except Exception as error:          # Transformers raises many types for unloadable tokenizers
        return None, 'Transformers cannot load its tokenizer (%s)' % type(error).__name__
    if not tokenizer.chat_template:
        return None, 'the Checkpoint has no chat template'
    return ChatTemplate(tokenizer), None


def messages(value):
    """The request's messages as chat templates expect them, or a ValueError naming the problem.
    Text content parts are joined into one string, as OpenAI-compatible servers do."""
    if not isinstance(value, list) or not value:
        raise ValueError('messages must be a non-empty array')
    result = []
    for i, message in enumerate(value):
        if not isinstance(message, dict):
            raise ValueError('messages[%d] must be an object' % i)
        if message.get('role') not in ROLES:
            raise ValueError('messages[%d].role must be one of %s' % (i, ', '.join(ROLES)))
        content = message.get('content')
        if isinstance(content, list):
            if not all(isinstance(p, dict) and p.get('type') == 'text' and isinstance(p.get('text'), str)
                       for p in content):
                raise ValueError('messages[%d].content: only text parts are supported' % i)
            content = ''.join(p['text'] for p in content)
        if content is None and not (message['role'] == 'assistant' and message.get('tool_calls')):
            raise ValueError('messages[%d].content is required' % i)
        if content is not None and not isinstance(content, str):
            raise ValueError('messages[%d].content must be a string or an array of text parts' % i)
        result.append({k: message[k] for k in FIELDS if message.get(k) is not None} | {'content': content})
    return result

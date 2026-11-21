"""Chat corpora: the model's chat template, pinned to the commit it comes from.

The Hub and the tokenizer download are replaced by stand-ins, so no network is used.
"""

from types import SimpleNamespace

import huggingface_hub
import pytest
import transformers

from openinterpretability.corpus_layouts import chat_formatter


class _Tokenizer:
    def __init__(self, template):
        self.chat_template = template

    def apply_chat_template(self, messages, tokenize):
        assert tokenize is False
        return "".join(f"<|{m['role']}|>{m['content']}" for m in messages)


def _offline(monkeypatch, template):
    loaded = []

    class Api:
        def model_info(self, model, revision=None):
            return SimpleNamespace(sha=f"{model}@{revision or 'main'}-sha")

    def from_pretrained(model, revision=None):
        loaded.append((model, revision))
        return _Tokenizer(template)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", from_pretrained)
    return loaded


def test_the_formatter_renders_messages_with_the_pinned_template(monkeypatch):
    loaded = _offline(monkeypatch, "{{ messages }}")
    render, revision = chat_formatter("org/chat")
    assert revision == "org/chat@main-sha"
    assert loaded == [("org/chat", "org/chat@main-sha")]  # the tokenizer is at that commit
    messages = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert render(messages) == "<|user|>hi<|assistant|>yo"
    assert render(None) == ""


def test_a_tokenizer_without_a_chat_template_is_refused(monkeypatch):
    _offline(monkeypatch, None)
    with pytest.raises(ValueError, match="org/base's tokenizer has no chat template"):
        chat_formatter("org/base", "abc")

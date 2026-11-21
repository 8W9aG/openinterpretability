"""``LLMAgent``: its default client and the temperature it passes on."""

import sys
from types import SimpleNamespace

from openinterpretability.agents.base import LLMAgent
from openinterpretability.agents.reviewer import ProbeSet


class Dummy(LLMAgent):
    pass


def test_the_client_is_made_from_litellm_once_on_first_use(monkeypatch):
    made = []
    completion = object()

    def from_litellm(fn):
        made.append(fn)
        return SimpleNamespace(name="instructor client")

    monkeypatch.setitem(sys.modules, "instructor", SimpleNamespace(from_litellm=from_litellm))
    monkeypatch.setitem(sys.modules, "litellm", SimpleNamespace(completion=completion))
    agent = Dummy("anthropic/m")
    client = agent.client
    assert client.name == "instructor client" and made == [completion]
    assert agent.client is client and len(made) == 1


def test_a_set_temperature_goes_to_the_completion(mocker):
    client = mocker.Mock(spec=["chat"])
    client.chat.completions.create.return_value = "answer"
    assert Dummy("openai/m", temperature=0.3, client=client).structured(ProbeSet, "hi") == "answer"
    assert client.chat.completions.create.call_args.kwargs["temperature"] == 0.3

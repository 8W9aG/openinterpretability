"""A unit's Neuronpedia link in the site tree (openinterpretability/builder/tree.py)."""

from openinterpretability.builder.tree import UnitNode

GPT2, TC = "openai-community/gpt2", "gpt2-small-transcoders"


def _node(**kw):
    return UnitNode(model=GPT2, layer=8, idx=5, href="x.html", **kw)


def test_only_a_feature_with_a_dictionary_has_a_neuronpedia_page():
    assert _node().neuronpedia is None  # a neuron
    assert _node(kind="dictionary_feature").neuronpedia is None  # no dictionary
    assert _node(kind="dictionary_feature", dictionary=TC).neuronpedia.endswith("/8-tres-dc/5")

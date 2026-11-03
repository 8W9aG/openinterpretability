"""WordNet categories of the peak words (`openinterp/wordnet.py`) in the agents' prompts."""

from __future__ import annotations

from openinterp import wordnet
from openinterp.agents.proposer import ProposerAgent
from openinterp.agents.reviewer import ReviewerAgent
from openinterp.core.schemas import ActivatingExample, NeuronIdentifier

# name: (lemma, hypernyms, frequency count); depth follows from the hypernyms.
TREE = {
    "entity.n.01": ("entity", [], 0),
    "physical_entity.n.01": ("physical entity", ["entity.n.01"], 0),
    "object.n.01": ("object", ["physical_entity.n.01"], 0),
    "whole.n.01": ("whole", ["object.n.01"], 0),
    "part.n.01": ("part", ["whole.n.01"], 0),
    "body_part.n.01": ("body part", ["part.n.01"], 0),
    "piece.n.01": ("piece", ["part.n.01"], 5),
    "fragment.n.01": ("fragment", ["part.n.01"], 2),
    "limb.n.01": ("limb", ["body_part.n.01"], 0),
    "arm.n.01": ("arm", ["limb.n.01"], 9),
    "leg.n.01": ("leg", ["limb.n.01"], 8),
    "hand.n.01": ("hand", ["body_part.n.01"], 7),
    "eye.n.01": ("eye", ["body_part.n.01"], 20),
    "toe.n.01": ("toe", ["body_part.n.01"], 3),
    "table.n.01": ("table", ["whole.n.01"], 4),
    "leg.n.02": ("leg", ["table.n.01"], 1),
    "hand.n.02": ("hand", ["whole.n.01"], 1),
}
SENSES = {"arm": ["arm.n.01"], "leg": ["leg.n.01", "leg.n.02"], "hand": ["hand.n.01", "hand.n.02"]}
SENSES |= {"table": ["table.n.01"], "eye": ["eye.n.01"]}
GLOSS = {"leg.n.02": "one of the supports for a piece of furniture", "hand.n.02": "a hired " * 20}


class FakeLexicon:
    def senses(self, word):
        return SENSES.get(word, [])

    def hypernyms(self, synset):
        return TREE[synset][1]

    def hyponyms(self, synset):
        return [s for s, (_, parents, _) in TREE.items() if synset in parents]

    def lemma(self, synset):
        return TREE[synset][0]

    def depth(self, synset):
        parents = TREE[synset][1]
        return 1 + min(self.depth(p) for p in parents) if parents else 0

    def gloss(self, synset):
        return GLOSS.get(synset, synset)

    def count(self, synset):
        return TREE[synset][2]


def example(words, peak):
    tokens = [" " + w for w in words]
    acts = [1.0 if i == peak else 0.0 for i in range(len(tokens))]
    return ActivatingExample(
        text="".join(tokens), tokens=tokens, activations=acts, max_activation=1.0,
        max_token_index=peak,
    )  # fmt: skip


def body_examples():
    return [example(["my", w, "hurts"], 1) for w in ["arm", "leg", "hand", "table", "the", "xyz"]]


def test_peak_word_joins_subword_tokens():
    ex = ActivatingExample(
        text=" Chicago!", tokens=[" Chic", "ago", "!"], activations=[0.0, 1.0, 0.0],
        max_activation=1.0, max_token_index=1,
    )  # fmt: skip
    assert wordnet.peak_word(ex) == "chicago"


def test_peak_word_is_none_without_letters():
    assert wordnet.peak_word(example(["a", "!", "b"], 1)) is None


def test_peak_words_are_distinct_and_long_enough():
    exs = [example(["x", w], 1) for w in ["Arm", "arm", "is", "leg"]]
    assert wordnet.peak_words(exs) == ["arm", "leg"]


def test_categorise_keeps_lowest_shared_hypernyms():
    found = wordnet.categorise(["arm", "leg", "hand", "table", "xyz"], FakeLexicon())
    assert found.known == ("arm", "leg", "hand", "table")
    assert found.missing == ("xyz",)
    by_synset = {g.synset: g for g in found.groups}
    # whole covers all four; part (same words as body part, shallower) is dropped.
    assert list(by_synset) == ["whole.n.01", "body_part.n.01"]
    assert by_synset["body_part.n.01"].words == ("arm", "leg", "hand")
    assert by_synset["body_part.n.01"].outside == ("table",)
    assert by_synset["body_part.n.01"].depth == 5


def test_categorise_needs_min_words():
    assert wordnet.categorise(["arm", "eye"], FakeLexicon()).groups == ()


def test_reviewer_material_for_the_largest_category():
    lex = FakeLexicon()
    body = wordnet.categorise(["arm", "leg", "hand"], lex).groups[0]
    assert body.synset == "body_part.n.01"
    assert wordnet.holdouts(lex, body) == ["eye", "toe", "limb"]
    assert wordnet.siblings(lex, body) == ["piece", "fragment"]
    senses = dict(wordnet.other_senses(lex, body))
    assert senses["leg"] == "one of the supports for a piece of furniture"
    assert len(senses["hand"]) == wordnet.MAX_GLOSS and senses["hand"].endswith("…")
    assert "arm" not in senses


def test_proposer_prompt_lists_categories(monkeypatch):
    monkeypatch.setattr(wordnet, "lexicon", FakeLexicon)
    ident = NeuronIdentifier(model_name="m", layer=0, neuron_idx=0, pole="positive")
    prompt = ProposerAgent.proposal_prompt(ident, body_examples())
    assert "- body part (body_part.n.01, depth 5): 3 of 4 (arm, leg, hand); outside: table" in (
        prompt
    )
    assert "Not in WordNet as nouns: the, xyz." in prompt


def test_reviewer_prompt_gives_probe_material(monkeypatch, submission):
    monkeypatch.setattr(wordnet, "lexicon", FakeLexicon)
    submission = submission.model_copy(update={"top_examples": body_examples()})
    prompt = ReviewerAgent.probe_prompt(submission)
    assert "under 'whole' (arm, leg, hand, table)" in prompt
    assert "for positive holdouts: eye, piece, toe" in prompt


def test_no_lexicon_no_note():
    assert wordnet.format_for_proposer(body_examples(), None) == ""
    assert wordnet.format_for_reviewer(body_examples(), None) == ""

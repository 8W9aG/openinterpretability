"""WordNet categories of the peak words (`openinterp/wordnet.py`) in the agents' prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from openinterp import wordnet
from openinterp.agents.proposer import ProposerAgent
from openinterp.agents.reviewer import ReviewerAgent
from openinterp.core.schemas import (
    ActivatingExample,
    NeuronIdentifier,
    WordnetSpecificity,
    WordStrength,
)
from openinterp.harness import specificity
from tests.site_pages import unit_pages

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
    def __init__(self, tree=TREE):
        self.tree = tree

    def has(self, synset):
        return synset in self.tree

    def senses(self, word):
        return SENSES.get(word, [])

    def hypernyms(self, synset):
        return self.tree[synset][1]

    def hyponyms(self, synset):
        return [s for s, (_, parents, _) in self.tree.items() if synset in parents]

    def lemma(self, synset):
        return self.tree[synset][0]

    def depth(self, synset):
        parents = self.tree[synset][1]
        return 1 + min(self.depth(p) for p in parents) if parents else 0

    def gloss(self, synset):
        return GLOSS.get(synset, synset)

    def count(self, synset):
        return self.tree[synset][2]


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
    assert wordnet.holdouts(lex, body.synset, body.words) == ["eye", "toe", "limb"]
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


# -- the specificity check ---------------------------------------------------------------

BIG = TREE | {f"finger{i}.n.01": (f"finger{i}", ["hand.n.01"], i) for i in range(6)}
BIG |= {f"shard{i}.n.01": (f"shard{i}", ["fragment.n.01"], i) for i in range(4)}
BIG |= {f"chunk{i}.n.01": (f"chunk{i}", ["piece.n.01"], i) for i in range(4)}


class FakeTokenizer:
    def __call__(self, text, return_offsets_mapping=False):
        return {"offset_mapping": [m.span() for m in re.finditer(r"\w+|[^\w\s]", text)]}


@dataclass(frozen=True)
class HotUnit:
    """Strength 1 on any token whose text starts with one of ``hot``, else 0."""

    hot: tuple[str, ...]
    skip_first: bool = False

    def strengths(self, runner, text):
        spans = runner.tokenizer(text)["offset_mapping"]
        return torch.tensor([[float(text[a:b].startswith(self.hot)) for a, b in spans]])


def test_lexicon_descendants_and_siblings():
    lex = FakeLexicon(BIG)
    assert set(wordnet.descendants(lex, "limb.n.01")) == {"arm.n.01", "leg.n.01"}
    near, words = wordnet.sibling_words(lex, "body_part.n.01", {"chunk3"}, 5)
    assert near == ["piece.n.01", "fragment.n.01"]
    # The siblings take turns; each gives its most frequent words first.
    assert words == ["piece", "shard3", "chunk2", "fragment", "chunk1"]


def test_word_strength_reads_only_the_words_tokens():
    runner = SimpleNamespace(tokenizer=FakeTokenizer())
    unit = HotUnit(("finger", "word"))
    # "Here is a word: x." has "word" hot, but only the word's own tokens count.
    assert specificity.word_strength(runner, unit, "Here is a word: {}.", "thumb") == 0.0
    assert specificity.word_strength(runner, unit, "Here is a word: {}.", "finger2") == 1.0
    first = HotUnit(("finger",), skip_first=True)
    assert specificity.word_strength(runner, first, "{} here", "finger2") is None


def test_auroc_counts_ties_half():
    assert specificity.auroc([1.0, 0.0], [0.0]) == 0.75


def test_specificity_compares_hyponyms_with_siblings():
    runner = SimpleNamespace(tokenizer=FakeTokenizer())
    result = specificity.specificity(
        runner, HotUnit(("finger", "eye")), FakeLexicon(BIG), "body_part.n.01", "proposer",
        scale=2.0, exclude=["Arm"],
    )  # fmt: skip
    category = [w.word for w in result.category]
    assert "arm" not in category and "eye" in category and "finger5" in category
    assert result.sibling_synsets == ["piece.n.01", "fragment.n.01"]
    assert all(w.strength == 0.0 for w in result.siblings)
    hot = sum(w.strength > 0 for w in result.category) / len(result.category)
    assert result.category_active == pytest.approx(hot, abs=1e-6)  # 1.0 >= 0.25 × 2.0
    assert result.siblings_active == 0.0
    assert result.auroc == pytest.approx(hot + (1 - hot) / 2, abs=1e-6)


def test_specificity_needs_enough_words_on_both_sides():
    runner = SimpleNamespace(tokenizer=FakeTokenizer())
    lex = FakeLexicon(BIG)
    assert (
        specificity.specificity(runner, HotUnit(("x",)), lex, "limb.n.01", "proposer", 1.0) is None
    )


def draft(synset=None):
    from openinterp.agents.proposer import HypothesisDraft

    pairs = [
        {
            "base_prompt": "a b",
            "counterfactual_prompt": "a c",
            "expected_delta": 0.5,
            "rationale": "r",
        }
    ] * 3
    return HypothesisDraft(
        explanation="x", activating_tokens=[], mechanism="retrospective",
        counterfactuals=pairs, wordnet_synset=synset,
    )  # fmt: skip


@pytest.fixture
def measured(monkeypatch):
    calls = []
    monkeypatch.setattr(wordnet, "lexicon", lambda: FakeLexicon(BIG))
    monkeypatch.setattr(
        "openinterp.harness.specificity.specificity",
        lambda runner, unit, lex, synset, source, scale, exclude: calls.append(
            (synset, source, scale, exclude)
        ),
    )
    return calls


def test_read_specificity_uses_the_proposers_synset(measured, submission):
    from openinterp.cli import cli  # noqa: F401  (loads the commands first)
    from openinterp.cli_submission import read_specificity

    sub = submission.model_copy(update={"top_examples": body_examples()})
    runner = SimpleNamespace(tokenizer=FakeTokenizer())
    read_specificity(runner, sub, draft("limb.n.01"))
    peaks = ["arm", "leg", "hand", "table", "the", "xyz"]
    assert measured == [("limb.n.01", "proposer", sub.max_activation, peaks)]


def test_read_specificity_falls_back_to_the_peak_words(measured, submission):
    from openinterp.cli import cli  # noqa: F401  (loads the commands first)
    from openinterp.cli_submission import read_specificity

    sub = submission.model_copy(update={"top_examples": body_examples()})
    runner = SimpleNamespace(tokenizer=FakeTokenizer())
    read_specificity(runner, sub, draft())
    assert measured[0][:2] == ("whole.n.01", "peak_words")
    assert read_specificity(runner, sub, draft("nonsense.n.09")) is None
    assert len(measured) == 1


def specificity_record(record):
    words = [WordStrength(word=w, strength=s) for w, s in [("eye", 1.5), ("toe", 0.25)]]
    result = WordnetSpecificity(
        synset="body_part.n.01", lemma="body part", source="proposer",
        sibling_synsets=["piece.n.01"], carriers=["Here is a word: {}."], scale=2.0,
        category=words, siblings=[WordStrength(word="chunk", strength=0.0)],
        category_active=0.5, siblings_active=0.0, auroc=1.0,
    )  # fmt: skip
    sub = record.submission.model_copy(update={"wordnet_specificity": result})
    return record.model_copy(update={"submission": sub})


def test_specificity_rendered(tmp_path, record):
    from openinterp.builder.compiler import compile_site
    from openinterp.builder.wiki import write_experiment_page

    rec = specificity_record(record)
    compile_site([rec], tmp_path / "site")
    (page,) = (tmp_path / "site" / "models").rglob("N117.html")
    html = unit_pages(page)
    assert "WordNet specificity" in html and "AUROC <strong>1.00</strong>" in html
    assert "named by the proposer" in html
    wiki = write_experiment_page(rec, tmp_path / "wiki").read_text()
    assert "- **body part:** eye 0.75, toe 0.12" in wiki
    assert "- **Siblings:** chunk 0.00" in wiki

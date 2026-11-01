"""Sister neurons across models (`openinterp sisters`), on random tiny Qwen2 and GPT-2 models
(no download)."""

import copy
import random

import pytest
import torch

from openinterp.builder.compiler import compile_site
from openinterp.core.schemas import NeuronIdentifier
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.sisters import Correlations, find_sisters, paired_rows, shared_boundaries
from openinterp.sisters import SIBLINGS, SisterRow, format_sisters, load_rows, pairs, sisters
from tests.conftest import FakeTokenizer
from tests.test_wiring import qwen_runner  # noqa: F401  (fixture)


@pytest.fixture
def texts():
    rng = random.Random(0)
    return [
        "".join(rng.choice("abcdefghijklmno") for _ in range(rng.randint(20, 40)))
        for _ in range(30)
    ]


def test_streaming_pearson_matches_corrcoef():
    g = torch.Generator().manual_seed(0)
    x, y = torch.rand(300, 3, generator=g), torch.rand(300, 5, generator=g)
    y[:, 0] = 2 * x[:, 1] + 0.1 * y[:, 0]
    corr = Correlations(3)
    for rows in (slice(0, 100), slice(100, 300)):  # two batches
        corr.add_sources(x[rows])
        corr.add_targets(7, x[rows], y[rows])
    expected = torch.corrcoef(torch.cat([x, y], 1).T)[:3, 3:]
    assert torch.allclose(corr.pearson(7), expected.double(), atol=1e-5)


def test_a_copy_of_the_model_is_its_own_sister(qwen_runner, texts):  # noqa: F811
    twin = NeuronHookRunner(copy.deepcopy(qwen_runner.model), FakeTokenizer())
    units = [(1, 5, "positive"), (2, 11, "negative")]
    found, tokens = find_sisters(qwen_runner, twin, units, texts, other_name="twin", k=3)
    assert tokens == sum(len(t) - 1 for t in texts)  # every token but each text's first
    for unit in units:
        s = found[unit]
        top = s.matches[0]
        assert (top.layer, top.neuron_idx, top.pole) == unit
        assert top.pearson == pytest.approx(1.0, abs=1e-4)
        assert len(s.matches) == 3 and s.matches[0].pearson >= s.matches[1].pearson
        back = s.back[0]
        assert (back.layer, back.neuron_idx, back.pole) == unit  # mutual
        assert s.rotated < 0.99  # a rotated basis doesn't contain the unit


class PairTokenizer(FakeTokenizer):
    """Two characters per token: shares every other boundary with ``FakeTokenizer``."""

    def encode(self, text, add_special_tokens=True):
        return [self._id(text[i]) for i in range(0, len(text), 2)]

    def __call__(self, texts, return_offsets_mapping=False, **kw):
        assert isinstance(texts, str)
        ids = self.encode(texts)
        spans = [(2 * i, min(2 * i + 2, len(texts))) for i in range(len(ids))]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "offset_mapping": spans}


def test_shared_boundaries_are_token_ends_both_tokenizations_have():
    # Ends 1..6 (one character a token) against 2, 4, 6; first tokens are left out.
    assert shared_boundaries([1, 2, 3, 4, 5, 6], [2, 4, 6]) == ([3, 5], [1, 2])
    # Where several tokens end at one character (a split byte), the last counts.
    assert shared_boundaries([1, 3, 3, 4], [1, 3, 4]) == ([2, 3], [1, 2])
    assert shared_boundaries([-1, 2, 4], [2, 4]) == ([2], [1])  # a special token


def test_models_that_tokenize_differently_pair_where_both_read_the_same_text(
    qwen_runner,  # noqa: F811
    tiny_gpt2,
    texts,
):
    gpt2 = NeuronHookRunner(tiny_gpt2, PairTokenizer())
    text = texts[0]
    (src, oth), *_ = paired_rows(qwen_runner, gpt2, [text], [1], [0])
    ends = [min(e, len(text)) for e in range(4, len(text) + 2, 2)]  # pair tokens 2, 3, ...
    assert src[1].shape[0] == oth[0].shape[0] == len(ends)
    _, cache = qwen_runner.run_with_cache(text, [1])
    assert torch.allclose(src[1], cache[1][0, [e - 1 for e in ends]])  # the char ending there
    _, cache = gpt2.run_with_cache({"input_ids": torch.tensor([gpt2.tokenizer.encode(text)])}, [0])
    assert torch.allclose(oth[0], cache[0][0, 1 : len(ends) + 1])

    found, tokens = find_sisters(
        qwen_runner, gpt2, [(1, 5, "positive")], texts, other_name="g", k=2
    )
    assert tokens == sum((len(t) + 1) // 2 - 1 for t in texts)
    assert len(found[(1, 5, "positive")].matches) == 2


def test_pairs_are_every_ordered_pair_and_each_models_siblings():
    a, b = "openai-community/gpt2", next(iter(SIBLINGS))
    assert set(pairs([a, b])) == {(a, b), (b, a)} | {(b, s) for s in SIBLINGS[b]}


def _record(record, model, layer, idx, pole, status="VERIFIED"):
    ident = NeuronIdentifier(model_name=model, layer=layer, neuron_idx=idx, pole=pole)
    return record.model_copy(
        update={
            "submission": record.submission.model_copy(update={"identifier": ident}),
            "verdict": record.verdict.model_copy(update={"status": status}),
        }
    )


def test_rows_cover_verified_poles_and_link_both_ways(tmp_path, qwen_runner, texts, record):  # noqa: F811
    twin = NeuronHookRunner(copy.deepcopy(qwen_runner.model), FakeTokenizer())
    records = [
        _record(record, "tiny", 1, 5, "positive"),
        _record(record, "tiny", 2, 3, "negative", status="REJECTED"),  # not VERIFIED
        _record(record, "tiny-twin", 1, 5, "positive"),  # the sister has a page too
    ]
    out = tmp_path / "sisters.jsonl"
    loads = []

    def load(name, revision):
        loads.append(name)
        return qwen_runner if name == "tiny" else twin

    kw = {"model_name": "tiny", "other": "tiny-twin", "corpus": "c", "corpus_sha256": "0" * 64}
    assert sisters(records, load, out, texts, k=2, echo=lambda _: None, **kw) == 1
    assert loads == ["tiny", "tiny-twin"]
    rows = load_rows(out)
    (row,) = rows["tiny/L1/N5-positive"]
    assert row.mutual and row.sister is not None and row.coverage == 1.0
    assert row.sister.slug("tiny-twin") == "tiny-twin/L1/N5-positive"
    assert sisters(records, load, out, texts, echo=lambda _: None, **kw) == 0  # already done

    site = tmp_path / "site"
    compile_site(records, site, sister_rows=rows)
    page = (site / "models/tiny/L1/N5.html").read_text()
    assert "Sister neurons" in page and "models/tiny-twin/L1/N5.html" in page
    back = (site / "models/tiny-twin/L1/N5.html").read_text()
    assert "Matched from" in back and "models/tiny/L1/N5.html" in back


def test_a_sister_must_be_mutual_and_beat_the_rotated_basis():
    def row(back_slug, pearson, rotated):
        return SisterRow.model_validate(
            {
                "slug": "m/L1/N5-positive",
                "other": "s",
                "model_revision": None,
                "other_revision": None,
                "corpus": "c",
                "corpus_sha256": "0",
                "tokens": 10,
                "coverage": 1.0,
                "matches": [
                    {
                        "layer": 2,
                        "neuron_idx": 9,
                        "pole": "positive",
                        "pearson": pearson,
                        "back_slug": back_slug,
                        "back_pearson": pearson,
                    }
                ],
                "rotated": rotated,
            }
        )

    assert row("m/L1/N5-positive", 0.8, 0.3).sister is not None
    assert row("m/L1/N6-positive", 0.8, 0.3).sister is None  # not mutual
    assert row("m/L1/N5-positive", 0.2, 0.3).sister is None  # below the rotated basis


def _sister_row(slug, other, matches, rotated=0.1):
    keys = ("layer", "neuron_idx", "pole", "pearson", "back_slug", "back_pearson")
    return SisterRow.model_validate(
        {
            "slug": slug,
            "other": other,
            "model_revision": None,
            "other_revision": None,
            "corpus": "c",
            "corpus_sha256": "0",
            "tokens": 10,
            "coverage": 1.0,
            "matches": [dict(zip(keys, m, strict=True)) for m in matches],
            "rotated": rotated,
        }
    )


def test_agents_see_studied_sisters_in_both_directions():
    rows = {
        "a/L1/N5-positive": [
            _sister_row(
                "a/L1/N5-positive",
                "b",
                [
                    (2, 9, "negative", 0.8, "a/L1/N5-positive", 0.8),  # mutual, studied
                    (2, 3, "positive", 0.55, "a/L0/N1-positive", 0.6),  # never studied
                    (3, 4, "positive", 0.3, "a/L0/N1-positive", 0.5),  # weak, not a sister
                ],
            )
        ],
        "c/L0/N2-positive": [
            _sister_row("c/L0/N2-positive", "a", [(1, 5, "positive", 0.6, "a/L9/N9-positive", 0.7)])
        ],
    }
    findings = {
        "b/L2/N9-negative": ("VERIFIED", "closing quotes"),
        "b/L3/N4-positive": ("REJECTED", "weak link"),
        "c/L0/N2-positive": ("VERIFIED", "speech verbs"),
    }
    text = format_sisters("a/L1/N5-positive", rows, findings, "reviewer")
    lines = text.splitlines()[1:]
    assert lines == [
        "- b/L2/N9-negative (r +0.80, sister; VERIFIED): closing quotes",
        "- c/L0/N2-positive (r +0.60; VERIFIED): speech verbs",  # matched a's pole, strongly
    ]
    assert "tell them apart" in text
    assert format_sisters("a/L1/N6-positive", rows, findings, "proposer") == ""


def test_the_proposer_prompt_ends_with_related_research(record):
    from openinterp.agents.proposer import ProposerAgent

    sub = record.submission
    prompt = ProposerAgent.proposal_prompt(sub.identifier, sub.top_examples, related="SISTERS\n")
    assert prompt.endswith("SISTERS\n")
    assert "SISTERS" not in ProposerAgent.proposal_prompt(sub.identifier, sub.top_examples)

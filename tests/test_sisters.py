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
    # A sister study's reviewer isn't shown the pole whose explanation is the hypothesis.
    tested = format_sisters("a/L1/N5-positive", rows, findings, "reviewer", ["b/L2/N9-negative"])
    assert "b/L2/N9" not in tested and "c/L0/N2" in tested


def test_the_proposer_prompt_ends_with_related_research(record):
    from openinterp.agents.proposer import ProposerAgent

    sub = record.submission
    prompt = ProposerAgent.proposal_prompt(sub.identifier, sub.top_examples, related="SISTERS\n")
    assert prompt.endswith("SISTERS\n")
    assert "SISTERS" not in ProposerAgent.proposal_prompt(sub.identifier, sub.top_examples)


def _sister_registry(tmp_path, record, other="o/m"):
    """A VERIFIED Qwen pole (the `record` fixture's) with a sister L2/N9- in ``other``."""
    from openinterp.builder.wiki import write_experiment_page

    write_experiment_page(record, tmp_path / "wiki")
    (tmp_path / "pyproject.toml").write_text(
        f'[tool.openinterp]\nmodels = ["Qwen/Qwen2.5-0.5B", "{other}"]\n'
        f'[tool.openinterp.window_tokens]\n"{other}" = 64\n'
    )
    slug = record.submission.identifier.slug
    row = _sister_row(slug, other, [(2, 9, "negative", 0.7, slug, 0.7)])
    rows = tmp_path / "docs/analyses/sister-neurons.jsonl"
    rows.parent.mkdir(parents=True)
    rows.write_text(row.model_dump_json() + "\n")
    return tmp_path / "wiki"


def test_the_planner_offers_unstudied_sisters_in_configured_models(tmp_path, record):
    from datetime import UTC, datetime

    from openinterp.draw_steps import render, steps
    from openinterp.planner import plan_next

    now = datetime(2026, 10, 20, 12, tzinfo=UTC)
    wiki = _sister_registry(tmp_path, record)
    plan = plan_next(wiki, seed=1, epsilon=0.0, now=now)
    (sister,) = [c for c in plan.candidates if c.kind == "sister"]
    assert sister.slug == "o--m/L2/N9-negative" and sister.model == "o/m"
    assert sister.audit.startswith("--model o/m --layer 2 --neuron 9 --pole negative")
    assert "--window-tokens 64 --sister-of " in sister.audit
    claimed = plan_next(wiki, seed=1, epsilon=0.0, now=now, claims=["o--m-L2-N9-negative"])
    (c,) = [c for c in claimed.candidates if c.kind == "sister"]
    assert c.weight < sister.weight and "claims the sister" in c.rationale

    plan.choice = sister
    text = render(steps(plan, corpus="c.jsonl"))
    assert "experiment/o--m-L2-N9-negative-sister" in text and "VERIFIED pole's history" in text
    scan, submit = [line for line in text.splitlines() if "openinterp audit" in line]
    assert scan.count("--sister-of") == submit.count("--sister-of") == 1

    other = _sister_registry(tmp_path / "x", record, other="not/configured")
    (tmp_path / "x/pyproject.toml").write_text(
        '[tool.openinterp]\nmodels = ["Qwen/Qwen2.5-0.5B"]\n'
    )
    plan = plan_next(other, seed=1, epsilon=0.0, now=now)
    assert not any(c.kind == "sister" for c in plan.candidates)  # CI couldn't reproduce it


def test_pages_link_a_sister_study_to_its_source(tmp_path, record):
    from openinterp.builder.wiki import experiment_id

    source = record  # VERIFIED, Qwen L3/N117+
    ref = f"{source.submission.identifier.slug}/{experiment_id(source)}"
    study = _record(record, "o/m", 2, 9, "negative")
    later = study.verdict.timestamp.replace(day=2)
    study = study.model_copy(
        update={
            "submission": study.submission.model_copy(update={"sister_of": ref}),
            "verdict": study.verdict.model_copy(update={"timestamp": later}),
        }
    )
    slug = source.submission.identifier.slug
    rows = {slug: [_sister_row(slug, "o/m", [(2, 9, "negative", 0.7, slug, 0.7)])]}
    site = tmp_path / "site"
    compile_site([source, study], site, sister_rows=rows)
    page = (site / "models/o--m/L2/N9.html").read_text()
    assert "Sister study: tests the explanation of" in page
    assert f"models/Qwen--Qwen2.5-0.5B/L3/N117.html#positive-{experiment_id(source)}" in page
    assert "tested here: VERIFIED" in page
    source_page = (site / "models/Qwen--Qwen2.5-0.5B/L3/N117.html").read_text()
    assert "models/o--m/L2/N9.html" in source_page

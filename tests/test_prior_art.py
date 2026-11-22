"""Prior art: published explanations found after the proposer answers."""

import pytest

from openinterpretability.agents.reviewer import ReviewerAgent
from openinterpretability.builder.compiler import compile_site
from openinterpretability.core.identifiers import FeatureIdentifier
from openinterpretability.core.schemas import NeuronIdentifier
from openinterpretability.crossref import AlignmentRow, Explanation
from openinterpretability.prior_art import TOP, find_prior_art, format_for_reviewer
from tests.site_pages import unit_pages

GPT2, TC = "openai-community/gpt2", "gpt2-small-transcoders"
FOUND = [
    Explanation(description="said by a person", explainer="human"),
    Explanation(description="closing quotes", explainer="gpt-4o-mini"),
]


def _sub(record, ident):
    return record.submission.model_copy(update={"identifier": ident})


def _fetch(asked):
    def fetch(dictionary, layer, idx):
        asked.append(idx)
        return FOUND

    return fetch


def test_a_latent_gets_its_own_machine_written_explanation(record, monkeypatch):
    monkeypatch.setenv("OPENINTERPRETABILITY_PRIOR_ART", "1")
    sub = _sub(record, FeatureIdentifier(model_name=GPT2, layer=8, dictionary=TC, feature_idx=5))
    (art,) = find_prior_art(sub, runner=None, fetch=_fetch([]))
    assert (art.relation, art.cosine, art.description) == ("self", None, "closing quotes")
    assert art.explainer == "gpt-4o-mini" and art.url.endswith("/gpt2-small/8-tres-dc/5")


def test_a_pole_gets_the_aligned_latents_above_chance(record, monkeypatch):
    monkeypatch.setenv("OPENINTERPRETABILITY_PRIOR_ART", "1")
    ident = NeuronIdentifier(model_name=GPT2, layer=8, neuron_idx=7, pole="positive")
    matches = [{"feature_idx": i, "cosine": c} for i, c in [(1, 0.9), (2, -0.5), (3, 0.1),
                                                              (4, 0.4), (5, 0.35)]]  # fmt: skip
    row = AlignmentRow.model_validate({
        "slug": ident.slug, "dictionary": TC, "repo": "r", "revision": "v", "path": "p",
        "model_revision": None, "writes": {"matches": matches, "chance": 0.2, "percentile": 0.9},
    })  # fmt: skip
    asked = []
    found = find_prior_art(_sub(record, ident), runner=None, rows={ident.slug: [row]},
                           fetch=_fetch(asked))  # fmt: skip
    assert asked == [1, 2, 4][:TOP]  # by |cosine|, above chance (0.2), at most TOP per side
    assert [(a.relation, a.cosine) for a in found] == [("writes", 0.9), ("writes", -0.5),
                                                        ("writes", 0.4)]  # fmt: skip


def test_prior_art_is_off_without_the_switch(record):
    # tests/conftest.py sets OPENINTERPRETABILITY_PRIOR_ART=0, as an offline user can.
    sub = _sub(record, FeatureIdentifier(model_name=GPT2, layer=8, dictionary=TC, feature_idx=5))
    assert find_prior_art(sub, runner=None, fetch=_fetch([])) is None


def test_the_reviewer_sees_prior_art_and_the_page_shows_it(record, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENINTERPRETABILITY_PRIOR_ART", "1")
    sub = _sub(record, FeatureIdentifier(model_name=GPT2, layer=8, dictionary=TC, feature_idx=5))
    art = find_prior_art(sub, runner=None, fetch=_fetch([]))
    sub = sub.model_copy(update={"prior_art": art})
    prompt = ReviewerAgent.probe_prompt(sub)
    assert format_for_reviewer(art) in prompt and "[neuronpedia, gpt-4o-mini]" in prompt
    assert "Prior art" not in ReviewerAgent.probe_prompt(sub.model_copy(update={"prior_art": []}))

    site = tmp_path / "site"
    compile_site([record.model_copy(update={"submission": sub})], site)
    page = unit_pages(site / "models/openai-community--gpt2/L8/gpt2-small-transcoders/F5.html")
    assert "Prior art" in page and "closing quotes" in page and art[0].url in page


def test_a_head_has_no_prior_art(record, monkeypatch):
    from openinterpretability.core.identifiers import HeadIdentifier

    monkeypatch.setenv("OPENINTERPRETABILITY_PRIOR_ART", "1")
    sub = _sub(record, HeadIdentifier(model_name=GPT2, layer=8, head_idx=1))
    assert find_prior_art(sub, runner=None, fetch=_fetch([])) is None


def test_a_latent_neuronpedia_does_not_host_or_has_no_explanation_gets_none(record, monkeypatch):
    monkeypatch.setenv("OPENINTERPRETABILITY_PRIOR_ART", "1")
    qwen = FeatureIdentifier(model_name="Qwen/Qwen2.5-0.5B", layer=3,
                             dictionary="qwen2.5-0.5b-transcoders", feature_idx=5)  # fmt: skip
    asked = []
    assert find_prior_art(_sub(record, qwen), runner=None, fetch=_fetch(asked)) == []
    assert asked == []  # no page to look up
    gpt2 = FeatureIdentifier(model_name=GPT2, layer=8, dictionary=TC, feature_idx=5)
    assert find_prior_art(_sub(record, gpt2), runner=None, fetch=lambda *a: None) == []


def test_a_pole_skips_dictionaries_neuronpedia_does_not_host(record, monkeypatch):
    monkeypatch.setenv("OPENINTERPRETABILITY_PRIOR_ART", "1")
    ident = NeuronIdentifier(model_name="Qwen/Qwen2.5-0.5B", layer=3, neuron_idx=7,
                             pole="positive")  # fmt: skip
    align = "openinterpretability.harness.crossref.align_layer"
    monkeypatch.setattr(align, lambda *a: pytest.fail("aligned"))
    assert find_prior_art(_sub(record, ident), runner=None, rows={}, fetch=_fetch([])) == []


def _matches(*pairs):
    from types import SimpleNamespace

    return [SimpleNamespace(feature_idx=i, cosine=c) for i, c in pairs]


def test_a_pole_without_a_stored_row_is_aligned_now_on_both_sides(record, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("OPENINTERPRETABILITY_PRIOR_ART", "1")
    ident = NeuronIdentifier(model_name=GPT2, layer=8, neuron_idx=7, pole="positive")
    write = SimpleNamespace(matches=_matches((1, 0.9), (2, 0.1)), chance=0.2)
    read = SimpleNamespace(matches=_matches((3, -0.6), (4, 0.5)), chance=0.2)
    seen = []

    def align_layer(runner, spec, layer, poles):
        seen.append((runner, spec, layer, poles))
        return {(7, "positive"): (write, read)}

    monkeypatch.setattr("openinterpretability.harness.crossref.align_layer", align_layer)

    def fetch(dictionary, layer, idx):
        return [] if idx == 4 else FOUND  # latent 4 has no explanation, so it is left out

    found = find_prior_art(_sub(record, ident), runner="runner", rows={}, fetch=fetch)
    from openinterpretability.harness.dictionaries import DICTIONARIES

    assert seen == [("runner", DICTIONARIES[TC], 8, [(7, "positive")])]
    assert [(a.relation, a.feature_idx, a.cosine) for a in found] == [
        ("writes", 1, 0.9), ("reads", 3, -0.6)]  # fmt: skip


def test_a_pole_whose_alignment_fails_gets_no_prior_art(record, monkeypatch):
    monkeypatch.setenv("OPENINTERPRETABILITY_PRIOR_ART", "1")
    ident = NeuronIdentifier(model_name=GPT2, layer=8, neuron_idx=7, pole="negative")

    def align_layer(*args):
        raise OSError("no weights offline")

    monkeypatch.setattr("openinterpretability.harness.crossref.align_layer", align_layer)
    assert find_prior_art(_sub(record, ident), runner=None, rows={}, fetch=_fetch([])) == []


def test_a_write_only_alignment_has_no_reads_side(record, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("OPENINTERPRETABILITY_PRIOR_ART", "1")
    ident = NeuronIdentifier(model_name=GPT2, layer=8, neuron_idx=7, pole="positive")
    write = SimpleNamespace(matches=_matches((1, 0.9)), chance=0.2)
    monkeypatch.setattr("openinterpretability.harness.crossref.align_layer",
                        lambda *a: {(7, "positive"): (write, None)})  # fmt: skip
    found = find_prior_art(_sub(record, ident), runner=None, rows={}, fetch=_fetch([]))
    assert [a.relation for a in found] == ["writes"]

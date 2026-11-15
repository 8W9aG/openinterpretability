"""Prior art: published explanations found after the proposer answers."""

from openinterp.agents.reviewer import ReviewerAgent
from openinterp.builder.compiler import compile_site
from openinterp.core.identifiers import FeatureIdentifier
from openinterp.core.schemas import NeuronIdentifier
from openinterp.crossref import AlignmentRow, Explanation
from openinterp.prior_art import TOP, find_prior_art, format_for_reviewer
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
    monkeypatch.setenv("OPENINTERP_PRIOR_ART", "1")
    sub = _sub(record, FeatureIdentifier(model_name=GPT2, layer=8, dictionary=TC, feature_idx=5))
    (art,) = find_prior_art(sub, runner=None, fetch=_fetch([]))
    assert (art.relation, art.cosine, art.description) == ("self", None, "closing quotes")
    assert art.explainer == "gpt-4o-mini" and art.url.endswith("/gpt2-small/8-tres-dc/5")


def test_a_pole_gets_the_aligned_latents_above_chance(record, monkeypatch):
    monkeypatch.setenv("OPENINTERP_PRIOR_ART", "1")
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
    # tests/conftest.py sets OPENINTERP_PRIOR_ART=0, as an offline user can.
    sub = _sub(record, FeatureIdentifier(model_name=GPT2, layer=8, dictionary=TC, feature_idx=5))
    assert find_prior_art(sub, runner=None, fetch=_fetch([])) is None


def test_the_reviewer_sees_prior_art_and_the_page_shows_it(record, monkeypatch, tmp_path):
    monkeypatch.setenv("OPENINTERP_PRIOR_ART", "1")
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

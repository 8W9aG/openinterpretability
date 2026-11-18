"""Broad units (#4545): VERIFIED, but active on at least half their held-out windows."""

from openinterpretability.builder.compiler import compile_site
from openinterpretability.core.explanation import (
    BROAD_SHARE,
    ExplanationScores,
    held_out_active_share,
    is_broad,
)
from tests.site_pages import unit_pages

MODEL = "Qwen--Qwen2.5-0.5B"


def _scores(record, active: int, candidates: int = 1000) -> ExplanationScores:
    return ExplanationScores(
        slug=record.submission.identifier.slug, explanation="x", threshold=0.6,
        window_tokens=32, quantiles=1, quantile_edges=[], seed=3, corpus="c",
        candidate_windows=candidates, active_windows=active,
    )  # fmt: skip


def _with(record, scores, status="VERIFIED"):
    verdict = record.verdict.model_copy(update={"explanation_scores": scores, "status": status})
    return record.model_copy(update={"verdict": verdict})


def test_broad_means_verified_and_active_on_half_the_held_out_windows(record):
    assert BROAD_SHARE == 0.5
    assert held_out_active_share(None) is None
    assert held_out_active_share(_scores(record, 0, candidates=0)) is None
    assert held_out_active_share(_scores(record, 250)) == 0.25
    assert is_broad(_with(record, _scores(record, 500)).verdict)  # exactly half counts
    assert not is_broad(_with(record, _scores(record, 499)).verdict)
    assert not is_broad(_with(record, _scores(record, 900), "NEEDS_REVISION").verdict)
    assert not is_broad(_with(record, None).verdict) and not is_broad(None)


def test_the_experiment_page_marks_a_broad_unit_in_words(tmp_path, record):
    compile_site([_with(record, _scores(record, 800))], tmp_path / "broad")
    page = unit_pages(tmp_path / "broad" / "models" / MODEL / "L3" / "N117.html")
    assert ">broad</span>" in page and "held-out text" in page
    compile_site([_with(record, _scores(record, 40))], tmp_path / "narrow")
    page = unit_pages(tmp_path / "narrow" / "models" / MODEL / "L3" / "N117.html")
    assert ">broad</span>" not in page


def test_the_explanation_page_marks_a_broad_cited_unit(tmp_path, record):
    from datetime import UTC, datetime
    from types import SimpleNamespace

    from openinterpretability.builder.explanation_pages import load_views
    from tests.test_explanation_records import SLUG, _record, _write

    wiki = tmp_path / "wiki"
    _write(wiki, _record(datetime(2026, 11, 17, tzinfo=UTC), status="VERIFIED"))
    scores = SimpleNamespace(candidate_windows=100, active_windows=90)
    verdict = SimpleNamespace(status="VERIFIED", explanation_scores=scores)
    states = {
        SLUG: SimpleNamespace(record=SimpleNamespace(verdict=verdict), standing=("VERIFIED", None))
    }
    (prompt,) = load_views(wiki, states=states)
    assert prompt.newest.broad == {SLUG}
    compile_site([record], tmp_path / "site", explanations=[prompt])
    page = (tmp_path / "site" / prompt.newest.href).read_text()
    assert ">broad</span>" in page and "A <em>broad</em> unit" in page

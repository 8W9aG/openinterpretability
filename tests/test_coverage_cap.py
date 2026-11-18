"""Prompt coverage is capped at 100%, and an overshoot is kept and shown (#4507)."""

import itertools
from datetime import UTC, datetime

import pytest

from openinterpretability import explain_coverage as ec


def _fake(monkeypatch, kept: float, drawn: float = -4.0):
    """Clean run −1, all ablated −5, the explained units ``kept``, random sets ``drawn``."""
    # Then the per-kind and pruning passes, which these tests don't check, read ``drawn``.
    values = itertools.chain([-1.0, -5.0, kept] + [drawn] * ec.DRAWS, itertools.repeat(drawn))
    monkeypatch.setattr(ec, "keep_hooks", lambda *a: None)
    monkeypatch.setattr(ec, "_log_prob", lambda *a: next(values))


def test_an_overshoot_is_capped_and_reported(monkeypatch):
    _fake(monkeypatch, kept=0.0, drawn=1.0)  # raw (0+5)/4 = 125%; random draws 150%
    order = [((1, 2, "positive"), 1.0, "VERIFIED", True), ((1, 3, "positive"), 0.5, "x", False)]
    found = ec.prompt_coverage(None, "p", None, 0, -1, order)
    assert found["coverage"] == 1.0 and found["raw"] == pytest.approx(1.25)
    assert found["overshoot"] and found["random_draws"] == [1.0] * ec.DRAWS


def test_coverage_under_100_percent_is_unchanged(monkeypatch):
    _fake(monkeypatch, kept=-3.0)  # raw (−3+5)/4 = 50%
    order = [((1, 2, "positive"), 1.0, "VERIFIED", True)]
    found = ec.prompt_coverage(None, "p", None, 0, -1, order)
    assert found["coverage"] == pytest.approx(0.5) == found["raw"] and not found["overshoot"]


def test_the_page_says_overshoot_in_words(tmp_path, record):
    from openinterpretability.builder.compiler import compile_site
    from openinterpretability.builder.explanation_pages import load_views
    from tests.test_explanation_records import _record, _write

    over = _record(datetime(2026, 11, 17, tzinfo=UTC)).model_copy(
        update={"prompt_coverage": 1.0, "raw_coverage": 5.36}
    )
    assert over.overshoot and not over.model_copy(update={"raw_coverage": 0.9}).overshoot
    _write(tmp_path / "wiki", over)
    (prompt,) = load_views(tmp_path / "wiki", states={})
    compile_site([record], tmp_path / "site", explanations=[prompt])
    page = (tmp_path / "site" / prompt.newest.href).read_text()
    assert "Overshoot: raw 536%, capped at 100%" in page
    assert "overshoot: raw 536%" in (tmp_path / "site" / prompt.href).read_text()

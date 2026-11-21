"""Each model's coverage view for its pages (openinterpretability/builder/coverage.py)."""

from openinterpretability.builder.coverage import coverage_view
from tests.test_coverage_view import _row


def test_a_model_with_no_rows_gets_no_coverage_view():
    view = coverage_view({"empty/model": [], "m": [_row("m", "2026-01-01", 0)]})
    assert list(view) == ["m"]
    assert [layer["layer"] for layer in view["m"]["layers"]] == [0]

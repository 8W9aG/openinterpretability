"""Chart helpers of the site (openinterpretability/builder/charts.py)."""

from openinterpretability.builder.charts import _nice_step


def test_an_empty_or_negative_span_gets_a_step_of_one():
    # Both callers pass a span of at least 1, so only a direct call reaches the guard.
    assert _nice_step(0) == 1.0 and _nice_step(-3.0) == 1.0
    assert _nice_step(7) == 2 and _nice_step(0.3) == 0.1

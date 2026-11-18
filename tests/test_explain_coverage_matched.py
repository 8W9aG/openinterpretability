"""Prompt coverage against a matched baseline, by kind, and its minimal set (#4777)."""

import random
from collections import Counter
from datetime import UTC, datetime

from openinterpretability import explain_coverage as ec
from openinterpretability.explain_groups import HEAD


def _units():
    """Two layers, each with 4 heads and 20 neuron poles."""
    out = []
    for layer in (0, 1):
        out += [(layer, h, HEAD) for h in range(4)]
        out += [(layer, n, "positive") for n in range(20)]
    return out


def test_a_matched_draw_keeps_each_layers_heads_and_poles():
    keep = [(0, 0, HEAD), (0, 1, HEAD), (0, 3, "positive"), (1, 2, HEAD)]
    for seed in range(5):
        drawn = ec.matched_draw(random.Random(seed), _units(), keep)
        kinds = Counter((u[0], u[2] == HEAD) for u in drawn)
        assert kinds == {(0, True): 2, (0, False): 1, (1, True): 1}
        assert len(set(drawn)) == len(drawn)


def test_the_minimal_set_keeps_ninety_percent_with_fewer_units():
    weights = {"a": 0.5, "b": 0.4, "c": 0.05, "d": 0.05}  # a + b is exactly 0.9 in floats

    def share(units):
        return min(1.0, sum(weights[u] for u in units))

    keep = list(weights)
    found = ec.minimal_set(share, keep, share(keep))
    assert share(found) >= 0.9 * share(keep) and len(found) < len(keep)
    assert set(found) == {"a", "b"}  # the two small units go; dropping b would fall below


def test_a_unit_that_carries_everything_is_the_minimal_set():
    found = ec.minimal_set(lambda units: 1.0 if "key" in units else 0.0, ["x", "key", "y"], 1.0)
    assert found == ["key"]


def test_the_page_cautions_and_marks_an_unmatched_baseline(tmp_path, record):
    from openinterpretability.builder.compiler import compile_site
    from openinterpretability.builder.explanation_pages import load_views
    from openinterpretability.core.explanation_records import CoverageDetail
    from tests.test_explanation_records import _record, _write

    old = _record(datetime(2026, 11, 17, tzinfo=UTC)).model_copy(
        update={"prompt_coverage": 1.0, "random_coverage": 0.0006, "explained_share": 0.099}
    )
    assert old.coverage_caution  # a low explained share, whatever the coverage
    detail = CoverageDetail(
        heads_only=0.97, poles_only=0.02, heads_kept={0: 6, 1: 5}, n_heads=8,
        minimal_units=["L0/H2", "L1/H4"], minimal_coverage=0.93, random_spread=0.05,
    )  # fmt: skip
    new = old.model_copy(
        update={"written": datetime(2026, 11, 18, tzinfo=UTC), "random_coverage": 0.95,
                "explained_share": 0.6, "coverage_detail": detail}
    )  # fmt: skip
    assert new.coverage_caution  # coverage within 0.2 of the matched baseline
    assert not new.model_copy(update={"random_coverage": 0.3}).coverage_caution
    for r in (old, new):
        _write(tmp_path / "wiki", r)
    (prompt,) = load_views(tmp_path / "wiki", states={})
    compile_site([record], tmp_path / "site", explanations=[prompt])
    newest = (tmp_path / "site" / prompt.newest.href).read_text()
    assert "Coverage here does not show that the answer is understood." in newest
    assert "matched by kind and layer" in newest and "L0/H2, L1/H4" in newest
    assert "6 of layer 0" in newest and "8 heads" in newest
    oldest = next(v for v in prompt.explanations if v.record.coverage_detail is None)
    assert "(not matched)" in (tmp_path / "site" / oldest.href).read_text()

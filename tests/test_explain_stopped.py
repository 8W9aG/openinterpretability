"""The stopped units of an explained prompt, ranked by coverage gain (explain_stopped.py)."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from openinterpretability import explain as ex
from openinterpretability.core.draws import UnauditableDraw, write_draw
from openinterpretability.core.identifiers import NeuronIdentifier
from openinterpretability.explain_coverage import _log_prob, _share, keep_hooks
from openinterpretability.explain_stopped import outcomes, rank
from tests.test_core_draws import PROVENANCE
from tests.test_explain import PROMPT, means  # noqa: F401  (a fixture)


def _stop(wiki, neuron, outcome, day):
    ident = NeuronIdentifier(model_name="toy", layer=0, neuron_idx=neuron, pole="positive")
    draw = UnauditableDraw(identifier=ident, outcome=outcome, provenance=PROVENANCE,
                           timestamp=datetime(2026, 11, day, tzinfo=UTC))  # fmt: skip
    write_draw(draw, wiki)
    return ident.slug


def test_the_newest_scan_outcome_of_each_unit_counts(tmp_path):
    slug = _stop(tmp_path, 3, "dead", 1)
    _stop(tmp_path, 3, "default_on", 2)
    assert outcomes(tmp_path) == {slug: "default_on"}


def test_blocked_units_are_ranked_by_the_coverage_they_add(toy_runner, means, tmp_path):  # noqa: F811
    _stop(tmp_path, 3, "default_on", 1)
    result = ex.explain(toy_runner, "toy", PROMPT, means, {}, top=10)
    units = [u for u, *_ in result.order if u != (0, 3, "positive")][:1] + [(0, 3, "positive")]
    labels = [f"L{la}/N{i}{'+' if p == 'positive' else '-'}" for la, i, p in units]
    planned = SimpleNamespace(blocked_units=[{"target": t, "share": 0.5} for t in labels])
    rows = rank(toy_runner, PROMPT, means, result.token_id, -1, result.order, planned, tmp_path,
                "toy")  # fmt: skip
    assert {r["target"] for r in rows} == set(labels)
    clean = _log_prob(toy_runner, PROMPT, result.token_id, -1)
    ablated = _log_prob(toy_runner, PROMPT, result.token_id, -1, keep_hooks(toy_runner, means, []))
    for r in rows:  # nothing is explained yet, so the gain is the unit's own coverage
        kept = _log_prob(toy_runner, PROMPT, result.token_id, -1,
                         keep_hooks(toy_runner, means, [r["unit"]]))  # fmt: skip
        assert r["gain"] == pytest.approx(_share(clean, ablated, kept), abs=1e-5)
        stopped = r["unit"] == (0, 3, "positive")
        assert r["reason"] == ("default_on" if stopped else "no follow-up")
    assert rows[0]["gain"] >= rows[1]["gain"]
    assert [r["reason"] for r in rows].count("default_on") == 1

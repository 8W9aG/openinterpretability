"""Sister draws in the planner (openinterpretability/planner_sisters.py)."""

from datetime import UTC, datetime

from openinterpretability.builder.wiki import write_experiment_page
from openinterpretability.core.schemas import NeuronIdentifier
from openinterpretability.planner import plan_next
from tests.test_sisters import _sister_registry

NOW = datetime(2026, 10, 20, 12, tzinfo=UTC)


def _sister_slugs(wiki):
    plan = plan_next(wiki, seed=1, epsilon=0.0, now=NOW)
    return {c.slug: c for c in plan.candidates if c.kind == "sister"}


def test_a_match_already_studied_is_not_drawn_again(tmp_path, record):
    wiki = _sister_registry(tmp_path, record)
    assert "o--m/L2/N9-negative" in _sister_slugs(wiki)
    studied = NeuronIdentifier(model_name="o/m", layer=2, neuron_idx=9, pole="negative")
    sub = record.submission.model_copy(update={"identifier": studied})
    write_experiment_page(record.model_copy(update={"submission": sub}), wiki)
    drawn = _sister_slugs(wiki)
    assert "o--m/L2/N9-negative" not in drawn and "o--m/L4/N1-positive" in drawn


def test_a_model_with_no_window_is_scanned_without_one(tmp_path, record):
    wiki = _sister_registry(tmp_path, record)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.openinterpretability]\nmodels = ["Qwen/Qwen2.5-0.5B", "o/m"]\n'
    )
    sister = _sister_slugs(wiki)["o--m/L2/N9-negative"]
    assert sister.audit.startswith("--model o/m --layer 2 --neuron 9 --pole negative --sister-of ")
    assert "--window-tokens" not in sister.audit

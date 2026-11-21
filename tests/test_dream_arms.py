"""The three dreaming arms and the review arm (scripts/dream_arms.py, #3779)."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("dream_arms", _SCRIPTS / "dream_arms.py")
arms = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(arms)


def _result(kind, prompt, activation):
    return SimpleNamespace(kind=kind, prompt=prompt, activation=activation)


def test_holdout_starts_are_the_strongest_firing_holdouts():
    record = SimpleNamespace(verdict=SimpleNamespace(adversarial_test_results=[
        _result("positive_holdout", "weak", 0.5),
        _result("positive_holdout", "silent", 0.1),
        _result("hard_negative", "negative", 9.0),
        _result("positive_holdout", "strong", 2.0),
        _result("positive_holdout", "middle", 1.0),
        _result("positive_holdout", "unmeasured", None),
    ]))  # fmt: skip
    assert arms.holdout_starts(record, 0.4) == ["strong", "middle"]  # at most STARTS
    assert arms.holdout_starts(record, 5.0) == []


def test_rates_count_what_matters_for_each_arm():
    items = [
        {"kind": "max", "judged_active": False},  # a miss
        {"kind": "max", "judged_active": True},
        {"kind": "min", "judged_active": True},  # a false positive
        {"kind": "hit", "judged_active": True},
        {"kind": "hit", "judged_active": False},
        {"kind": "control", "judged_active": False},
    ]
    r = arms.rates(items)
    assert (r["max"]["k"], r["max"]["n"]) == (1, 2)
    assert (r["min"]["k"], r["min"]["n"]) == (1, 1)
    assert r["hit"]["share"] == pytest.approx(0.5)
    assert r["control"]["share"] == 0.0
    assert arms.rates([])["max"] == {"n": 0, "k": 0, "share": None, "ci95": [0.0, 1.0]}


def test_judged_reads_the_answer_by_item_number(tmp_path):
    key = {"items": [{"kind": "max", "text": "a"}, {"kind": "control", "text": "b"}]}
    (tmp_path / "p.key.json").write_text(json.dumps(key))
    (tmp_path / "p.json").write_text(json.dumps({"active": [2]}))
    items = arms.judged(tmp_path / "p.key.json")
    assert [i["judged_active"] for i in items] == [False, True]


def test_report_writes_models_reviews_and_poles(tmp_path):
    key = {"slug": "m/L0/N1-positive", "model": "m", "threshold": 0.5,
           "items": [{"kind": "max", "text": "a"}, {"kind": "min", "text": "b"}]}  # fmt: skip
    (tmp_path / "p.key.json").write_text(json.dumps(key))
    (tmp_path / "p.json").write_text(json.dumps({"active": [2]}))
    (tmp_path / "q.key.json").write_text(json.dumps(key | {"slug": "unanswered"}))
    review = {"slug": "m/L0/N1-positive", "dreams_added": 1,
              "before": {"outcome": "VERIFIED", "balanced_accuracy": 0.9},
              "after": {"outcome": "NEEDS_REVISION", "balanced_accuracy": 0.8}}  # fmt: skip
    (tmp_path / "p.review.json").write_text(json.dumps(review))
    from click.testing import CliRunner

    out = tmp_path / "rows.jsonl"
    result = CliRunner().invoke(arms.cli, ["report", "--dir", str(tmp_path), "--out", str(out)])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    model, moves, pole = rows
    assert model["poles"] == 1 and model["rates"]["max"]["k"] == 1  # the unanswered key skipped
    assert (moves["verified_before"], moves["verified_after"]) == (1, 0)
    assert pole["rates"]["min"]["k"] == 1

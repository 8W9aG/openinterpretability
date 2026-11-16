"""``scripts/score_by_range.py``: explanation scores by bin of the activation range (#3238)."""

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "score_by_range", Path(__file__).parents[1] / "scripts" / "score_by_range.py"
)
sbr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sbr)


def _record(tmp_path, name, detection, edges=(1.0, 2.0, 3.0, 4.0), peak=10.0, judge="external/x"):
    path = tmp_path / "experiments" / "m" / "L0" / "N1-positive" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    scores = {"slug": "m/L0/N1-positive", "quantiles": 5, "quantile_edges": list(edges),
              "threshold": 2.5, "judge_model": judge,
              "detection": {"recall_by_quantile": detection}, "fuzzing": None}  # fmt: skip
    path.write_text(json.dumps({"submission": {"max_activation": peak},
                                "verdict": {"explanation_scores": scores}}))  # fmt: skip


def test_rows_give_edges_and_line_as_a_share_of_the_maximum(tmp_path):
    _record(tmp_path, "a", [0.0, 0.2, 0.4, 0.6, 0.8])
    (row,) = sbr.score_rows(tmp_path)
    assert row["edges"] == [0.1, 0.2, 0.3, 0.4] and row["line"] == 0.25
    assert row["detection"] == [0.0, 0.2, 0.4, 0.6, 0.8]


def test_records_without_five_bins_are_left_out(tmp_path):
    _record(tmp_path, "a", [0.2, 0.4])
    _record(tmp_path, "b", [0.2] * 5, peak=0.0)
    assert sbr.score_rows(tmp_path) == []


def test_the_top_only_gap_is_the_top_bin_minus_the_mean(tmp_path):
    _record(tmp_path, "a", [0.0, 0.2, 0.4, 0.6, 0.8])  # mean 0.4, top 0.8
    _record(tmp_path, "b", [0.4] * 5)  # no gap
    _record(tmp_path, "c", [None, 0.2, 0.2, 0.2, 1.0])  # an empty bin: left out of the gap
    summary = {r["table"]: r for r in sbr.summary_rows(sbr.score_rows(tmp_path))}
    top = summary["top_only"]
    assert top["records"] == 2 and top["gap"] == 0.2 and top["top_bin"] == 0.6
    recall = next(r for r in sbr.summary_rows(sbr.score_rows(tmp_path))
                  if r["table"] == "recall" and r["kind"] == "detection")  # fmt: skip
    assert recall["windows_per_bin"] == [2, 3, 3, 3, 3]
    assert recall["mean"][0] == 0.2  # the empty bin counts in no mean


def test_judge_families_drop_provider_and_context():
    assert sbr.judge_family("external/claude-haiku-5-5") == "haiku"
    assert sbr.judge_family("external/deepseek-flash[1m]") == "deepseek"

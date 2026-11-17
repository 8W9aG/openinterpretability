"""``scripts/rescore_pilot.py``: the single-judge rescore pilot (#4102)."""

import importlib.util
import sys
from pathlib import Path

import pytest

_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(_DIR))
_SPEC = importlib.util.spec_from_file_location("rescore_pilot", _DIR / "rescore_pilot.py")
rp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rp)


def _rows(model: str, status: str, n: int) -> list[dict]:
    return [{"model": model, "status": status, "slug": f"{model}{status}{i}"} for i in range(n)]


def test_quotas_split_the_total_in_proportion_and_sum_to_it():
    rows = _rows("a", "VERIFIED", 6) + _rows("b", "VERIFIED", 3) + _rows("c", "VERIFIED", 1)
    q = rp.quotas(rows, 7)
    assert sum(q.values()) == 7
    assert q["a"] >= q["b"] >= q["c"]


def test_the_draw_takes_each_arm_apart_and_is_repeatable():
    rows = (
        _rows("a", "VERIFIED", 50) + _rows("a", "REJECTED", 30) + _rows("b", "NEEDS_REVISION", 30)
    )
    first = rp.draw(rows, per_arm=10)
    assert sum(r["status"] == "VERIFIED" for r in first) == 10
    assert sum(r["status"] != "VERIFIED" for r in first) == 10
    assert first == rp.draw(rows, per_arm=10)
    assert len({r["slug"] for r in first}) == 20


def test_spearman_brown_raises_reliability_with_more_seeds():
    assert rp.spearman_brown(0.5, 1) == pytest.approx(0.5)
    assert rp.spearman_brown(0.5, 2) == pytest.approx(2 / 3)
    assert rp.spearman_brown(0.5, 4) > rp.spearman_brown(0.5, 2)


def test_separation_of_perfectly_split_scores():
    pairs = [(0.9, True)] * 10 + [(0.5, False)] * 10
    out = rp.separation(pairs)
    assert out["auroc"] == 1.0 and out["difference"] == pytest.approx(0.4)
    assert out["n_verified"] == 10 and out["n_other"] == 10


def test_seed_reliability_and_old_scores_in_the_kind_report():
    sample, rows = {}, []
    for i in range(12):
        status = "VERIFIED" if i % 2 else "REJECTED"
        record = f"r{i}"
        sample[record] = {"status": status, "old": {"detection": 0.5 + i / 100}}
        rows += [{"record": record, "seed": k, "detection": 0.5 + i / 40 + k / 100}
                 for k in range(rp.SEEDS)]  # fmt: skip
    out = rp.kind_report(rows, sample, "detection")
    assert out["records"] == 12
    assert out["seed_reliability"] == pytest.approx(1.0)
    assert out["old_vs_new"] == pytest.approx(1.0)
    assert out["old_same_records"]["n_verified"] == 6


def test_a_record_on_a_bad_corpus_is_swapped_for_one_of_its_model_and_arm():
    def row(record: str, status: str, sha: str) -> dict:
        return {"record": record, "model": "a", "status": status, "corpus_sha256": sha}

    rows = [row(f"v{i}", "VERIFIED", "bad" if i < 2 else "ok") for i in range(6)]
    rows.append(row("o", "REJECTED", "ok"))
    chosen = [rows[0], rows[2], rows[6]]
    out = rp.replace(chosen, rows, {"bad"})
    assert [r["record"] for r in out][1:] == ["v2", "o"]
    assert out[0]["status"] == "VERIFIED" and out[0]["corpus_sha256"] == "ok"
    assert out[0]["record"] not in {"v0", "v2"}
    assert out == rp.replace(chosen, rows, {"bad"})

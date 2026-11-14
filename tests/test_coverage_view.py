"""Registry coverage on the site (openinterp/builder/coverage.py)."""

import json

import pytest

from openinterp.builder.coverage import load_rows, model_summary


def _row(model, date, layer, clean=2.0, ablated=3.0, kept=2.5, rand=0.1, poles=4, width=8):
    return {"model": model, "date": date, "layer": layer, "width": width, "verified_poles": poles,
            "loss_clean": clean, "loss_ablated": ablated, "loss_kept_verified": kept,
            "recovered_random": rand}  # fmt: skip


def test_load_rows_keeps_each_models_newest_run(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [_row("a", "2026-11-01", 1), _row("a", "2026-11-13", 1), _row("a", "2026-11-13", 0)]
    rows.append(_row("b", "2026-10-01", 0))
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = load_rows(path)
    assert [r["layer"] for r in out["a"]] == [0, 1] and {r["date"] for r in out["a"]} == {
        "2026-11-13"
    }
    assert len(out["b"]) == 1
    assert load_rows(tmp_path / "missing.jsonl") == {}


def test_model_summary_weights_layers_by_their_loss_rise():
    rows = [
        _row("a", "d", 0, ablated=3.0, kept=2.5),
        _row("a", "d", 1, ablated=2.2, kept=2.2, rand=0.5),
    ]
    s = model_summary(rows)
    # Rises 1.0 and 0.2: 0.5 + 0 kept of 1.2. Random: 0.1·1.0 + 0.5·0.2 of 1.2.
    assert s["recovered"] == pytest.approx(0.5 / 1.2)
    assert s["recovered_random"] == pytest.approx(0.2 / 1.2)
    assert (s["verified_poles"], s["poles"], s["layers"]) == (8, 32, 2)


def test_home_page_shows_coverage(tmp_path, record):
    from openinterp.builder.compiler import compile_site

    rows = {"toy": [_row("toy", "2026-11-13", 0)]}
    compile_site([record], tmp_path, coverage_rows=rows)
    home = (tmp_path / "index.html").read_text()
    assert "How much of each model is explained" in home and "50.00%" in home

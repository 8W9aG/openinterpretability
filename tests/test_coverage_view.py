"""Registry coverage on the site (openinterpretability/builder/coverage.py)."""

import json

import pytest

from openinterpretability.builder.coverage import load_rows, model_summary


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


def test_model_pages_show_coverage_and_the_home_page_does_not(tmp_path, record):
    # The coverage moved from the home page to the Models page and each model page (#3837).
    from openinterpretability.builder.compiler import compile_site

    model = record.submission.identifier.model_name
    row = _row(model, "2026-11-13", 0) | {"recovered": 0.5, "corpus": "toy-corpus"}
    compile_site([record], tmp_path, coverage_rows={model: [row]})
    home = (tmp_path / "index.html").read_text()
    assert "How much of each model is explained" not in home and 'id="coverage"' not in home
    models = (tmp_path / "models" / "index.html").read_text()
    assert "Loss effect kept" in models and "50.00%" in models and "random 10.00%" in models
    slug = record.submission.identifier.model_slug
    page = (tmp_path / "models" / slug / "index.html").read_text()
    assert "How much of this model is explained" in page and "toy-corpus" in page
    assert "4 of 16" in page and "1.000" in page  # L0: VERIFIED poles of 2×width, and the rise


def test_a_model_with_no_coverage_run_shows_a_dash(tmp_path, record):
    from openinterpretability.builder.compiler import compile_site

    compile_site([record], tmp_path)
    models = (tmp_path / "models" / "index.html").read_text()
    row = models[models.index("<tbody") :]
    assert "Loss effect kept" in models and "—</td>" in row
    slug = record.submission.identifier.model_slug
    assert "How much of this model" not in (tmp_path / "models" / slug / "index.html").read_text()

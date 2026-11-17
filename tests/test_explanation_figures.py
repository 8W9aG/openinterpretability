"""An explanation page's figures, from the record's measured data (#4479)."""

import xml.etree.ElementTree as ET
from types import SimpleNamespace

from openinterpretability import explain as ex
from openinterpretability.builder import explanation_figures as ef
from openinterpretability.core.explanation_records import ExplainRun, ExplanationFigures, FigureUnit
from openinterpretability.explain_figure_data import READ, TOP_TOKENS, figure_data
from tests.test_explain import PROMPT, means  # noqa: F401  (a fixture)


def _data(groups=()):
    units = [FigureUnit(label="L3/N7+", layer=3, kind="pole", attribution=2.0, status="VERIFIED",
                        explained=True),
             FigureUnit(label="L5/H1", layer=5, kind="head", attribution=-1.0, status="unstudied",
                        explained=False, blocked=True)]  # fmt: skip
    return ExplanationFigures(
        tokens=["a", " b"], logit=10.0, units=units, net=1.5,
        edges=[{"source": "L3/N7+", "target": "L5/H1", "weight": -0.4}],
        coverage=0.3, random_draws=[0.01, -0.02], reading={"L3/N7+": [0.0, 2.0]},
        ablation=[{"token": " x", "before": 0.5, "after": 0.1}], groups=list(groups),
    )  # fmt: skip


def test_the_record_holds_what_the_figures_need(toy_runner, means):  # noqa: F811
    result = ex.explain(toy_runner, "toy", PROMPT, means, {}, top=10, edges=3)
    planned = SimpleNamespace(blocked_units=[])
    coverage = {"coverage": 0.2, "random_draws": [0.1, 0.0]}
    data = figure_data(toy_runner, means, result, coverage, planned)
    assert data.tokens and len(data.units) == len(result.poles) and data.net == round(result.net, 4)
    assert len(data.reading) == min(READ, len(result.poles))
    assert all(len(v) == len(data.tokens) for v in data.reading.values())
    assert len(data.ablation) == TOP_TOKENS and data.random_draws == [0.1, 0.0]


def test_each_figure_draws_and_the_moving_ones_have_a_still():
    drawn = ef.figures(_data(), "x")
    assert [d.name for d in drawn] == ["graph", "waterfall", "coverage", "reading", "ablation"]
    for d in drawn:
        ET.fromstring(d.svg)  # well-formed
        assert d.alt and d.table and d.table[0]
    moving = {d.name: d for d in drawn if d.static}
    assert set(moving) == {"graph", "coverage"}
    for d in moving.values():
        assert "@keyframes" in d.svg and ef.STATIC in d.static and "prefers-reduced-motion" in d.svg
    assert "dashed" in drawn[0].svg  # a negative edge has its own line style
    assert "fhatch" in drawn[1].svg and "blocked" in drawn[1].svg  # pattern and a direct label


def test_groups_and_a_run_add_to_the_figures():
    data = _data([{"name": "C4", "kind": "circuit", "status": "VERIFIED", "members": ["L3/N7+"]}])
    run = ExplainRun(share_before=0.1, share_after=0.3, stopped="x", shares=[0.1, 0.2, 0.3])
    drawn = ef.with_run(data, run, "x")
    assert drawn[-1].name == "groups" and "C4" in drawn[-1].svg
    assert "The explained share before each draw" in drawn[2].svg and "step2" in drawn[2].svg


def test_the_page_shows_the_figures_with_a_pause_and_a_still(tmp_path, monkeypatch):
    from openinterpretability.builder import explanation_pages
    from tests.test_explanation_records import _record

    record = _record(
        __import__("datetime").datetime(2026, 11, 17, tzinfo=__import__("datetime").UTC)
    )
    source = f"wiki/explanations/m/{record.prompt_id}/x.json"
    view = explanation_pages.ExplanationView(record.model_copy(update={"figures": _data()}), source)
    rendered = {}
    render = SimpleNamespace(output_dir=tmp_path, __call__=None)
    drawn = explanation_pages.write_figures(render, view)
    folder = tmp_path / view.href.removesuffix(".html")
    assert (folder / "fig-graph.svg").exists() and (folder / "fig-graph-static.svg").exists()
    assert len(drawn) == 5 and not (folder / "fig-reading-static.svg").exists()
    bare = view.__class__(record, view.source)
    assert explanation_pages.write_figures(render, bare) == [] and not rendered


def test_the_built_page_has_the_figures_section(tmp_path, record):
    import datetime

    from openinterpretability.builder.compiler import compile_site
    from openinterpretability.builder.explanation_pages import ExplanationView
    from tests.test_explanation_records import _record

    rec = _record(datetime.datetime(2026, 11, 17, tzinfo=datetime.UTC))
    view = ExplanationView(rec.model_copy(update={"figures": _data()}),
                           f"wiki/explanations/m/{rec.prompt_id}/x.json")  # fmt: skip
    compile_site([record], tmp_path / "site", explanations=[view])
    page = (tmp_path / "site" / view.href).read_text()
    assert 'id="figures"' in page and 'media="(prefers-reduced-motion: reduce)"' in page
    assert "data-pause" in page and f"{rec.prompt_id}/fig-graph-static.svg" in page
    plain = ExplanationView(rec, f"wiki/explanations/m/{rec.prompt_id}/x.json")
    compile_site([record], tmp_path / "bare", explanations=[plain])
    assert 'id="figures"' not in (tmp_path / "bare" / plain.href).read_text()

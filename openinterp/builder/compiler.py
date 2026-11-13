"""Compile every archived experiment into a static site under ``dist/``.

Pages follow the registry hierarchy: the index, then a page per model, layer and unit.
A neuron's page is ``models/<model>/L<layer>/N<idx>.html``, and a head's is
``H<idx>.html``. Each unit page lists all of its experiments, most conclusive first. Null
results are published alongside positive ones.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from pydantic import ValidationError

from openinterp import __version__
from openinterp.builder import charts, pieces, statuses
from openinterp.builder.agents import (
    agent_href,
    agents_view,
    contributors_view,
    legacy_redirects,
)
from openinterp.builder.aligned import aligned_latents
from openinterp.builder.circuits import HUB_RULE, circuits_and_hubs, unit_anchors, wiring_view
from openinterp.builder.corpora import corpora_view, language_mix
from openinterp.builder.dashboard import dashboard
from openinterp.builder.heat import HeatToken, heat_html, token_heat, values_rows
from openinterp.builder.model_facts import ModelFacts, render_models
from openinterp.builder.neuronpedia_export import export_neuronpedia
from openinterp.builder.parallel import in_background, render_in_parallel
from openinterp.builder.parquet_export import export_draws, export_registry
from openinterp.builder.pii import save_spans, scrub, scrub_record
from openinterp.builder.registry_stats import registry_stats  # noqa: F401  (re-exported)
from openinterp.builder.sections import render_reference_pages
from openinterp.builder.significance import P_USEFUL, evidence_strength
from openinterp.builder.sister_graph import sister_graphs
from openinterp.builder.standing import ROLES, contested_units, reviewer_agreement
from openinterp.builder.stopped import add_stopped
from openinterp.builder.stops import pageless_stops, stops_by_unit
from openinterp.builder.tree import (
    Experiment,
    build_tree,
    circuits_href,
    explore_focus,
    false_discovery,
    neuron_href,
)
from openinterp.core.agent_names import canonical_agent
from openinterp.core.draws import ChosenStop
from openinterp.core.errata import Erratum
from openinterp.core.exceptions import RegistryError
from openinterp.core.project import project_links
from openinterp.core.schemas import ActivatingExample, AuditRecord
from openinterp.planner import configured_licenses
from openinterp.sensitive import is_offensive
from openinterp.sisters import by_match_slug, sister_source_href, sister_tests, sister_view
from openinterp.wiring_backfill import record_wiring


@dataclass
class BuildReport:
    """What a build wrote."""

    output_dir: Path
    records_loaded: int
    experiments_published: int
    pages: list[Path] = field(default_factory=list)
    parquet: Path | None = None


def load_records(audits_dir: Path) -> list[AuditRecord]:
    """Load every ``*.json`` audit record under ``audits_dir`` (recursively)."""
    records = []
    for path in sorted(Path(audits_dir).rglob("*.json")):
        try:
            records.append(AuditRecord.model_validate_json(path.read_text()))
        except ValidationError as exc:
            raise RegistryError(f"{path}: invalid audit record\n{exc}") from exc
    return records


def _environment() -> Environment:
    env = Environment(
        loader=PackageLoader("openinterp.builder", "templates"),
        autoescape=select_autoescape(["html"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pct"] = lambda v: "—" if v is None else f"{100 * v:.0f}%"
    env.filters["pct_fine"] = _pct_fine
    env.filters["num"] = lambda v, d=3: "—" if v is None else f"{v:.{d}f}"
    env.filters["thousands"] = lambda v: "?" if v is None else f"{v:,}"
    # p-values below 0.001 would round to 0.000, so show them in scientific notation.
    env.filters["pval"] = lambda v: "—" if v is None else f"{v:.3f}" if v >= 0.001 else f"{v:.1e}"
    env.globals["roles"] = ROLES
    env.globals["charts"] = charts  # page-level charts: the layer map
    env.globals["pieces"] = pieces  # the pieces bars of a model and of a layer
    env.globals["statuses"] = statuses  # unit statuses, their order and colours
    env.globals["agent_href"] = agent_href  # links from agent names to their pages
    env.globals["agent_name"] = canonical_agent  # one name per model, however it was recorded
    env.globals["evidence_strength"] = evidence_strength  # strong / moderate / weak bands
    env.globals["language_mix"] = language_mix  # a scan's tokens per language
    env.globals["heat_html"] = heat_html  # token heatmaps, built in Python for speed
    env.globals["values_rows"] = values_rows  # and the rows of their values tables
    return env


def _pct_fine(v: float | None) -> str:
    """A percentage with two significant figures, so 1 neuron in 100k isn't shown as 0%."""
    if v is None:
        return "—"
    if v == 0:
        return "0%"
    decimals = max(0, 1 - math.floor(math.log10(100 * v)))
    return f"{100 * v:.{decimals}f}%"


def _root_for(rel_path: str) -> str:
    return "../" * (len(Path(rel_path).parts) - 1)


# Detection outcomes, the explanation's mistakes first.
OUTCOMES = {
    (True, False): "missed",
    (False, True): "false alarm",
    (True, True): "hit",
    (False, False): "correct rejection",
}


def explanation_view(scores, scale: float) -> dict | None:
    """Held-out windows grouped by how the judge did, ready for the token heatmap."""
    if scores is None:
        return None
    scale = scale or 1.0

    def window(w) -> dict:
        peak = max(range(len(w.activations)), key=w.activations.__getitem__, default=-1)
        marks = set(w.highlighted)
        tokens = [
            HeatToken(
                tok.replace("\n", "↵"),
                round(min(1.0, max(0.0, act / scale)), 3),
                act or 0.0,
                i == peak and act >= scores.threshold,
                i in marks,
            )
            for i, (tok, act) in enumerate(zip(w.tokens, w.activations, strict=True))
        ]
        outcome = OUTCOMES[(w.label, w.judged)] if w.judged is not None else "not judged"
        return {
            "w": w,
            "tokens": tokens,
            "outcome": outcome,
            "peak": max(w.activations, default=0),
            "offensive": is_offensive(w.text),
        }

    order = list(OUTCOMES.values())
    detection = sorted(
        (window(w) for w in scores.detection_windows),
        key=lambda r: (order.index(r["outcome"]) if r["outcome"] in order else 9, -r["peak"]),
    )
    fuzzing = sorted(
        (window(w) for w in scores.fuzzing_windows),
        key=lambda r: (r["w"].label == r["w"].judged, -r["peak"]),
    )
    counts = {o: sum(r["outcome"] == o for r in detection) for o in order}
    return {"s": scores, "detection": detection, "fuzzing": fuzzing, "counts": counts}


# Paths the builder (and the Pagefind step after it) owns inside the output dir.
# They are removed before each build so deleted records leave no stale pages.
OWNED_PATHS = (
    "index.html",
    "about.html",
    "models",
    "agents",
    "neurons",
    "research",
    "data",
    "pagefind",
)


def _clean_owned_paths(output_dir: Path) -> None:
    for name in OWNED_PATHS:
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _experiment_view(
    e: Experiment, studied: set[str], rows: dict | None = None, circuit: str | None = None
) -> dict:
    """Render a unit's page.

    ``rows`` is backfilled wiring (see :mod:`openinterp.wiring_backfill`). ``circuit`` is
    the link to the unit's circuit, if any.
    """
    sub, verdict = e.record.submission, e.record.verdict
    weights, activations, backfilled = record_wiring(e.record, rows or {})
    ident = sub.identifier
    correlated = []
    for c in sub.correlated_neurons:  # neuron poles only
        href = neuron_href(ident.model_slug, ident.layer, c.neuron_idx)
        correlated.append({"c": c, "href": href if href in studied else None})
    return {
        "e": e,
        "sub": sub,
        "verdict": verdict,
        "ident": ident,
        "c": verdict.causal_metrics,
        "bars": charts.histogram_bars(sub.activation_histogram),
        "probes": charts.probe_chart(verdict, sub.max_activation),
        "causal": charts.causal_chart(verdict.causal_metrics, sub.counterfactuals),
        "vocab": charts.vocab_chart(sub.vocabulary_lens),
        "examples": [token_heat(ex, sub.max_activation or 0.0) for ex in sub.top_examples[:12]],
        "offensive_examples": [i for i, x in enumerate(sub.top_examples) if is_offensive(x.text)],
        "correlated": correlated,
        "scores": explanation_view(verdict.explanation_scores, sub.max_activation or 0.0),
        "wiring": [
            view
            for w in (weights, activations)
            if (view := wiring_view(w, ident.model_slug, studied)) is not None
        ],
        "wiring_backfilled": backfilled,
        "circuit": circuit,
        "sister_of": sub.sister_of
        and {"ref": sub.sister_of, "href": sister_source_href(sub.sister_of)},
    }


def pole_scans(neuron, n: int = 3) -> dict | None:
    """Both poles' top examples for a neuron page, side by side.

    The newest experiment that recorded its other pole supplies both, so they come from
    one scan. Without one (records from before both poles were scanned), each pole's
    newest experiment supplies its own, and a pole with none has no column.
    """
    if not neuron.is_neuron:
        return None
    newest = sorted(neuron.experiments, key=lambda e: e.record.verdict.timestamp, reverse=True)
    both = next((e for e in newest if e.record.submission.opposite_pole), None)
    found: dict[str, tuple[list[ActivatingExample], Experiment]] = {}
    if both is not None:
        sub = both.record.submission
        found[sub.identifier.pole] = (sub.top_examples, both)
        found[sub.opposite_pole.pole] = (sub.opposite_pole.top_examples, both)
    else:
        for e in newest:
            found.setdefault(e.side, (e.record.submission.top_examples, e))
    poles = []
    for pole in neuron.sides:
        examples, e = found.get(pole, ([], None))
        scale = max((ex.max_activation for ex in examples), default=0.0)
        poles.append(
            {
                "pole": pole,
                "max": scale,
                "examples": [token_heat(ex, scale, before=20, after=5) for ex in examples[:n]],
                "source": e,
            }
        )
    if not any(p["examples"] for p in poles):
        return None
    return {"poles": poles, "shared": both}


def compile_site(
    records: Sequence[AuditRecord],
    output_dir: Path,
    *,
    site_title: str = "Open Interpretability",
    wiring_rows: dict | None = None,
    draws: Sequence = (),
    corpus_sources: dict | None = None,
    alignment_rows: dict | None = None,
    sister_rows: dict | None = None,
    papers_dir: Path | None = None,
    stops: Sequence[ChosenStop] = (),
    model_facts: dict[str, ModelFacts] | None = None,
    jobs: int = 1,
    errata: Sequence[Erratum] = (),
) -> BuildReport:
    """Render every reviewed experiment in ``records``, normally the wiki archive.

    Rows fill in what old records lack. ``wiring_rows`` and ``corpus_sources`` come from
    :func:`openinterp.wiring_backfill.load_rows` and :func:`openinterp.corpus.load_sources`.
    ``alignment_rows`` (:func:`openinterp.crossref.load_rows`) are each pole's aligned
    latents. ``sister_rows`` (:func:`openinterp.sisters.load_rows`) are its matches.
    ``papers_dir`` holds the Research section (:mod:`openinterp.builder.research`).
    ``stops`` (:func:`openinterp.core.draws.load_stops`) go on their unit's page, or the home's.
    ``model_facts`` (:func:`openinterp.builder.model_facts.load_model_facts`) fill the
    Models page, which lists every model in ``[tool.openinterp] models``.
    ``jobs`` is the number of worker processes that render the unit pages.
    ``errata`` (:func:`openinterp.core.errata.load_errata`) alter the experiments they cover:
    each shows the erratum, and one set aside counts toward no standing.
    """
    rows = wiring_rows or {}
    output_dir = Path(output_dir)
    _clean_owned_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = build_tree(records, errata)
    add_stopped(models, (*draws, *stops))  # neurons that only stopped scans cover
    experiments = [
        e for m in models for layer in m.layers for n in layer.units for e in n.experiments
    ]
    # Statistics read the true text: build_tree counts near-duplicate probes, and the queries
    # group readings by explanation. Only what the site publishes is scrubbed. The two
    # exports run in the background while the pages render (builder/parallel.py).
    stats_result = in_background(lambda: _true_text_stats(experiments, draws, corpus_sources), jobs)
    for experiment in experiments:
        experiment.record = scrub_record(experiment.record)
    report = BuildReport(output_dir, len(records), len(experiments))
    parquet = in_background(lambda: export_registry(experiments, output_dir, corpus_sources), jobs)
    export_draws(draws, output_dir)  # random draws no experiment could audit
    export_neuronpedia(experiments, output_dir, project_links()["repo_url"])

    common = _site_context(site_title, corpus_sources, experiments)
    render = _Renderer(_environment(), output_dir, common, report, jobs)
    studied = {n.href for m in models for layer in m.layers for n in layer.units}
    by_unit = stops_by_unit(stops)
    for model in models:
        sisters = (sister_rows or {}, sister_tests(experiments))
        _render_model(render, model, rows, studied, alignment_rows or {}, sisters, by_unit)
    render_models(render, models, model_facts or {}, _render_model)
    agents = _render_agents(render, experiments, output_dir, report)
    stats = stats_result()
    stats["totals"]["neurons"] += sum(len(layer.stopped) for m in models for layer in m.layers)
    _render_index(render, models, experiments, agents, stats, (*draws, *stops), by_unit, studied)
    report.parquet = parquet()
    render_reference_pages(render, papers_dir, report)
    _save_memos()
    return report


def _save_memos() -> None:
    """Keep the matches of new text, and the significance of new records, for the next build."""
    save_spans()
    P_USEFUL.save()


def _true_text_stats(experiments, draws, corpus_sources) -> dict:
    """Return the home page's totals and leaderboards, read from the records' true text.

    Only the leaderboards' text is scrubbed, because they show explanations.
    """
    with TemporaryDirectory() as unpublished:
        true_text = export_registry(experiments, Path(unpublished), corpus_sources)
        return scrub(registry_stats(true_text, draws))


def _render_agents(render, experiments, output_dir, report) -> list:
    """Write the agents page, each agent's page and redirects from old agent paths."""
    agents = agents_view(experiments)
    contributors = contributors_view(experiments)
    render("agents/index.html", "agents.html", agents=agents, contributors=contributors)
    for agent in agents:
        render(agent.href, "agent.html", agent=agent)
    for old, new in legacy_redirects(agents).items():
        _write_redirect(output_dir / old, Path(new).name, report)
    return agents


def _render_index(render, models, experiments, agents, stats, draws, by_unit, studied) -> None:
    render(
        "index.html",
        "index.html",
        models=models,
        explore=explore_focus(models, draws),  # draws and stops: proven dead counts as verified
        stats=stats,
        dashboard=dashboard(stats["verdicts"], models),
        agents=agents,
        fdr=false_discovery(models),
        agreement=reviewer_agreement(models),
        corpora=corpora_view(experiments),
        contested=contested_units(models),
        pageless_stops=pageless_stops(by_unit, studied),
    )


def _site_context(site_title: str, corpus_sources: dict | None, experiments) -> dict:
    """What every page's template gets."""
    return {
        **project_links(),
        "site_title": site_title,
        "corpus_sources": corpus_sources or {},  # for records with no corpus_source of their own
        "model_licenses": configured_licenses(),
        "version": __version__,
        # Derived from the data, never the clock: an unchanged registry must rebuild
        # byte-for-byte so the deploy uploads (and purges) nothing.
        "updated_at": max((e.record.verdict.timestamp for e in experiments), default=None),
    }


def _write_redirect(out: Path, target: str, report: BuildReport) -> None:
    """A page that used to live at ``out`` and now lives at ``target`` (relative to it)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f'<!doctype html><meta charset="utf-8"><title>Moved</title>'
        f'<link rel="canonical" href="{target}"><meta name="robots" content="noindex">'
        f'<meta http-equiv="refresh" content="0; url={target}">'
        f'<p>This page moved to <a href="{target}">{target}</a>.</p>\n'
    )
    report.pages.append(out)


@dataclass
class _Renderer:
    """Renders a template to a path under the output dir, with the site-wide context."""

    env: Environment
    output_dir: Path
    common: dict
    report: BuildReport
    jobs: int = 1  # worker processes for the unit pages (builder/parallel.py)

    def __call__(self, rel: str, template: str, **context) -> None:
        out = self.output_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        page = self.env.get_template(template)
        url = self.common["site_url"] + rel
        out.write_text(page.render(**self.common, root=_root_for(rel), canonical=url, **context))
        self.report.pages.append(out)


def _render_model(
    render: _Renderer,
    model,
    rows: dict,
    studied: set,
    alignment: dict | None = None,
    sisters: tuple[dict, dict] = ({}, {}),
    stops: dict | None = None,
) -> None:
    """A model's page, its circuits page (if any), and its layer and unit pages."""
    found, found_hubs = circuits_and_hubs(model, rows)
    sister_rows, tests = sisters
    named_by = by_match_slug(sister_rows)
    anchor_of = unit_anchors(found, found_hubs)
    page = circuits_href(model.slug)
    graphs = sister_graphs(model, sister_rows, studied)
    shown = {"circuits": found, "hubs": found_hubs, "circuits_page": page, "sister_graphs": graphs}
    render(model.href, "model.html", model=model, **shown)
    if found or found_hubs:
        n_units = sum(len(c.units) for c in found)
        view = {"found": found, "hubs": found_hubs, "n_units": n_units, **HUB_RULE}
        render(page, "circuits.html", model=model, **view)
    for layer in model.layers:
        render(layer.href, "layer.html", model=model, layer=layer)

    def unit_page(layer, neuron) -> None:
        anchor = anchor_of.get(neuron.href)
        circuit = f"{page}#{anchor}" if anchor else None
        views = [_experiment_view(e, studied, rows, circuit) for e in neuron.experiments]
        render(
            neuron.href,
            "neuron.html",
            model=model,
            layer=layer,
            neuron=neuron,
            views=views,
            poles=pole_scans(neuron),
            aligned=aligned_latents(model, neuron, alignment or {}, studied),
            sisters=sister_view(model.slug, neuron, sister_rows, named_by, studied, tests),
            stops=(stops or {}).get(neuron.href, []),
        )

    units = [(layer, neuron) for layer in model.layers for neuron in layer.units]
    render_in_parallel(render, unit_page, units)

"""Compile every archived experiment into a static site under ``dist/``.

Pages follow the registry hierarchy: the index, then a page per model, layer and unit.
A neuron's page is ``models/<model>/L<layer>/N<idx>.html``, and a head's ``H<idx>.html``.
Each unit page lists all of its experiments, null results too, most conclusive first.
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

from openinterpretability import __version__
from openinterpretability import sisters as sister_lib
from openinterpretability.builder import charts, palette, pieces, statuses
from openinterpretability.builder.agents import (
    agent_href,
    agents_view,
    contributors_view,
    legacy_redirects,
)
from openinterpretability.builder.aligned import aligned_latents
from openinterpretability.builder.circuit_pages import render_circuits
from openinterpretability.builder.circuits import circuits_and_hubs, wiring_view
from openinterpretability.builder.contents import add_contents
from openinterpretability.builder.corpora import language_mix
from openinterpretability.builder.dashboard import dashboard
from openinterpretability.builder.feeds import write_feeds
from openinterpretability.builder.heat import HeatToken, heat_html, token_heat, values_rows
from openinterpretability.builder.model_facts import ModelFacts, model_page_extras, render_models
from openinterpretability.builder.neuronpedia_export import export_neuronpedia
from openinterpretability.builder.parallel import in_background, render_in_parallel
from openinterpretability.builder.parquet_export import export_draws, export_registry
from openinterpretability.builder.pii import save_spans, scrub, scrub_experiments
from openinterpretability.builder.recent import recent_experiments
from openinterpretability.builder.registry_stats import registry_stats  # noqa: F401  (re-exported)
from openinterpretability.builder.sections import render_reference_pages, with_explanations
from openinterpretability.builder.significance import P_USEFUL, evidence_strength
from openinterpretability.builder.sister_graph import sister_graphs
from openinterpretability.builder.slim import SlimArchive, full_records, hand_over
from openinterpretability.builder.standing import ROLES, contested_units, reviewer_agreement
from openinterpretability.builder.stopped import add_stopped, explore_focus, stopped_unit
from openinterpretability.builder.stops import stops_by_unit
from openinterpretability.builder.superpositions import add_members, count_pageless_neurons
from openinterpretability.builder.tree import (
    Experiment,
    build_tree,
    circuits_href,
    false_discovery,
    neuron_href,
)
from openinterpretability.core.agent_names import canonical_agent
from openinterpretability.core.draws import ChosenStop
from openinterpretability.core.errata import Erratum
from openinterpretability.core.exceptions import RegistryError
from openinterpretability.core.explanation import is_broad
from openinterpretability.core.project import project_links
from openinterpretability.core.schemas import ActivatingExample, AuditRecord
from openinterpretability.planner import configured_licenses
from openinterpretability.sensitive import is_offensive
from openinterpretability.wiring_backfill import record_wiring


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
        loader=PackageLoader("openinterpretability.builder", "templates"),
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
    env.globals["is_broad"] = is_broad  # a VERIFIED unit active on most held-out text (#4545)
    env.globals["palette"] = palette  # the status and pole colours
    env.globals["agent_href"] = agent_href  # links from agent names to their pages
    env.globals["agent_name"] = canonical_agent  # one name per model, however it was recorded
    env.globals["evidence_strength"] = evidence_strength  # strong / moderate / weak bands
    env.globals["language_mix"] = language_mix  # a scan's tokens per language
    env.globals["heat_html"] = heat_html  # token heatmaps, built in Python for speed
    env.globals["values_rows"] = values_rows  # and the rows of their values tables
    return env


def _pct_fine(v: float | None) -> str:
    """A signed percentage with two significant figures, so 1 in 100k isn't shown as 0%."""
    if v is None:
        return "—"
    if v == 0:
        return "0%"
    decimals = max(0, 1 - math.floor(math.log10(100 * abs(v))))
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
OWNED_PATHS = ("index.html", "about.html", "learned-baseline.html", "models", "agents", "users",
               "neurons", "research", "posts", "data", "pagefind")  # fmt: skip


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
    """What an experiment's page shows.

    ``rows`` is backfilled wiring (see :mod:`openinterpretability.wiring_backfill`). ``circuit`` is
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
        and {"ref": sub.sister_of, "href": sister_lib.sister_source_href(sub.sister_of)},
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
    posts_dir: Path | None = None,
    stops: Sequence[ChosenStop] = (),
    model_facts: dict[str, ModelFacts] | None = None,
    jobs: int = 1,
    errata: Sequence[Erratum] = (),
    users: dict[str, str] | None = None,
    coverage_rows: dict | None = None,
    paper_pdfs: bool = False,
    pending: Sequence = (),
    explanations: Sequence = (),  # builder/explanation_pages.load_views
    circuits: dict | None = None,  # builder/circuit_units.load_circuit_data
    concept_map: dict | None = None,  # builder/concept_map_view.load_maps
    archive: SlimArchive | None = None,  # the records were loaded slim by it (slim.py)
) -> BuildReport:
    """Render every reviewed experiment in ``records``, normally the wiki archive.

    Rows fill in what old records lack. ``wiring_rows`` and ``corpus_sources`` come from
    ``wiring_backfill.load_rows`` and ``corpus.load_sources``. ``alignment_rows``
    (``crossref.load_rows``) are each pole's aligned latents. ``sister_rows``
    (``sisters.load_rows``) are its matches. ``papers_dir`` holds the Research section, and
    ``posts_dir`` the Posts section (#4650). ``paper_pdfs`` adds each paper's PDF, and
    ``pending`` (``builder/pending.py``) lists open experiment PRs on the pages. ``stops``
    and ``draws`` (``core.draws``) go on their unit's page. ``model_facts``
    (``builder.model_facts.load_model_facts``) fill the Models page.
    ``jobs`` is the unit pages' worker processes. ``circuits`` feed the circuit and hub pages.
    ``errata`` (``core.errata.load_errata``) alter the experiments they cover: each shows the
    erratum, and one set aside counts toward no standing. ``users`` (``builder.users``) name
    who made records that name no contributor, each with a page under ``users/``.
    ``coverage_rows`` (``builder.coverage.load_rows``) give each model page its coverage.
    """
    rows = wiring_rows or {}
    output_dir = Path(output_dir)
    _clean_owned_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = build_tree(records, errata, users, archive and archive.claims)
    add_stopped(models, (*draws, *stops))  # neurons that only stopped scans cover
    add_members(models)  # the neurons of each superposition, linked back to it
    experiments = [e for m in models for la in m.layers for n in la.units for e in n.experiments]
    report = BuildReport(output_dir, len(records), len(experiments))
    hand_over(records, experiments, archive and archive.paths)  # each slim record's file
    # Statistics read the true text: build_tree counts near-duplicate probes, and the queries
    # group readings by explanation. Only what the site publishes is scrubbed. The two
    # exports run in the background while the pages render (builder/parallel.py).
    stats_result = in_background(lambda: _true_text_stats(experiments, draws, corpus_sources), jobs)
    scrub_experiments(experiments)  # what the pages show: the record and its errata notes
    parquet = in_background(lambda: export_registry(experiments, output_dir, corpus_sources), jobs)
    export_draws(draws, output_dir)  # random draws no experiment could audit
    export_neuronpedia(experiments, output_dir, project_links()["repo_url"])

    common = _site_context(site_title, corpus_sources, experiments, pending)
    common |= model_page_extras(coverage_rows or {}, explanations, circuits or {}, concept_map)
    render = _Renderer(_environment(), output_dir, common, report, jobs)
    studied = {n.href for m in models for layer in m.layers for n in (*layer.pages, *layer.stopped)}
    by_unit = stops_by_unit((*draws, *stops))
    for model in models:
        sisters = (sister_rows or {}, sister_lib.sister_tests(experiments))
        _render_model(render, model, rows, studied, alignment_rows or {}, sisters, by_unit)
    render_models(render, models, model_facts or {}, _render_model, (*draws, *stops))
    agents = _render_agents(render, experiments, output_dir, report)
    stats = with_explanations(stats_result(), explanations)  # the home page's tile (#4574)
    count_pageless_neurons(stats["totals"], models)  # stopped ones, and superposition members
    _render_index(render, models, agents, stats, (*draws, *stops))
    report.parquet = parquet()
    render_reference_pages(render, papers_dir, report, paper_pdfs, explanations, posts_dir, studied)
    write_feeds(report, experiments, common)  # sitemap.xml, feed.xml and robots.txt
    save_spans()  # keep the matches of new text for the next build
    P_USEFUL.save()  # and the significance of new records
    return report


def _true_text_stats(experiments, draws, corpus_sources) -> dict:
    """The home page's totals and leaderboards, from true text. Leaderboards are scrubbed."""
    with TemporaryDirectory() as unpublished:
        true_text = export_registry(experiments, Path(unpublished), corpus_sources, with_json=False)
        return scrub(registry_stats(true_text, draws))


def _render_agents(render, experiments, output_dir, report) -> list:
    """Write the Agents and Users pages, each agent's and person's page, and redirects."""
    agents = agents_view(experiments)
    contributors = contributors_view(experiments)
    render("agents/index.html", "agents.html", agents=agents, contributors=contributors)
    for person in contributors:
        render(person.href, "user.html", user=person)
    for agent in agents:
        render(agent.href, "agent.html", agent=agent)
    for old, new in legacy_redirects(agents).items():
        _write_redirect(output_dir / old, Path(new).name, report)
    return agents


def _render_index(render, models, agents, stats, draws) -> None:
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
        contested=contested_units(models),
    )
    if stats["baseline"]:  # the home page links to it (#3272)
        render("learned-baseline.html", "learned-baseline.html", stats=stats)


def _site_context(site_title: str, corpus_sources: dict | None, experiments, pending=()) -> dict:
    """What every page's template gets."""
    return {
        **project_links(),
        "site_title": site_title,
        "corpus_sources": corpus_sources or {},  # for records with no corpus_source of their own
        "model_licenses": configured_licenses(),
        "version": __version__,
        "pending": list(pending),  # open experiment PRs, for the home and model pages
        # From the data, not the clock, so an unchanged registry rebuilds byte for byte.
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
        html = page.render(**self.common, root=_root_for(rel), canonical=url, **context)
        out.write_text(add_contents(html))  # ids on headings, and contents on a long page
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
    named_by = sister_lib.by_match_slug(sister_rows)
    page = circuits_href(model.slug)
    graphs = sister_graphs(model, sister_rows, studied)
    shown = {"circuits": found, "hubs": found_hubs, "circuits_page": page, "sister_graphs": graphs}
    render(model.href, "model.html", model=model, recent=recent_experiments(model), **shown)
    circuit_of = render_circuits(render, model, found, found_hubs)
    for layer in model.layers:
        render(layer.href, "layer.html", model=model, layer=layer)

    def unit_page(layer, neuron) -> None:
        with full_records(neuron.experiments):  # only this unit's evidence in memory (#4652)
            _unit_pages(layer, neuron)

    def _unit_pages(layer, neuron) -> None:
        circuit = circuit_of.get(neuron.href)
        for rank, e in enumerate(neuron.experiments, 1):
            view = _experiment_view(e, studied, rows, circuit)
            unit = {"model": model, "layer": layer, "neuron": neuron}
            render(e.href, "experiment.html", **unit, v=view, rank=rank)
        render(
            neuron.href,
            "superposition.html" if neuron.is_superposition else "neuron.html",  # #4006
            model=model,
            layer=layer,
            neuron=neuron,
            poles=pole_scans(neuron),
            aligned=aligned_latents(model, neuron, alignment or {}, studied),
            sisters=sister_lib.sister_view(
                model.slug, neuron, sister_rows, named_by, studied, tests
            ),
            stops=(stops or {}).get(neuron.href, []),
        )

    units = [(layer, neuron) for layer in model.layers for neuron in layer.pages]
    units += [(layer, stopped_unit(model, n)) for layer in model.layers for n in layer.stopped]
    render_in_parallel(render, unit_page, units)

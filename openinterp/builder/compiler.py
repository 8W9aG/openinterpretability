"""Compile every archived experiment into a static site under ``dist/``.

Pages follow the registry hierarchy: ``index.html`` → ``models/<model>/index.html``
→ ``models/<model>/L<layer>/index.html`` → ``models/<model>/L<layer>/N<idx>.html`` (a
neuron) or ``H<idx>.html`` (an attention head). Each unit page lists all of its
experiments, most conclusive first. Null results are published alongside positive ones.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from email.utils import parseaddr
from importlib.metadata import metadata
from pathlib import Path

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from pydantic import ValidationError

from openinterp import __version__
from openinterp.builder import charts, statuses
from openinterp.builder.agents import (
    agent_href,
    agents_view,
    contributors_view,
    legacy_redirects,
)
from openinterp.builder.circuits import HUB_RULE, circuits_and_hubs, unit_anchors, wiring_view
from openinterp.builder.corpora import corpora_view, language_mix
from openinterp.builder.parquet_export import export_draws, export_registry
from openinterp.builder.registry_stats import registry_stats  # noqa: F401  (re-exported)
from openinterp.builder.significance import evidence_strength
from openinterp.builder.standing import ROLES, native_standings, reviewer_agreement
from openinterp.builder.tree import (
    Experiment,
    build_tree,
    circuits_href,
    false_discovery,
    feature_href,
    neuron_href,
)
from openinterp.core.agent_names import canonical_agent
from openinterp.core.exceptions import RegistryError
from openinterp.core.schemas import ActivatingExample, AuditRecord
from openinterp.crossref import neuronpedia_url
from openinterp.sisters import by_match_slug, sister_source_href, sister_tests, sister_view
from openinterp.wiring_backfill import record_wiring


@dataclass
class BuildReport:
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
    env.globals["charts"] = charts  # page-level charts: the layer map and pieces bars
    env.globals["statuses"] = statuses  # unit statuses, their order and colours
    env.globals["agent_href"] = agent_href  # links from agent names to their pages
    env.globals["agent_name"] = canonical_agent  # one name per model, however it was recorded
    env.globals["evidence_strength"] = evidence_strength  # strong / moderate / weak bands
    env.globals["language_mix"] = language_mix  # a scan's tokens per language
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


@dataclass(frozen=True)
class HeatToken:
    text: str
    shade: float  # 0..1 of the pole's max activation; the template caps the tint
    value: float  # the stored pole strength
    peak: bool
    marked: bool = False  # fuzzing: a token the judge was told is active


@dataclass(frozen=True)
class Attended:
    text: str
    weight: float
    offset: int  # tokens from the peak (≤ 0)


@dataclass(frozen=True)
class HeatExample:
    """One top-activating example, ready for the token heatmap."""

    max_activation: float
    window: list[HeatToken]  # tokens around the peak
    tokens: list[HeatToken]  # the whole snippet
    clipped_start: bool
    clipped_end: bool
    attended: list[Attended] = field(default_factory=list)  # a head's peak attention


def token_heat(
    example: ActivatingExample, scale: float, before: int = 30, after: int = 10
) -> HeatExample:
    """Shade every token by its pole strength relative to ``scale`` (the neuron's max).

    The window keeps ``before`` tokens before the peak and ``after`` after it, so long
    snippets stay readable; the full snippet is kept for an expandable view.
    """
    scale = scale or example.max_activation or 1.0
    tokens = [
        HeatToken(
            tok.replace("\n", "↵"),  # a newline the neuron fires on must stay visible
            round(min(1.0, max(0.0, act / scale)), 3),
            act or 0.0,  # strengths of inactive tokens can be stored as -0.0
            i == example.max_token_index,
        )
        for i, (tok, act) in enumerate(zip(example.tokens, example.activations, strict=True))
    ]
    start = max(0, example.max_token_index - before)
    end = min(len(tokens), example.max_token_index + after + 1)
    return HeatExample(
        max_activation=example.max_activation,
        window=tokens[start:end],
        tokens=tokens,
        clipped_start=start > 0,
        clipped_end=end < len(tokens),
        attended=[
            Attended(tokens[i].text, w, i - example.max_token_index)
            for i, w in zip(example.attends_to, example.attention_weights, strict=True)
        ],
    )


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
        return {"w": w, "tokens": tokens, "outcome": outcome, "peak": max(w.activations, default=0)}

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
OWNED_PATHS = ("index.html", "models", "agents", "neurons", "data", "pagefind")


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
    """``rows``: backfilled wiring (see :mod:`openinterp.wiring_backfill`); ``circuit``:
    the link to the circuit the unit is in, if any."""
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
) -> BuildReport:
    """Render every reviewed experiment in ``records`` (normally the wiki archive), with
    ``wiring_rows`` (from :func:`openinterp.wiring_backfill.load_rows`) and ``corpus_sources``
    (:func:`openinterp.corpus.load_sources`) for records that predate wiring or sidecars,
    each neuron pole's aligned dictionary latents, ``alignment_rows``
    (:func:`openinterp.crossref.load_rows`), and its matches in other models,
    ``sister_rows`` (:func:`openinterp.sisters.load_rows`)."""
    rows = wiring_rows or {}
    output_dir = Path(output_dir)
    _clean_owned_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = build_tree(records)
    experiments = [
        e for m in models for layer in m.layers for n in layer.units for e in n.experiments
    ]
    report = BuildReport(output_dir, len(records), len(experiments))
    report.parquet = export_registry(experiments, output_dir, corpus_sources)
    export_draws(draws, output_dir)  # random draws no experiment could audit

    common = _site_context(site_title, corpus_sources, experiments)
    render = _Renderer(_environment(), output_dir, common, report)
    studied = {n.href for m in models for layer in m.layers for n in layer.units}
    for model in models:
        sisters = (sister_rows or {}, sister_tests(experiments))
        _render_model(render, model, rows, studied, alignment_rows or {}, sisters)
    agents = agents_view(experiments)
    contributors = contributors_view(experiments)
    render("agents/index.html", "agents.html", agents=agents, contributors=contributors)
    for agent in agents:
        render(agent.href, "agent.html", agent=agent)
    for old, new in legacy_redirects(agents).items():
        _write_redirect(output_dir / old, Path(new).name, report)
    _render_index(render, models, experiments, agents, registry_stats(report.parquet, draws))
    return report


def _render_index(render, models, experiments, agents, stats) -> None:
    render(
        "index.html",
        "index.html",
        models=models,
        stats=stats,
        agents=agents,
        fdr=false_discovery(models),
        agreement=reviewer_agreement(models),
        corpora=corpora_view(experiments),
        contested=[
            {"model": n.model, "unit": n, "side": side, "standing": st}
            for n, side, st in native_standings(models)
            if st.contested
        ],
    )


def _project_links() -> dict:
    """The repo URL and contact email for the footer, from pyproject.toml's metadata."""
    meta = metadata("openinterp")
    urls = dict(u.split(", ", 1) for u in meta.get_all("Project-URL") or [])
    return {"repo_url": urls["Repository"], "contact_email": parseaddr(meta["Author-email"])[1]}


def _site_context(site_title: str, corpus_sources: dict | None, experiments) -> dict:
    """What every page's template gets."""
    return {
        **_project_links(),
        "site_title": site_title,
        "corpus_sources": corpus_sources or {},  # for records with no corpus_source of their own
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

    def __call__(self, rel: str, template: str, **context) -> None:
        out = self.output_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        page = self.env.get_template(template)
        out.write_text(page.render(**self.common, root=_root_for(rel), **context))
        self.report.pages.append(out)


def aligned_latents(model, neuron, alignment: dict, studied: set) -> list[dict]:
    """Each pole's aligned dictionary latents for a neuron page, with each latent's page
    here (if it was audited) and on Neuronpedia (if hosted there)."""
    if not neuron.is_neuron:
        return []
    poles = []
    for pole in neuron.sides:
        rows = alignment.get(f"{model.slug}/L{neuron.layer}/N{neuron.idx}-{pole}", [])
        views = []
        for row in rows:
            sides = []
            for name in ("writes", "reads"):
                side = getattr(row, name)
                if side is None:
                    continue
                matches = []
                for m in side.matches:
                    href = feature_href(model.slug, neuron.layer, row.dictionary, m.feature_idx)
                    matches.append(
                        {
                            "m": m,
                            "href": href if href in studied else None,
                            "neuronpedia": neuronpedia_url(row, m.feature_idx),
                            "above": m.cosine > side.chance,
                        }
                    )
                sides.append({"name": name, "side": side, "matches": matches})
            views.append({"row": row, "sides": sides})
        if views:
            poles.append({"pole": pole, "rows": views})
    return poles


def _render_model(
    render: _Renderer,
    model,
    rows: dict,
    studied: set,
    alignment: dict | None = None,
    sisters: tuple[dict, dict] = ({}, {}),
) -> None:
    """A model's page, its circuits page (if any), and its layer and unit pages."""
    found, found_hubs = circuits_and_hubs(model, rows)
    sister_rows, tests = sisters
    named_by = by_match_slug(sister_rows)
    anchor_of = unit_anchors(found, found_hubs)
    page = circuits_href(model.slug)
    render(
        model.href, "model.html", model=model, circuits=found, hubs=found_hubs, circuits_page=page
    )
    if found or found_hubs:
        n_units = sum(len(c.units) for c in found)
        view = {"found": found, "hubs": found_hubs, "n_units": n_units, **HUB_RULE}
        render(page, "circuits.html", model=model, **view)
    for layer in model.layers:
        render(layer.href, "layer.html", model=model, layer=layer)
        for neuron in layer.units:
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
            )

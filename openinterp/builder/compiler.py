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
from pathlib import Path

import duckdb
from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from pydantic import ValidationError

from openinterp import __version__
from openinterp.builder.charts import causal_chart, histogram_bars, probe_chart, vocab_chart
from openinterp.builder.parquet_export import export_registry
from openinterp.builder.standing import ROLES, native_standings, reviewer_agreement
from openinterp.builder.tree import (
    LABEL_PREFIX,
    SIDES,
    Experiment,
    build_tree,
    false_discovery,
    feature_href,
    head_href,
    neuron_href,
)
from openinterp.core.exceptions import RegistryError
from openinterp.core.schemas import ActivatingExample, AuditRecord


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
    env.globals["roles"] = ROLES
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


def _anchor_href(row: dict) -> str:
    model_slug = row["slug"].split("/", 1)[0]
    kind, layer = row["kind"], row["layer"]
    if kind == "attn_head":
        page = head_href(model_slug, layer, row["head_idx"])
    elif kind == "dictionary_feature":
        page = feature_href(model_slug, layer, row["dictionary"], row["feature_idx"])
    else:
        page = neuron_href(model_slug, layer, row["neuron_idx"])
    return f"{page}#{row['pole'] or SIDES[kind][0]}-{row['experiment_id']}"


def _unit_label(row: dict) -> str:
    """``N12``, ``H3`` or ``F40``; templates add the layer and the pole mark."""
    column = {"attn_head": "head_idx", "dictionary_feature": "feature_idx"}
    return f"{LABEL_PREFIX[row['kind']]}{row[column.get(row['kind'], 'neuron_idx')]}"


RECENT_EXPERIMENTS = 10  # rows in the home page's "Most recent experiments"


def registry_stats(parquet: Path) -> dict:
    """Totals and leaderboards, queried with DuckDB over the exported parquet."""
    con = duckdb.connect()
    con.execute(f"CREATE VIEW r AS SELECT * FROM read_parquet('{parquet.as_posix()}')")

    def rows(sql: str) -> list[dict]:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    totals = rows(
        """
        SELECT count(*) AS experiments,
               count(DISTINCT (model_name, layer, neuron_idx))
                   FILTER (WHERE kind = 'mlp_neuron_pole') AS neurons,
               count(DISTINCT (model_name, layer, head_idx))
                   FILTER (WHERE kind = 'attn_head') AS heads,
               count(DISTINCT slug) FILTER (WHERE kind = 'dictionary_feature') AS features,
               count(DISTINCT model_name) AS models,
               -- dictionary features are a baseline, not claims about native units
               count(DISTINCT slug) FILTER (WHERE unit_status = 'VERIFIED'
                                            AND kind != 'dictionary_feature') AS verified,
               count(DISTINCT slug) FILTER (WHERE unit_status = 'VERIFIED'
                                            AND kind = 'dictionary_feature') AS verified_features,
               count(DISTINCT slug) FILTER (WHERE unit_status = 'CONTESTED'
                                            AND kind != 'dictionary_feature') AS contested,
               count(*) FILTER (WHERE conclusiveness >= 0.9 AND evidence_direction = 'refutes')
                   AS conclusive_nulls,
               count(*) FILTER (WHERE self_reviewed) AS self_reviewed,
               -- a standing rests on self-reviews only when no independent review exists
               count(DISTINCT slug) FILTER (WHERE counts_toward_status AND self_reviewed
                                            AND unit_status = 'VERIFIED'
                                            AND kind != 'dictionary_feature')
                   AS verified_self_reviewed
        FROM r
        """
    )[0]
    verified = rows(
        """
        SELECT slug, model_name, kind, layer, neuron_idx, pole, head_idx, dictionary, feature_idx,
               experiment_id,
               explanation, self_reviewed,
               confidence_score, sufficiency_score, necessity_score, is_lookahead
        FROM r
        WHERE counts_toward_status AND unit_status = 'VERIFIED' AND kind != 'dictionary_feature'
        QUALIFY row_number() OVER (
            PARTITION BY slug ORDER BY reviewed_at DESC, experiment_id
        ) = 1  -- one row per unit: its newest agreeing review
        ORDER BY confidence_score DESC, slug
        """
    )
    conclusive = rows(
        """
        SELECT slug, model_name, kind, layer, neuron_idx, pole, head_idx, dictionary, feature_idx,
               experiment_id,
               explanation, status,
               conclusiveness, evidence_direction
        FROM r
        WHERE reuses_probes_of IS NULL  -- re-measurements are not independent evidence
        QUALIFY row_number() OVER (
            PARTITION BY slug ORDER BY conclusiveness DESC, reviewed_at DESC, experiment_id
        ) = 1  -- one row per neuron pole, so repeated experiments don't crowd the list
        ORDER BY conclusiveness DESC, reviewed_at DESC, slug LIMIT 15
        """
    )
    recent = rows(
        f"""
        SELECT slug, model_name, kind, layer, neuron_idx, pole, head_idx, dictionary, feature_idx,
               experiment_id,
               explanation, status,
               conclusiveness, evidence_direction, submitting_agent_model,
               -- formatted in SQL: returning TIMESTAMPTZ to Python would need pytz
               strftime(timezone('UTC', reviewed_at), '%Y-%m-%dT%H:%M:%SZ') AS reviewed_iso,
               strftime(timezone('UTC', reviewed_at), '%Y-%m-%d') AS reviewed_date,
               strftime(timezone('UTC', reviewed_at), '%H:%M UTC') AS reviewed_time,
               reviewer_agent_model, self_reviewed,
               reuses_probes_of IS NOT NULL AS remeasurement
        FROM r ORDER BY reviewed_at DESC, slug, experiment_id LIMIT {RECENT_EXPERIMENTS}
        """
    )
    agents = rows(
        """
        SELECT submitting_agent_model AS agent, count(*) AS experiments,
               count(DISTINCT slug)
                   FILTER (WHERE counts_toward_status AND unit_status = 'VERIFIED') AS verified,
               count(*) FILTER (WHERE self_reviewed) AS self_reviewed,
               avg(conclusiveness) AS mean_conclusiveness
        FROM r GROUP BY ALL ORDER BY experiments DESC, verified DESC, agent
        """
    )
    con.close()
    for row in verified + conclusive + recent:
        row["href"] = _anchor_href(row)
        row["unit"] = _unit_label(row)
    return {
        "totals": totals,
        "verified": verified,
        "conclusive": conclusive,
        "recent": recent,
        "agents": agents,
    }


# Paths the builder (and the Pagefind step after it) owns inside the output dir.
# They are removed before each build so deleted records leave no stale pages.
OWNED_PATHS = ("index.html", "models", "neurons", "data", "pagefind")


def _clean_owned_paths(output_dir: Path) -> None:
    for name in OWNED_PATHS:
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _experiment_view(e: Experiment, studied: set[str]) -> dict:
    sub, verdict = e.record.submission, e.record.verdict
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
        "bars": histogram_bars(sub.activation_histogram),
        "probes": probe_chart(verdict, sub.max_activation),
        "causal": causal_chart(verdict.causal_metrics, sub.counterfactuals),
        "vocab": vocab_chart(sub.vocabulary_lens),
        "examples": [token_heat(ex, sub.max_activation or 0.0) for ex in sub.top_examples[:12]],
        "correlated": correlated,
        "scores": explanation_view(verdict.explanation_scores, sub.max_activation or 0.0),
    }


def compile_site(
    records: Sequence[AuditRecord],
    output_dir: Path,
    *,
    site_title: str = "Open Interpretability",
) -> BuildReport:
    """Render every reviewed experiment in ``records`` (normally the wiki archive)."""
    output_dir = Path(output_dir)
    _clean_owned_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = build_tree(records)
    experiments = [
        e for m in models for layer in m.layers for n in layer.units for e in n.experiments
    ]
    report = BuildReport(output_dir, len(records), len(experiments))
    report.parquet = export_registry(experiments, output_dir)

    env = _environment()
    common = {
        "site_title": site_title,
        "version": __version__,
        # Derived from the data, never the clock: an unchanged registry must rebuild
        # byte-for-byte so the deploy uploads (and purges) nothing.
        "updated_at": max((e.record.verdict.timestamp for e in experiments), default=None),
    }

    def render(rel: str, template: str, **context) -> None:
        out = output_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(env.get_template(template).render(**common, root=_root_for(rel), **context))
        report.pages.append(out)

    studied = {n.href for m in models for layer in m.layers for n in layer.units}
    for model in models:
        render(model.href, "model.html", model=model)
        for layer in model.layers:
            render(layer.href, "layer.html", model=model, layer=layer)
            for neuron in layer.units:
                views = [_experiment_view(e, studied) for e in neuron.experiments]
                render(
                    neuron.href, "neuron.html", model=model, layer=layer, neuron=neuron, views=views
                )
    render(
        "index.html",
        "index.html",
        models=models,
        stats=registry_stats(report.parquet),
        fdr=false_discovery(models),
        agreement=reviewer_agreement(models),
        contested=[
            {"model": n.model, "unit": n, "side": side, "standing": st}
            for n, side, st in native_standings(models)
            if st.contested
        ],
    )
    return report

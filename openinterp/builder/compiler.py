"""Compile every archived experiment into a static site under ``dist/``.

Pages follow the registry hierarchy: ``index.html`` → ``models/<model>/index.html``
→ ``models/<model>/L<layer>/index.html`` → ``models/<model>/L<layer>/N<idx>.html``.
Each neuron page lists all of its experiments, most conclusive first. Null results
are published alongside positive ones.
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
from openinterp.builder.parquet_export import export_registry
from openinterp.builder.tree import Experiment, build_tree, false_discovery, neuron_href
from openinterp.core.exceptions import RegistryError
from openinterp.core.schemas import ActivatingExample, AuditRecord, Histogram


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


def histogram_bars(hist: Histogram | None, width: int = 600, height: int = 140) -> list[dict]:
    """SVG bar geometry; heights are log-scaled because activations are heavy-tailed."""
    if hist is None or not hist.counts:
        return []
    peak = math.log1p(max(hist.counts)) or 1.0
    bar_w = width / len(hist.counts)
    bars = []
    for i, count in enumerate(hist.counts):
        h = height * math.log1p(count) / peak
        bars.append(
            {
                "x": round(i * bar_w, 2),
                "y": round(height - h, 2),
                "w": round(max(bar_w - 1, 0.5), 2),
                "h": round(h, 2),
                "label": f"[{hist.bin_edges[i]:.3g}, {hist.bin_edges[i + 1]:.3g}): {count}",
            }
        )
    return bars


def token_heat(example: ActivatingExample, scale: float) -> list[tuple[str, float]]:
    scale = scale or example.max_activation or 1.0
    return [
        (tok, round(min(1.0, act / scale), 3))
        for tok, act in zip(example.tokens, example.activations, strict=True)
    ]


def _anchor_href(row: dict) -> str:
    model_slug = row["slug"].split("/", 1)[0]
    page = neuron_href(model_slug, row["layer"], row["neuron_idx"])
    return f"{page}#{row['pole']}-{row['experiment_id']}"


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
               count(DISTINCT (model_name, layer, neuron_idx)) AS neurons,
               count(DISTINCT model_name) AS models,
               count(*) FILTER (WHERE is_latest AND status = 'VERIFIED') AS verified,
               count(*) FILTER (WHERE conclusiveness >= 0.9 AND evidence_direction = 'refutes')
                   AS conclusive_nulls
        FROM r
        """
    )[0]
    verified = rows(
        """
        SELECT slug, model_name, layer, neuron_idx, pole, experiment_id, explanation,
               confidence_score, sufficiency_score, necessity_score, is_lookahead
        FROM r WHERE is_latest AND status = 'VERIFIED'
        ORDER BY confidence_score DESC, slug
        """
    )
    conclusive = rows(
        """
        SELECT slug, model_name, layer, neuron_idx, pole, experiment_id, explanation, status,
               conclusiveness, evidence_direction
        FROM r
        WHERE reuses_probes_of IS NULL  -- re-measurements are not independent evidence
        QUALIFY row_number() OVER (
            PARTITION BY slug ORDER BY conclusiveness DESC, reviewed_at DESC, experiment_id
        ) = 1  -- one row per neuron pole, so repeated experiments don't crowd the list
        ORDER BY conclusiveness DESC, reviewed_at DESC, slug LIMIT 15
        """
    )
    agents = rows(
        """
        SELECT submitting_agent_model AS agent, count(*) AS experiments,
               count(*) FILTER (WHERE is_latest AND status = 'VERIFIED') AS verified,
               avg(conclusiveness) AS mean_conclusiveness
        FROM r GROUP BY ALL ORDER BY experiments DESC, verified DESC, agent
        """
    )
    con.close()
    for row in verified + conclusive:
        row["href"] = _anchor_href(row)
    return {"totals": totals, "verified": verified, "conclusive": conclusive, "agents": agents}


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
    for c in sub.correlated_neurons:
        href = neuron_href(ident.model_slug, ident.layer, c.neuron_idx)
        correlated.append({"c": c, "href": href if href in studied else None})
    return {
        "e": e,
        "sub": sub,
        "verdict": verdict,
        "ident": ident,
        "c": verdict.causal_metrics,
        "bars": histogram_bars(sub.activation_histogram),
        "examples": [
            (ex, token_heat(ex, sub.max_activation or 0.0)) for ex in sub.top_examples[:12]
        ],
        "correlated": correlated,
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
        e for m in models for layer in m.layers for n in layer.neurons for e in n.experiments
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

    studied = {n.href for m in models for layer in m.layers for n in layer.neurons}
    for model in models:
        render(model.href, "model.html", model=model)
        for layer in model.layers:
            render(layer.href, "layer.html", model=model, layer=layer)
            for neuron in layer.neurons:
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
    )
    return report

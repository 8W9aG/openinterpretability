"""Compile verified audit records into a static site under ``dist/``."""

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
from openinterp.core.exceptions import RegistryError
from openinterp.core.schemas import ActivatingExample, AuditRecord, Histogram


@dataclass
class BuildReport:
    output_dir: Path
    records_loaded: int
    records_published: int
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
    env.filters["num"] = lambda v, d=3: "—" if v is None else f"{v:.{d}f}"
    return env


def neuron_page_path(record: AuditRecord) -> str:
    return f"neurons/{record.submission.identifier.slug}.html"


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


def registry_stats(parquet: Path) -> dict:
    """Leaderboard and coverage tables, queried with DuckDB over the exported parquet."""
    con = duckdb.connect()
    con.execute(f"CREATE VIEW r AS SELECT * FROM read_parquet('{parquet.as_posix()}')")

    def rows(sql: str) -> list[dict]:
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]

    models = rows(
        """
        SELECT model_name,
               count(*) AS neurons,
               count(DISTINCT layer) AS layers_covered,
               coalesce(max(model_num_layers), max(layer) + 1) AS n_layers,
               avg(confidence_score) AS mean_confidence,
               avg(CASE WHEN is_lookahead THEN 1 ELSE 0 END) AS lookahead_share
        FROM r GROUP BY model_name ORDER BY neurons DESC, model_name
        """
    )
    per_layer = rows(
        "SELECT model_name, layer, count(*) AS n FROM r GROUP BY ALL ORDER BY model_name, layer"
    )
    for model in models:
        counts = {
            row["layer"]: row["n"] for row in per_layer if row["model_name"] == model["model_name"]
        }
        model["coverage"] = model["layers_covered"] / model["n_layers"]
        model["layer_counts"] = [counts.get(i, 0) for i in range(model["n_layers"])]
        model["layer_max"] = max(model["layer_counts"]) or 1

    agents = rows(
        """
        SELECT submitting_agent_model AS agent, count(*) AS verified,
               avg(confidence_score) AS mean_confidence
        FROM r GROUP BY ALL ORDER BY verified DESC, mean_confidence DESC, agent
        """
    )
    totals = rows(
        """
        SELECT count(*) AS neurons, count(DISTINCT model_name) AS models,
               avg(confidence_score) AS mean_confidence,
               sum(CASE WHEN is_lookahead THEN 1 ELSE 0 END) AS lookahead
        FROM r
        """
    )[0]
    con.close()
    return {"models": models, "agents": agents, "totals": totals}


# Paths the builder (and the Pagefind step after it) owns inside the output dir.
# They are removed before each build so deleted records leave no stale pages.
OWNED_PATHS = ("index.html", "neurons", "data", "pagefind")


def _clean_owned_paths(output_dir: Path) -> None:
    for name in OWNED_PATHS:
        path = output_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def compile_site(
    records: Sequence[AuditRecord],
    output_dir: Path,
    *,
    include_unverified: bool = False,
    site_title: str = "Open Interpretability",
) -> BuildReport:
    output_dir = Path(output_dir)
    _clean_owned_paths(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    published = [r for r in records if include_unverified or r.is_verified]
    # Total order, so identical records always produce byte-identical output.
    published.sort(
        key=lambda r: (
            -(r.verdict.confidence_score if r.verdict else 0.0),
            r.submission.identifier.slug,
        )
    )

    report = BuildReport(output_dir, len(records), len(published))
    report.parquet = export_registry(published, output_dir)

    env = _environment()
    common = {
        "site_title": site_title,
        "version": __version__,
        # Derived from the data, never the clock: an unchanged registry must rebuild
        # byte-for-byte so the deploy uploads (and purges) nothing.
        "updated_at": max((r.verdict.timestamp for r in published if r.verdict), default=None),
    }

    neurons = []
    for record in published:
        rel = neuron_page_path(record)
        sub = record.submission
        page = env.get_template("neuron.html").render(
            **common,
            root=_root_for(rel),
            record=record,
            sub=sub,
            verdict=record.verdict,
            ident=sub.identifier,
            bars=histogram_bars(sub.activation_histogram),
            examples=[
                (ex, token_heat(ex, sub.max_activation or 0.0)) for ex in sub.top_examples[:12]
            ],
        )
        out = output_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(page)
        report.pages.append(out)
        neurons.append({"record": record, "href": rel})

    index = env.get_template("index.html").render(
        **common,
        root="",
        neurons=neurons,
        stats=registry_stats(report.parquet),
        total_loaded=len(records),
    )
    (output_dir / "index.html").write_text(index)
    report.pages.append(output_dir / "index.html")
    return report

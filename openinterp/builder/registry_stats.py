"""The home page's totals and leaderboards, queried with DuckDB over the exported
registry parquet (see :func:`openinterp.builder.parquet_export.export_registry`)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import duckdb

from openinterp.builder.learned_baseline import learned_baseline
from openinterp.builder.tree import LABEL_PREFIX, SIDES, feature_href, head_href, neuron_href

RECENT_EXPERIMENTS = 10  # rows in the home page's "Most recent experiments"


TOTALS = """
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

VERIFIED = """
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

CONCLUSIVE = """
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

RECENT = f"""
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

# The queries in the order the page's stats come in; TOTALS is one row.
QUERIES = {
    "totals": TOTALS,
    "verified": VERIFIED,
    "conclusive": CONCLUSIVE,
    "recent": RECENT,
}


def registry_stats(parquet: Path, draws: Sequence = ()) -> dict:
    """Totals and leaderboards, queried with DuckDB over the exported parquet."""
    con = duckdb.connect()
    con.execute(f"CREATE VIEW r AS SELECT * FROM read_parquet('{parquet.as_posix()}')")
    stats = {name: _rows(con, sql) for name, sql in QUERIES.items()}
    stats["totals"] = stats["totals"][0]
    stats["baseline"] = learned_baseline(con, draws)
    con.close()
    for row in stats["verified"] + stats["conclusive"] + stats["recent"]:
        row["href"] = _anchor_href(row)
        row["unit"] = _unit_label(row)
    return stats


def _rows(con, sql: str) -> list[dict]:
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


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

"""The home page's totals and leaderboards.

They are queried with DuckDB over the exported registry parquet (see
:func:`openinterp.builder.parquet_export.export_registry`).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import duckdb

from openinterp.builder.hrefs import experiment_href
from openinterp.builder.learned_baseline import learned_baseline
from openinterp.builder.significance import STRONG_EVIDENCE
from openinterp.builder.tree import LABEL_PREFIX, NATIVE_KINDS
from openinterp.core.schemas import PROTOCOL_VERSION

RECENT_EXPERIMENTS = 10  # rows in the home page's "Most recent experiments"
# Dictionary features and superpositions are not claims about one native unit. They are
# counted apart.
NATIVE = "kind IN ({})".format(", ".join(f"'{kind}'" for kind in NATIVE_KINDS))


TOTALS = f"""
    SELECT count(*) AS experiments,
           count(DISTINCT (model_name, layer, neuron_idx))
               FILTER (WHERE kind = 'mlp_neuron_pole') AS neurons,
           count(DISTINCT (model_name, layer, head_idx))
               FILTER (WHERE kind = 'attn_head') AS heads,
           count(DISTINCT slug) FILTER (WHERE kind = 'dictionary_feature') AS features,
           count(DISTINCT slug) FILTER (WHERE kind = 'superposition') AS superpositions,
           count(DISTINCT model_name) AS models,
           count(DISTINCT slug) FILTER (WHERE unit_status = 'VERIFIED' AND {NATIVE}) AS verified,
           count(DISTINCT slug) FILTER (WHERE unit_status = 'VERIFIED'
                                        AND kind = 'dictionary_feature') AS verified_features,
           count(DISTINCT slug) FILTER (WHERE unit_status = 'VERIFIED'
                                        AND kind = 'superposition') AS verified_superpositions,
           count(DISTINCT slug) FILTER (WHERE unit_status = 'CONTESTED' AND {NATIVE}) AS contested,
           count(*) FILTER (WHERE conclusiveness >= {STRONG_EVIDENCE}
                                  AND evidence_direction = 'refutes') AS conclusive_nulls,
           count(*) FILTER (WHERE self_reviewed) AS self_reviewed,
           -- a standing rests on self-reviews only when no independent review exists
           count(DISTINCT slug) FILTER (WHERE counts_toward_status AND self_reviewed
                                        AND unit_status = 'VERIFIED' AND {NATIVE})
               AS verified_self_reviewed
    FROM r
    """  # noqa: S608  (interpolates module constants only)

# The protocol version that each verified unit's standing rests on: its newest record
# that agrees with the standing (issue #3207). Older verdicts were judged by older rules.
PROTOCOL_MIX = f"""
    SELECT coalesce(protocol_version, 'unknown') AS protocol_version, count(*) AS units
    FROM (
        SELECT slug, protocol_version FROM r
        WHERE counts_toward_status AND unit_status = 'VERIFIED' AND {NATIVE}
        QUALIFY row_number() OVER (PARTITION BY slug ORDER BY reviewed_at DESC, experiment_id) = 1
    )
    GROUP BY ALL ORDER BY protocol_version DESC
    """  # noqa: S608  (interpolates module constants only)

CONCLUSIVE = """
    SELECT slug, model_name, kind, layer, neuron_idx, pole, head_idx, dictionary, feature_idx,
           superposition_id,
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
           superposition_id,
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
    """  # noqa: S608  (interpolates module constants only)

# The dashboard's input: verdict counts per UTC day, model and layer, of native units only.
VERDICTS = f"""
    SELECT strftime(timezone('UTC', reviewed_at), '%Y-%m-%d') AS day,
           model_name, layer, status, count(*) AS n
    FROM r
    WHERE {NATIVE} AND status IS NOT NULL AND reviewed_at IS NOT NULL
    GROUP BY ALL ORDER BY ALL
    """  # noqa: S608  (interpolates module constants only)

# The queries in the order the page's stats come in; TOTALS is one row.
QUERIES = {
    "totals": TOTALS,
    "protocol_mix": PROTOCOL_MIX,
    "conclusive": CONCLUSIVE,
    "recent": RECENT,
    "verdicts": VERDICTS,
}


def registry_stats(parquet: Path, draws: Sequence = ()) -> dict:
    """Totals and leaderboards, queried with DuckDB over the exported parquet."""
    con = duckdb.connect()
    # The path is a local file the build wrote, not user input.
    con.execute(f"CREATE VIEW r AS SELECT * FROM read_parquet('{parquet.as_posix()}')")  # noqa: S608
    stats = {name: _rows(con, sql) for name, sql in QUERIES.items()}
    stats["totals"] = stats["totals"][0]
    stats["current_protocol"] = PROTOCOL_VERSION
    stats["baseline"] = learned_baseline(con, draws)
    con.close()
    for row in stats["conclusive"] + stats["recent"]:
        row["href"] = _experiment_href(row)
        row["unit"] = _unit_label(row)
    return stats


def _rows(con, sql: str) -> list[dict]:
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]


def _experiment_href(row: dict) -> str:
    return experiment_href(row["slug"], row["experiment_id"])


def _unit_label(row: dict) -> str:
    """``N12``, ``H3``, ``F40`` or ``S0123456789ab``; templates add the layer and the pole mark."""
    column = {
        "attn_head": "head_idx",
        "dictionary_feature": "feature_idx",
        "superposition": "superposition_id",
    }
    return f"{LABEL_PREFIX[row['kind']]}{row[column.get(row['kind'], 'neuron_idx')]}"

"""A read-only search of the registry parquet, for people and agents (#3242).

``open-interpretability query`` and the ``query_registry`` tool of ``open-interpretability
mcp`` both call :func:`search`. It reads ``registry.parquet`` (one row per experiment,
:mod:`openinterpretability.builder.parquet_export`) from a local path or from the site.
DuckDB reads a URL by HTTP range requests. So only the columns and row groups that a query
needs come over the network.

Every filter is a bound parameter, so no input can change the SQL. By default only each
unit's newest experiment is returned (``is_latest``).
"""

from __future__ import annotations

from pathlib import Path

from openinterpretability.builder.parquet_export import LATEST

COLUMNS = (
    "experiment_id", "slug", "model_name", "kind", "layer", "neuron_idx", "head_idx", "pole",
    "unit_status", "status", "explanation", "activating_tokens", "balanced_accuracy",
    "reviewed_at",
)  # fmt: skip
"""The columns that a search returns: enough to name a unit and say what it does."""
MAX_LIMIT = 500
"""The most rows one search returns."""
TOKENS_SHOWN = 10
"""The activating tokens kept per row."""


ENV = "OPENINTERPRETABILITY_REGISTRY"
"""An environment variable that names the parquet to search, as a path or a URL."""


def default_source() -> str:
    """The parquet that :data:`ENV` names, else the local build's, else the site's.

    Returns:
        A path or a URL.

    >>> default_source().endswith("data/registry.parquet")
    True
    """
    import os

    from openinterpretability.core.project import site_url

    if os.environ.get(ENV):
        return os.environ[ENV]
    local = Path("dist") / LATEST
    return str(local) if local.exists() else site_url().rstrip("/") + "/" + LATEST


def _sample(folder: str | Path) -> str:
    """Write a one-row registry parquet into ``folder``, for the examples here.

    Args:
        folder: An existing folder.

    Returns:
        The parquet's path: one VERIFIED pole of ``m/a``, L0/N7+, that fires on " def".
    """
    import duckdb

    path = str(Path(folder) / "registry.parquet")
    duckdb.sql(
        "SELECT 'e1' AS experiment_id, 'm--a/L0/N7-positive' AS slug, 'm/a' AS model_name, "
        "'mlp_neuron_pole' AS kind, 0 AS layer, 7 AS neuron_idx, NULL::INT AS head_idx, "
        "'positive' AS pole, 'VERIFIED' AS unit_status, 'VERIFIED' AS status, "
        "'Fires on Python keywords' AS explanation, [' def'] AS activating_tokens, "
        "0.9 AS balanced_accuracy, TIMESTAMPTZ '2026-11-20 12:00:00+00' AS reviewed_at, "
        "TRUE AS is_latest"
    ).write_parquet(path)
    return path


def _like(text: str) -> str:
    """A case-insensitive substring pattern for ``text``, with ``%``, ``_`` and ``\\`` escaped.

    Args:
        text: The text to find.

    Returns:
        The pattern for ``ILIKE … ESCAPE '\\'``.

    >>> _like("100%_done")
    '%100\\\\%\\\\_done%'
    """
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def build_query(
    *,
    model: str | None = None,
    layer: int | None = None,
    kind: str | None = None,
    status: str | None = None,
    text: str | None = None,
    token: str | None = None,
    all_records: bool = False,
    limit: int = 20,
) -> tuple[str, list]:
    """The SQL of a search and its parameters, with ``?`` for the parquet's source.

    Args:
        model: The model's name, such as ``openai-community/gpt2``.
        layer: The layer.
        kind: The unit kind, such as ``mlp_neuron_pole`` or ``attn_head``.
        status: The unit's standing, such as ``VERIFIED``.
        text: Text that the explanation contains, in any case.
        token: A token that the unit fires on, exactly as stored.
        all_records: Return every experiment, not only each unit's newest.
        limit: The most rows to return, at most :data:`MAX_LIMIT`.

    Returns:
        The SQL and its parameters, the source first.

    Raises:
        ValueError: The limit is not between 1 and :data:`MAX_LIMIT`.

    >>> sql, params = build_query(status="VERIFIED", text="cat")
    >>> "unit_status = ?" in sql and "cat" not in sql, params
    (True, ['VERIFIED', '%cat%', 20])
    """
    if not 1 <= limit <= MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}, not {limit}")
    where, params = [], []
    filters = (("model_name = ?", model), ("layer = ?", layer), ("kind = ?", kind),
               ("unit_status = ?", status))  # fmt: skip
    for clause, value in filters:
        if value is not None:
            where.append(clause)
            params.append(value)
    if text:
        where.append("explanation ILIKE ? ESCAPE '\\'")
        params.append(_like(text))
    if token is not None:
        where.append("list_contains(activating_tokens, ?)")
        params.append(token)
    if not all_records:
        where.append("is_latest")
    # The review date comes back as text, since a time zone would need pytz in Python. The
    # score comes back as a float, which JSON can write, whatever type the file stored.
    casts = {"reviewed_at": "VARCHAR", "balanced_accuracy": "DOUBLE"}
    selected = [f"CAST({c} AS {casts[c]})" if c in casts else c for c in COLUMNS]
    sql = f"SELECT {', '.join(selected)} FROM read_parquet(?)"  # noqa: S608  (constants only)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY model_name, layer, neuron_idx, head_idx, pole, reviewed_at DESC LIMIT ?"
    return sql, [*params, limit]


def search(source: str | Path | None = None, **filters) -> list[dict]:
    """The registry's experiments that match every filter given, as dicts.

    Args:
        source: A path or URL of ``registry.parquet``. Default: :func:`default_source`.
        **filters: The keyword arguments of :func:`build_query`.

    Returns:
        One dict per experiment, with the :data:`COLUMNS`. The review date is text, and only
        the first :data:`TOKENS_SHOWN` activating tokens are kept.

    >>> import tempfile
    >>> [(r["experiment_id"], r["activating_tokens"])
    ...  for r in search(_sample(tempfile.mkdtemp()), token=" def")]
    [('e1', [' def'])]
    """
    import duckdb

    sql, params = build_query(**filters)
    source = str(source or default_source())
    con = duckdb.connect()
    try:
        con.execute("SET TimeZone = 'UTC'")  # review dates read the same on every machine
        rows = con.execute(sql, [source, *params]).fetchall()
    finally:
        con.close()
    out = []
    for values in rows:
        row = dict(zip(COLUMNS, values, strict=True))
        row["activating_tokens"] = list(row["activating_tokens"] or [])[:TOKENS_SHOWN]
        out.append(row)
    return out


def unit_label(row: dict) -> str:
    """A row's unit, as the site writes it: ``L3/N7+`` or ``L2/H3``.

    Args:
        row: A row from :func:`search`.

    Returns:
        The unit's short label.

    >>> unit_label({"layer": 3, "neuron_idx": 7, "head_idx": None, "pole": "positive"})
    'L3/N7+'
    >>> unit_label({"layer": 2, "neuron_idx": None, "head_idx": 3, "pole": None})
    'L2/H3'
    """
    if row.get("head_idx") is not None:
        return f"L{row['layer']}/H{row['head_idx']}"
    sign = {"positive": "+", "negative": "−"}.get(row.get("pole") or "", "")
    return f"L{row['layer']}/N{row['neuron_idx']}{sign}"

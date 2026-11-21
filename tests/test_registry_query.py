"""The registry search, its CLI and its MCP tool (#3242)."""

import asyncio
import json

import duckdb
import pytest
from click.testing import CliRunner

from openinterpretability import mcp_server, registry_query
from openinterpretability.cli import cli

ROWS = [
    # experiment, model, kind, layer, neuron, head, pole, status, explanation, tokens, latest
    ("e1", "m/a", "mlp_neuron_pole", 0, 7, None, "positive", "VERIFIED",
     "Fires on 100% of code_blocks", [" def", " class"], True),
    ("e2", "m/a", "mlp_neuron_pole", 0, 7, None, "positive", "VERIFIED",
     "An older claim about code", [" def"], False),
    ("e3", "m/a", "attn_head", 2, None, 3, None, "REJECTED",
     "Copies the previous name", [" Mary"], True),
    ("e4", "m/b", "mlp_neuron_pole", 1, 9, None, "negative", "NEEDS_REVISION",
     "Rests below zero on prose", None, True),
]  # fmt: skip


@pytest.fixture
def parquet(tmp_path):
    path = tmp_path / "registry.parquet"
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE r (experiment_id VARCHAR, model_name VARCHAR, kind VARCHAR, layer INT, "
        "neuron_idx INT, head_idx INT, pole VARCHAR, unit_status VARCHAR, explanation VARCHAR, "
        "activating_tokens VARCHAR[], is_latest BOOLEAN)"
    )
    con.executemany("INSERT INTO r VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ROWS)
    con.execute(
        "COPY (SELECT *, model_name || '/L' || layer AS slug, unit_status AS status, "
        "0.9 AS balanced_accuracy, TIMESTAMPTZ '2026-11-20 12:00:00+00' AS reviewed_at "
        f"FROM r) TO '{path}' (FORMAT parquet)"
    )
    con.close()
    return path


def ids(rows):
    return [r["experiment_id"] for r in rows]


def test_each_filter_narrows_the_newest_records(parquet):
    assert ids(registry_query.search(parquet)) == ["e1", "e3", "e4"]  # e2 isn't the newest
    assert ids(registry_query.search(parquet, model="m/b")) == ["e4"]
    assert ids(registry_query.search(parquet, layer=2)) == ["e3"]
    assert ids(registry_query.search(parquet, kind="attn_head")) == ["e3"]
    assert ids(registry_query.search(parquet, status="VERIFIED")) == ["e1"]
    assert ids(registry_query.search(parquet, token=" Mary")) == ["e3"]
    assert ids(registry_query.search(parquet, text="CODE")) == ["e1"]  # any case
    assert ids(registry_query.search(parquet, text="code", all_records=True)) == ["e1", "e2"]
    assert ids(registry_query.search(parquet, limit=1)) == ["e1"]


def test_text_is_matched_literally_not_as_a_pattern(parquet):
    # "%" and "_" are wildcards in LIKE; escaped, they only match themselves.
    assert ids(registry_query.search(parquet, text="100%")) == ["e1"]
    assert ids(registry_query.search(parquet, text="code_blocks")) == ["e1"]
    assert ids(registry_query.search(parquet, text="codeXblocks")) == []
    assert ids(registry_query.search(parquet, text="%")) == ["e1"]


def test_input_can_not_change_the_sql(parquet):
    attack = "x' OR 1=1 --"
    sql, params = registry_query.build_query(model=attack, text=attack)
    assert attack not in sql and attack in params
    assert registry_query.search(parquet, model=attack) == []


def test_rows_hold_text_dates_and_at_most_ten_tokens(parquet):
    (row,) = registry_query.search(parquet, model="m/b")
    assert row["activating_tokens"] == [] and row["reviewed_at"].startswith("2026-11-20 12:00")
    assert set(row) == set(registry_query.COLUMNS)


@pytest.mark.parametrize("limit", [0, registry_query.MAX_LIMIT + 1])
def test_a_limit_out_of_range_is_refused(limit):
    with pytest.raises(ValueError, match="limit must be between"):
        registry_query.build_query(limit=limit)


def test_the_default_source_is_the_local_build_or_the_site(tmp_path, monkeypatch):
    monkeypatch.delenv(registry_query.ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    assert registry_query.default_source().endswith("/data/registry.parquet")
    assert registry_query.default_source().startswith("http")
    (tmp_path / "dist" / "data").mkdir(parents=True)
    (tmp_path / "dist" / "data" / "registry.parquet").write_bytes(b"")
    assert registry_query.default_source() == "dist/data/registry.parquet"


def test_the_environment_names_the_source(parquet, monkeypatch):
    monkeypatch.setenv(registry_query.ENV, str(parquet))
    assert registry_query.default_source() == str(parquet)
    assert ids(mcp_server.query_registry(token=" def")) == ["e1"]


def test_unit_labels():
    assert registry_query.unit_label({"layer": 1, "neuron_idx": 9, "pole": "negative"}) == "L1/N9−"
    assert registry_query.unit_label({"layer": 1, "neuron_idx": 9, "pole": None}) == "L1/N9"


def test_the_cli_prints_a_table_json_or_no_match(parquet):
    runner = CliRunner()
    out = runner.invoke(cli, ["query", "--source", str(parquet), "--status", "REJECTED"])
    assert out.exit_code == 0, out.output
    assert "m/a L2/H3  REJECTED  e3" in out.output and "tokens: ' Mary'" in out.output
    out = runner.invoke(cli, ["query", "--source", str(parquet), "--json", "--layer", "1"])
    assert json.loads(out.output)["experiment_id"] == "e4"
    out = runner.invoke(cli, ["query", "--source", str(parquet), "--model", "none"])
    assert out.output.strip() == "No experiment matches."
    out = runner.invoke(cli, ["query", "--source", str(parquet), "--limit", "0"])
    assert out.exit_code != 0


def test_the_mcp_tool_is_read_only_and_searches(parquet, monkeypatch):
    monkeypatch.setattr(registry_query, "default_source", lambda: str(parquet))
    assert ids(mcp_server.query_registry(kind="attn_head")) == ["e3"]
    app = mcp_server.server()
    (tool,) = asyncio.run(app.list_tools())
    assert tool.name == "query_registry" and tool.annotations.read_only_hint
    assert {"model", "layer", "text", "token", "limit"} <= set(tool.input_schema["properties"])
    result = asyncio.run(app.call_tool("query_registry", {"status": "NEEDS_REVISION"}))
    assert "e4" in json.dumps(result.model_dump(mode="json"))


def test_the_mcp_command_runs_stdio_or_names_the_extra(monkeypatch):
    calls = []

    class App:
        def run(self, transport):
            calls.append(transport)

    monkeypatch.setattr(mcp_server, "server", App)
    assert CliRunner().invoke(cli, ["mcp"]).exit_code == 0 and calls == ["stdio"]

    def missing():
        raise ModuleNotFoundError("mcp")

    monkeypatch.setattr(mcp_server, "server", missing)
    out = CliRunner().invoke(cli, ["mcp"])
    assert out.exit_code != 0 and "open-interpretability[mcp]" in out.output


def test_the_table_prints_no_control_characters_from_a_record(monkeypatch):
    # Anyone can write a record's text, and a terminal runs the escape codes in it.
    row = {"model_name": "m/a\x1b]0;title\x07", "layer": 1, "kind": "mlp_neuron_pole",
           "neuron_idx": 2, "pole": "positive", "head_idx": None, "unit_status": "REJECTED",
           "experiment_id": "e1\x1b[1A", "activating_tokens": [],
           "explanation": "Fires on dates.\x1b[2K\r\x1b[1AVERIFIED fake line"}  # fmt: skip
    monkeypatch.setattr("openinterpretability.cli_query.search", lambda *a, **k: [row])
    result = CliRunner().invoke(cli, ["query"])
    assert result.exit_code == 0, result.output
    assert not any(c in result.output for c in "\x1b\x07\r")
    assert "Fires on dates.\\x1b[2K \\x1b[1AVERIFIED fake line" in result.output

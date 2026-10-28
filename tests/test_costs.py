"""Experiment costs: the ledger every command appends to, and `openinterp costs`."""

import json
import os

from click.testing import CliRunner

from openinterp.cli import cli
from openinterp.cli_costs import read_ledger
from openinterp.core import costs


def test_merge_skips_missing_parts_and_keeps_order():
    a = costs.ExperimentCosts(stages=[costs.StageTiming(stage="scan", seconds=2.0)])
    b = costs.ExperimentCosts(
        stages=[costs.StageTiming(stage="gates", seconds=0.5)],
        llm_calls=[costs.LLMCall(agent="reviewer", model="m", cost_usd=0.25)],
    )
    assert costs.merge(None, None) is None
    merged = costs.merge(a, None, b)
    assert [s.stage for s in merged.stages] == ["scan", "gates"]
    assert (merged.seconds, merged.cost_usd) == (2.5, 0.25)


def test_external_answer_is_timed_from_file_mtimes(tmp_path):
    prompt, answer = tmp_path / "ev.prompt.md", tmp_path / "ev.hypothesis.json"
    prompt.write_text("x" * 50)
    answer.write_text("{}")
    os.utime(prompt, (1000, 1000))
    os.utime(answer, (1090, 1090))
    costs.reset()
    costs.add_external("proposer", "external/p", prompt, answer)
    got = costs.measured()
    assert [(s.stage, s.seconds) for s in got.stages] == [("proposer", 90.0)]
    assert got.llm_calls[0].prompt_chars == 50 and got.llm_calls[0].external


def test_ledger_and_summary(tmp_path, monkeypatch):
    ledger = tmp_path / "costs.jsonl"
    monkeypatch.setenv(costs.LEDGER_ENV, str(ledger))
    CliRunner().invoke(cli, ["next", "--help"])  # --help exits before the command runs
    session = tmp_path / "session.json"
    session.write_text(
        json.dumps(
            {
                "duration_ms": 120000,
                "num_turns": 30,
                "total_cost_usd": 1.5,
                "usage": {
                    "input_tokens": 10,
                    "cache_read_input_tokens": 1000,
                    "cache_creation_input_tokens": 100,
                    "output_tokens": 50,
                },
            }
        )
    )
    entry = {
        "command": "review",
        "seconds": 90.0,
        "stages": [{"stage": "reviewer", "seconds": 60.0}],
        "llm_calls": [
            {
                "agent": "reviewer",
                "model": "anthropic/m",
                "prompt_chars": 30000,
                "prompt_tokens": 8000,
                "completion_tokens": 900,
                "cost_usd": 0.5,
            }
        ],
    }
    ledger.write_text(json.dumps(entry) + "\n")
    result = CliRunner().invoke(cli, ["costs", str(ledger), "--session", str(session)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "review" in out and "reviewer" in out and "8,000 + 900 tokens" in out
    assert "1 session, 2.0 min, 30 turns, 1,110 input (1,000 cache reads)" in out
    assert "$2.00 reported" in out  # LLM calls plus the driving session
    assert len(read_ledger([ledger])) == 1  # `costs` itself is not an experiment step


def test_pricing_normalises_names_and_skips_unknown_models():
    from openinterp.core import pricing

    assert pricing.price_key("external/anthropic/claude-opus-5-5[1m]") == "claude-opus-5-5"
    # 1M input at $4, 1M cache reads at $0.20, 1M 5-minute writes at 1.25 × $4, 1M output
    cost = pricing.cost_usd(
        "external/claude-opus-5-5",
        input_tokens=10**6,
        output_tokens=10**6,
        cache_read_tokens=10**6,
        cache_write_5m_tokens=10**6,
        cache_write_1h_tokens=10**6,
    )
    assert cost == 4 + 20 + 0.2 + 5 + 8
    assert pricing.cost_usd("external/deepseek-flash", input_tokens=1, output_tokens=1) is None


def _transcript(path, model="claude-sonnet-5-5"):
    def line(msg_id, output, model=model, write_1h=0):
        usage = {
            "input_tokens": 2,
            "cache_read_input_tokens": 100,
            "cache_creation_input_tokens": 50,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 50 - write_1h,
                "ephemeral_1h_input_tokens": write_1h,
            },
            "output_tokens": output,
        }
        message = {"id": msg_id, "model": model, "usage": usage}
        return json.dumps({"type": "assistant", "message": message})

    lines = [
        json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}}),
        line("m1", 5),
        line("m1", 300),  # the same response's last part carries its final output count
        line("m2", 40, write_1h=50),
        line("m3", 0, model="<synthetic>"),
        "{cut off",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def test_usage_from_transcript_counts_each_response_once(tmp_path):
    from openinterp.core.transcripts import usage_from_transcript

    u = usage_from_transcript(_transcript(tmp_path / "agent-abc.jsonl"))
    assert (u.model, u.requests, u.input_tokens, u.cache_read_tokens) == (
        "claude-sonnet-5-5",
        2,
        4,
        200,
    )
    assert (u.cache_write_5m_tokens, u.cache_write_1h_tokens, u.output_tokens) == (50, 50, 340)
    assert u.source == "claude-code:agent-abc"


def test_external_answer_with_usage_file_is_priced(tmp_path):
    from openinterp.core import pricing

    prompt, answer = tmp_path / "review.prompt.md", tmp_path / "review.probes.json"
    prompt.write_text("x")
    answer.write_text("{}")
    usage = costs.ExternalUsage(
        model="claude-sonnet-5-5",
        requests=2,
        input_tokens=4,
        cache_read_tokens=200,
        cache_write_5m_tokens=50,
        output_tokens=340,
    )
    costs.usage_path(answer).write_text(usage.model_dump_json())
    assert costs.usage_path(answer).name == "review.probes.usage.json"
    costs.reset()
    costs.add_external("reviewer", "external/claude-sonnet-5-5", prompt, answer)
    call = costs.measured().llm_calls[0]
    assert (call.prompt_tokens, call.completion_tokens) == (254, 340)
    assert (call.cached_tokens, call.cache_write_tokens) == (200, 50)
    assert call.price_table == pricing.TABLE
    assert call.cost_usd == round((4 * 2 + 200 * 0.2 + 50 * 2.5 + 340 * 10) / 1e6, 6)

    # A provider's own cost wins, and an unknown model stays unpriced.
    reported = usage.model_copy(update={"cost_usd": 0.5})
    costs.usage_path(answer).write_text(reported.model_dump_json())
    costs.add_external("reviewer", "external/claude-sonnet-5-5", prompt, answer)
    costs.usage_path(answer).write_text(usage.model_dump_json())
    costs.add_external("reviewer", "external/deepseek-flash", prompt, answer)
    _, own, unknown = costs.measured().llm_calls
    assert (own.cost_usd, own.price_table) == (0.5, None)
    assert (unknown.cost_usd, unknown.price_table, unknown.prompt_tokens) == (None, None, 254)


def test_usage_command_writes_the_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    _transcript(tmp_path / "claude/projects/p/session/subagents/agent-a1b2.jsonl")
    answer = tmp_path / "ev.hypothesis.json"
    answer.write_text("{}")
    result = CliRunner().invoke(cli, ["usage", "a1b2", "--answer", str(answer)])
    assert result.exit_code == 0, result.output
    written = costs.ExternalUsage.model_validate_json(
        (tmp_path / "ev.hypothesis.usage.json").read_text()
    )
    assert written.output_tokens == 340
    missing = CliRunner().invoke(cli, ["usage", "nope", "--answer", str(answer)])
    assert missing.exit_code != 0 and "No Claude Code transcript" in missing.output


def test_by_agent_reports_mean_and_cost_per_verified():
    def run(*calls):
        return costs.ExperimentCosts(
            llm_calls=[costs.LLMCall(agent=a, model=m, cost_usd=c) for a, m, c in calls]
        )

    rows = costs.by_agent(
        [
            ("VERIFIED", run(("judge", "j", 0.1), ("judge", "j", 0.1), ("proposer", "p", 1.0))),
            ("REJECTED", run(("judge", "j", 0.4), ("proposer", "p", None))),
            ("VERIFIED", None),
        ]
    )
    assert [(r["agent"], r["model"]) for r in rows] == [("proposer", "p"), ("judge", "j")]
    proposer, judge = rows
    assert (proposer["experiments"], proposer["priced"], proposer["mean_usd"]) == (2, 1, 1.0)
    assert proposer["usd_per_verified"] == 1.0
    assert (judge["priced"], judge["verified"]) == (2, 1)
    assert round(judge["mean_usd"], 6) == 0.3 and round(judge["usd_per_verified"], 6) == 0.6

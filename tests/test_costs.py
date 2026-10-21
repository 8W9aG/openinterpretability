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

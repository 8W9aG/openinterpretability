"""Spend caps and the stop file of scripts/experiment-loop.sh, and `openinterp costs --usd`."""

import json
import os
import subprocess
from pathlib import Path

from click.testing import CliRunner

from openinterp.cli import cli

SCRIPT = (Path(__file__).parents[1] / "scripts" / "experiment-loop.sh").read_text()
FUNCTIONS = ("stop_file", "stop_requested", "over_budget", "spent", "todays_logs", "check_limits")


def _functions() -> str:
    """The script's limit functions, as bash source."""
    out = []
    for name in FUNCTIONS:
        start = SCRIPT.index(f"{name}() {{")
        one_line = SCRIPT.index("\n", start)
        end = (
            one_line + 1
            if SCRIPT[start:one_line].rstrip().endswith("}")
            else SCRIPT.index("\n}\n", start) + 3
        )
        out.append(SCRIPT[start:end])
    return "\n".join(out)


def run(tmp_path, body: str, spend: str = "0", **env) -> subprocess.CompletedProcess:
    """Run ``body`` in /bin/bash (3.2 on macOS) after the limit functions, with stubs."""
    stub = (
        f'uv() {{ echo "{spend}"; }}\n'
        f"WORKTREE={tmp_path}\nLOG_DIR={tmp_path}/logs\n"
        'LEDGERS=() SESSIONS=() BUDGET="" DAILY_BUDGET=""\nset -euo pipefail\n'
    )
    environ = {**os.environ, "OPENINTERP_STOP_FILE": str(tmp_path / "stop"), **env}
    return subprocess.run(
        ["/bin/bash", "-c", stub + _functions() + "\n" + body],
        capture_output=True,
        text=True,
        env=environ,
    )


def test_over_budget_needs_a_cap_and_compares_dollars(tmp_path):
    out = run(
        tmp_path,
        'for a in "0.5 1" "1 1" "2.5 1" "9 \'\'"; do eval over_budget $a && echo y || echo n; done',
    )
    assert out.stdout.split() == ["n", "y", "y", "n"]


def test_stop_file_stops_before_the_next_experiment(tmp_path):
    assert run(tmp_path, "check_limits 3; echo went-on").stdout.strip() == "went-on"
    (tmp_path / "stop").touch()
    out = run(tmp_path, "check_limits 3; echo went-on")
    assert out.returncode == 0 and "stop file" in out.stdout and "went-on" not in out.stdout


def test_loop_budget_counts_only_after_an_experiment(tmp_path):
    body = (
        "BUDGET=1; check_limits 1; echo first; "
        "LEDGERS=(a); SESSIONS=(b); check_limits 2; echo second"
    )
    out = run(tmp_path, body, spend="1.25")
    assert "first" in out.stdout and "second" not in out.stdout and "--budget $1" in out.stdout


def test_daily_budget_reads_todays_logs(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    out = run(tmp_path, "DAILY_BUDGET=5; check_limits 1; echo went-on", spend="9")
    assert "went-on" in out.stdout  # no logs today: nothing spent
    day = subprocess.run(["date", "-u", "+%Y%m%d"], capture_output=True, text=True).stdout.strip()
    (logs / f"{day}T000000Z-1.costs.jsonl").write_text("")
    (logs / f"{day}T000000Z-1.json").write_text("{}")
    listed = run(tmp_path, "todays_logs").stdout.split("\n")
    assert listed[0].endswith("-1.costs.jsonl") and listed[1] == "--session"
    out = run(tmp_path, "DAILY_BUDGET=5; check_limits 1; echo went-on", spend="9")
    assert "went-on" not in out.stdout and "--daily-budget $5" in out.stdout


def test_costs_usd_sums_priced_calls_and_sessions(tmp_path):
    ledger = tmp_path / "l.jsonl"
    calls = [
        {"agent": "proposer", "model": "m", "cost_usd": 0.25},
        {"agent": "judge", "model": "x"},
    ]
    ledger.write_text(json.dumps({"command": "audit", "seconds": 1, "llm_calls": calls}) + "\n")
    session = tmp_path / "s.json"
    session.write_text(json.dumps({"total_cost_usd": 1.5}))
    out = CliRunner().invoke(cli, ["costs", "--usd", str(ledger), "--session", str(session)])
    assert out.exit_code == 0 and out.stdout.strip() == "1.7500"
    assert "1 unpriced" in out.stderr
    assert CliRunner().invoke(cli, ["costs", "--usd"]).stdout.strip() == "0.0000"

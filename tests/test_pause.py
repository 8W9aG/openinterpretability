"""``open-interpretability pause``: new experiments wait while a measurement runs.

The measurement needs a quiet machine.
"""

import getpass
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from click.testing import CliRunner

from openinterpretability.cli import cli
from openinterpretability.core import pause

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)


def test_a_pause_is_in_force_until_its_time_ends():
    started = pause.start("timing devices", minutes=30, now=NOW)
    assert pause.read(NOW + timedelta(minutes=29)) == started
    assert pause.read(NOW + timedelta(minutes=31)) is None
    assert not pause.pause_file().exists()  # an ended pause is removed


def test_resume_ends_a_pause():
    pause.start("timing devices", now=NOW)
    assert pause.stop() is True
    assert pause.read(NOW) is None
    assert pause.stop() is False


def test_a_file_that_cant_be_read_is_a_pause_that_ends():
    path = pause.pause_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not json")
    written = datetime.fromtimestamp(path.stat().st_mtime, UTC)
    assert pause.read(written + timedelta(minutes=1)).reason == "the pause file can't be read"
    later = written + timedelta(minutes=pause.DEFAULT_MINUTES + 1)
    assert pause.read(later) is None


def test_the_description_names_the_reason_and_how_to_resume():
    text = pause.start("timing devices", now=NOW).describe()
    assert "timing devices" in text and "open-interpretability resume" in text


def test_next_draws_nothing_while_paused(tmp_path):
    pause.start("timing devices")
    result = CliRunner().invoke(cli, ["next", "--wiki-dir", str(tmp_path), "--no-claims"])
    assert result.exit_code == pause.EXIT_PAUSED
    assert "paused" in result.output and "Next experiment" not in result.output


def test_status_exits_75_while_paused_and_0_otherwise():
    runner = CliRunner()
    assert runner.invoke(cli, ["pause", "--status"]).exit_code == 0
    assert runner.invoke(cli, ["pause", "--reason", "timing", "--for", "5"]).exit_code == 0
    status = runner.invoke(cli, ["pause", "--status"])
    assert status.exit_code == pause.EXIT_PAUSED and "timing" in status.output
    assert "Resumed" in runner.invoke(cli, ["resume"]).output
    assert runner.invoke(cli, ["pause", "--status"]).exit_code == 0


def test_the_loop_script_waits_only_for_exit_75():
    script = (Path(__file__).parents[1] / "scripts" / "experiment-loop.sh").read_text()
    body = script[script.index("wait_while_paused() {") :]
    body = body[: body.index("\n}\n") + 3]

    def run(code: int) -> str:
        stub = f"uv() {{ return {code}; }}\nsleep() {{ echo slept; exit 9; }}\nWORKTREE=.\n"
        out = subprocess.run(
            ["bash", "-c", stub + body + "wait_while_paused 1; echo returned"],
            capture_output=True,
            text=True,
            env={**os.environ},
        )
        return out.stdout

    assert "slept" in run(75)  # paused: it waits
    assert run(0).strip().endswith("returned")  # not paused
    assert run(2).strip().endswith("returned")  # an older openinterpretability without `pause`


def test_a_bare_pause_while_one_stands_only_shows_it(monkeypatch):
    # An agent that runs `pause` to check must not extend the pause (#6839).
    runner = CliRunner()
    runner.invoke(cli, ["pause", "--reason", "timing devices", "--for", "5"])
    before = pause.pause_file().read_bytes()
    result = runner.invoke(cli, ["pause"])
    assert result.exit_code == pause.EXIT_PAUSED and "timing devices" in result.output
    assert pause.pause_file().read_bytes() == before  # nothing changed


def test_a_bare_pause_with_none_standing_sets_one_with_the_defaults():
    result = CliRunner().invoke(cli, ["pause"])
    assert result.exit_code == 0 and "Replaced" not in result.output
    current = pause.read()
    assert current.reason == pause.DEFAULT_REASON
    span = datetime.fromisoformat(current.until) - datetime.fromisoformat(current.since)
    assert span == timedelta(minutes=pause.DEFAULT_MINUTES)


def test_an_explicit_option_replaces_a_standing_pause():
    runner = CliRunner()
    runner.invoke(cli, ["pause", "--reason", "timing devices", "--for", "5"])
    longer = runner.invoke(cli, ["pause", "--for", "30"])
    assert longer.exit_code == 0 and "Replaced the earlier pause" in longer.output
    current = pause.read()
    assert current.reason == pause.DEFAULT_REASON
    span = datetime.fromisoformat(current.until) - datetime.fromisoformat(current.since)
    assert span == timedelta(minutes=30)
    other = runner.invoke(cli, ["pause", "--reason", "a benchmark"])
    assert other.exit_code == 0 and pause.read().reason == "a benchmark"


def test_the_pause_says_where_it_came_from(monkeypatch, tmp_path):
    from labinacan.config import env

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(env("AGENT"), raising=False)
    by = pause.start("timing devices", now=NOW).by
    assert by == f"{getpass.getuser()}, process {os.getpid()}, in {tmp_path}"
    monkeypatch.setenv(env("AGENT"), " agent-1a2b ")
    assert pause.start("timing devices", now=NOW).by == f"{by}, agent agent-1a2b"


def test_a_pause_makes_the_folders_of_its_file(monkeypatch, tmp_path):
    monkeypatch.setenv(pause.ENV, str(tmp_path / "not" / "made" / "yet" / "pause"))
    started = pause.start("timing devices", now=NOW)
    assert pause.read(NOW) == started

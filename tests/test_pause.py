"""``openinterp pause``: new experiments wait while a measurement needs a quiet machine."""

import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from click.testing import CliRunner

from openinterp.cli import cli
from openinterp.core import pause

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
    assert "timing devices" in text and "openinterp resume" in text


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
    assert run(2).strip().endswith("returned")  # an older openinterp without `pause`

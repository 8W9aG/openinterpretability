"""Asking before `explain` spends tokens, and its stage lines (#5830)."""

import json
from pathlib import Path

import click
import pytest

from openinterpretability.explain_confirm import Estimate, confirm, estimate, warning
from openinterpretability.harness.progress import QUIET_ENV, Progress


def _experiment(wiki: Path, name: str, calls: list[dict] | None) -> None:
    path = wiki / "experiments" / "a--m" / "L0" / "N1-positive" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    costs = None if calls is None else {"llm_calls": calls}
    path.write_text(json.dumps({"submission": {"provenance": {"costs": costs}}}))


def test_the_estimate_is_the_mean_of_the_newest_records_with_calls(tmp_path):
    _experiment(tmp_path, "20261101T000000Z.json", [{"prompt_tokens": 900, "completion_tokens": 100,
                                                       "cost_usd": 0.5}])  # fmt: skip
    _experiment(tmp_path, "20261102T000000Z.json", [{"prompt_tokens": 2000, "cost_usd": None}])
    _experiment(tmp_path, "20261103T000000Z.json", None)  # no calls: not counted
    (tmp_path / "experiments" / "a--m" / "bad.json").write_text("{not json")
    found = estimate(tmp_path, "a/m")
    assert found == Estimate(tokens=1500, usd=0.5, records=2)
    assert estimate(tmp_path, "a/m", recent=1) == Estimate(tokens=2000, usd=None, records=1)


def test_the_warning_without_history_or_a_run():
    lines = warning(0, 5, True, Estimate(10, 0.1, 3))
    assert lines[0] == "This calls LLMs: the narrator."
    assert lines[1] == "Estimate: none from the registry yet. This can use a lot of tokens."
    lines = warning(2, 3, False, Estimate(10, None, 3))
    assert lines[1] == "Estimate: 10 tokens an experiment (the mean of 3 records)."
    assert lines[2] == "So 20 to 60 tokens."


def test_yes_goes_on_without_asking():
    confirm(["x"], yes=True, tty=False, ask=lambda *a, **k: pytest.fail("asked"))


def test_with_no_terminal_it_refuses_and_names_the_flag():
    with pytest.raises(click.UsageError, match=r"Add --yes \(-y\)"):
        confirm(["x"], yes=False, tty=False)


def test_on_a_terminal_it_asks_and_a_no_stops():
    asked = []
    confirm(["x"], yes=False, tty=True, ask=lambda *a, **k: asked.append(k) or True)
    assert asked == [{"default": False, "err": True}]  # the default answer is no
    with pytest.raises(click.Abort):
        confirm(["x"], yes=False, tty=True, ask=lambda *a, **k: False)


def test_on_a_terminal_progress_is_a_bar_redrawn_in_place(capsys, monkeypatch):
    monkeypatch.delenv(QUIET_ENV, raising=False)
    now = [0.0]
    progress = Progress("units", 2, clock=lambda: now[0], tty=True)
    now[0] = 1.0
    progress.step()
    now[0] = 1.05  # too soon to redraw, but the last item always draws
    progress.step()
    err = capsys.readouterr().err
    assert err.startswith("\r[##########..........] units: 1/2")
    assert err.endswith("units: 2/2, 1 s so far\n")


def test_quiet_progress_prints_nothing(capsys, monkeypatch):
    monkeypatch.setenv(QUIET_ENV, "1")
    assert Progress("units", 1, interval=0, tty=False).step() is None
    assert capsys.readouterr().err == ""

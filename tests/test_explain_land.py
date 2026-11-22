"""explain --record lands the record as its own PR (explain_land.py, #4475)."""

import subprocess
from datetime import UTC, datetime

import pytest

from openinterpretability import experiment_loop as el
from openinterpretability.core.explanation_records import ExplainRun
from openinterpretability.explain_land import branch_name, land_record, pr_body, write_local
from tests.test_experiment_loop import URL, repo  # noqa: F401  (a fixture)
from tests.test_explanation_records import _record

WHEN = datetime(2026, 11, 17, tzinfo=UTC)


def _landed():
    run = ExplainRun(share_before=0.1, share_after=0.4, stopped="the target was reached",
                     landed=["m/L1/N2-positive VERIFIED https://x/pull/1"],
                     pending=["m/L1/N3-positive VERIFIED https://x/pull/2"],
                     local=["m/L1/N4-positive REJECTED wiki/experiments/r.json"])  # fmt: skip
    return _record(WHEN).model_copy(update={"run": run})


def test_the_pr_body_gives_the_shares_and_the_landed_experiments():
    body = pr_body(_landed())
    assert "**Explained share:** 0.3%" in body and "against 0.0% for random units" in body
    assert "from 10.0% to 40.0%" in body
    assert "landed: m/L1/N2-positive VERIFIED https://x/pull/1" in body
    assert "pending, counted before it merges: m/L1/N3-positive VERIFIED https://x/pull/2" in body
    assert "shared from a local run: m/L1/N4-positive REJECTED wiki/experiments/r.json" in body
    name = branch_name(_landed(), WHEN)
    assert name.startswith("explanation/openai-community--gpt2-")
    assert name.endswith("20261117T000000Z")


def test_the_pr_body_leaves_out_the_coverage_and_the_run_when_the_record_has_neither():
    record = _record(WHEN).model_copy(update={"prompt_coverage": None, "run": None})
    body = pr_body(record)
    assert "**Explained share:** 0.3%" in body
    assert "Prompt coverage" not in body and "The run" not in body
    assert body.endswith("Written by `open-interpretability explain --record` (#4475).")


def test_the_record_lands_on_its_own_branch_without_waiting(repo, monkeypatch):  # noqa: F811
    checked = []
    monkeypatch.setattr(el, "openinterpretability", lambda args: checked.append(args))
    url, merged = land_record(_landed())
    assert (url, merged) == (URL, False) and checked == [["check", "--base-ref", "origin/main"]]
    gh = repo.gh_log.read_text()
    assert "pr create --title Explanation: openai-community/gpt2" in gh
    assert "pr checks" not in gh and "pr merge" not in gh  # auto-merge.yml merges it (#5826)
    files = subprocess.run(["git", "show", "--name-only", "--format=", "HEAD"], capture_output=True,
                           text=True, check=True).stdout  # fmt: skip
    assert files.strip().startswith("wiki/explanations/openai-community--gpt2/")


def test_no_land_writes_locally_and_a_second_run_updates_one_page(tmp_path):
    from openinterpretability.builder.explanation_pages import load_views

    first = write_local(_record(WHEN), tmp_path / "wiki")
    second = write_local(_record(WHEN.replace(hour=5)), tmp_path / "wiki")
    assert first.parent == second.parent and first != second
    (view,) = load_views(tmp_path / "wiki", states={})  # one page, from the newer record
    assert view.record.written.hour == 5


def test_explanation_pages_are_scrubbed_of_personal_data(tmp_path):
    """The site publishes explanations scrubbed, as it does experiments (#4750)."""
    from openinterpretability.builder.explanation_pages import load_views
    from openinterpretability.builder.pii import REMOVED_TEXT

    record = _record(WHEN)
    told = record.narration.model_copy(update={"paragraph": "Login from 8.8.4.4 failed."})
    write_local(record.model_copy(update={"narration": told}), tmp_path / "wiki")
    (view,) = load_views(tmp_path / "wiki", states={})
    assert view.record.narration.paragraph == f"Login from {REMOVED_TEXT} failed."


def test_a_record_that_fails_to_land_is_kept_with_how_to_land_it(tmp_path, monkeypatch):
    # #5965: the record, and its paid-for narration, survive a failed landing.
    import click

    from openinterpretability import cli_explain, explain_land
    from openinterpretability.core.explanation_records import record_path
    from openinterpretability.experiment_loop import LoopError

    def fail(record):
        raise LoopError("git worktree failed")

    monkeypatch.setattr(explain_land, "land_record", fail)
    record = _landed()
    with pytest.raises(click.ClickException, match="explain-share") as caught:
        cli_explain._keep(record, tmp_path, land=True)
    path = record_path(tmp_path, record)
    assert path.exists() and str(path) in caught.value.message
    monkeypatch.setattr(explain_land, "land_record", lambda r: ("https://x/pull/3", False))
    cli_explain._keep(record, tmp_path, land=True)
    assert not path.exists()  # landed: no untracked copy for a later pull to meet

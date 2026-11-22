"""`open-interpretability explain-share` (openinterpretability/explain_share.py, #5827)."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from click.testing import CliRunner

from openinterpretability import cli_loop
from openinterpretability import experiment_loop as el
from openinterpretability.cli import cli
from openinterpretability.core.explanation_records import ExplainRun, ExplanationRecord
from openinterpretability.explain_share import _share_body, _shared, landed_since, share
from tests.test_explanation_records import _record

UNIT = Path("wiki/experiments/toy--m/L1/N2-positive")
LOCAL = UNIT / "20261120T000000Z.json"
URL = "https://github.com/o/r/pull/9"


def _local_record() -> ExplanationRecord:
    run = ExplainRun(share_before=0.1, share_after=0.5, stopped="done",
                     local=[f"toy/m/L1/N2-positive VERIFIED {LOCAL.as_posix()}"])  # fmt: skip
    return _record(datetime(2026, 11, 20, tzinfo=UTC)).model_copy(update={"run": run})


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """The user's checkout with one local record, and a loop worktree with a newer one."""
    home, worktree = tmp_path / "home", tmp_path / "wt"
    (home / UNIT).mkdir(parents=True)
    (home / LOCAL).write_text('{"local": true}')
    (home / LOCAL).with_name(LOCAL.name + ".sig").write_text("sig")
    (worktree / UNIT).mkdir(parents=True)
    (worktree / UNIT / "20261121T000000Z.json").write_text("{}")  # landed on main since
    record_file = home / "explanation.json"
    record_file.write_text(_local_record().model_dump_json())
    calls = []
    monkeypatch.chdir(home)
    monkeypatch.setattr(cli_loop, "loop_worktree", lambda *a: (home, worktree))
    monkeypatch.setattr(el, "reset", lambda: calls.append(["reset"]))
    monkeypatch.setattr(el, "sh", lambda args, **k: calls.append(args) or "")
    monkeypatch.setattr(el, "openinterpretability", lambda args: calls.append(["oi", *args]))
    monkeypatch.setattr(el, "pr_url", lambda: URL)
    monkeypatch.setattr(el, "land", lambda url: calls.append(["land", url]) or False)
    return home, worktree, record_file, calls


def test_share_opens_one_pr_with_the_record_and_its_experiments(checkout):
    home, worktree, record_file, calls = checkout
    assert share(record_file) == (URL, False)
    assert (worktree / LOCAL).read_text() == '{"local": true}'
    assert (worktree / LOCAL).with_name(LOCAL.name + ".sig").read_text() == "sig"
    (written,) = (worktree / "wiki" / "explanations").rglob("*.json")
    moved = ExplanationRecord.model_validate_json(written.read_text()).run
    assert moved.local == []
    assert moved.landed == ["toy/m/L1/N2-positive VERIFIED shared with this explanation"]
    (create,) = [c for c in calls if c[:3] == ["gh", "pr", "create"]]
    body = create[-1]
    assert LOCAL.as_posix() in body and "**Landed since.**" in body
    assert calls[-1] == ["land", URL]  # handed over: no wait for CI and no merge (#5826)
    assert not any("merge" in c or "checks" in c for c in calls)


def test_a_record_the_run_lists_but_the_checkout_lacks_is_an_error(checkout):
    home, _, record_file, _ = checkout
    (home / LOCAL).unlink()
    result = CliRunner().invoke(cli, ["explain-share", str(record_file)])
    assert result.exit_code == 1 and "isn't in this checkout" in result.output


def test_the_command_prints_the_pr(checkout):
    _, _, record_file, _ = checkout
    result = CliRunner().invoke(cli, ["explain-share", str(record_file)])
    assert result.exit_code == 0, result.output
    assert f"Shared: {URL} (open)" in result.output


def test_a_record_with_no_run_is_shared_as_it_is():
    record = _record(datetime(2026, 11, 20, tzinfo=UTC))
    assert _shared(record) is record


def test_only_newer_records_count_as_landed_since(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / UNIT).mkdir(parents=True)
    (tmp_path / UNIT / "20261119T000000Z.json").write_text("{}")  # older: not a clash
    missing = Path("wiki/experiments/other/L0/N1-positive/20261120T000000Z.json")
    assert landed_since([LOCAL, LOCAL.with_name(LOCAL.name + ".sig"), missing]) == []


def test_a_body_with_nothing_landed_since_has_no_such_section():
    body = _share_body(_shared(_local_record()), [LOCAL], [])
    assert "1 experiment record(s)" in body and "Landed since" not in body

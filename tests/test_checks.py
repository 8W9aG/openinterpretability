import subprocess

import pytest

from openinterp.builder.wiki import write_experiment_page
from openinterp.checks import check_registry
from openinterp.core.schemas import AuditRecord


@pytest.fixture
def repo(tmp_path, monkeypatch, record):
    """A minimal registry: one audit record with its wiki page, committed to git."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "audits").mkdir()
    (tmp_path / "audits" / "a.json").write_text(record.model_dump_json())
    write_experiment_page(record, tmp_path / "wiki")
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q", "-b", "main"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "base"], check=True)
    return tmp_path


def messages(report):
    return " | ".join(p.message for p in report.problems)


def page_of(repo):
    return next(p for p in (repo / "wiki").rglob("2*.md"))


def test_consistent_registry_passes(repo):
    report = check_registry(base_ref="main")
    assert report.ok, messages(report)
    assert (report.records, report.pages) == (1, 1)


def test_audit_without_wiki_page(repo):
    page_of(repo).unlink()
    report = check_registry()
    assert "missing wiki page" in messages(report)


def test_wiki_page_without_audit(repo):
    (repo / "audits" / "a.json").unlink()
    assert "has no audit record" in messages(check_registry())


def test_unreviewed_record(repo, submission):
    (repo / "audits" / "a.json").write_text(AuditRecord(submission=submission).model_dump_json())
    assert "no verdict" in messages(check_registry())


def test_invalid_and_duplicate_records(repo, record):
    (repo / "audits" / "dup.json").write_text(record.model_dump_json())
    (repo / "audits" / "bad.json").write_text('{"submission": {}}')
    text = messages(check_registry())
    assert "duplicate record" in text and "invalid audit record" in text


def test_front_matter_must_match_record(repo):
    page = page_of(repo)
    page.write_text(page.read_text().replace('status: "VERIFIED"', 'status: "REJECTED"'))
    text = messages(check_registry())
    assert "front matter status" in text


def test_stale_index(repo):
    (repo / "wiki" / "README.md").write_text("# hand edited\n")
    assert "out of date" in messages(check_registry())


def test_discussion_edits_allowed(repo):
    page = page_of(repo)
    page.write_text(page.read_text() + "\nA note from a reviewer.\n")
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "note"], check=True
    )
    report = check_registry(base_ref="main~1")
    assert report.ok, messages(report)


def test_body_edits_and_deletions_rejected(repo):
    page = page_of(repo)
    page.write_text(page.read_text().replace("## Hypothesis", "## Hypothesis (edited)"))
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "edit"], check=True
    )
    assert "only the Discussion section" in messages(check_registry(base_ref="main~1"))

    subprocess.run(["git", "rm", "-q", str(page)], check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "rm"], check=True
    )
    assert "must not be deleted" in messages(check_registry(base_ref="main~2"))


def test_cli_exit_code(repo):
    from click.testing import CliRunner

    from openinterp.cli import cli

    assert CliRunner().invoke(cli, ["check"]).exit_code == 0
    page_of(repo).unlink()
    result = CliRunner().invoke(cli, ["check"])
    assert result.exit_code == 1
    assert "missing wiki page" in result.output

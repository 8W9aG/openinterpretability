import subprocess
from datetime import timedelta

import pytest

from openinterp.builder.wiki import archive_path, write_experiment_page
from openinterp.checks import check_registry
from openinterp.core.schemas import AuditRecord


@pytest.fixture
def repo(tmp_path, monkeypatch, record):
    """A minimal registry: one experiment (record + page), committed to git."""
    monkeypatch.chdir(tmp_path)
    write_experiment_page(record, tmp_path / "wiki")
    (tmp_path / ".gitignore").write_text("wiki/README.md\nwiki/experiments/**/README.md\n")
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


def _later(record, hours):
    verdict = record.verdict.model_copy(
        update={"timestamp": record.verdict.timestamp + timedelta(hours=hours)}
    )
    return record.model_copy(update={"verdict": verdict})


def test_many_experiments_per_neuron(repo, record):
    for hours in (1, 2):
        write_experiment_page(_later(record, hours), repo / "wiki", kind="reevaluation")
    report = check_registry(base_ref="main")
    assert report.ok, messages(report)
    assert (report.records, report.pages) == (3, 3)


def _revision(record, revises, hours=1):
    later = _later(record, hours)
    sub = later.submission.model_copy(update={"supersedes": revises})
    return later.model_copy(update={"submission": sub})


def test_revision_links_an_earlier_experiment(repo, record):
    first = page_of(repo).stem
    page = write_experiment_page(_revision(record, first), repo / "wiki")
    assert check_registry(base_ref="main").ok
    text = page.read_text()
    assert f'supersedes: "{first}"' in text and f"| Revises | [{first}]({first}.md) |" in text
    write_experiment_page(_revision(record, None, hours=2), repo / "wiki")  # rebuild the index
    neuron = (page.parent / "README.md").read_text()
    assert f"revises [{first}]({first}.md)" in neuron


@pytest.mark.parametrize(
    "revises, message",
    [("20990101T000000Z", "not an earlier experiment"), ("20000101T000000Z", "no such experiment")],
)
def test_revision_must_name_an_earlier_experiment_on_the_unit(repo, record, revises, message):
    write_experiment_page(_revision(record, revises), repo / "wiki")
    assert message in messages(check_registry())


def test_page_and_record_agree_on_supersedes(repo):
    page = page_of(repo)
    page.write_text(page.read_text().replace("---\n", '---\nsupersedes: "x"\n', 1))
    assert "front matter supersedes" in messages(check_registry())


def test_record_without_page(repo):
    page_of(repo).unlink()
    assert "has no page" in messages(check_registry())


def test_page_without_record(repo):
    page_of(repo).with_suffix(".json").unlink()
    assert "missing experiment record" in messages(check_registry())


def test_unreviewed_record(repo, record, submission):
    stray = archive_path(repo / "wiki", record).with_name("x.json")
    stray.write_text(AuditRecord(submission=submission).model_dump_json())
    assert "no verdict" in messages(check_registry())


def test_invalid_and_misplaced_records(repo, record):
    folder = archive_path(repo / "wiki", record).parent
    (folder / "bad.json").write_text('{"submission": {}}')
    (folder / "20200101T000000Z.json").write_text(record.model_dump_json())
    text = messages(check_registry())
    assert "invalid experiment record" in text and "belongs at" in text


def test_legacy_audits_dir_rejected(repo, record):
    (repo / "audits").mkdir()
    (repo / "audits" / "a.json").write_text(record.model_dump_json())
    assert "audits/ is retired" in messages(check_registry())


def test_front_matter_must_match_record(repo):
    page = page_of(repo)
    page.write_text(page.read_text().replace('status: "VERIFIED"', 'status: "REJECTED"'))
    text = messages(check_registry())
    assert "front matter status" in text


def test_committed_index_rejected(repo):
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "add", "-f", "wiki"], check=True)
    subprocess.run([*git, "commit", "-qm", "index"], check=True)
    assert "not committed" in messages(check_registry(base_ref="main~1"))


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
    assert "has no page" in result.output


def test_archive_edits_rejected_by_write_once(repo):
    archived = page_of(repo).with_suffix(".json")
    archived.write_text(archived.read_text() + "\n")
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qam", "touch"], check=True
    )
    assert "archived experiment records are write-once" in messages(
        check_registry(base_ref="main~1")
    )


def test_new_records_spell_agents_canonically(repo, record):
    sub = record.submission.model_copy(
        update={"submitting_agent_model": "external/deepseek-flash[1m]"}
    )
    write_experiment_page(_later(record.model_copy(update={"submission": sub}), 1), repo / "wiki")
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "new"], check=True)
    assert "submitting_agent_model='external/deepseek-flash[1m]': spell it " in messages(
        check_registry(base_ref="main~1")
    )
    assert check_registry().ok  # without a base ref, old records keep their spellings

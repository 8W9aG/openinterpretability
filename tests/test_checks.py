import json
import subprocess
from datetime import timedelta

import pytest

from openinterp.builder.wiki import archive_path, write_experiment_page
from openinterp.checks import check_registry
from openinterp.core.schemas import AuditRecord


@pytest.fixture
def submission(submission):
    """The shared submission, with the model fields that a new record must have."""
    return submission.model_copy(update={"model_num_layers": 2, "model_mlp_width": 8})


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
    report = check_registry()
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
    report = check_registry()
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


def _spelled(record, submitter, reviewer):
    sub = record.submission.model_copy(update={"submitting_agent_model": submitter})
    verdict = record.verdict.model_copy(update={"reviewer_agent_model": reviewer})
    return record.model_copy(update={"submission": sub, "verdict": verdict})


def test_follow_ups_may_keep_the_spelling_of_their_unit(repo, record):
    old_name, git = (
        "external/deepseek-flash[1m]",
        ["git", "-c", "user.email=t@t", "-c", "user.name=t"],
    )
    write_experiment_page(_later(_spelled(record, old_name, old_name), 1), repo / "wiki")
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "old spelling"], check=True)
    subprocess.run([*git, "branch", "base"], check=True)

    # A reevaluation copies both names. An independent review adds a new reviewer.
    write_experiment_page(_later(_spelled(record, old_name, old_name), 2), repo / "wiki")
    write_experiment_page(
        _later(_spelled(record, old_name, "external/claude-opus-5-5"), 3), repo / "wiki"
    )
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "follow-ups"], check=True)
    report = check_registry(base_ref="base")
    assert report.ok, messages(report)


def test_inherited_spelling_needs_the_same_field_in_an_older_record(repo, record):
    old_name, git = (
        "external/deepseek-flash[1m]",
        ["git", "-c", "user.email=t@t", "-c", "user.name=t"],
    )
    write_experiment_page(
        _later(_spelled(record, old_name, "external/claude-opus-5-5"), 1), repo / "wiki"
    )
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "old proposer"], check=True)
    subprocess.run([*git, "branch", "base"], check=True)

    # The reviewer spelling is new. A record of this change cannot excuse itself.
    write_experiment_page(_later(_spelled(record, old_name, old_name), 2), repo / "wiki")
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "new reviewer"], check=True)
    problems = messages(check_registry(base_ref="base"))
    assert "reviewer_agent_model=" in problems
    assert "submitting_agent_model=" not in problems


def _commit_new_record(repo, record):
    """Commit ``record`` one hour later than the base. Return its path."""
    later = _later(record, 1)
    write_experiment_page(later, repo / "wiki")
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "new"], check=True)
    return archive_path(repo / "wiki", later)


def _with_fields(record, **fields):
    sub = record.submission.model_copy(update=fields)
    return record.model_copy(update={"submission": sub})


def test_new_records_need_the_model_fields(repo, record):
    _commit_new_record(repo, _with_fields(record, model_mlp_width=None, model_num_layers=None))
    problems = messages(check_registry(base_ref="main~1"))
    assert "model_num_layers is null" in problems
    assert "model_mlp_width is null" in problems
    assert check_registry().ok  # without a base ref, old records keep their shape


def test_new_records_need_every_model_field_in_the_file(repo, record):
    path = _commit_new_record(repo, record)
    raw = json.loads(path.read_text())
    del raw["submission"]["model_n_heads"]
    path.write_text(json.dumps(raw) + "\n")
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "commit", "-qam", "absent"], check=True)
    assert "model_n_heads is missing" in messages(check_registry(base_ref="main~2"))


def test_only_head_records_have_a_head_count(repo, record):
    _commit_new_record(repo, _with_fields(record, model_n_heads=4))
    assert "model_n_heads must be null on a mlp_neuron_pole" in messages(
        check_registry(base_ref="main~1")
    )


def test_re_review_may_keep_a_null_that_its_unit_has(tmp_path, monkeypatch, record):
    monkeypatch.chdir(tmp_path)
    old = _with_fields(record, model_num_layers=2, model_mlp_width=None)
    write_experiment_page(old, tmp_path / "wiki")
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q", "-b", "main"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", "base"], check=True)
    _commit_new_record(tmp_path, old)
    report = check_registry(base_ref="main~1")
    assert report.ok, messages(report)


def _commit(message="next"):
    git = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-qm", message], check=True)


@pytest.fixture
def old_problem(repo):
    """A registry whose base already holds an invalid record, on a new branch from it."""
    (page_of(repo).parent / "20200101T000000Z.json").write_text("{}")
    _commit("base with a problem")
    subprocess.run(["git", "checkout", "-qb", "branch"], check=True)
    return repo


def test_a_branch_check_reads_only_what_the_branch_changes(old_problem, record):
    assert "invalid experiment record" in messages(check_registry())
    report = check_registry(base_ref="main")
    assert report.ok, messages(report)
    assert report.changed_only and (report.records, report.pages) == (0, 0)

    write_experiment_page(_later(record, 1), old_problem / "wiki", kind="reevaluation")
    _commit()
    report = check_registry(base_ref="main")
    assert report.ok, messages(report)
    assert (report.records, report.pages) == (1, 1)


def test_a_change_outside_the_wiki_checks_every_record(old_problem):
    (old_problem / ".gitignore").write_text("wiki/README.md\n")  # tracked, not yet committed
    report = check_registry(base_ref="main")
    assert not report.changed_only and "invalid experiment record" in messages(report)
    _commit()
    report = check_registry(base_ref="main")
    assert not report.changed_only and "invalid experiment record" in messages(report)


def test_untracked_scratch_files_do_not_force_a_full_check(old_problem):
    (old_problem / "ev.json").write_text("{}")
    assert check_registry(base_ref="main").changed_only


def test_a_branch_check_sees_uncommitted_records(repo, record):
    page = write_experiment_page(_later(record, 1), repo / "wiki", kind="reevaluation")
    page.unlink()
    assert "has no page" in messages(check_registry(base_ref="main"))


def test_a_branch_check_sees_a_page_without_its_record(repo, record):
    page = write_experiment_page(_later(record, 1), repo / "wiki", kind="reevaluation")
    page.with_suffix(".json").unlink()
    assert "missing experiment record" in messages(check_registry(base_ref="main"))


def test_a_branch_check_compares_a_changed_page_with_its_record(repo):
    page = page_of(repo)
    page.write_text(page.read_text().replace('"VERIFIED"', '"REJECTED"', 1))
    text = messages(check_registry(base_ref="main"))
    assert "front matter status" in text and "only the Discussion section" in text


def test_a_branch_check_sees_a_signature_without_a_record(repo):
    (page_of(repo).parent / "20200101T000000Z.json.sig").write_text("sig")
    assert "signature for a missing record" in messages(check_registry(base_ref="main"))


def test_a_branch_check_sees_uncommitted_deletions(repo):
    page = page_of(repo)
    page.with_suffix(".json").unlink()
    assert "write-once" in messages(check_registry(base_ref="main"))
    page.unlink()
    assert "must not be deleted" in messages(check_registry(base_ref="main"))


def test_new_draws_and_stops_are_checked_and_old_ones_skipped(repo):
    draw = repo / "wiki" / "draws" / "m" / "old.json"
    draw.parent.mkdir(parents=True)
    draw.write_text("{}")
    _commit()
    report = check_registry(base_ref="main")
    assert report.ok and report.draws == 0
    new_draw = draw.with_name("new.json")
    new_draw.write_text("{}")
    stop = repo / "wiki" / "stops" / "m" / "new.json"
    stop.parent.mkdir(parents=True)
    stop.write_text("{}")
    text = messages(check_registry(base_ref="main"))
    assert "invalid draw record" in text and "invalid stop record" in text


def test_cli_names_the_scope_and_explains_a_missing_ref(repo):
    from click.testing import CliRunner

    from openinterp.cli import cli

    result = CliRunner().invoke(cli, ["check", "--base-ref", "main"])
    assert result.exit_code == 0 and "changed since main" in result.output
    result = CliRunner().invoke(cli, ["check", "--base-ref", "origin/nope"])
    assert result.exit_code != 0 and "can't diff against origin/nope" in result.output
    assert "Traceback" not in result.output

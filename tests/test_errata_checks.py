from openinterpretability.builder.wiki import write_experiment_page
from openinterpretability.checks import CheckReport
from openinterpretability.core.errata import errata_path, record_ref
from openinterpretability.core.paths import experiments_dir
from openinterpretability.errata_checks import _check_new_errata, changed_file, check_errata
from openinterpretability.signing import signature_path
from tests.test_errata import erratum


def messages(report):
    return [p.message for p in report.problems]


def test_a_signature_without_its_errata_file_is_a_problem(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    signature_path(errata_path(wiki)).write_text("sig")
    report = CheckReport()
    check_errata(wiki, report, None, [], tmp_path / "allowed_signers")
    assert messages(report) == ["signature of an errata file that does not exist"]


def test_an_errata_file_that_does_not_parse_is_a_problem(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    errata_path(wiki).write_text("not json\n")
    report = CheckReport()
    check_errata(wiki, report, None, [], tmp_path / "allowed_signers")
    (message,) = messages(report)
    assert message.startswith("invalid erratum: line 1:")


def test_with_a_base_ref_an_unchanged_file_is_not_changed(tmp_path):
    assert changed_file(tmp_path / "errata.jsonl", "main", []) is False


def test_an_invalid_record_is_skipped_when_new_errata_are_checked(tmp_path, monkeypatch, record):
    monkeypatch.chdir(tmp_path)  # not a git clone, so no commit is found
    wiki = tmp_path / "wiki"
    write_experiment_page(record, wiki)
    (experiments_dir(wiki) / "broken.json").write_text("{")
    report = CheckReport()
    x = erratum(records=[record_ref(record)])
    _check_new_errata([x], experiments_dir(wiki), report, errata_path(wiki))
    # The valid record is covered, and the broken file does not stop the check.
    assert messages(report) == ["erratum scale-bug: no commit abcdef1 in this clone"]

"""The takedown file's part of `open-interpretability check` (takedown_checks.py)."""

from openinterpretability.checks import CheckReport
from openinterpretability.core.takedowns import takedowns_path
from openinterpretability.signing import signature_path
from openinterpretability.takedown_checks import check_takedowns


def test_deleting_the_takedown_file_is_a_problem(tmp_path):
    path = takedowns_path(tmp_path)
    report = CheckReport()
    check_takedowns(tmp_path, report, None, [("D", path)], tmp_path / "allowed_signers")
    assert [(p.path, p.message) for p in report.problems] == [
        (path, "the takedown file is append-only and must not be deleted")
    ]


def test_a_signature_without_its_takedown_file_is_a_problem(tmp_path):
    sig = signature_path(takedowns_path(tmp_path))
    sig.write_text("signature")
    report = CheckReport()
    check_takedowns(tmp_path, report, None, [], tmp_path / "allowed_signers")
    assert [(p.path, p.message) for p in report.problems] == [
        (sig, "signature of a takedown file that does not exist")
    ]

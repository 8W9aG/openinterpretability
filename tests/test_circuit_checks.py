"""Circuit record checks: scope, invalid files and records of an older rule version."""

from openinterpretability.checks import CheckReport
from openinterpretability.circuit_checks import check_circuits
from openinterpretability.core import circuit_records as cr
from tests.test_circuit_records import _record, _write


def test_records_outside_the_scope_are_not_checked(tmp_path):
    bad = tmp_path / cr.CIRCUITS_WIKI / "toy--model" / "C1" / "bad.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{}")
    report = CheckReport()
    check_circuits(tmp_path, report, scope=set())
    assert report.ok


def test_a_file_that_is_not_a_circuit_record_is_reported(tmp_path):
    bad = tmp_path / cr.CIRCUITS_WIKI / "toy--model" / "C1" / "bad.json"
    bad.parent.mkdir(parents=True)
    bad.write_text('{"hypothesis": 1}')
    report = CheckReport()
    check_circuits(tmp_path, report, scope={bad})
    assert not report.ok and "not a valid circuit record" in str(report)


def test_a_record_of_an_older_rule_version_keeps_its_status(tmp_path):
    record = _record(status="REJECTED")  # the measures give VERIFIED under the current rules
    policy = record.verdict.policy.model_copy(update={"protocol_version": "0.0"})
    verdict = record.verdict.model_copy(update={"policy": policy})
    _write(tmp_path, record.model_copy(update={"verdict": verdict}))
    report = CheckReport()
    check_circuits(tmp_path, report)
    assert report.ok

"""Loading and checking answer records (core/answer_records.py)."""

from openinterpretability.checks import CheckReport
from openinterpretability.core import answer_records as ar
from tests.test_answer_records import _measures, _record


def test_an_invalid_answer_file_is_left_out_when_loading(tmp_path):
    good = _record()
    path = ar.record_path(tmp_path, good)
    path.parent.mkdir(parents=True)
    path.write_text(good.model_dump_json())
    (path.parent / "broken.json").write_text('{"claim": {}}')
    assert ar.load_paths(tmp_path, good.claim.model_name, good.claim.prompt_id) == [(path, good)]


def test_check_reports_a_file_that_is_not_an_answer_record(tmp_path):
    folder = tmp_path / ar.ANSWERS_WIKI / "m" / "p"
    folder.mkdir(parents=True)
    (folder / "bad.json").write_text('{"claim": {}}')
    (folder / "binary.json").write_bytes(b"\xff\xfe\x00")
    report = CheckReport()
    ar.check_answers(tmp_path, report)
    found = {p.path.name: p.message for p in report.problems}
    assert set(found) == {"bad.json", "binary.json"}
    assert all(m.startswith("not a valid answer record: ") for m in found.values())


def test_a_record_under_an_older_rule_is_not_rederived(tmp_path):
    record = _record(_measures(patching=0.0))  # the current rule would reject it
    policy = record.verdict.policy.model_copy(update={"protocol_version": "1.0"})
    old = record.model_copy(update={"verdict": record.verdict.model_copy(
        update={"policy": policy, "status": "VERIFIED"})})  # fmt: skip
    path = ar.record_path(tmp_path, old)
    path.parent.mkdir(parents=True)
    path.write_text(old.model_dump_json())
    report = CheckReport()
    ar.check_answers(tmp_path, report)
    assert report.problems == []

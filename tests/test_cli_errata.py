"""``open-interpretability erratum draft``: edits given as text, bad ids and unknown records."""

import json

from openinterpretability.cli_errata import _edit
from tests.test_errata_readers import _draft
from tests.test_planner import _record, _registry


def _wiki(tmp_path, record):
    return _registry(tmp_path, _record(record, status="VERIFIED", idx=1))


def test_an_edit_value_that_is_not_json_is_kept_as_text():
    assert _edit("a.b=3") == {"path": "a.b", "value": 3}
    assert _edit("a.b=Fires on x") == {"path": "a.b", "value": "Fires on x"}


def test_a_draft_with_a_text_edit_prints_it_as_a_string(tmp_path, record):
    model = record.submission.identifier.model_name
    path = "submission.natural_language_explanation"
    done = _draft(_wiki(tmp_path, record), "--id", "x", "--model", model,
                  "--edit", f"{path}=Fires on quotes")  # fmt: skip
    assert done.exit_code == 0, done.output
    line = json.loads(done.output.strip().splitlines()[-1])
    assert line["edits"] == [{"path": path, "value": "Fires on quotes"}]


def test_an_edit_without_an_equals_sign_is_refused(tmp_path, record):
    done = _draft(_wiki(tmp_path, record), "--id", "x", "--edit", "verdict.status")
    assert done.exit_code == 2 and "'verdict.status' is not PATH=VALUE" in done.output


def test_an_invalid_erratum_is_refused(tmp_path, record):
    model = record.submission.identifier.model_name
    done = _draft(_wiki(tmp_path, record), "--id", "Not_Valid", "--model", model, "--set-aside")
    assert done.exit_code == 1 and "the erratum is not valid" in done.output


def test_a_listed_record_not_in_the_archive_is_refused(tmp_path, record):
    missing = "wiki/experiments/no-such/record.json"
    done = _draft(_wiki(tmp_path, record), "--id", "x", "--record", missing, "--set-aside")
    assert done.exit_code == 1
    assert f"listed records not in the archive: {missing}" in done.output

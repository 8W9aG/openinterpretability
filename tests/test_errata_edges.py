"""Edges of the errata rules and edits that mutation testing found untested (#6810).

These tests are fast, so ``[tool.mutmut]`` selects them for ``core/errata.py``.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from openinterpretability.core.errata import (
    Erratum,
    ErratumRule,
    _replace,
    apply_errata,
    corrected,
    load_errata,
    parse_errata,
    record_ref,
    standing_records,
)
from openinterpretability.core.schemas import AuditRecord, ExperimentProvenance, ReviewPolicy


def with_provenance(record: AuditRecord, harness: str) -> AuditRecord:
    prov = ExperimentProvenance(
        dtype="float32", corpus="test", n_snippets=1, tokens_scanned=1, selection="random",
        harness_version=harness,
    )  # fmt: skip
    return record.model_copy(
        update={"submission": record.submission.model_copy(update={"provenance": prov})}
    )


def with_protocol(record: AuditRecord, version: str) -> AuditRecord:
    verdict = record.verdict.model_copy(update={"policy": ReviewPolicy(protocol_version=version)})
    return record.model_copy(update={"verdict": verdict})


def test_a_harness_rule_reads_the_records_provenance(record):
    stamped = with_provenance(record, "1.2.3")
    assert ErratumRule(harness_versions=["1.2.3"]).matches(stamped)
    assert not ErratumRule(harness_versions=["9.9.9"]).matches(stamped)
    assert not ErratumRule(harness_versions=["1.2.3"]).matches(record)  # no provenance


def test_a_protocol_rule_reads_the_verdicts_policy(record):
    judged = with_protocol(record, "0.5")
    assert ErratumRule(protocol_versions=["0.5"]).matches(judged)
    assert not ErratumRule(protocol_versions=["0.6"]).matches(judged)


def test_a_record_with_no_verdict_matches_no_version_or_date_rule(record):
    bare = record.model_copy(update={"verdict": None})
    assert not ErratumRule(protocol_versions=["0.5"]).matches(bare)
    assert not ErratumRule(reviewed_from=record.verdict.timestamp).matches(bare)
    assert not ErratumRule(reviewed_before=record.verdict.timestamp).matches(bare)
    assert ErratumRule(models=[record.submission.identifier.model_name]).matches(bare)


def test_an_edit_can_end_at_a_list_index():
    data = {"a": [1, 2, {"b": 3}]}
    assert _replace(data, "a.1", 9) == 2
    assert _replace(data, "a.2.b", 4) == 3
    assert data == {"a": [1, 9, {"b": 4}]}


@pytest.mark.parametrize("path", ["a.5", "a.x", "missing", "a.0.b"])
def test_an_edit_outside_the_record_is_refused(path):
    with pytest.raises(ValueError, match=f"^the record has no field {path}$"):
        _replace({"a": [1]}, path, 0)


def erratum(id_: str = "scale-bug", **fields) -> Erratum:
    base = {"id": id_, "date": "2026-11-13", "title": "A bug", "bug": "What it did.",
            "fixed_in": "abcdef1"}  # fmt: skip
    if "edits" not in fields:
        base["set_aside"] = True
    return Erratum.model_validate_json(json.dumps(base | fields))


def test_edits_apply_in_order_and_say_what_they_changed(record):
    first = erratum("first", records=[record_ref(record)],
                    edits=[{"path": "verdict.status", "value": "REJECTED"}])  # fmt: skip
    second = erratum("second", records=[record_ref(record)],
                     edits=[{"path": "verdict.status", "value": "NEEDS_REVISION"}])  # fmt: skip
    edited, applied = apply_errata(record, [first, second])
    assert edited.verdict.status == "NEEDS_REVISION" and record.verdict.status == "VERIFIED"
    assert [(a.erratum, a.path, a.before, a.after) for a in applied] == [
        ("first", "verdict.status", "VERIFIED", "REJECTED"),
        ("second", "verdict.status", "REJECTED", "NEEDS_REVISION"),
    ]


def test_an_erratum_with_no_edit_returns_the_record_itself(record):
    edited, applied = apply_errata(record, [erratum(records=[record_ref(record)])])
    assert edited is record and applied == []


def test_an_edit_that_breaks_the_record_is_refused(record):
    bad = erratum(records=[record_ref(record)], edits=[{"path": "verdict.status", "value": "X"}])
    with pytest.raises(ValueError, match="^the edited record is not valid: "):
        apply_errata(record, [bad])


def test_corrected_says_whether_a_covering_erratum_sets_the_record_aside(record):
    ref = record_ref(record)
    fix = erratum("fix", records=[ref], edits=[{"path": "verdict.status", "value": "REJECTED"}])
    assert corrected(record, [fix])[0].verdict.status == "REJECTED"
    assert corrected(record, [fix])[1] is False
    assert corrected(record, [erratum("aside", records=[ref])]) == (record, True)
    assert corrected(record, [erratum("other", records=["elsewhere.json"])]) == (record, False)


def other_unit(record: AuditRecord) -> AuditRecord:
    ident = record.submission.identifier.model_copy(update={"neuron_idx": 999})
    return record.model_copy(
        update={"submission": record.submission.model_copy(update={"identifier": ident})}
    )


def test_standing_records_leave_out_set_aside_records_unless_the_unit_has_no_other(record):
    later = record.model_copy(update={"verdict": record.verdict.model_copy(
        update={"timestamp": record.verdict.timestamp.replace(year=2099)})})  # fmt: skip
    lone = other_unit(record)
    aside = erratum(records=[record_ref(record), record_ref(lone)])
    # The first unit keeps its later record; the second has only a set-aside one, so it stays.
    assert standing_records([record, later, lone], [aside]) == [later, lone]
    assert standing_records([lone, record], []) == [lone, record]


def test_an_errata_file_is_read_line_by_line(tmp_path):
    lines = [erratum("a", records=["x.json"]), erratum("b", records=["y.json"])]
    (tmp_path / "errata.jsonl").write_text(
        "\n".join(e.model_dump_json() for e in lines) + "\n", encoding="utf-8"
    )
    assert [e.id for e in load_errata(tmp_path)] == ["a", "b"]
    assert load_errata(tmp_path / "none") == []
    with pytest.raises(ValueError, match="^line 2: "):
        parse_errata(lines[0].model_dump_json() + "\n{}")


def test_a_date_rule_takes_reviewed_from_and_leaves_out_reviewed_before(record):
    t = record.verdict.timestamp
    second = timedelta(seconds=1)
    assert ErratumRule(reviewed_from=t).matches(record)
    assert not ErratumRule(reviewed_from=t + second).matches(record)
    assert ErratumRule(reviewed_before=t + second).matches(record)
    assert not ErratumRule(reviewed_before=t).matches(record)

"""Errata reach every reader of records, and ``erratum draft`` writes them (issue #3271)."""

import ast
import json
from datetime import timedelta
from pathlib import Path

from click.testing import CliRunner

from openinterp.cli import cli
from openinterp.core.errata import errata_path, load_errata, record_ref, standing_records
from openinterp.planner import load_states, plan_next
from tests.test_errata import erratum, later
from tests.test_planner import NOW, _kinds, _record, _registry

ROOT = Path(__file__).parents[1]


def test_standing_records_leave_out_set_aside_records_unless_all_are(record):
    first = _record(record, status="VERIFIED", idx=1)
    second = later(first, 1, status="REJECTED")
    alone = _record(record, status="VERIFIED", idx=2)
    errata = [erratum(records=[record_ref(second), record_ref(alone)])]
    kept = standing_records([first, second, alone], errata)
    assert kept == [first, alone]  # alone's only record is set aside, so it still counts


def test_standing_records_apply_edits(record):
    rec = _record(record, status="VERIFIED", idx=1)
    edit = {"path": "verdict.status", "value": "REJECTED"}
    (edited,) = standing_records([rec], [erratum(records=[record_ref(rec)], edits=[edit])])
    assert edited.verdict.status == "REJECTED" and rec.verdict.status == "VERIFIED"


def test_the_planner_skips_a_set_aside_record(tmp_path, record):
    older = _record(record, status="NEEDS_REVISION", gates={"sensitivity": False}, idx=1)
    newer = _record(record, status="VERIFIED", gates={}, idx=1)
    when = older.verdict.timestamp + timedelta(hours=1)
    verdict = newer.verdict.model_copy(update={"timestamp": when})
    newer = newer.model_copy(update={"verdict": verdict})
    wiki = _registry(tmp_path, newer, older)
    errata_path(wiki).write_text(erratum(records=[record_ref(newer)]).model_dump_json() + "\n")
    state = load_states(wiki)[newer.submission.identifier.slug]
    assert state.record.verdict.status == "NEEDS_REVISION" and state.experiments == 2
    kinds = {kind for kind, _ in _kinds(plan_next(wiki, seed=1, epsilon=0.0, now=NOW))}
    assert "replicate" not in kinds and "revise" in kinds


def test_every_build_tree_call_passes_errata():
    """``build_tree`` has no default for ``errata``; this also catches a call by keyword."""
    missing = []
    for folder in ("openinterp", "scripts"):
        for path in (ROOT / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "id", getattr(node.func, "attr", None))
                given = len(node.args) >= 2 or any(k.arg == "errata" for k in node.keywords)
                if name == "build_tree" and not given:
                    missing.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not missing, f"build_tree without errata: {missing}"


def _draft(wiki, *args):
    base = ["erratum", "draft", "--wiki-dir", str(wiki), "--title", "t", "--bug", "b"]
    return CliRunner().invoke(cli, [*base, "--fixed-in", "HEAD", *args])


def test_erratum_draft_lists_the_covered_records_and_writes_only_when_asked(tmp_path, record):
    model = record.submission.identifier.model_name
    wiki = _registry(tmp_path, _record(record, status="VERIFIED", idx=1),
                     _record(record, status="REJECTED", idx=2))  # fmt: skip
    done = _draft(wiki, "--id", "scale-bug", "--model", model, "--set-aside")
    assert done.exit_code == 0, done.output
    assert "covers 2 records" in done.output
    line = json.loads(done.output.strip().splitlines()[-1])
    assert line["rule"] == {"models": [model]} and len(line["fixed_in"]) == 40
    assert not errata_path(wiki).exists()
    written = _draft(wiki, "--id", "scale-bug", "--model", model, "--set-aside", "--write")
    assert written.exit_code == 0, written.output
    assert [e.id for e in load_errata(wiki)] == ["scale-bug"]
    again = _draft(wiki, "--id", "scale-bug", "--model", model, "--set-aside")
    assert again.exit_code != 0 and "already in the errata file" in again.output


def test_erratum_draft_refuses_what_check_would(tmp_path, record):
    model = record.submission.identifier.model_name
    wiki = _registry(tmp_path, _record(record, status="VERIFIED", idx=1))
    bad_edit = _draft(wiki, "--id", "x", "--model", model, "--edit", "verdict.nothing=1")
    assert bad_edit.exit_code != 0 and "has no field verdict.nothing" in bad_edit.output
    nothing = _draft(wiki, "--id", "x", "--model", "no/such-model", "--set-aside")
    assert nothing.exit_code != 0 and "covers no record" in nothing.output
    no_commit = CliRunner().invoke(cli, ["erratum", "draft", "--wiki-dir", str(wiki), "--id", "x",
                                         "--title", "t", "--bug", "b", "--fixed-in", "0000000",
                                         "--model", model, "--set-aside"])  # fmt: skip
    assert no_commit.exit_code != 0 and "is not a commit" in no_commit.output

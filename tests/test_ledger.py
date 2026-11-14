"""The draw ledger (issue #3205): written at draw time, committed by the claim, resolved later."""

import json
from datetime import UTC, datetime, timedelta

from click.testing import CliRunner

from openinterp.cli import cli
from openinterp.core.ledger import (
    DrawEntry,
    ledger_path,
    load_ledger,
    resolve,
    summary,
    write_entry,
)
from openinterp.evidence import ScanRequest, pole_forced

T = datetime(2026, 11, 14, 1, 0, tzinfo=UTC)
MODEL = "Qwen/Qwen2.5-0.5B"
SLUG = "Qwen--Qwen2.5-0.5B/L3/N7-positive"


def explore(seed=11, audit_seed=42, slug=None):
    audit = f"--model {MODEL} --random --seed {audit_seed}"
    return DrawEntry(timestamp=T, seed=seed, kind="explore", model=MODEL, slug=slug, audit=audit)


def test_entries_are_write_once_by_day_and_seed(tmp_path):
    entry = explore()
    path = write_entry(entry, tmp_path)
    assert path == tmp_path / "ledger" / "20261114" / "20261114T010000Z-11.json"
    assert ledger_path(tmp_path, entry) == path
    write_entry(entry.model_copy(update={"kind": "revise"}), tmp_path)  # left as it was
    assert load_ledger(tmp_path) == [entry]


def test_a_random_draw_matches_its_audit_seed_and_model():
    entry = explore()
    record = (SLUG, 42, T + timedelta(hours=1), "VERIFIED")
    other_model = ("openai-community--gpt2/L3/N7-positive", 42, T, "VERIFIED")
    assert resolve(entry, [record], []) == ("record", "VERIFIED")
    assert resolve(entry, [other_model], []) == ("none", None)


def test_a_known_slug_matches_records_from_its_time_on():
    entry = explore(audit_seed=0, slug=SLUG).model_copy(update={"audit": None, "kind": "revise"})
    before = (SLUG, None, T - timedelta(days=1), "REJECTED")
    after = (SLUG, None, T + timedelta(hours=2), "NEEDS_REVISION")
    assert resolve(entry, [before], []) == ("none", None)
    assert resolve(entry, [before, after], []) == ("record", "NEEDS_REVISION")


def test_a_stopped_draw_and_a_dropped_one():
    entry = explore()
    assert resolve(entry, [], [(SLUG, 42, T, None)]) == ("stopped", None)
    assert resolve(entry, [], []) == ("none", None)


def test_summary_brackets_the_share_with_dropped_draws():
    a, b, c = explore(seed=1), explore(seed=2), explore(seed=3)
    got = summary([(a, "record", "VERIFIED"), (b, "record", "REJECTED"), (c, "none", None)])
    assert got["outcomes"] == {"record": 2, "stopped": 0, "none": 1}
    assert got["verified_share_of_all_explore"] == 1 / 3
    assert got["verified_share_of_finished_explore"] == 1 / 2


def test_provenance_says_whether_the_pole_was_forced():
    assert pole_forced(ScanRequest(model_name=MODEL, random_pick=True, pole="negative"))
    assert pole_forced(ScanRequest(model_name=MODEL, random_pick=True)) is False
    assert pole_forced(ScanRequest(model_name=MODEL, layer=3, neuron=7)) is None
    head = ScanRequest(model_name=MODEL, random_pick=True, unit="head")
    assert pole_forced(head) is None


def test_next_steps_writes_the_entry_and_the_claim_commits_it(tmp_path):
    args = ["next", "--wiki-dir", str(tmp_path), "--no-claims", "--steps", "--seed", "5"]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, result.output
    (path,) = list((tmp_path / "ledger").rglob("*.json"))
    entry = DrawEntry.model_validate_json(path.read_text())
    assert entry.seed == 5 and entry.kind == "explore"
    assert f"git add {path}" in result.output
    assert "--allow-empty" not in result.output
    assert "land it with only its ledger entry" in result.output


def test_next_without_steps_records_nothing(tmp_path):
    result = CliRunner().invoke(cli, ["next", "--wiki-dir", str(tmp_path), "--no-claims"])
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "ledger").exists()


def test_ledger_command_reports_outcomes(tmp_path):
    write_entry(explore(), tmp_path)
    result = CliRunner().invoke(cli, ["ledger", "--wiki-dir", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    got = json.loads(result.output)
    assert got["entries"] == 1 and got["outcomes"]["none"] == 1


def test_check_reads_ledger_entries_and_flags_a_misplaced_one(tmp_path):
    from openinterp.checks import check_registry

    path = write_entry(explore(), tmp_path)
    legacy = tmp_path / "no-legacy"
    report = check_registry(tmp_path, legacy_audits_dir=legacy)
    assert report.ok and report.ledger == 1
    path.rename(path.with_name("elsewhere.json"))
    report = check_registry(tmp_path, legacy_audits_dir=legacy)
    assert not report.ok and "ledger entry belongs at" in str(report.problems)

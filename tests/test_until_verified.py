"""Until-verified draws keep going on one unit, on one PR (until_verified.py, #4485)."""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest

from openinterpretability import experiment_loop as el
from openinterpretability import until_verified as uv
from openinterpretability.core.paths import experiments_dir
from openinterpretability.revision import previous_attempt

SLUG = "toy/model/L0/N1-positive"


def test_the_mode_is_seeded_forced_or_drawn_at_the_share():
    assert uv.decide(7) == uv.decide(7)  # the same seed draws the same mode
    assert uv.decide(7, True) and not uv.decide(7, False)
    share = sum(uv.decide(seed) for seed in range(4000)) / 4000
    assert abs(share - uv.UNTIL_VERIFIED_SHARE) < 0.03


@pytest.mark.parametrize(
    ("kind", "verdict", "turn", "expected"),
    [
        ("explore", "NEEDS_REVISION", 1, True),
        ("explore", "VERIFIED", 1, False),  # success
        ("explore", "STOPPED", 1, False),  # the unit is dead or default-on
        ("explanation", "EXPLAINED", 1, False),  # not a unit experiment
        ("revise", "REJECTED", 3, False),  # the last turn
    ],
)
def test_what_ends_a_sequence(kind, verdict, turn, expected):
    assert uv.keep_going(kind, verdict, turn, max_turns=3) is expected


def _turns(monkeypatch, verdicts, follow_up=True):
    """Run more_turns after a first verdict; later turns take ``verdicts`` in order."""
    later = iter(verdicts[1:])
    calls: list[str] = []
    plan = SimpleNamespace(choice=SimpleNamespace(kind="revise"), seed=0)
    monkeypatch.setattr(el, "next_turn", lambda slug, wiki: plan if follow_up else None)
    monkeypatch.setattr(el, "steps", lambda *a, **k: ["claim", "work", "post"])
    monkeypatch.setattr(el, "say", lambda text: None)

    def run_work(work, kind, key):
        calls.append(kind)
        return next(later)

    monkeypatch.setattr(el, "run_work", run_work)
    options = el.Options("p", "r", until_verified=True, max_turns=3)
    return el.more_turns(options, "explore", SLUG, verdicts[0]), calls


def test_verified_on_the_first_turn_takes_no_more(monkeypatch):
    assert _turns(monkeypatch, ["VERIFIED"]) == ("VERIFIED", [])


def test_verified_on_a_later_turn_stops_there(monkeypatch):
    assert _turns(monkeypatch, ["NEEDS_REVISION", "VERIFIED"]) == ("VERIFIED", ["revise"])


def test_max_turns_without_verified_stops_at_the_last(monkeypatch):
    status, calls = _turns(monkeypatch, ["NEEDS_REVISION", "NEEDS_REVISION", "REJECTED"])
    assert status == "REJECTED" and calls == ["revise", "revise"]  # turns 2 and 3 of 3


def test_a_terminal_verdict_or_no_follow_up_ends_early(monkeypatch):
    assert _turns(monkeypatch, ["NEEDS_REVISION", "STOPPED"]) == ("STOPPED", ["revise"])
    assert _turns(monkeypatch, ["NEEDS_REVISION"], follow_up=False) == ("NEEDS_REVISION", [])


def test_each_turns_proposer_sees_every_earlier_turn(record, tmp_path):
    """Turn 3's brief covers turn 2 (the superseded record) and turn 1, through supersedes."""
    from openinterpretability.builder.wiki import experiment_id

    def attempt(text, shift, supersedes=None):
        sub = record.submission.model_copy(
            update={"natural_language_explanation": text, "supersedes": supersedes}
        )
        verdict = record.verdict.model_copy(
            update={"timestamp": record.verdict.timestamp + timedelta(minutes=shift)}
        )
        return record.model_copy(update={"submission": sub, "verdict": verdict})

    first = attempt("Turn one: fires on dates.", 0)
    second = attempt("Turn two: fires on years.", 1, experiment_id(first))
    folder = experiments_dir(tmp_path) / first.submission.identifier.slug
    folder.mkdir(parents=True)
    for r in (first, second):
        (folder / f"{experiment_id(r)}.json").write_text(r.model_dump_json(), encoding="utf-8")
    brief = previous_attempt(second, tmp_path)
    assert "Turn two" in brief and "Attempt 2 before this one" in brief and "Turn one" in brief
    assert brief.index("Turn two") < brief.index("Turn one")  # newest first
    assert "Turn one" not in previous_attempt(second)  # with no wiki, only the superseded one


def test_next_steps_prints_the_rule_and_records_the_mode(tmp_path, monkeypatch):
    import json

    from click.testing import CliRunner

    from openinterpretability.cli import cli
    from openinterpretability.core.ledger import load_ledger

    wiki = tmp_path / "wiki"
    args = ["next", "--wiki-dir", str(wiki), "--seed", "3", "--no-claims", "--model", "gpt2"]
    out = CliRunner().invoke(cli, [*args, "--until-verified", "--max-turns", "2", "--steps"])
    assert out.exit_code == 0, out.output
    assert "Until verified: yes, at most 2 turns" in out.output
    assert "next --continue" in out.output and "turn 2" in out.output
    (entry,) = load_ledger(wiki)
    assert entry.until_verified and entry.max_turns == 2
    plain = CliRunner().invoke(cli, [*args, "--no-until-verified", "--json"])
    assert json.loads(plain.output)["until_verified"] is False


def test_the_next_turn_revises_the_last_record(record, tmp_path):
    from click.testing import CliRunner

    from openinterpretability.builder.wiki import experiment_id
    from openinterpretability.cli import cli

    verdict = record.verdict.model_copy(update={"status": "NEEDS_REVISION"})
    needs = record.model_copy(update={"verdict": verdict})
    slug = needs.submission.identifier.slug
    folder = experiments_dir(tmp_path) / slug
    folder.mkdir(parents=True)
    (folder / f"{experiment_id(needs)}.json").write_text(needs.model_dump_json(), encoding="utf-8")
    plan = uv.next_turn(slug, tmp_path)
    assert plan is not None and plan.choice.kind == "revise"
    out = CliRunner().invoke(cli, ["next", "--wiki-dir", str(tmp_path), "--continue", slug])
    assert out.exit_code == 0, out.output
    assert "Next turn: revise" in out.output and "no claim" in out.output
    assert "Claim" not in out.output.split("Steps", 1)[1]
    missing = CliRunner().invoke(
        cli, ["next", "--wiki-dir", str(tmp_path), "--continue", "x/L0/N1"]
    )
    assert missing.exit_code != 0 and "no superseding follow-up" in missing.output


def test_a_verified_unit_has_no_next_turn(record, tmp_path):
    from openinterpretability.builder.wiki import experiment_id

    assert record.verdict.status == "VERIFIED"
    slug = record.submission.identifier.slug
    folder = experiments_dir(tmp_path) / slug
    folder.mkdir(parents=True)
    (folder / f"{experiment_id(record)}.json").write_text(
        record.model_dump_json(), encoding="utf-8"
    )
    assert uv.next_turn(slug, tmp_path) is None
    assert uv.next_turn("x/L0/N1", tmp_path) is None  # no such unit

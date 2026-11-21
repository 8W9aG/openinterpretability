"""``circuit`` draws: circuits with no experiment, or whose members moved (#4286)."""

import json
from pathlib import Path

import pytest

from openinterpretability import draw_steps
from openinterpretability import experiment_loop as el
from openinterpretability.planner import Plan, claim_key
from openinterpretability.planner_circuits import CIRCUIT_WEIGHT, circuit_candidates

C4 = next(Path("wiki/circuits/openai-community--gpt2/C4").glob("*.json"))
SLUG = "openai-community--gpt2"


def _root(tmp_path, circuits: dict, record_members=None):
    """A repo root with these circuits, and C4's real record made for ``record_members``."""
    data = tmp_path / "data" / "circuits"
    data.mkdir(parents=True)
    lines = [{"date": "2026-11-16", "event": "new", "id": cid, "members": members}
             for cid, members in circuits.items()]  # fmt: skip
    (data / f"{SLUG}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    if record_members is not None:
        record = json.loads(C4.read_text())
        record["hypothesis"]["cluster_members"] = record_members
        out = tmp_path / "wiki" / "circuits" / SLUG / "C4" / C4.name
        out.parent.mkdir(parents=True)
        out.write_text(json.dumps(record))
    return tmp_path


def _page(unit):
    return f"models/{SLUG}/{unit}.html"


NOW = [_page("L4/N2999"), _page("L5/N1208"), _page("L5/N2567"), _page("L6/H3")]


def test_a_circuit_with_no_experiment_is_drawn_at_full_weight(tmp_path):
    (c,) = circuit_candidates([], _root(tmp_path, {"C1": NOW}))
    assert (c.kind, c.slug, c.weight, c.record) == (
        "circuit",
        f"{SLUG}/circuits/C1",
        CIRCUIT_WEIGHT,
        None,
    )
    assert c.model == "openai-community/gpt2" and "--circuit C1" in c.commands[0]


def test_a_current_circuit_is_not_drawn_and_a_moved_one_is_by_its_change(tmp_path):
    current = _root(tmp_path / "a", {"C4": NOW}, record_members=NOW)
    assert circuit_candidates([], current) == []
    moved = _root(tmp_path / "b", {"C4": NOW}, record_members=[*NOW[:3], _page("L7/N1")])
    (c,) = circuit_candidates([], moved)
    assert c.weight == pytest.approx(CIRCUIT_WEIGHT * 2 / 5)  # L6/H3 joined, L7/N1 left, of 5
    assert c.rationale.startswith("stale: 2 of 5 members changed") and c.record.endswith(C4.name)


def test_a_claimed_circuit_is_not_drawn(tmp_path):
    root = _root(tmp_path, {"C1": NOW})
    assert circuit_candidates([claim_key(f"{SLUG}/circuits/C1")], root) == []


def test_a_circuit_draw_writes_the_task_proposes_and_reviews(tmp_path):
    (c,) = circuit_candidates([], _root(tmp_path, {"C1": NOW}))
    steps = draw_steps.steps(Plan(3, c, False, [c]))
    titles = [s.title for s in steps]
    assert "Write the circuit's task" in titles and titles[-1] == "Post the result, whatever it is"
    commands = [cmd for s in steps for cmd in s.commands]
    assert any("circuits propose" in cmd for cmd in commands)
    assert any("circuits review" in cmd for cmd in commands)
    assert any(cmd.startswith(f"git switch -c experiment/{claim_key(c.slug)}") for cmd in commands)


def test_the_loop_posts_a_circuit_draw_by_its_verdict(monkeypatch):
    monkeypatch.setattr(el, "new_files", lambda prefix: [C4] if "circuits" in str(prefix) else [])
    assert el.verdict_of("circuit", "k") == (
        "NEEDS_REVISION",
        "Circuit study of k (NEEDS_REVISION)",
    )
    monkeypatch.setattr(el, "new_files", lambda prefix: [])
    with pytest.raises(el.LoopError, match="no circuit record"):
        el.verdict_of("circuit", "k")


def test_a_circuit_draw_names_the_configured_model_of_its_slug(tmp_path):
    # The slug alone reads as a/b/c. The configured model a/b--c has the same slug.
    data = tmp_path / "data" / "circuits"
    data.mkdir(parents=True)
    line = {"date": "2026-11-16", "event": "new", "id": "C1", "members": NOW}
    (data / "a--b--c.jsonl").write_text(json.dumps(line) + "\n")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.openinterpretability]\nmodels = ["x/y", "a/b--c"]\n'
    )
    (c,) = circuit_candidates([], tmp_path)
    assert c.model == "a/b--c" and "--model a/b--c " in c.commands[0]

"""The narrated levels of `explain` (agents/narrator.py, explain_narrate.py, #3743)."""

import json

import pytest
from click.testing import CliRunner

from openinterpretability import explain as ex
from openinterpretability import explain_plan as ep
from openinterpretability.agents.narrator import Narrative, Narrator
from openinterpretability.explain_narrate import check_citations, cited, evidence
from tests.test_explain import PROMPT, TEXTS, means  # noqa: F401  (a fixture)

COVERAGE = {"coverage": 0.0, "random": None}


def _story(text: str) -> Narrative:
    return Narrative(sections=[{"title": "Step", "text": text}], paragraph=text)


def test_the_evidence_holds_only_verified_explanations_fenced(toy_runner, means):  # noqa: F811
    first = ex.explain(toy_runner, "toy", PROMPT, means, {}, top=2, edges=0)
    a, b = first.poles
    labels = {ex._slug("toy", a.key): ("VERIFIED", "fires on b", 1.0),
              ex._slug("toy", b.key): ("REJECTED", "a wrong idea", 1.0)}  # fmt: skip
    result = ex.explain(toy_runner, "toy", PROMPT, means, labels, top=2, edges=0)
    ev = evidence(result, COVERAGE, ep.plan(result, {}, "toy"))
    rows = {row["unit"]: row for row in ev["units"]}
    assert "<data-" in rows[a.label]["explanation"] and "fires on b" in rows[a.label]["explanation"]
    assert "explanation" not in rows[b.label]  # not VERIFIED: no explanation goes in
    assert ev["unexplained_share"] == pytest.approx(1 - ev["explained_share"], abs=1e-4)
    assert "<data-" in ev["prompt"] and ev["next_experiments"]


def test_citations_must_be_in_the_evidence(toy_runner, means):  # noqa: F811
    result = ex.explain(toy_runner, "toy", PROMPT, means, {}, top=2, edges=0)
    ev = evidence(result, COVERAGE, ep.plan(result, {}, "toy"))
    known = ev["units"][0]["unit"]
    assert cited("as L3/N7− and L2/H10 and C4 show") == {"L3/N7-", "L2/H10", "C4"}
    assert check_citations(_story(f"{known} carries it."), ev) == []
    assert check_citations(_story(f"{known} and L9/N99+ carry it, with C77."), ev) == [
        "C77",
        "L9/N99+",
    ]


class _Stub:
    """An instructor client that returns a fixed narrative and records the request."""

    def __init__(self, narrative):
        self.narrative, self.calls = narrative, []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.narrative


def test_the_narrator_reads_the_evidence_under_its_rules():
    story = _story("L0/N1+ carries it. Most of the answer is not known yet.")
    stub = _Stub(story)
    ev = {"units": [{"unit": "L0/N1+"}], "groups": [], "paths": [], "next_experiments": []}
    assert Narrator("anthropic/x", client=stub).narrate(ev) == story
    system, user = stub.calls[0]["messages"]
    assert "Use only the evidence" in json.dumps(system) and '"L0/N1+"' in user["content"]


def test_an_external_narrator_writes_its_prompt(tmp_path):
    narrator = Narrator("external/someone", exchange_dir=tmp_path, timeout=0.01, poll_interval=0.01)
    with pytest.raises(Exception, match="no response"):
        narrator.narrate({"units": []})
    prompt = (tmp_path / "Narrator.prompt.md").read_text()
    assert "Cite a unit for every claim" in prompt and "paragraph" in prompt


def test_cli_narrates_a_paragraph(toy_runner, tmp_path, monkeypatch):
    from openinterpretability.cli import cli

    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    monkeypatch.setattr("openinterpretability.cli_explain._load_runner", lambda *a, **k: toy_runner)
    monkeypatch.setattr(Narrator, "narrate", lambda self, ev: _story("Nothing is explained (C99)."))
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS))
    args = ["explain", PROMPT, "--model", "toy", "--corpus", str(corpus), "--mean-tokens", "200",
            "--wiki-dir", str(tmp_path / "wiki"), "--edges", "0", "--level", "paragraph", "--yes",
            "--narrator-model", "anthropic/x"]  # fmt: skip
    out = CliRunner().invoke(cli, args)
    assert out.exit_code == 0, out.output
    assert "Nothing is explained" in out.output and "Plan:" not in out.output
    assert "cites units that the evidence doesn't hold: C99" in out.output
    as_json = json.loads(CliRunner().invoke(cli, [*args, "--json"]).stdout)  # the warning is stderr
    assert as_json["narrator_model"] == "anthropic/x" and as_json["citations_not_in_evidence"] == [
        "C99"
    ]
    assert as_json["evidence"]["units"] and as_json["narrative"]["paragraph"]


def test_record_at_the_long_level_asks_the_narrator_once(toy_runner, tmp_path, monkeypatch):
    """``explain --record --level long`` narrates once, and records that narration (#4623)."""
    from openinterpretability.cli import cli
    from openinterpretability.core.explanation_records import load_all

    asked = []

    def narrate(self, ev):
        asked.append(ev)
        return _story(f"Narration {len(asked)}.")

    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    monkeypatch.setattr("openinterpretability.cli_explain._load_runner", lambda *a, **k: toy_runner)
    monkeypatch.setattr(Narrator, "narrate", narrate)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps({"text": t}) + "\n" for t in TEXTS))
    wiki = tmp_path / "wiki"
    args = ["explain", PROMPT, "--model", "toy", "--corpus", str(corpus), "--mean-tokens", "200",
            "--wiki-dir", str(wiki), "--edges", "0", "--level", "long", "--yes",
            "--narrator-model", "anthropic/x", "--record", "--no-land"]  # fmt: skip
    out = CliRunner().invoke(cli, args)
    assert out.exit_code == 0, out.output
    assert len(asked) == 1
    (((_, record),),) = load_all(wiki).values()
    assert record.narration.paragraph == "Narration 1."


def test_groups_and_path_ends_go_into_the_evidence_and_count_as_cited(toy_runner, means):  # noqa: F811
    from dataclasses import replace

    from openinterpretability.explain_groups import GroupAttribution

    base = ex.explain(toy_runner, "toy", PROMPT, means, {}, top=2, edges=0)
    group = GroupAttribution("circuit", "C4", 3, base.total / 2, "VERIFIED", "a copy circuit")
    path = {"source": "L0/N1+", "target": "L2/N5−", "weight": 0.5}
    result = replace(base, groups=[group], edges=[path])
    ev = evidence(result, COVERAGE, ep.plan(result, {}, "toy"))
    (row,) = ev["groups"]
    assert row["unit"] == "C4" and row["kind"] == "circuit" and row["members"] == 3
    assert row["share"] == pytest.approx(0.5) and "a copy circuit" in row["explanation"]
    assert check_citations(_story("C4 feeds L0/N1+ then L2/N5- and L8/N8+."), ev) == ["L8/N8+"]

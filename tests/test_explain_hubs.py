"""explain names the hubs among the units that carry the answer (#4472)."""

from pathlib import Path
from types import SimpleNamespace

from openinterpretability import hubs
from openinterpretability.cli_explain import _explained_units, mark_hubs
from openinterpretability.core.explanation_records import ExplainedUnit
from openinterpretability.explain import HEAD, PoleAttribution
from openinterpretability.explain_narrate import evidence
from openinterpretability.hubs import hub_line

MODEL = "Qwen/Qwen2.5-0.5B"
SLUG = "Qwen--Qwen2.5-0.5B"


def _norms(folder: Path) -> None:
    folder.mkdir(parents=True)
    row = hubs.NormRow(hub=f"{SLUG}/L5/H1/HUB", date="2026-11-17", norm=21.3,
                       layer_median=20.2, percentile=0.75)  # fmt: skip
    hubs.path_for(MODEL, folder).write_text(row.model_dump_json() + "\n")


def _result():
    head = PoleAttribution(5, 1, HEAD, attribution=2.0, strength=1.0, status="VERIFIED")
    pole = PoleAttribution(3, 7, "positive", attribution=1.0, strength=1.0)
    plan = SimpleNamespace(experiments=[], groups=[], blocked=0.0)
    result = SimpleNamespace(poles=[head, pole], groups=[], total=3.0, prompt="p", token="t",
                             verified_share=0.5, joint={}, edges=[])  # fmt: skip
    return result, plan


def test_a_hub_center_gets_its_note_and_other_units_none(tmp_path, record, monkeypatch):
    folder = tmp_path / "hubs"
    _norms(folder)
    state = SimpleNamespace(record=record)
    notes = hubs.explain_notes(MODEL, {f"{SLUG}/L5/H1": state}, folder)
    (note,) = notes.values()
    assert note.hub == f"{SLUG}/L5/H1/HUB" and note.href == f"models/{SLUG}/hubs/L5-H1.html"
    assert (note.norm, note.percentile) == (21.3, 0.75) and note.default_on is not None
    monkeypatch.setattr(hubs.explain_notes, "__defaults__", (folder,))
    result, plan = _result()
    mark_hubs(result, MODEL, {f"{SLUG}/L5/H1": state})
    head, pole = result.poles
    assert head.hub["hub"] == note.hub and pole.hub is None
    line = hub_line(head.hub)
    assert "output norm 21.3 against its layer's median 20.2 (p75)" in line
    assert line.endswith("A hub can be an artefact.")


def test_the_hub_goes_into_the_evidence_and_the_record(tmp_path, record, monkeypatch):
    folder = tmp_path / "hubs"
    _norms(folder)
    monkeypatch.setattr(hubs.explain_notes, "__defaults__", (folder,))
    result, plan = _result()
    mark_hubs(result, MODEL, {})
    ev = evidence(result, {}, plan)
    head_row, pole_row = ev["units"]
    assert head_row["hub"]["hub"] == f"{SLUG}/L5/H1/HUB" and "hub" not in pole_row
    assert "artefact" in ev["notes"]
    head_unit, pole_unit = _explained_units(MODEL, result)
    assert head_unit.hub.href == f"models/{SLUG}/hubs/L5-H1.html" and pole_unit.hub is None


def test_an_older_record_without_hub_still_validates():
    unit = ExplainedUnit(unit="L3/N7+", kind="pole", share=0.1, attribution=1.0, status="unstudied")
    assert unit.hub is None

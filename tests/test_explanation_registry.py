"""A new explanation record must agree with the registry (explanation_registry.py, #5173)."""

from types import SimpleNamespace

from openinterpretability.explanation_registry import registry_mismatches
from openinterpretability.reproduce_explanation import check_path
from tests.test_reproduce_explanation import recorded  # noqa: F401  (fixture)


def _wiki(path):
    return path.resolve().parents[3]


def test_an_honest_record_agrees_with_the_registry(recorded):  # noqa: F811
    path, record, _ = recorded
    assert registry_mismatches(record, _wiki(path), groups=[]) == []


def test_a_record_cannot_call_an_unstudied_unit_explained(recorded):  # noqa: F811
    path, record, _ = recorded
    other = next(u.unit for u in record.units if u.status != "VERIFIED")
    forged = record.model_copy(update={"inputs": record.inputs.model_copy(
        update={"explained": [*record.inputs.explained, other]})})  # fmt: skip
    wrong = registry_mismatches(forged, _wiki(path), groups=[])
    assert any(other in w and "explained set" in w for w in wrong)


def test_a_record_cannot_block_an_unstudied_unit(recorded):  # noqa: F811
    path, record, _ = recorded
    other = next(u.unit for u in record.units if u.status != "VERIFIED")
    forged = record.model_copy(update={"inputs": record.inputs.model_copy(
        update={"blocked": [other]})})  # fmt: skip
    assert any("blocked set" in w for w in registry_mismatches(forged, _wiki(path), groups=[]))


def test_a_record_cannot_change_a_units_status_or_text(recorded):  # noqa: F811
    path, record, _ = recorded
    i = next(i for i, u in enumerate(record.units) if u.status != "VERIFIED")
    lie = record.units[i].model_copy(update={"status": "VERIFIED", "explanation": "anything"})
    forged = record.model_copy(update={"units": [*record.units[:i], lie, *record.units[i + 1 :]]})
    assert any("registry says" in w for w in registry_mismatches(forged, _wiki(path), groups=[]))


def test_reproduce_fails_a_new_forged_record_before_it_loads_a_model(recorded):  # noqa: F811
    path, record, _ = recorded
    other = next(u.unit for u in record.units if u.status != "VERIFIED")
    forged = record.model_copy(update={"inputs": record.inputs.model_copy(
        update={"explained": [*record.inputs.explained, other]})})  # fmt: skip
    path.write_text(forged.model_dump_json(), encoding="utf-8")

    def refuse(*a, **k):
        raise AssertionError("loaded a model")

    reproducer = SimpleNamespace(new={path.resolve()}, load=refuse, model_refusal=refuse)
    status, wrong, _ = check_path(reproducer, path)
    assert status == "FAILED" and any("explained set" in w for w in wrong)


def _group_record(explained, units):
    inputs = SimpleNamespace(explained=explained, blocked=[])
    return SimpleNamespace(model_name="toy", inputs=inputs, units=units)


def _unit(kind, unit, status, explanation=None):
    return SimpleNamespace(kind=kind, unit=unit, status=status, explanation=explanation)


def _groups_registry(monkeypatch, superposition_status):
    from openinterpretability import explain as ex
    from openinterpretability import explain_groups
    from openinterpretability.explain_groups import GroupDef

    labels = {
        ex._slug("toy", (0, 1, "positive")): ("VERIFIED", "fires on a", 1.0),
        "toy-superposition": (superposition_status, "counts nouns", 1.0),
    }
    monkeypatch.setattr(ex, "registry_labels", lambda *a, **k: labels)
    monkeypatch.setattr(explain_groups, "circuit_labels", lambda *a: {"c4": ("REJECTED", None)})
    return [
        GroupDef("superposition", "L1/Sabc", "toy-superposition", ((1, 2, 1.0),)),
        GroupDef("circuit", "C4", "c4", ((2, 3, 1.0),)),
    ]


def test_group_units_are_checked_against_the_registry(tmp_path, monkeypatch):
    groups = _groups_registry(monkeypatch, "VERIFIED")
    record = _group_record(
        ["L0/N1+", "L1/N2+", "nonsense"],
        [
            _unit("pole", "L0/N1+", "VERIFIED", "fires on a"),
            _unit("superposition", "L1/Sabc", "VERIFIED", "counts nouns"),
            _unit("circuit", "C4", "REJECTED"),
            _unit("circuit", "C9", "VERIFIED", "made up"),
            _unit("pole", "garbage", "VERIFIED", "made up"),
        ],
    )
    # L1/N2+ is a member of a VERIFIED superposition, so it may be in the explained set.
    assert registry_mismatches(record, tmp_path, groups=groups) == [
        "nonsense: not a unit label, in the explained set",
        "C9: no such circuit in the repository",
        "garbage: the record says VERIFIED, the registry says unknown",
    ]


def test_a_member_of_an_unverified_superposition_is_not_explained(tmp_path, monkeypatch):
    groups = _groups_registry(monkeypatch, "REJECTED")
    record = _group_record(["L1/N2+"], [_unit("superposition", "L1/Sabc", "REJECTED")])
    assert registry_mismatches(record, tmp_path, groups=groups) == [
        "L1/N2+ is in the explained set, but the registry has it unstudied"
    ]

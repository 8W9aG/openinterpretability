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

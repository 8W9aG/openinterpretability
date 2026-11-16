"""Circuit experiments: the decision, the review and the record checks (#4284)."""

from datetime import UTC, datetime

import pytest

from openinterp.checks import CheckReport
from openinterp.circuit_checks import check_circuits
from openinterp.circuit_review import review
from openinterp.core import circuit_records as cr
from openinterp.harness.hooks import NeuronHookRunner
from tests.conftest import FakeTokenizer


def _measures(necessity=0.2, nulls=(0.0, 0.01, 0.02), negatives=None):
    return cr.CircuitMeasures(
        holdout_faithfulness=0.1, holdout_necessity=necessity, null_necessities=list(nulls),
        negative_necessities=negatives if negatives is not None else {"a": 0.01, "b": 0.05},
    )  # fmt: skip


def test_a_circuit_that_carries_its_task_and_only_it_is_verified():
    status, gates = cr.decide(_measures(), cr.CircuitPolicy())
    assert status == "VERIFIED" and all(gates.values())


def test_one_that_also_carries_other_tasks_needs_revision():
    status, gates = cr.decide(_measures(negatives={"a": 0.5, "b": 0.6}), cr.CircuitPolicy())
    assert status == "NEEDS_REVISION" and not gates["specificity"]


def test_one_that_does_no_better_than_random_sets_is_rejected():
    status, gates = cr.decide(
        _measures(necessity=0.015, nulls=(0.0, 0.02, 0.03)), cr.CircuitPolicy()
    )
    assert status == "REJECTED" and not gates["beats_null"] and not gates["necessity_floor"]


def test_the_null_threshold_is_the_quantile():
    assert cr.null_threshold([0.3, 0.1, 0.2, 0.4], 0.5) == 0.2
    assert cr.null_threshold([], 0.95) == 0.0


PAIR = cr.PromptPair(clean="abcd", corrupted="xbcd", good="e", bad="f")


def _record(status="VERIFIED"):
    hypothesis = cr.CircuitHypothesis(
        model_name="toy/model", circuit_id="C1", members=["L0/N1", "L1/H0"],
        task=cr.CircuitTask(name="t", description="d", pairs=[PAIR]), explanation="e",
        submitting_agent_model="external/a",
    )  # fmt: skip
    negative = cr.CircuitTask(name="n", description="d", pairs=[PAIR])
    probes = cr.CircuitProbes(
        holdouts=[PAIR], reviewer_agent_model="external/b", negative_tasks=[negative]
    )
    measures = _measures()
    _, gates = cr.decide(measures, cr.CircuitPolicy())
    verdict = cr.CircuitVerdict(status=status, gates=gates, policy=cr.CircuitPolicy(),
                                measures=measures, reviewer_agent_model="external/b",
                                timestamp=datetime(2026, 11, 16, 23, 0, tzinfo=UTC))  # fmt: skip
    return cr.CircuitRecord(hypothesis=hypothesis, probes=probes, verdict=verdict)


def _write(wiki, record, path=None):
    path = path or cr.record_path(wiki, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json())
    return path


def test_a_record_at_its_path_with_its_status_passes(tmp_path):
    path = _write(tmp_path, _record())
    assert path.relative_to(tmp_path).as_posix() == "circuits/toy--model/C1/20261116T230000Z.json"
    report = CheckReport()
    check_circuits(tmp_path, report)
    assert report.ok


def test_a_misplaced_or_altered_record_fails(tmp_path):
    _write(tmp_path, _record(), tmp_path / "circuits" / "toy--model" / "C2" / "x.json")
    _write(tmp_path, _record(status="REJECTED"))
    report = CheckReport()
    check_circuits(tmp_path, report)
    text = str(report)
    assert "should be at" in text and "the measures give VERIFIED, not REJECTED" in text


def test_a_review_measures_and_decides(tiny_gpt2):
    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    record = review(runner, _record().hypothesis, _record().probes,
                    cr.CircuitPolicy(null_sets=4))  # fmt: skip
    measures = record.verdict.measures
    assert len(measures.null_necessities) == 4 and set(measures.negative_necessities) == {"n"}
    assert set(measures.member_necessities) == {"L0/N1", "L1/H0"}
    assert record.verdict.status in {"VERIFIED", "REJECTED", "NEEDS_REVISION"}
    assert cr.decide(measures, record.verdict.policy)[0] == record.verdict.status


def test_prompts_of_different_lengths_are_refused(tiny_gpt2):
    from openinterp.circuit_review import pair_task

    runner = NeuronHookRunner(tiny_gpt2, FakeTokenizer())
    with pytest.raises(ValueError, match="same token count"):
        pair_task(runner, [PAIR, cr.PromptPair(clean="ab", corrupted="xb", good="e", bad="f")])

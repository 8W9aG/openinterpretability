"""Route rows of the interference report: which records it reads, and answer routes (#5230)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from openinterpretability import explain as ex
from openinterpretability.answer_review import review
from openinterpretability.core import answer_records as ar
from openinterpretability.interference import routes
from openinterpretability.interference.floors import Plain
from tests.test_answer_records import TEXTS, _claim, _probes, _record
from tests.test_explain_reuse import INPUTS
from tests.test_explanation_records import _record as _explanation
from tests.test_explanation_records import _write as _write_explanation

MODEL = "EleutherAI/pythia-70m"


def _store(wiki, record):
    path = ar.record_path(wiki, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json())


def test_answer_records_keep_the_newest_verified_record_of_each_claim(tmp_path):
    wiki = tmp_path / "wiki"
    old = _record(when=datetime(2026, 11, 17, tzinfo=UTC))
    new = _record(when=datetime(2026, 11, 18, tzinfo=UTC))
    rejected = _record(ar.AnswerMeasures(**(_record().verdict.measures.model_dump()
                                            | {"patching": 0.0})),
                       when=datetime(2026, 11, 19, tzinfo=UTC), attempt=2)  # fmt: skip
    other_prompt = _record(prompt="wxyz", corrupted="wayz")
    for record in (old, new, rejected, other_prompt):
        _store(wiki, record)
    assert rejected.verdict.status == "REJECTED" and new.verdict.status == "VERIFIED"
    found = routes.answer_records(wiki, MODEL)
    # Attempt 1's newest record stands; attempt 2 was rejected, so it is left out.
    assert sorted((r.claim.prompt, r.id) for r in found) == [
        ("abcd", new.id),
        ("wxyz", other_prompt.id),
    ]
    assert routes.answer_records(wiki, "other/model") == []
    assert routes.answer_records(tmp_path / "missing", MODEL) == []


def test_explanation_records_keep_the_best_of_each_prompt_with_inputs(tmp_path):
    wiki = tmp_path / "wiki"
    low = _explanation(datetime(2026, 11, 16, tzinfo=UTC)).model_copy(
        update={"prompt_coverage": 0.2, "inputs": INPUTS}
    )
    high = _explanation(datetime(2026, 11, 17, tzinfo=UTC)).model_copy(
        update={"prompt_coverage": 0.6, "inputs": INPUTS}
    )
    no_inputs = _explanation(datetime(2026, 11, 18, tzinfo=UTC)).model_copy(
        update={"prompt_coverage": 0.9}
    )
    for record in (low, high, no_inputs):
        _write_explanation(wiki, record)
    (chosen,) = routes.explanation_records(wiki, "openai-community/gpt2")
    assert chosen.written == high.written and chosen.prompt_coverage == 0.6
    assert routes.explanation_records(wiki, "other/model") == []


@pytest.fixture
def gpt2(tiny_gpt2, tmp_path, monkeypatch):
    from openinterpretability.harness.hooks import NeuronHookRunner
    from tests.conftest import FakeTokenizer

    monkeypatch.setenv(ex.MEANS_ENV, str(tmp_path / "means"))
    monkeypatch.setattr(ex, "WINDOW", 4)
    return NeuronHookRunner(tiny_gpt2.double(), FakeTokenizer())


def test_an_answer_route_is_reviewed_again_on_every_side(gpt2):
    sides = {"base": Plain(gpt2, "base"), "floor:self": Plain(gpt2, "self"),
             "changed": Plain(gpt2, "patch")}  # fmt: skip
    means = routes.side_means(sides, "tiny", TEXTS, 200)
    policy = ar.AnswerPolicy(null_sets=2)
    first = review(gpt2, means["base"], _claim(), _probes(), policy, seed=0)
    (row,) = routes.answer_routes(sides, [first], means)
    assert (row["table"], row["kind"], row["id"], row["prompt"]) == (
        "route", "answer", first.id, "abcd"
    )  # fmt: skip
    failed = [g for g, ok in first.verdict.gates.items() if not ok]
    expected = {"status": first.verdict.status, "failed": failed}
    assert row["status"] == {name: expected for name in sides}
    # Every side is the same model, so no route flips and none counts.
    assert row["flips"] == {"floor:self": False, "changed": False}
    assert row["counts"] is False

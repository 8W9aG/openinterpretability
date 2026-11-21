"""The Contributors page views in ``builder.agents``: costs, review lists and page names."""

from datetime import timedelta

import pytest

from openinterpretability.builder.agents import RECENT_PER_USER, agents_view, contributors_view
from openinterpretability.builder.tree import build_tree
from openinterpretability.core.costs import ExperimentCosts, LLMCall
from openinterpretability.core.exceptions import RegistryError
from openinterpretability.core.schemas import ExperimentProvenance


def experiments_of(records):
    (model,) = build_tree(records, [])
    return model.layers[0].units[0].experiments


def with_costs(record, calls):
    provenance = ExperimentProvenance(
        dtype="float32",
        corpus="c",
        n_snippets=1,
        tokens_scanned=1,
        selection="random",
        harness_version="t",
        costs=ExperimentCosts(llm_calls=calls),
    )
    sub = record.submission.model_copy(update={"provenance": provenance})
    return record.model_copy(update={"submission": sub})


def test_costs_of_a_role_outside_the_page_are_not_shown(record):
    model = record.submission.submitting_agent_model
    rec = with_costs(
        record,
        [
            LLMCall(agent="proposer", model=model, cost_usd=0.5),
            LLMCall(agent="narrator", model=model, cost_usd=2.0),
        ],
    )
    views = {v.model: v for v in agents_view(experiments_of([rec]))}
    roles = next(v.roles for v in views.values() if "proposer" in v.roles)
    assert "narrator" not in roles
    assert roles["proposer"].costs["usd"] == 0.5


def test_a_reviewers_list_keeps_only_the_newest_reviews(record):
    records = []
    for hour in range(RECENT_PER_USER + 2):
        verdict = record.verdict.model_copy(
            update={
                "timestamp": record.verdict.timestamp + timedelta(hours=hour),
                "contributor": "carol",
            }
        )
        records.append(record.model_copy(update={"verdict": verdict}))
    (carol,) = contributors_view(experiments_of(records))
    assert carol.handle == "carol" and carol.reviews == RECENT_PER_USER + 2
    assert len(carol.reviewed) == RECENT_PER_USER
    newest = record.verdict.timestamp + timedelta(hours=RECENT_PER_USER + 1)
    assert carol.reviewed[0]["e"].record.verdict.timestamp == newest


def test_two_models_with_one_page_name_are_an_error(record):
    sub = record.submission.model_copy(update={"submitting_agent_model": "gpt test"})
    verdict = record.verdict.model_copy(update={"reviewer_agent_model": "gpt-test"})
    rec = record.model_copy(update={"submission": sub, "verdict": verdict})
    with pytest.raises(RegistryError, match="share the page agents/gpt-test.html"):
        agents_view(experiments_of([rec]))

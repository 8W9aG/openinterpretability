"""The proposer's prompt for an answer experiment (openinterpretability/answer_prompts.py)."""

from openinterpretability.answer_prompts import proposer_prompt
from tests.test_reproduce_explanation import recorded  # noqa: F401  (fixture)


def test_the_proposer_prompt_leaves_out_a_minimal_set_and_paths_it_does_not_have(
    recorded,  # noqa: F811
):
    _, record, _ = recorded
    full = proposer_prompt("c.json", "toy", "a b c", 3, " d", record, [], 1)
    assert "- Minimal set (" in full and "Strongest paths between top units" in full
    bare = record.model_copy(update={"coverage_detail": None, "figures": None})
    text = proposer_prompt("c.json", "toy", "a b c", 3, " d", bare, [], 1)
    assert "- Minimal set (" not in text and "Strongest paths between top units" not in text
    assert "Top units (share of |attribution|" in text

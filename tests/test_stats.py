from openinterp.cli_registry import registry_stats
from openinterp.core.schemas import ReviewPolicy


def test_registry_stats(record):
    def variant(neuron_idx, status, reviewer, version, model="Qwen/Qwen2.5-0.5B"):
        ident = record.submission.identifier.model_copy(
            update={"neuron_idx": neuron_idx, "model_name": model}
        )
        policy = ReviewPolicy(protocol_version=version) if version else None
        return record.model_copy(
            update={
                "submission": record.submission.model_copy(update={"identifier": ident}),
                "verdict": record.verdict.model_copy(
                    update={"status": status, "reviewer_agent_model": reviewer, "policy": policy}
                ),
            }
        )

    proposer = record.submission.submitting_agent_model
    records = [
        variant(1, "VERIFIED", proposer, "0.9"),
        variant(1, "REJECTED", "other", "0.10"),
        variant(2, "NEEDS_REVISION", "other", None),
        variant(1, "VERIFIED", "other", "0.9", model="openai-community/gpt2"),
    ]
    assert registry_stats(records).splitlines() == [
        "Registry: 4 experiments on 3 units; VERIFIED 2/4 (50%), self-reviewed 1/4 (25%), 2 models",
        "  Qwen/Qwen2.5-0.5B: 3 experiments on 2 units; VERIFIED 1/3 (33%), "
        "self-reviewed 1/3 (33%)",
        "  openai-community/gpt2: 1 experiments on 1 units; VERIFIED 1/1 (100%), "
        "self-reviewed 0/1 (0%)",
        "Protocol versions: 0.10 ×1, 0.9 ×2, < 0.2 ×1",
    ]


def test_registry_stats_empty():
    assert registry_stats([]).splitlines() == [
        "Registry: 0 experiments on 0 units; VERIFIED 0/0, self-reviewed 0/0, 0 models",
        "Protocol versions: none",
    ]

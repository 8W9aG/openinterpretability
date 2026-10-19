"""Proposer agent: turns top-activating snippets into a falsifiable hypothesis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from openinterp.agents.base import LLMAgent
from openinterp.core.schemas import (
    ActivatingExample,
    ConfoundProfile,
    CorrelatedNeuron,
    CounterfactualPair,
    ExperimentProvenance,
    Histogram,
    LookaheadProfile,
    NeuronAuditSubmission,
    NeuronIdentifier,
    VocabularyLens,
)
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.lookahead import measure_lookahead


class CounterfactualDraft(BaseModel):
    base_prompt: str = Field(
        description="A prompt the hypothesis predicts strongly activates the pole."
    )
    counterfactual_prompt: str = Field(
        description="A minimal edit of base_prompt that the hypothesis predicts changes activation."
    )
    expected_delta: float = Field(
        ge=-1.0,
        le=1.0,
        description="Predicted (base - counterfactual) activation as a fraction of max activation.",
    )
    rationale: str


class HypothesisDraft(BaseModel):
    explanation: str = Field(
        description="One or two precise sentences: what input feature or upcoming-token "
        "preparation makes this pole fire."
    )
    activating_tokens: list[str] = Field(description="Representative tokens where the pole peaks.")
    mechanism: Literal["retrospective", "lookahead"] = Field(
        description="retrospective = reacts to tokens already seen; "
        "lookahead = prepares a specific upcoming token."
    )
    target_upcoming_tokens: list[str] = Field(
        default_factory=list, description="For lookahead neurons: the tokens being prepared."
    )
    counterfactuals: list[CounterfactualDraft] = Field(min_length=3, max_length=10)


def format_examples(examples: Sequence[ActivatingExample], max_examples: int = 20) -> str:
    """Render examples with the peak token in [[double brackets]] and per-token strengths."""
    lines = []
    for n, ex in enumerate(examples[:max_examples], 1):
        marked = "".join(
            f"[[{tok}]]" if i == ex.max_token_index else tok for i, tok in enumerate(ex.tokens)
        )
        window = range(max(0, ex.max_token_index - 3), min(len(ex.tokens), ex.max_token_index + 2))
        context = ", ".join(f"{ex.tokens[i]!r}:{ex.activations[i]:.2f}" for i in window)
        lines.append(f"{n}. (max {ex.max_activation:.3f}) {marked}\n   around peak: {context}")
    return "\n".join(lines)


class ProposerAgent(LLMAgent):
    system_prompt = (
        "You are a mechanistic interpretability researcher auditing a single native MLP "
        "neuron pole in an open-weight language model. Activations are pole strengths "
        "(non-negative; larger = more active). Propose the most specific hypothesis that "
        "explains the evidence and that could be falsified. Decide explicitly whether the "
        "pole reacts to what has already been read (retrospective) or prepares a particular "
        "upcoming token (lookahead): lookahead neurons often peak on the token *before* the "
        "predicted one, and their promoted-token list is informative. Write minimal-pair "
        "counterfactuals that change as little text as possible, including at least one "
        "control pair with expected_delta near 0."
    )

    def propose(
        self,
        identifier: NeuronIdentifier,
        examples: Sequence[ActivatingExample],
        lens: VocabularyLens | None = None,
        confounds: ConfoundProfile | None = None,
        correlated: Sequence[CorrelatedNeuron] = (),
    ) -> HypothesisDraft:
        prompt = (
            f"Neuron: {identifier}\n\n"
            f"Top-activating snippets (peak token in [[ ]]):\n{format_examples(examples)}\n\n"
        )
        if lens is not None:
            prompt += (
                f"Tokens this pole promotes via the Jacobian lens (its output direction "
                f"transported to the final layer; reliable in middle layers): "
                f"{', '.join(map(repr, lens.jacobian_promoted))}; suppresses: "
                f"{', '.join(map(repr, lens.jacobian_suppressed))}. "
                f"Logit lens (no transport): {', '.join(map(repr, lens.logit_promoted))}. "
                f"Lens agreement {lens.overlap:.0%}; peakedness (kurtosis) "
                f"{lens.jacobian_kurtosis:.1f}, where ~3 means no specific token set.\n"
            )
        if confounds is not None:
            top = ", ".join(repr(t.token) for t in confounds.trigger_tokens[:10])
            prompt += (
                f"\nConfounds: token identity explains {confounds.token_r2:.0%} of activation "
                f"variance, position {confounds.position_r2:.0%}. Highest mean-activation "
                f"tokens: {top}. Your hypothesis must predict activity better than "
                f"'fires on these tokens'.\n"
            )
        if correlated:
            pairs = ", ".join(
                f"N{c.neuron_idx}{'+' if c.pole == 'positive' else '-'} "
                f"(co-activation lift >= {c.coactivation_lift_lower:.1f}, rho={c.spearman:+.2f})"
                for c in correlated[:5]
            )
            prompt += f"Neuron poles in this layer that co-activate with it: {pairs}.\n"
        return self.structured(HypothesisDraft, prompt)

    def build_submission(
        self,
        identifier: NeuronIdentifier,
        draft: HypothesisDraft,
        *,
        runner: NeuronHookRunner | None = None,
        examples: Sequence[ActivatingExample] = (),
        histogram: Histogram | None = None,
        confounds: ConfoundProfile | None = None,
        correlated: Sequence[CorrelatedNeuron] = (),
        provenance: ExperimentProvenance | None = None,
        vocabulary_lens: VocabularyLens | None = None,
    ) -> NeuronAuditSubmission:
        """Convert a draft into a submission, measuring evidence with ``runner`` if given."""
        max_act = max((e.max_activation for e in examples), default=0.0) or None
        pairs = [
            CounterfactualPair(
                base_prompt=cf.base_prompt,
                counterfactual_prompt=cf.counterfactual_prompt,
                expected_delta=cf.expected_delta,
            )
            for cf in draft.counterfactuals
            if cf.base_prompt != cf.counterfactual_prompt
        ]
        lookahead = LookaheadProfile(
            is_lookahead=draft.mechanism == "lookahead",
            target_upcoming_tokens=list(draft.target_upcoming_tokens),
            kl_divergence=0.0,
        )
        if runner is not None and max_act:
            pairs = [
                pair.model_copy(
                    update={"observed_delta": observed_delta(runner, identifier, pair, max_act)}
                )
                for pair in pairs
            ]
            measured = measure_lookahead(
                runner,
                [p.base_prompt for p in pairs],
                identifier.layer,
                identifier.neuron_idx,
                identifier.pole,
            )
            # Keep the proposer's claim; record the measured KL as evidence for review.
            lookahead = lookahead.model_copy(update={"kl_divergence": measured.kl_divergence})

        return NeuronAuditSubmission(
            identifier=identifier,
            natural_language_explanation=draft.explanation,
            activating_tokens=list(draft.activating_tokens),
            counterfactuals=pairs,
            lookahead=lookahead,
            submitting_agent_model=self.model,
            max_activation=max_act,
            top_examples=list(examples),
            activation_histogram=histogram,
            model_num_layers=runner.n_layers if runner is not None else None,
            provenance=provenance,
            confounds=confounds,
            correlated_neurons=list(correlated),
            vocabulary_lens=vocabulary_lens,
        )


def observed_delta(
    runner: NeuronHookRunner, ident: NeuronIdentifier, pair: CounterfactualPair, scale: float
) -> float:
    """Measured (base - counterfactual) max pole strength, normalised by ``scale``."""
    args = (ident.layer, ident.neuron_idx, ident.pole)
    base = runner.max_pole_strength(pair.base_prompt, *args)
    cf = runner.max_pole_strength(pair.counterfactual_prompt, *args)
    return round((base - cf) / scale, 6)

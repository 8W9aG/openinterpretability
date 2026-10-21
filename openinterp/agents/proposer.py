"""Proposer agent: turns top-activating snippets into a falsifiable hypothesis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from openinterp.agents.base import LLMAgent
from openinterp.core.attribution import InputAttribution
from openinterp.core.schemas import (
    ActivatingExample,
    ComponentIdentifier,
    ConfoundProfile,
    CorrelatedNeuron,
    CounterfactualPair,
    ExperimentProvenance,
    HeadProfile,
    Histogram,
    LookaheadProfile,
    NeuronAuditSubmission,
    VocabularyLens,
)
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.lookahead import ablation_profile
from openinterp.harness.units import Unit, max_strength, unit_of


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
        if ex.attends_to:
            attended = ", ".join(
                f"{ex.tokens[i]!r} (token {i - ex.max_token_index:+d}, weight {w:.2f})"
                for i, w in zip(ex.attends_to, ex.attention_weights, strict=True)
            )
            lines.append(f"   peak attends to: {attended}")
    return "\n".join(lines)


def format_head_profile(profile: HeadProfile) -> str:
    return (
        f"Strength is |W_O (z - mean z)|: how far the head's write to the residual stream is "
        f"from its average write. Attention-pattern scores on {profile.pattern_sequences} "
        f"random {profile.pattern_length}-token sequences repeated twice (0-1; mean attention "
        f"weight): previous token {profile.previous_token_score:.2f}, duplicate token "
        f"{profile.duplicate_token_score:.2f}, induction (to the token after an earlier copy "
        f"of the current one) {profile.induction_score:.2f}.\n"
    )


def format_input_attribution(attribution: InputAttribution) -> str:
    """Where in the context the pole's peaks come from, for the proposer and the reviewer."""
    if not attribution.examples:
        return ""
    n = len(attribution.examples)
    text = (
        f"Input attribution (integrated gradients over the input tokens, at the peak of "
        f"{n} top examples; the path integral converged on {attribution.converged}): "
    )
    if attribution.distances:
        shares = {d.min_distance: d.share for d in attribution.distances}
        far = sum(d.share for d in attribution.distances if d.min_distance > 16)
        text += (
            f"{shares[0]:.0%} of it is on the peak token itself, {shares[1]:.0%} on the "
            f"token before and {far:.0%} more than 16 tokens back (mean distance "
            f"{attribution.mean_distance:.1f} tokens). "
        )
    if attribution.ablation_agreement is not None:
        text += (
            f"Replacing one token at a time agrees with it at rho = "
            f"{attribution.ablation_agreement:+.2f}. "
        )
    lines = [
        text + "Tokens driving each peak most, as offset from the peak: share of the "
        "activation (attribution / effect of replacing that token alone; trust the second "
        "where an example did not converge):"
    ]
    for ex in attribution.examples:
        tokens = ", ".join(
            f"{t.token!r} ({t.offset:+d}) {t.attribution:.2f}/"
            + ("?" if t.ablation_effect is None else f"{t.ablation_effect:.2f}")
            for t in ex.top
        )
        note = "" if ex.converged else " (not converged)"
        lines.append(f"  example {ex.example + 1}, peak {ex.peak_token!r}{note}: {tokens}")
    return "\n".join(lines) + "\n"


class ProposerAgent(LLMAgent):
    role = "proposer"
    system_prompt = (
        "You are a mechanistic interpretability researcher auditing a single native unit, an "
        "MLP neuron pole or an attention head, in an open-weight language model. Activations "
        "are strengths (non-negative; larger = more active). Propose the most specific "
        "hypothesis that explains the evidence and that could be falsified. Decide explicitly "
        "whether the unit reacts to what has already been read (retrospective) or prepares a "
        "particular upcoming token (lookahead): lookahead neurons often peak on the token "
        "*before* the predicted one, and their promoted-token list is informative. Write "
        "minimal-pair "
        "counterfactuals that change as little text as possible, including at least one "
        "control pair with expected_delta near 0."
    )

    def propose(
        self,
        identifier: ComponentIdentifier,
        examples: Sequence[ActivatingExample],
        lens: VocabularyLens | None = None,
        confounds: ConfoundProfile | None = None,
        correlated: Sequence[CorrelatedNeuron] = (),
        head_profile: HeadProfile | None = None,
        attribution: InputAttribution | None = None,
    ) -> HypothesisDraft:
        prompt = self.proposal_prompt(
            identifier, examples, lens, confounds, correlated, head_profile, attribution
        )
        return self.structured(HypothesisDraft, prompt)

    @staticmethod
    def proposal_prompt(
        identifier: ComponentIdentifier,
        examples: Sequence[ActivatingExample],
        lens: VocabularyLens | None = None,
        confounds: ConfoundProfile | None = None,
        correlated: Sequence[CorrelatedNeuron] = (),
        head_profile: HeadProfile | None = None,
        attribution: InputAttribution | None = None,
    ) -> str:
        """The task text the proposer answers: the scan evidence for one unit."""
        prompt = f"{'Attention head' if head_profile else 'Neuron'}: {identifier}\n\n"
        if head_profile is not None:
            prompt += format_head_profile(head_profile) + "\n"
        prompt += f"Top-activating snippets (peak token in [[ ]]):\n{format_examples(examples)}\n\n"
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
        if attribution is not None:
            prompt += "\n" + format_input_attribution(attribution)
        return prompt

    def build_submission(
        self,
        identifier: ComponentIdentifier,
        draft: HypothesisDraft,
        *,
        runner: NeuronHookRunner | None = None,
        examples: Sequence[ActivatingExample] = (),
        histogram: Histogram | None = None,
        confounds: ConfoundProfile | None = None,
        correlated: Sequence[CorrelatedNeuron] = (),
        provenance: ExperimentProvenance | None = None,
        vocabulary_lens: VocabularyLens | None = None,
        input_attribution: InputAttribution | None = None,
        head_profile: HeadProfile | None = None,
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
        is_head = identifier.kind == "attn_head"
        if runner is not None and max_act:
            unit = unit_of(identifier, head_profile)
            pairs = [
                pair.model_copy(
                    update={"observed_delta": observed_delta(runner, unit, pair, max_act)}
                )
                for pair in pairs
            ]
            prompts = [p.base_prompt for p in pairs]
            measured = ablation_profile(runner, prompts, unit.ablation(runner))
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
            model_mlp_width=runner.mlp_width(identifier.layer) if runner is not None else None,
            model_n_heads=runner.n_heads if runner is not None and is_head else None,
            provenance=provenance,
            confounds=confounds,
            correlated_neurons=list(correlated),
            vocabulary_lens=vocabulary_lens,
            input_attribution=input_attribution,
            head_profile=head_profile,
        )


def observed_delta(
    runner: NeuronHookRunner, unit: Unit, pair: CounterfactualPair, scale: float
) -> float:
    """Measured (base - counterfactual) max strength, normalised by ``scale``."""
    base = max_strength(runner, unit, pair.base_prompt)
    cf = max_strength(runner, unit, pair.counterfactual_prompt)
    return round((base - cf) / scale, 6)

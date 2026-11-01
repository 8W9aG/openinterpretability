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
    DictionaryProfile,
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


# Agents see a window of tokens around each example's peak, not the whole scanned
# document, and not near-duplicates of an earlier example (--full-snippets turns both off).
SNIPPET_BEFORE = 48
SNIPPET_AFTER = 16
NEAR_DUPLICATE = 0.8


def snippet_window(ex: ActivatingExample, full: bool = False) -> range:
    if full:
        return range(len(ex.tokens))
    peak = ex.max_token_index
    return range(max(0, peak - SNIPPET_BEFORE), min(len(ex.tokens), peak + SNIPPET_AFTER + 1))


def snippet(ex: ActivatingExample, full: bool = False) -> str:
    """The example's text with the peak token in [[ ]], cut to its window with … marks."""
    window = snippet_window(ex, full)
    marked = "".join(
        f"[[{ex.tokens[i]}]]" if i == ex.max_token_index else ex.tokens[i] for i in window
    )
    start = "…" if window.start > 0 else ""
    end = "…" if window.stop < len(ex.tokens) else ""
    return start + marked + end


def label_note(ex: ActivatingExample) -> str:
    """The peak token's labels (``"language fr"``), or ``""`` if the scan had none."""
    return ", ".join(f"{name} {label}" for name, label in (ex.peak_labels or {}).items())


def distinct_examples(
    examples: Sequence[ActivatingExample], limit: int, full: bool = False
) -> tuple[list[tuple[int, ActivatingExample]], list[int]]:
    """The first ``limit`` examples, numbered from 1, without those whose window shares at
    least NEAR_DUPLICATE of its distinct tokens with an earlier kept one's; and the numbers
    dropped. Numbers stay those of the scan's order, which input attribution cites."""
    kept: list[tuple[int, ActivatingExample]] = []
    dropped: list[int] = []
    seen: list[set[str]] = []
    for n, ex in enumerate(examples[:limit], 1):
        tokens = {ex.tokens[i] for i in snippet_window(ex)}
        if not full and any(len(tokens & s) >= NEAR_DUPLICATE * len(tokens | s) for s in seen):
            dropped.append(n)
            continue
        seen.append(tokens)
        kept.append((n, ex))
    return kept, dropped


def dropped_note(dropped: Sequence[int]) -> str:
    if not dropped:
        return ""
    numbers = ", ".join(map(str, dropped))
    return f"(Omitted as near-duplicates of an earlier example: {numbers}.)\n"


def format_review_examples(examples: Sequence[ActivatingExample], full: bool = False) -> str:
    """The reviewer's view of the top 10 examples: text only, no strengths."""
    if full:  # the prompt before windowing, for A/B comparison
        listed = "\n".join(f"{n}. {_noted(e)}{e.text}" for n, e in enumerate(examples[:10], 1))
        return f"Top-activating examples:\n{listed or '(none)'}\n\n"
    kept, dropped = distinct_examples(examples, 10)
    listed = "\n".join(f"{n}. {_noted(e)}{snippet(e)}" for n, e in kept) or "(none)"
    note = "\n" + dropped_note(dropped).rstrip() if dropped else ""
    return f"Top-activating examples (peak token in [[ ]], … where cut):\n{listed}{note}\n\n"


def _noted(ex: ActivatingExample) -> str:
    note = label_note(ex)
    return f"({note}) " if note else ""


def format_examples(
    examples: Sequence[ActivatingExample], max_examples: int = 20, full: bool = False
) -> str:
    """Render examples with the peak token in [[double brackets]] and per-token strengths."""
    lines = []
    kept, dropped = distinct_examples(examples, max_examples, full)
    for n, ex in kept:
        marked = snippet(ex, full)
        window = range(max(0, ex.max_token_index - 3), min(len(ex.tokens), ex.max_token_index + 2))
        context = ", ".join(f"{ex.tokens[i]!r}:{ex.activations[i]:.2f}" for i in window)
        note = label_note(ex)
        head = f"max {ex.max_activation:.3f}" + (f", {note}" if note else "")
        lines.append(f"{n}. ({head}) {marked}\n   around peak: {context}")
        if ex.attends_to:
            attended = ", ".join(
                f"{ex.tokens[i]!r} (token {i - ex.max_token_index:+d}, weight {w:.2f})"
                for i, w in zip(ex.attends_to, ex.attention_weights, strict=True)
            )
            lines.append(f"   peak attends to: {attended}")
    return "\n".join(lines) + ("\n" + dropped_note(dropped).rstrip() if dropped else "")


def format_head_profile(profile: HeadProfile) -> str:
    return (
        f"Strength is |W_O (z - mean z)|: how far the head's write to the residual stream is "
        f"from its average write. Attention-pattern scores on {profile.pattern_sequences} "
        f"random {profile.pattern_length}-token sequences repeated twice (0-1; mean attention "
        f"weight): previous token {profile.previous_token_score:.2f}, duplicate token "
        f"{profile.duplicate_token_score:.2f}, induction (to the token after an earlier copy "
        f"of the current one) {profile.induction_score:.2f}.\n"
    )


def format_dictionary_profile(profile: DictionaryProfile) -> str:
    if profile.kind == "basis":  # says nothing of which basis, so a null arm stays blind
        return (
            f"This is one direction in this layer's MLP hidden space (one of a basis of "
            f"{profile.n_features}), read where native neurons are: the input to the down "
            f"projection. Strength is the positive part of the hidden activations' "
            f"projection onto it; it writes that strength x the direction, through the "
            f"down projection, to the MLP output. It is non-zero on {profile.density:.2%} of "
            f"scanned tokens.\n"
        )
    residual = profile.output_site == "resid_post"
    output = "the residual stream after this layer" if residual else "the MLP output"
    fit = ""
    if profile.explained_variance is not None:
        fit = (
            f" On the scan the whole dictionary reconstructs "
            f"{profile.explained_variance:.0%} of {output}'s variance with "
            f"{profile.mean_l0:.0f} active latents per token."
        )
    trained = output if residual else "this layer's MLP"
    return (
        f"This is one latent of a sparse {profile.kind} ({profile.n_features} latents) "
        f"trained on {trained}, not a native neuron. Strength is the latent's "
        f"activation (non-negative); it writes activation x its decoder vector to "
        f"{output}. It is non-zero on {profile.density:.2%} of scanned tokens.{fit}\n"
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


# Protocol 0.8: how a unit with a first-token floor is read, for every agent that writes
# prompts for one.
FIRST_TOKEN_UNMEASURED = (
    "This unit has a first-token floor: at a text's first token it is active whatever the "
    "token is, so it is not measured there. A prompt's strength is its peak over every "
    "token but the first, so every prompt needs at least two tokens, and a trigger that "
    "opens a prompt is not seen. Its examples show the first token's strength, but it "
    "never counts.\n"
)


class ProposerAgent(LLMAgent):
    role = "proposer"
    system_prompt = (
        "You are a mechanistic interpretability researcher auditing a single unit, a native "
        "MLP neuron pole or attention head, or a latent of a sparse dictionary trained on an "
        "MLP layer, in an open-weight language model. Activations "
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
        dictionary_profile: DictionaryProfile | None = None,
        first_token_floor: bool = False,
        full_snippets: bool = False,
        related: str = "",
    ) -> HypothesisDraft:
        prompt = self.proposal_prompt(
            identifier,
            examples,
            lens,
            confounds,
            correlated,
            head_profile,
            attribution,
            dictionary_profile,
            first_token_floor,
            full_snippets,
            related,
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
        dictionary_profile: DictionaryProfile | None = None,
        first_token_floor: bool = False,
        full_snippets: bool = False,
        related: str = "",
    ) -> str:
        """The task text the proposer answers: the scan evidence for one unit, then what
        other research links to it (``related``: its studied sister neurons,
        :func:`openinterp.sisters.format_sisters`)."""
        prompt = f"{identifier.noun.capitalize()}: {identifier}\n\n"
        if first_token_floor:
            prompt += FIRST_TOKEN_UNMEASURED + "\n"
        if head_profile is not None:
            prompt += format_head_profile(head_profile) + "\n"
        if dictionary_profile is not None:
            prompt += format_dictionary_profile(dictionary_profile) + "\n"
        window = (
            ""
            if full_snippets
            else f"; up to {SNIPPET_BEFORE} tokens before it and {SNIPPET_AFTER} after, … where cut"
        )
        prompt += (
            f"Top-activating snippets (peak token in [[ ]]{window}):\n"
            f"{format_examples(examples, full=full_snippets)}\n\n"
        )
        if lens is not None:
            prompt += format_lens(lens, identifier)
        if confounds is not None:
            prompt += format_confounds(confounds)
        if correlated:
            prompt += format_correlated(correlated)
        if attribution is not None:
            prompt += "\n" + format_input_attribution(attribution)
        if related:
            prompt += "\n" + related
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
        dictionary_profile: DictionaryProfile | None = None,
        first_token_floor: bool | None = None,
    ) -> NeuronAuditSubmission:
        """Convert a draft into a submission, measuring evidence with ``runner`` if given."""
        max_act = max((e.max_activation for e in examples), default=0.0) or None
        pairs, lookahead = draft_claims(draft)
        is_head = identifier.kind == "attn_head"
        if runner is not None and max_act:
            unit = unit_of(identifier, head_profile, dictionary_profile, bool(first_token_floor))
            pairs, lookahead = measure_claims(runner, unit, pairs, lookahead, max_act)

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
            first_token_floor=first_token_floor,
            model_num_layers=runner.n_layers if runner is not None else None,
            model_mlp_width=runner.mlp_width(identifier.layer) if runner is not None else None,
            model_n_heads=runner.n_heads if runner is not None and is_head else None,
            provenance=provenance,
            confounds=confounds,
            correlated_neurons=list(correlated),
            vocabulary_lens=vocabulary_lens,
            input_attribution=input_attribution,
            head_profile=head_profile,
            dictionary_profile=dictionary_profile,
        )


def format_lens(lens: VocabularyLens, identifier: ComponentIdentifier) -> str:
    return (
        f"Tokens this {'pole' if identifier.pole else identifier.noun} promotes via the "
        f"Jacobian lens (its output direction "
        f"transported to the final layer; reliable in middle layers): "
        f"{', '.join(map(repr, lens.jacobian_promoted))}; suppresses: "
        f"{', '.join(map(repr, lens.jacobian_suppressed))}. "
        f"Logit lens (no transport): {', '.join(map(repr, lens.logit_promoted))}. "
        f"Lens agreement {lens.overlap:.0%}; peakedness (kurtosis) "
        f"{lens.jacobian_kurtosis:.1f}, where ~3 means no specific token set.\n"
    )


def format_confounds(confounds: ConfoundProfile) -> str:
    top = ", ".join(repr(t.token) for t in confounds.trigger_tokens[:10])
    labels = "".join(
        f" {name.capitalize()} explains {c.r2:.0%} ({c.mutual_information_bits:.3f} bits); "
        f"active tokens by {name}: "
        f"{', '.join(f'{s.label} {s.active}/{s.count}' for s in c.levels[:10])}."
        for name, c in (confounds.labels or {}).items()
    )
    return (
        f"\nConfounds: token identity explains {confounds.token_r2:.0%} of activation "
        f"variance, position {confounds.position_r2:.0%}.{labels} Highest mean-activation "
        f"tokens: {top}. Your hypothesis must predict activity better than "
        f"'fires on these tokens'.\n"
    )


def format_correlated(correlated: Sequence[CorrelatedNeuron]) -> str:
    pairs = ", ".join(
        f"N{c.neuron_idx}{'+' if c.pole == 'positive' else '-'} "
        f"(co-activation lift >= {c.coactivation_lift_lower:.1f}, rho={c.spearman:+.2f})"
        for c in correlated[:5]
    )
    return f"Neuron poles in this layer that co-activate with it: {pairs}.\n"


def draft_claims(draft: HypothesisDraft) -> tuple[list[CounterfactualPair], LookaheadProfile]:
    """The draft's counterfactual pairs (dropping unedited ones) and mechanism, unmeasured."""
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
    return pairs, lookahead


def measure_claims(runner, unit: Unit, pairs, lookahead: LookaheadProfile, max_act: float):
    """Each pair's observed delta, and the ablation KL on the base prompts."""
    pairs = [
        pair.model_copy(update={"observed_delta": observed_delta(runner, unit, pair, max_act)})
        for pair in pairs
    ]
    prompts = [p.base_prompt for p in pairs]
    measured = ablation_profile(runner, prompts, unit.ablation(runner))
    # Keep the proposer's claim; record the measured KL as evidence for review.
    return pairs, lookahead.model_copy(update={"kl_divergence": measured.kl_divergence})


def observed_delta(
    runner: NeuronHookRunner, unit: Unit, pair: CounterfactualPair, scale: float
) -> float:
    """Measured (base - counterfactual) max strength, normalised by ``scale``."""
    base = max_strength(runner, unit, pair.base_prompt)
    cf = max_strength(runner, unit, pair.counterfactual_prompt)
    return round((base - cf) / scale, 6)

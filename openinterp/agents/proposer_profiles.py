"""How the proposer's prompt describes a unit that is not a neuron pole.

A head is described by its attention-pattern scores, and a dictionary feature by its
dictionary's fit. A superposition is described by its member neurons and their weights.
"""

from __future__ import annotations

from openinterp.core.schemas import DictionaryProfile, HeadProfile, SuperpositionProfile

SHOWN_MEMBERS = 20  # the largest members by weight that the prompt lists


def format_head_profile(profile: HeadProfile) -> str:
    """The proposer's note on what a head's strength is, and its pattern scores."""
    return (
        f"Strength is |W_O (z - mean z)|: how far the head's write to the residual stream is "
        f"from its average write. Attention-pattern scores on {profile.pattern_sequences} "
        f"random {profile.pattern_length}-token sequences repeated twice (0-1; mean attention "
        f"weight): previous token {profile.previous_token_score:.2f}, duplicate token "
        f"{profile.duplicate_token_score:.2f}, induction (to the token after an earlier copy "
        f"of the current one) {profile.induction_score:.2f}.\n"
    )


def format_dictionary_profile(profile: DictionaryProfile) -> str:
    """The proposer's note on what a dictionary latent's strength is, and its fit."""
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
    if profile.kind == "spd":
        return (
            f"This is one latent ({profile.n_features} in this layer) of a parameter "
            f"decomposition of this layer's MLP down projection into rank-1 pieces, not a "
            f"native neuron. It reads where native neurons are: the input to the down "
            f"projection. Strength is the positive part of the hidden activations' "
            f"projection onto the piece's input direction, scaled by the piece's causal "
            f"importance on that token (0 to 1: how much ablating the piece would change "
            f"the model's output). It writes strength x the piece's output direction to the "
            f"MLP output. It is non-zero on {profile.density:.2%} of scanned tokens.{fit}\n"
        )
    trained = output if residual else "this layer's MLP"
    return (
        f"This is one latent of a sparse {profile.kind} ({profile.n_features} latents) "
        f"trained on {trained}, not a native neuron. Strength is the latent's "
        f"activation (non-negative); it writes activation x its decoder vector to "
        f"{output}. It is non-zero on {profile.density:.2%} of scanned tokens.{fit}\n"
    )


def format_superposition_profile(profile: SuperpositionProfile) -> str:
    """The superposition's members, largest weight first, and how it was found."""
    members = sorted(profile.members, key=lambda m: -abs(m.weight))
    top = abs(members[0].weight)
    shown = ", ".join(f"N{m.neuron_idx} ({m.weight / top:+.2f})" for m in members[:SHOWN_MEMBERS])
    more = len(members) - SHOWN_MEMBERS
    rest = f", and {more} more" if more > 0 else ""
    return (
        f"This unit is a superposition: one direction over {len(members)} of the layer's "
        f"neurons, active where the weighted sum of their activations is positive. Members, "
        f"by relative weight: {shown}{rest}. It was found by {profile.method} "
        f"({profile.source}). Explain what the direction as a whole responds to, not one "
        "neuron."
    )

"""What defines the strength of a unit that is not a neuron pole: an attention head's mean
output, or a dictionary feature's dictionary. Stored in the record, so every later
measurement of the experiment uses the same definition."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from openinterp.core.strict import StrictModel, UnitFloat


class HeadProfile(StrictModel):
    """What defines an attention head's activity, and its standard attention-pattern scores.

    A head's strength at token t is ``|W_O (z_t − mean_output)|``: how far its write to
    the residual stream is from its mean write over the scan, i.e. how far mean-ablating
    the head would move the residual stream there. ``mean_output`` (z̄, one value per
    head dimension) is measured once, by the scan, and every later measurement of the
    experiment uses this stored value, so a record reproduces without the corpus.

    The pattern scores are mean attention weights on ``pattern_sequences`` sequences of
    ``pattern_length`` uniformly random tokens repeated twice (Olsson et al., 2022):
    ``previous_token`` to the token before, ``duplicate_token`` from the second copy to
    the same token in the first, ``induction`` from the second copy to the token *after*
    the same token in the first. Each is between 0 and 1; reported, not gated.
    """

    mean_output: list[float] = Field(min_length=1)
    mean_tokens: int = Field(gt=0, description="Tokens z̄ was averaged over.")
    previous_token_score: UnitFloat
    duplicate_token_score: UnitFloat
    induction_score: UnitFloat
    pattern_sequences: int = Field(gt=0)
    pattern_length: int = Field(gt=1)
    pattern_seed: int


DictionarySite = Literal["mlp_in", "mlp_in_normalized", "mlp_out"]


class DictionaryProfile(StrictModel):
    """The dictionary a feature belongs to, pinned, and how well it fits the scan corpus.

    A latent's strength is its activation ``act(W_enc[:, f] · x + b_enc[f])``, where x is
    read at ``input_site`` of the feature's layer: the MLP's input (``mlp_in``), that
    input's normalisation without the norm's gain and bias (``mlp_in_normalized``, the
    TransformerLens ``ln2.hook_normalized`` convention most public transcoders were
    trained on), or the MLP's output (``mlp_out``, an SAE). With
    ``subtract_decoder_bias``, ``b_dec`` is subtracted from x first, as SAEs often do.
    The latent writes ``a_f · W_dec[f]`` to the MLP's output; ablating it removes that
    write. Every measurement reads the weights at ``repo``/``revision``/``path``, so a
    record reproduces even if the preset that named them changes.

    ``explained_variance`` (the fraction of the MLP output's variance the whole
    dictionary reconstructs) and ``mean_l0`` (active latents per token) are measured on
    the scan's tokens: reported, not gated. A poorly fitting dictionary makes a weak
    baseline. So does one with many ``dead_share`` latents, which a random draw can't
    audit. ``density`` is the share of scanned tokens on which this latent is non-zero.
    """

    kind: Literal["transcoder", "sae"]
    repo: str = Field(min_length=1, description="Hugging Face repository of the weights.")
    revision: str | None = Field(default=None, description="Commit hash of ``repo``.")
    path: str = Field(min_length=1, description="Safetensors file within ``repo``.")
    input_site: DictionarySite
    output_site: Literal["mlp_out"] = "mlp_out"
    activation: Literal["relu", "jumprelu"]
    subtract_decoder_bias: bool = False
    n_features: int = Field(gt=0)
    density: UnitFloat
    explained_variance: float | None = Field(default=None, le=1.0)
    mean_l0: float | None = Field(default=None, ge=0.0)
    dead_share: UnitFloat | None = Field(
        default=None, description="Share of the latents inactive on every scanned token."
    )
    tokens: int = Field(gt=0, description="Scanned tokens the fit statistics cover.")

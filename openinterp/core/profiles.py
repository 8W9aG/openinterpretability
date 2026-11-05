"""What defines the strength of a unit that is not a neuron pole.

For an attention head it is the head's mean output. For a dictionary feature it is the
dictionary. Stored in the record, so every later
measurement of the experiment uses the same definition.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from openinterp.core.strict import StrictModel, UnitFloat


class HeadProfile(StrictModel):
    """What defines an attention head's activity, and its standard attention-pattern scores.

    A head's strength at token t is ``|W_O (z_t − mean_output)|``. This is the distance
    from its write to the residual stream to its mean write over the scan. It is also how
    far mean-ablating the head would move the residual stream at t. The scan measures
    ``mean_output`` (z̄, one value per head dimension) once. Every later measurement of
    the experiment uses this stored value, so a record reproduces without the corpus.

    The pattern scores are mean attention weights on ``pattern_sequences`` sequences.
    Each sequence is ``pattern_length`` uniformly random tokens, repeated twice (Olsson et
    al., 2022). The scores are:

    - ``previous_token``: attention to the token before.
    - ``duplicate_token``: attention from the second copy to the same token in the first.
    - ``induction``: attention from the second copy to the token *after* the same token
      in the first.

    Each score is between 0 and 1. Reported, not gated.
    """

    mean_output: list[float] = Field(min_length=1)
    mean_tokens: int = Field(gt=0, description="Tokens z̄ was averaged over.")
    previous_token_score: UnitFloat
    duplicate_token_score: UnitFloat
    induction_score: UnitFloat
    pattern_sequences: int = Field(gt=0)
    pattern_length: int = Field(gt=1)
    pattern_seed: int


DictionarySite = Literal[
    "resid_mid", "mlp_in", "mlp_in_normalized", "mlp_out", "resid_post", "mlp_hidden"
]
DictionaryOutputSite = Literal["mlp_out", "resid_post"]


def output_site_of(input_site: DictionarySite) -> DictionaryOutputSite:
    """Return where a dictionary that reads ``input_site`` writes.

    A residual-stream SAE reconstructs the residual stream it reads. Every other
    dictionary reconstructs the MLP's output.
    """
    return "resid_post" if input_site == "resid_post" else "mlp_out"


class DictionaryProfile(StrictModel):
    """The dictionary a feature belongs to, pinned, and how well it fits the scan corpus.

    A latent's strength is its activation ``act(W_enc[:, f] · x + b_enc[f])``. The
    harness reads x at ``input_site`` of the feature's layer:

    - ``resid_mid``: the residual stream that the MLP reads, before its norm.
    - ``mlp_in``: the MLP's input.
    - ``mlp_in_normalized``: the norm's input, normalised without its gain and bias. This
      is the TransformerLens ``ln2.hook_normalized`` convention, which most public
      transcoders were trained on.
    - ``mlp_out``: the MLP's output, for an SAE.
    - ``resid_post``: the residual stream after the feature's layer, for a residual-stream
      SAE (TransformerLens's ``hook_resid_post``).
    - ``mlp_hidden``: the MLP's hidden activations, for a ``basis``. These are the down
      projection's input, where native neurons live.

    With ``subtract_decoder_bias``, the harness first subtracts ``b_dec`` from x, as SAEs
    often do. The MLP's output is what the MLP adds to the residual stream. In Gemma 2
    this is after the norm that follows the MLP (TransformerLens's ``hook_mlp_out``).

    Each latent of a ``basis`` is the positive part of one direction in the hidden
    activations, and writes through the down projection. With the identity basis, the
    latents are the native positive poles. With a random orthogonal basis, they are
    rotated units (``docs/analyses/null-calibration.md``).

    An ``spd`` dictionary holds the rank-1 subcomponents of a Stochastic Parameter
    Decomposition of the down projection (``scripts/train_spd.py``). It reads
    ``mlp_hidden``, and its activation is ``causal_importance``. Each subcomponent is
    two latents: the positive and the negative part of ``g · (W_enc[:, f] · x)``. Here
    ``g`` in [0, 1] is the subcomponent's causal importance on that token.

    The latent writes ``a_f · W_dec[f]`` to ``output_site``. That is the MLP's output, or
    for a residual-stream SAE, the residual stream that it reads. Ablating the latent
    removes that write. A skip transcoder's linear skip term is part of its
    reconstruction, not of a latent's write. Every measurement reads the weights at
    ``repo``/``revision``/``path``. So a record reproduces even if its preset changes.

    The harness measures two fits on the scan's tokens. ``explained_variance`` is the
    fraction of ``output_site``'s variance that the whole dictionary reconstructs.
    ``mean_l0`` is the number of active latents per token. Reported, not gated.

    A dictionary that fits poorly makes a weak baseline. So does a dictionary with a
    high ``dead_share``, because a random draw cannot audit a dead latent. ``density`` is
    the share of scanned tokens on which this latent is non-zero.
    """

    kind: Literal["transcoder", "sae", "basis", "spd"]
    repo: str = Field(min_length=1, description="Hugging Face repository of the weights.")
    revision: str | None = Field(default=None, description="Commit hash of ``repo``.")
    path: str = Field(min_length=1, description="Safetensors file within ``repo``.")
    input_site: DictionarySite
    output_site: DictionaryOutputSite = "mlp_out"
    activation: Literal["relu", "jumprelu", "causal_importance"]
    subtract_decoder_bias: bool = False
    n_features: int = Field(gt=0)
    density: UnitFloat
    explained_variance: float | None = Field(default=None, le=1.0)
    mean_l0: float | None = Field(default=None, ge=0.0)
    dead_share: UnitFloat | None = Field(
        default=None, description="Share of the latents inactive on every scanned token."
    )
    tokens: int = Field(
        gt=0, description="Scanned tokens; the fit statistics leave out each piece's first."
    )

    @model_validator(mode="after")
    def _sites_agree(self) -> DictionaryProfile:
        if self.output_site != output_site_of(self.input_site):
            raise ValueError(
                f"a dictionary reading {self.input_site} writes to "
                f"{output_site_of(self.input_site)}, not {self.output_site}"
            )
        return self

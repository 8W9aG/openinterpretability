"""Published sparse dictionaries (transcoders, SAEs) whose latents can be audited.

The learned-basis baseline for the native-neuron premise: audit randomly chosen latents
of a dictionary trained on the same layers, under the same protocol, and compare the
share VERIFIED with native neurons' (Dunefsky et al., 2024, arXiv:2406.11944; Paulo et
al., 2025, arXiv:2501.18823).

A preset (:data:`DICTIONARIES`) names a dictionary family: where its per-layer weights
live and how to read them. A submission pins the exact file in its
:class:`~openinterp.core.schemas.DictionaryProfile`, which is all a later measurement
reads, so changing a preset never changes an old record.

Weights are one safetensors file per layer with ``W_enc`` ``[d_in, n]``, ``b_enc``
``[n]``, ``W_dec`` ``[n, d_out]``, ``b_dec`` ``[d_out]`` (and ``threshold`` ``[n]``
for JumpReLU). Matrices stored the other way round are transposed on load.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from openinterp.core.schemas import DictionaryProfile, DictionarySite


@dataclass(frozen=True)
class DictionarySpec:
    """A family of per-layer dictionaries for one model."""

    model_name: str
    kind: str  # "transcoder" | "sae"
    repo: str
    revision: str
    path: str  # file within ``repo``; ``{layer}`` is filled in
    layers: tuple[int, ...]
    input_site: DictionarySite
    activation: str = "relu"
    subtract_decoder_bias: bool = False
    # Longest context the dictionary fits; feature scans must use windows no longer.
    context_tokens: int | None = None
    reference: str = ""

    def file(self, layer: int) -> str:
        if layer not in self.layers:
            raise ValueError(f"the dictionary has no layer {layer}; it covers {self.layers}")
        return self.path.format(layer=layer)


DICTIONARIES: dict[str, DictionarySpec] = {
    # Checked against openai-community/gpt2 at layer 8: on 16 corpus documents the
    # dictionary explains 66% of the MLP output's variance with ~69 active latents per
    # token when it reads the LayerNorm-standardised input, and nothing when it reads the
    # MLP's (affine) input, which is why the site matters. Read through position 128 the
    # fit holds (layers 0–2: 45–90% of variance), then falls; past position ~200 it goes
    # negative there (−154% at layer 2 over 512-token windows), so scans use 128-token
    # windows, as if the dictionaries were trained on 128-token contexts.
    "gpt2-small-transcoders": DictionarySpec(
        model_name="openai-community/gpt2",
        kind="transcoder",
        repo="jacobdunefsky/gpt2small-transcoders",
        revision="8f254a3a8cda641cb078137415b6b9503e10fc6b",
        path="gpt2-small-dun-chl-mlp-tc{layer}/sae.safetensors",
        layers=tuple(range(12)),
        input_site="mlp_in_normalized",
        context_tokens=128,
        reference="Dunefsky, Chlenski & Nanda, 2024, arXiv:2406.11944",
    ),
}


def dictionary_spec(name: str) -> DictionarySpec:
    try:
        return DICTIONARIES[name]
    except KeyError:
        known = ", ".join(sorted(DICTIONARIES))
        raise ValueError(f"unknown dictionary {name!r}; known: {known}") from None


@dataclass(frozen=True)
class Dictionary:
    """One layer's dictionary, in float32 on one device."""

    w_enc: torch.Tensor  # [d_in, n]
    b_enc: torch.Tensor  # [n]
    w_dec: torch.Tensor  # [n, d_out]
    b_dec: torch.Tensor  # [d_out]
    threshold: torch.Tensor | None
    activation: str
    subtract_decoder_bias: bool

    @property
    def n_features(self) -> int:
        return self.w_enc.shape[1]

    def _input(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().to(self.w_enc.device)
        return x - self.b_dec if self.subtract_decoder_bias else x

    def _activate(self, pre: torch.Tensor, threshold: torch.Tensor | None) -> torch.Tensor:
        if self.activation == "jumprelu":
            return pre * (pre > threshold)
        return torch.relu(pre)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Every latent's activation ``[..., n]``."""
        return self._activate(self._input(x) @ self.w_enc + self.b_enc, self.threshold)

    def feature(self, x: torch.Tensor, f: int | torch.Tensor) -> torch.Tensor:
        """Latent ``f``'s activation ``[...]`` (``[..., k]`` for a tensor of k indices),
        without encoding the rest."""
        pre = self._input(x) @ self.w_enc[:, f] + self.b_enc[f]
        return self._activate(pre, None if self.threshold is None else self.threshold[f])

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.w_dec + self.b_dec


def load_dictionary(profile: DictionaryProfile, device: torch.device | str = "cpu") -> Dictionary:
    """The weights ``profile`` pins, on ``device``."""
    reading = (profile.activation, profile.subtract_decoder_bias)
    return _load(profile.repo, profile.revision, profile.path, str(device), *reading)


def load_spec(spec: DictionarySpec, layer: int, device: torch.device | str = "cpu") -> Dictionary:
    """``spec``'s dictionary for ``layer``, on ``device``."""
    reading = (spec.activation, spec.subtract_decoder_bias)
    return _load(spec.repo, spec.revision, spec.file(layer), str(device), *reading)


@lru_cache(maxsize=2)  # one layer's weights can be hundreds of MB
def _load(repo, revision, path, device, activation, subtract_decoder_bias) -> Dictionary:
    from safetensors.torch import load_file

    local = Path(repo) / path
    if not local.exists():  # a Hugging Face repository; a local directory in tests
        from huggingface_hub import hf_hub_download

        local = Path(hf_hub_download(repo, path, revision=revision))
    # Copied: a tensor left on the file's memory map can be misaligned (see cli._load).
    w = {k: v.to(device, torch.float32, copy=True) for k, v in load_file(str(local)).items()}
    n = w["b_enc"].shape[0]
    w_enc = w["W_enc"] if w["W_enc"].shape[1] == n else w["W_enc"].T
    w_dec = w["W_dec"] if w["W_dec"].shape[0] == n else w["W_dec"].T
    if activation == "jumprelu" and "threshold" not in w:
        raise ValueError(f"{path} is a JumpReLU dictionary without a threshold tensor")
    return Dictionary(
        w_enc=w_enc.contiguous(),
        b_enc=w["b_enc"],
        w_dec=w_dec.contiguous(),
        b_dec=w["b_dec"],
        threshold=w.get("threshold") if activation == "jumprelu" else None,
        activation=activation,
        subtract_decoder_bias=subtract_decoder_bias,
    )


def spec_profile(spec: DictionarySpec, layer: int, **measured) -> DictionaryProfile:
    """A profile pinning ``spec``'s file for ``layer``, with the scan's fit statistics."""
    return DictionaryProfile(
        kind=spec.kind,
        repo=spec.repo,
        revision=spec.revision,
        path=spec.file(layer),
        input_site=spec.input_site,
        activation=spec.activation,
        subtract_decoder_bias=spec.subtract_decoder_bias,
        **measured,
    )

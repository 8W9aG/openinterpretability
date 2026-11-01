"""Published sparse dictionaries (transcoders, SAEs) whose latents can be audited.

The learned-basis baseline for the native-neuron premise: audit randomly chosen latents
of a dictionary trained on the same layers, under the same protocol, and compare the
share VERIFIED with native neurons' (Dunefsky et al., 2024, arXiv:2406.11944; Paulo et
al., 2025, arXiv:2501.18823).

A preset (:data:`DICTIONARIES`) names a dictionary family: where its per-layer weights
live and how to read them. A submission pins the exact file in its
:class:`~openinterp.core.schemas.DictionaryProfile`, which is all a later measurement
reads, so changing a preset never changes an old record.

Weights are one safetensors or NumPy ``.npz`` file per layer with ``W_enc``
``[d_in, n]``, ``b_enc`` ``[n]``, ``W_dec`` ``[n, d_out]``, ``b_dec`` ``[d_out]`` (and
``threshold`` ``[n]`` for JumpReLU). Matrices stored the other way round are transposed on
load. An OpenMOSS ``lm_sae`` checkpoint (``encoder.weight`` …, with ``hyperparams.json``
beside its ``checkpoints/`` directory, as Llama Scope ships) is converted on load; see
:func:`_from_lm_sae`. A skip transcoder also has ``W_skip`` ``[d_out, d_in]``, a linear
map from its input that it adds to its reconstruction (circuit-tracer's convention); a
latent's write doesn't include it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from openinterp.core.profiles import output_site_of
from openinterp.core.schemas import DictionaryProfile, DictionarySite


@dataclass(frozen=True)
class DictionarySpec:
    """A family of per-layer dictionaries for one model."""

    model_name: str
    kind: str  # "transcoder" | "sae"
    repo: str
    revision: str
    path: str | Mapping[int, str]  # file within ``repo``: ``{layer}`` is filled in, or by layer
    layers: tuple[int, ...]
    input_site: DictionarySite
    activation: str = "relu"
    subtract_decoder_bias: bool = False
    # Longest context the dictionary fits; feature scans must use windows no longer.
    context_tokens: int | None = None
    reference: str = ""
    # Neuronpedia's source for a layer (``{model}/{layer}-{set}``), where it hosts these
    # same latents under the same indices; ``None`` where it doesn't or that's unchecked.
    neuronpedia: str | None = None

    def neuronpedia_source(self, layer: int) -> str | None:
        return self.neuronpedia.format(layer=layer) if self.neuronpedia else None

    def file(self, layer: int) -> str:
        if layer not in self.layers:
            raise ValueError(f"the dictionary has no layer {layer}; it covers {self.layers}")
        if isinstance(self.path, Mapping):
            return self.path[layer]
        return self.path.format(layer=layer)


DICTIONARIES: dict[str, DictionarySpec] = {
    # Checked against openai-community/gpt2 at layer 8: on 16 corpus documents the
    # dictionary explains 66% of the MLP output's variance with ~69 active latents per
    # token when it reads the LayerNorm-standardised input, and nothing when it reads the
    # MLP's (affine) input, which is why the site matters. Read through position 128 the
    # fit holds (layers 0–2: 45–90% of variance), then falls; past position ~200 it goes
    # negative there (−154% at layer 2 over 512-token windows), so scans use 128-token
    # windows, as if the dictionaries were trained on 128-token contexts. Neuronpedia's
    # tres-dc are these latents: on the top example of 24 latents across layers 2–11, each
    # peaks on the same token here as there (activations correlate at 0.94–1.0; the peak's
    # size differs because Neuronpedia's snippets are cut from longer contexts).
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
        neuronpedia="gpt2-small/{layer}-tres-dc",
    ),
    # The checks below each read 16 corpus documents through position 1024 with the first
    # token left out, and found the fit flat with position. Each site was the only one of
    # resid_mid, mlp_in and mlp_in_normalized that fit; the others explain less than 0.
    #
    # Read at mlp_in_normalized: 63% of the MLP output's variance at layer 5 with 88 active
    # latents per token (the file says 87) and 32% at layer 12 (L0 6). Gemma 2 normalises
    # its MLP's output before adding it to the residual stream, and the transcoders
    # reconstruct that normalised output: against the down projection's output, the fit is
    # −180. These are the lowest-L0 transcoders circuit-tracer uses, layer by layer.
    "gemma-scope-2b-transcoders": DictionarySpec(
        model_name="google/gemma-2-2b",
        kind="transcoder",
        repo="google/gemma-scope-2b-pt-transcoders",
        revision="50eec2f25c60545a9a74c1c3a26a0afdd0b4b872",
        path={
            layer: f"layer_{layer}/width_16k/average_l0_{l0}/params.npz"
            for layer, l0 in enumerate(
                (76, 65, 49, 54, 88, 87, 95, 70, 52, 72, 88, 5, 6)
                + (8, 8, 8, 10, 12, 13, 12, 11, 13, 15, 25, 37, 41)
            )
        },
        layers=tuple(range(26)),
        input_site="mlp_in_normalized",
        activation="jumprelu",
        context_tokens=1024,
        reference="Lieberum et al., 2024, arXiv:2408.05147",
    ),
    # An MLP-output SAE: 70% of the variance at layer 5 (L0 114), 63% at layer 12 (L0
    # 113), reading the normalised output Gemma 2 adds to the residual stream; read
    # before that norm, the fit is −14. Each layer has Gemma Scope's canonical 16k SAE,
    # the one whose L0 is closest to 100 (Neuronpedia's gemmascope-mlp-16k).
    "gemma-scope-2b-mlp-saes": DictionarySpec(
        model_name="google/gemma-2-2b",
        kind="sae",
        repo="google/gemma-scope-2b-pt-mlp",
        revision="4c9df613baa9660b522f4c3908b13c9aee41bcad",
        path={
            layer: f"layer_{layer}/width_16k/average_l0_{l0}/params.npz"
            for layer, l0 in enumerate(
                (119, 105, 95, 95, 85, 114, 133, 60, 136, 88, 110, 98, 108)
                + (112, 97, 80, 72, 68, 106, 109, 109, 113, 121, 128, 73, 126)
            )
        },
        layers=tuple(range(26)),
        input_site="mlp_out",
        activation="jumprelu",
        context_tokens=1024,
        reference="Lieberum et al., 2024, arXiv:2408.05147",
        neuronpedia="gemma-2-2b/{layer}-gemmascope-mlp-16k",
    ),
    # A residual-stream SAE, reading and writing the residual stream after each layer
    # (hook_resid_post): 76–93% of its variance at layers 0, 5, 12, 20 and 25 with L0
    # 65–116; read one step earlier, before the MLP (resid_mid), the fit is −3 to −8.
    # Gemma Scope's canonical 16k SAEs, as for the MLP SAEs (Neuronpedia's
    # gemmascope-res-16k). A latent here isn't an MLP unit: compared with native neurons,
    # it is a different site, not only a different basis.
    "gemma-scope-2b-res-saes": DictionarySpec(
        model_name="google/gemma-2-2b",
        kind="sae",
        repo="google/gemma-scope-2b-pt-res",
        revision="fd571b47c1c64851e9b1989792367b9babb4af63",
        path={
            layer: f"layer_{layer}/width_16k/average_l0_{l0}/params.npz"
            for layer, l0 in enumerate(
                (105, 102, 141, 59, 124, 68, 70, 69, 71, 73, 77, 80, 82)
                + (84, 84, 78, 78, 77, 74, 73, 71, 70, 72, 75, 73, 116)
            )
        },
        layers=tuple(range(26)),
        input_site="resid_post",
        activation="jumprelu",
        context_tokens=1024,
        reference="Lieberum et al., 2024, arXiv:2408.05147",
        neuronpedia="gemma-2-2b/{layer}-gemmascope-res-16k",
    ),
    # Llama Scope (He et al., 2024, arXiv:2410.20526): top-k SAEs and transcoders for
    # Llama-3.1-8B, converted to JumpReLU, in OpenMOSS's lm_sae format (see _from_lm_sae).
    # Read as Neuronpedia reads them: on the sentence "The quick brown fox jumps over the
    # lazy dog. Paris is the capital of France.", L15R latents 1791 and 2290 fire on the
    # same tokens with the same values as Neuronpedia's 15-llamascope-res-32k, to bf16
    # precision. At layer 15 that reading fits 4 points worse than taking θ unscaled
    # (residual: 62.8% vs 66.5%), with L0 33 instead of 228; the scaling, measured on the
    # corpus, matches the hyperparameters' (resid_post 10.80 vs 10.81, mlp_out 5.34 vs
    # 5.28). Over layers 0, 8, 15, 24 and 31 through position 1024 the residual SAEs fit
    # 54–86%, the MLP SAEs 30–84% (weakest mid-network) and the transcoders, reading
    # mlp_in_normalized as their hook_point_in says, 44–87%, the best of the three input
    # sites at every layer. Fits fall 2–8 points from the first 256 positions to 512–1024.
    # The 8x (32k-latent) dictionaries, which the Llama Scope README recommends.
    "llama-scope-8b-res-saes": DictionarySpec(
        model_name="meta-llama/Llama-3.1-8B",
        kind="sae",
        repo="fnlp/Llama3_1-8B-Base-LXR-8x",
        revision="8dbc1d85edfced43081c03c38b05514dbab1368b",
        path="Llama3_1-8B-Base-L{layer}R-8x/checkpoints/final.safetensors",
        layers=tuple(range(32)),
        input_site="resid_post",
        activation="jumprelu",
        context_tokens=1024,
        reference="He et al., 2024, arXiv:2410.20526",
        neuronpedia="llama3.1-8b/{layer}-llamascope-res-32k",
    ),
    "llama-scope-8b-mlp-saes": DictionarySpec(
        model_name="meta-llama/Llama-3.1-8B",
        kind="sae",
        repo="fnlp/Llama3_1-8B-Base-LXM-8x",
        revision="69284f67f460bc42c91abe494b3166810179bbba",
        path="Llama3_1-8B-Base-L{layer}M-8x/checkpoints/final.safetensors",
        layers=tuple(range(32)),
        input_site="mlp_out",
        activation="jumprelu",
        context_tokens=1024,
        reference="He et al., 2024, arXiv:2410.20526",
    ),
    "llama-scope-8b-transcoders": DictionarySpec(
        model_name="meta-llama/Llama-3.1-8B",
        kind="transcoder",
        repo="fnlp/Llama3_1-8B-Base-LXTC-8x",
        revision="a51e5499031b69022363a73f1fd1dece523668dd",
        path="Llama3_1-8B-Base-L{layer}TC-8x/checkpoints/final.safetensors",
        layers=tuple(range(32)),
        input_site="mlp_in_normalized",
        activation="jumprelu",
        context_tokens=1024,
        reference="He et al., 2024, arXiv:2410.20526",
    ),
    # circuit-tracer's skip transcoders read the residual stream before the MLP's norm
    # (hook_resid_mid): 78% of the variance at layer 5 with L0 47. The skip term carries
    # most of it: without it, the latents alone explain 9%. Applied the other way round
    # (x @ W_skip rather than x @ W_skip.T), it explains −0.70.
    "llama-3.2-1b-transcoders": DictionarySpec(
        model_name="meta-llama/Llama-3.2-1B",
        kind="transcoder",
        repo="mntss/transcoder-Llama-3.2-1B",
        revision="c37a82c1ec4cea30d424850d159b17b720ce19e2",
        path="layer_{layer}.safetensors",
        layers=tuple(range(16)),
        input_site="resid_mid",
        context_tokens=1024,
        reference="circuit-tracer, github.com/safety-research/circuit-tracer; skip "
        "transcoders: Paulo, Shabalin & Belrose, 2025, arXiv:2501.18823",
    ),
    # Read at mlp_in, the MLP's input after its norm's gain: 60% of the variance at
    # layer 5 with L0 11.
    "qwen3-0.6b-transcoders": DictionarySpec(
        model_name="Qwen/Qwen3-0.6B",
        kind="transcoder",
        repo="mwhanna/qwen3-0.6b-transcoders-lowl0",
        revision="f28e5268fa1f3f685e028a8ba64a553e862dc7c3",
        path="layer_{layer}.safetensors",
        layers=tuple(range(28)),
        input_site="mlp_in",
        context_tokens=1024,
        reference="circuit-tracer, github.com/safety-research/circuit-tracer",
    ),
    # No public dictionary covers Qwen2.5-0.5B, so these were trained here, with
    # scripts/train_transcoders.py on 13.5M pile-10k tokens that leave out the scan
    # corpus's documents. They fit 59–93% of the scan corpus's MLP output (L0 ~32);
    # without the skip path, at most 34%. The card is in hf_models/.
    "qwen2.5-0.5b-transcoders": DictionarySpec(
        model_name="Qwen/Qwen2.5-0.5B",
        kind="transcoder",
        repo="sackfield/qwen2.5-0.5b-transcoders",
        revision="d10da828bcd803ffac8cf3db6550f5a1dff4eb13",
        path="layer_{layer}.safetensors",
        layers=tuple(range(24)),
        input_site="mlp_in",
        activation="jumprelu",
        context_tokens=1024,
        reference="openinterp, scripts/train_transcoders.py",
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
    w_skip: torch.Tensor | None = None  # [d_in, d_out]

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

    def reconstruct(self, x: torch.Tensor, acts: torch.Tensor | None = None) -> torch.Tensor:
        """The dictionary's estimate of its output from its input ``x`` (whose latent
        activations are ``acts``, if already encoded), with any skip connection."""
        out = self.decode(self.encode(x) if acts is None else acts)
        if self.w_skip is not None:
            out = out + x.float().to(self.w_skip.device) @ self.w_skip
        return out


def load_dictionary(profile: DictionaryProfile, device: torch.device | str = "cpu") -> Dictionary:
    """The weights ``profile`` pins, on ``device``."""
    reading = (profile.activation, profile.subtract_decoder_bias)
    return _load(profile.repo, profile.revision, profile.path, str(device), *reading)


def load_spec(spec: DictionarySpec, layer: int, device: torch.device | str = "cpu") -> Dictionary:
    """``spec``'s dictionary for ``layer``, on ``device``."""
    reading = (spec.activation, spec.subtract_decoder_bias)
    return _load(spec.repo, spec.revision, spec.file(layer), str(device), *reading)


def _read(local: Path) -> dict[str, torch.Tensor]:
    if local.suffix == ".npz":
        import numpy as np

        with np.load(local) as arrays:
            return {k: torch.from_numpy(arrays[k]) for k in arrays.files}
    from safetensors.torch import load_file

    return load_file(str(local))


def _fetch(repo: str, revision: str | None, path: str) -> Path:
    local = Path(repo) / path
    if local.exists():  # a local directory in tests
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, path, revision=revision))


def _from_lm_sae(w: dict[str, torch.Tensor], hyperparams: dict) -> dict[str, torch.Tensor]:
    """An OpenMOSS ``lm_sae`` checkpoint (Llama Scope's format) as plain JumpReLU weights.

    lm_sae scales its input by ``s_in = sqrt(d) / dataset_average_activation_norm["in"]``
    and its reconstruction back by ``s_out``, and has one ``jump_relu_threshold`` θ.
    Neuronpedia (through SAELens) reads it with a latent active where its pre-activation
    ``h = s_in·x·W_enc + b_enc`` exceeds ``θ·s_in``, at strength ``h·‖W_dec[f]‖ / s_in``.
    Folding those into the weights gives the same activations, so a latent here reads as
    it does there.
    """
    d = w["decoder.bias"].shape[0]
    norms = hyperparams["dataset_average_activation_norm"]
    s_in, s_out = d**0.5 / norms["in"], d**0.5 / norms["out"]
    w_enc, w_dec = w["encoder.weight"].T, w["decoder.weight"].T  # [d_in, n], [n, d_out]
    dec_norm = w_dec.norm(dim=-1)  # [n]
    return {
        # a = h·‖W_dec[f]‖ / s_in, so a > θ·‖W_dec[f]‖ exactly where h > θ·s_in
        "W_enc": w_enc * dec_norm,
        "b_enc": w["encoder.bias"] * dec_norm / s_in,
        "threshold": hyperparams["jump_relu_threshold"] * dec_norm,
        # h·W_dec / s_out = a·W_dec·s_in / (‖W_dec[f]‖·s_out)
        "W_dec": w_dec * (s_in / s_out) / dec_norm[:, None],
        "b_dec": w["decoder.bias"] / s_out,
    }


@lru_cache(maxsize=2)  # one layer's weights can be hundreds of MB
def _load(repo, revision, path, device, activation, subtract_decoder_bias) -> Dictionary:
    local = _fetch(repo, revision, path)
    # Copied: a tensor left on the file's memory map can be misaligned (see cli._load).
    w = {k: v.to(device, torch.float32, copy=True) for k, v in _read(local).items()}
    if "encoder.weight" in w:  # lm_sae: its hyperparameters sit beside "checkpoints/"
        hyperparams_path = str(Path(path).parent.parent / "hyperparams.json")
        hyperparams = json.loads(_fetch(repo, revision, hyperparams_path).read_text())
        w = _from_lm_sae(w, hyperparams)
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
        w_skip=w["W_skip"].T.contiguous() if "W_skip" in w else None,
    )


def spec_profile(spec: DictionarySpec, layer: int, **measured) -> DictionaryProfile:
    """A profile pinning ``spec``'s file for ``layer``, with the scan's fit statistics."""
    return DictionaryProfile(
        kind=spec.kind,
        repo=spec.repo,
        revision=spec.revision,
        path=spec.file(layer),
        input_site=spec.input_site,
        output_site=output_site_of(spec.input_site),
        activation=spec.activation,
        subtract_decoder_bias=spec.subtract_decoder_bias,
        **measured,
    )

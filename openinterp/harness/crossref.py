"""Dictionary latents aligned with a neuron pole's weights.

A pole writes ``±w_out`` (its down-projection column, negated for the negative pole) and
reads its MLP's input with ``gate``/``up`` (or ``in``). A latent writes ``W_dec[f]`` and,
when its dictionary reads the MLP's input (a transcoder), reads it with ``W_enc[:, f]``.
Both are compared by cosine in residual-stream coordinates, with each norm folded in as
the weight wiring folds it (:func:`openinterp.harness.wiring._fold`), and oriented so that
+ means the latent writes what the pole writes or reads what drives the pole
(:func:`orientation`): such a latent is a candidate for the same feature.

Two references put a cosine in context. ``chance`` is the 99th percentile, over random
directions, of the best cosine any latent reaches. ``percentile`` is the share of the
layer's other neurons whose best match is weaker than this one's, so a pole the dictionary
happens to have a latent for stands out from a layer the dictionary fits well everywhere.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

from openinterp.core.exceptions import ArchitectureNotSupportedError
from openinterp.harness.dictionaries import Dictionary, DictionarySpec, load_spec
from openinterp.harness.hooks import NeuronHookRunner
from openinterp.harness.wiring import _fold, _matrix, mlp_reads

# Sites where a dictionary's encoder reads what the MLP reads.
READS_MLP_INPUT = ("resid_mid", "mlp_in", "mlp_in_normalized")
CHANCE_DRAWS = 256
CHUNK = 512


@dataclass(frozen=True)
class Match:
    feature_idx: int
    cosine: float
    channel: str | None = None  # the neuron's read channel, for a read


@dataclass(frozen=True)
class Side:
    """One pole's best-aligned latents on one side (write or read)."""

    matches: list[Match]
    chance: float
    percentile: float


def _unit(rows: torch.Tensor) -> torch.Tensor:
    return rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def latent_writes(runner: NeuronHookRunner, spec: DictionarySpec, layer: int, d: Dictionary):
    """Each latent's write ``[n, d_model]``, in the space a pole writes to."""
    if runner.mlp_write(layer) is not runner.down_proj(layer):
        # Gemma 2 normalises the MLP's output, and its dictionaries decode after that norm.
        raise ArchitectureNotSupportedError(
            "the MLP's output is normalised before the residual stream, so a neuron's write "
            "and a latent's are not in the same space"
        )
    return d.w_dec.float()


def latent_reads(
    runner: NeuronHookRunner, spec: DictionarySpec, layer: int, d: Dictionary
) -> torch.Tensor | None:
    """Each latent's read ``[n, d_model]`` of the residual stream, or ``None`` when its
    dictionary doesn't read the MLP's input."""
    if spec.input_site not in READS_MLP_INPUT:
        return None
    read = d.w_enc.T.float()
    if spec.input_site == "resid_mid":
        return read
    return _fold(runner.mlp_norm(layer), read, gain=spec.input_site == "mlp_in")


def _seed(name: str) -> int:
    """A seed no training run is likely to share. A dictionary's dead latents keep their
    initial weights, and with a small seed such as 0 the "random" directions would be
    those weights (cosine 0.93 with Qwen2.5-0.5B's layer-0 transcoder)."""
    return int.from_bytes(hashlib.sha256(f"openinterp crossref {name}".encode()).digest()[:8])


def _null(runner: NeuronHookRunner, layer: int, latents: torch.Tensor, name: str) -> float:
    """The 99th percentile of the best cosine between a random direction and any latent.
    Directions are centred as LayerNorm centres a read, so they live where reads do."""
    g = torch.Generator().manual_seed(_seed(f"{name} {layer}"))
    r = torch.randn(CHANCE_DRAWS, latents.shape[1], generator=g).to(latents.device)
    if isinstance(runner.mlp_norm(layer), nn.LayerNorm):
        r = r - r.mean(-1, keepdim=True)
    best = (_unit(r) @ latents.T).amax(-1)
    return float(best.quantile(0.99))


def orientation(channel: str, pole: str) -> float:
    """The sign of a vector that drives ``pole`` along ``channel``. A latent fires
    non-negatively, so only its own direction counts. A pole writes ``±w_out``. A gated
    neuron fires when its gate opens, whatever the pole, with the sign of ``up``; an
    ungated one with the sign of its pre-activation."""
    return 1.0 if channel == "gate" or pole == "positive" else -1.0


def _align(
    channels: dict[str, torch.Tensor],
    latents: torch.Tensor,
    poles: Iterable[tuple[int, str]],
    k: int,
) -> tuple[dict[tuple[int, str], list[Match]], dict[str, torch.Tensor]]:
    """The top ``k`` latents by oriented cosine for each ``(neuron, pole)``, over all the
    neuron's channels (a latent appears once, at its best channel), and every neuron's
    best oriented cosine, by pole."""
    poles = sorted(set(poles))
    width = next(iter(channels.values())).shape[0]
    best = {p: torch.full((width,), -1.0) for p in ("positive", "negative")}
    found: dict[tuple[int, str], list[Match]] = {p: [] for p in poles}
    for channel, rows in channels.items():
        for start in range(0, width, CHUNK):
            cos = _unit(rows[start : start + CHUNK].float()) @ latents.T  # [c, n]
            highest, lowest = cos.amax(-1).cpu(), (-cos).amax(-1).cpu()
            for pole, b in best.items():
                reach = highest if orientation(channel, pole) > 0 else lowest
                b[start : start + CHUNK] = torch.maximum(b[start : start + CHUNK], reach)
            for i, pole in poles:
                if start <= i < start + CHUNK:
                    row = orientation(channel, pole) * cos[i - start]
                    top = row.topk(k).indices
                    found[(i, pole)] += [Match(int(f), float(row[f]), channel) for f in top]
    for key, matches in found.items():
        by_feature: dict[int, Match] = {}
        for m in sorted(matches, key=lambda m: -m.cosine):
            by_feature.setdefault(m.feature_idx, m)
        found[key] = list(by_feature.values())[:k]
    return found, best


def align_layer(
    runner: NeuronHookRunner,
    spec: DictionarySpec,
    layer: int,
    poles: Iterable[tuple[int, str]],
    *,
    k: int = 5,
) -> dict[tuple[int, str], tuple[Side, Side | None]]:
    """``{(neuron, pole): (write, read)}`` against ``spec``'s dictionary for ``layer``;
    ``read`` is ``None`` when the dictionary doesn't read the MLP's input."""
    poles = list(poles)
    device = runner.device
    d = load_spec(spec, layer, device)
    writes = _unit(latent_writes(runner, spec, layer, d))
    w_out = _matrix(runner.down_proj(layer)).T.to(device)  # [width, d_model]
    write_null = _null(runner, layer, writes, "write")
    by_write, write_best = _align({"out": w_out}, writes, poles, k)

    reads = latent_reads(runner, spec, layer, d)
    by_read = read_best = read_null = None
    if reads is not None:
        reads = _unit(reads)
        read_null = _null(runner, layer, reads, "read")
        neuron_reads = {c: r.to(device) for c, r in mlp_reads(runner, layer).items()}
        by_read, read_best = _align(neuron_reads, reads, poles, k)

    def share_below(best: dict[str, torch.Tensor], i: int, pole: str) -> float:
        return float((best[pole] < best[pole][i]).float().mean())

    out = {}
    for i, pole in poles:
        write_matches = [Match(m.feature_idx, m.cosine) for m in by_write[(i, pole)]]
        write = Side(write_matches, write_null, share_below(write_best, i, pole))
        read = None
        if by_read is not None:
            read = Side(by_read[(i, pole)], read_null, share_below(read_best, i, pole))
        out[(i, pole)] = (write, read)
    return out

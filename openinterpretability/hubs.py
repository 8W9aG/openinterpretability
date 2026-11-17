"""Hubs as entities: identity, derived standing and artefact facts (issue #4285).

A hub is a neuron or head that the wiring joins to many studied units
(:func:`openinterpretability.builder.circuits.hubs`). It has no experiments of its own. So
its standing is its center's: the same ``Standing`` per pole, or for the head, that the
center's page shows. Nothing is stored.

A hub can be an artefact, so its facts are reported, not gated (``docs/circuits-and-hubs.md``):

- *Default-on.* The share of scan tokens that the center is active on, from its lead
  experiment, and whether that reaches the protocol's ``max_active_share``.
- *Output norm against its layer.* The center's output-weight norm, the median of its layer,
  and its percentile there (``neuron-family-triage``). For a neuron, the norm is |W_out
  column| over the layer's neurons. For a head, it is |W_O,h| over its heads. These come
  from weights, so ``open-interpretability circuits
  hubs`` records them in ``data/hubs/<model>.jsonl``, and the build reads the newest row.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from openinterpretability.core.identifiers import HubIdentifier, model_slug
from openinterpretability.core.strict import StrictModel

HUBS_DIR = Path("data/hubs")


class NormRow(StrictModel):
    """One hub center's output-weight norm against its layer, on one day."""

    hub: str  # the HubIdentifier slug
    date: str
    norm: float
    layer_median: float
    percentile: float  # the share of the layer's neurons or heads with a smaller norm


@dataclass
class HubEntity:
    """A hub with its identity, derived standing and artefact facts."""

    identifier: HubIdentifier
    neighbours: int  # the studied units wired to it
    standings: dict = field(default_factory=dict)  # pole (or "head") → the center's Standing
    active_share: dict[str, float | None] = field(default_factory=dict)  # per side
    default_on: dict[str, bool] = field(default_factory=dict)  # per side
    norm: NormRow | None = None


def identifier(unit) -> HubIdentifier:
    """The hub identifier of a center, a :class:`~openinterpretability.builder.tree.UnitNode`."""
    center = "head" if unit.is_head else "neuron"
    return HubIdentifier(model_name=unit.model, layer=unit.layer, center=center, index=unit.idx)


def _facts(standing) -> tuple[float | None, bool]:
    """The active share and the default-on flag from a standing's lead experiment."""
    from openinterpretability.core.schemas import ReviewPolicy, scan_active_share
    from openinterpretability.planner import _default_on

    record = standing.lead.record
    metrics = record.verdict.metrics
    share = metrics.active_share if metrics and metrics.active_share is not None else None
    if share is None:
        share = scan_active_share(record.submission, ReviewPolicy())
    return share, _default_on(record)


def hub_entities(found_hubs: Iterable, norms: dict[str, NormRow] | None = None) -> list[HubEntity]:
    """A :class:`HubEntity` for each hub that the site finds, in the same order."""
    out = []
    for hub in found_hubs:
        ident = identifier(hub.unit)
        entity = HubEntity(ident, len(hub.neighbours), dict(hub.unit.standings))
        for side, standing in hub.unit.standings.items():
            entity.active_share[side], entity.default_on[side] = _facts(standing)
        entity.norm = (norms or {}).get(ident.slug)
        out.append(entity)
    return out


def path_for(model: str, folder: Path = HUBS_DIR) -> Path:
    """``data/hubs/<model slug>.jsonl``."""
    return folder / f"{model_slug(model)}.jsonl"


def load_norms(path: Path) -> dict[str, NormRow]:
    """The newest row per hub, or none when the file is missing."""
    if not path.exists():
        return {}
    newest: dict[str, NormRow] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = NormRow.model_validate_json(line)
            if row.hub not in newest or row.date >= newest[row.hub].date:
                newest[row.hub] = row
    return newest


def layer_norms(runner, layer: int, center: str):
    """Every neuron's |W_out column|, or every head's |W_O,h|, in ``layer``."""
    import torch

    if center == "head":
        heads = range(runner.n_heads)
        return torch.stack([runner.head_output_weights(layer, h).float().norm() for h in heads])
    from openinterpretability.harness.memory import loaded

    proj = runner.down_proj(layer)
    with loaded(proj):  # streamed: the weights are real only inside this block
        w = proj.weight.detach().float()
        return (w.norm(dim=0) if hasattr(proj, "in_features") else w.norm(dim=1)).cpu()


def norm_row(runner, ident: HubIdentifier, date: str) -> NormRow:
    """The center's norm, its layer's median and its percentile, from the weights."""
    norms = layer_norms(runner, ident.layer, ident.center)
    own = float(norms[ident.index])
    return NormRow(
        hub=ident.slug, date=date, norm=round(own, 6), layer_median=round(float(norms.median()), 6),
        percentile=round(float((norms < own).float().mean()), 6),
    )  # fmt: skip

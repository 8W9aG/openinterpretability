"""The interference report: rows, footprint and verdict (#5230).

It compares a base side with a changed side (:mod:`.floors`). Every comparison is also run
with each noise floor in place of the changed side: the base against itself, and a matched
random change. A change counts only when it is above both floors.

- **Keep flips.** A VERIFIED pole flips when its probes pass on the base and fail on the
  changed side. It counts when no floor flips it too.
- **Footprint.** It holds every neuron whose activation drifts more than under either floor.
  The drift is one minus its correlation with the base. It also holds every neuron whose
  output column changes more than under either floor. A neuron that a patch edits directly
  needs only to move above the self floor. The random floor edits the same weights by the
  same amount.
- **Verdict.** It is *interference* when a counted flip names a VERIFIED pole. Otherwise it
  is *footprint not yet studied* when the footprint holds a neuron the registry hasn't
  studied. Otherwise it is *no known interference*.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from openinterpretability.interference import behaviour, keep, units

TOL = 1e-6  # how far above a floor a measure must be to count


def studied(wiki_dir: Path, model: str) -> set[tuple[int, int]]:
    """Each (layer, neuron) with an experiment in the registry."""
    from openinterpretability.core.identifiers import model_slug
    from openinterpretability.core.paths import experiments_dir

    root = experiments_dir(Path(wiki_dir)) / model_slug(model)
    found = set()
    for path in root.glob("L*/N*-*") if root.is_dir() else []:
        if (m := re.fullmatch(r"N(\d+)-\w+", path.name)) and path.parent.name[1:].isdigit():
            found.add((int(path.parent.name[1:]), int(m.group(1))))
    return found


def edited(ops) -> set[tuple[int, int | None]]:
    """The neurons a patch edits directly: (layer, neuron), or (layer, None) for all of them."""
    from openinterpretability.patches.graph import parse

    out: set[tuple[int, int | None]] = set()
    for _, op in ops:
        for target in op.targets:
            try:
                name = parse(target)
            except ValueError:
                continue  # a ref to another op, not a site
            if name.layer is None or not name.part.startswith("mlp."):
                continue
            index = op.params.get("index")
            if op.type == "sparse_add" and isinstance(index, list):
                out |= {(name.layer, int(i)) for i in index}
            elif op.type in ("prune", "clamp") and isinstance(index, int):
                out.add((name.layer, index))
            elif op.type in ("dense_add", "lowrank_add", "replace", "scale"):
                out.add((name.layer, None))
    return out


def _keep(sides: dict, records: list, batches) -> dict[str, dict]:
    """Each VERIFIED pole's keep readings on every side, with its scale ratio there."""
    by_layer: dict[int, list] = {}
    for r in records:
        by_layer.setdefault(r.submission.identifier.layer, []).append(r)
    out: dict[str, dict] = {}
    for layer, group in sorted(by_layer.items()):
        neurons = sorted({r.submission.identifier.neuron_idx for r in group})
        cols = {}
        for name, side in sides.items():
            with side.entered() as runner:
                cols[name] = keep.layer_columns(runner, batches, layer, neurons)
        for record in group:
            ident = record.submission.identifier
            j = neurons.index(ident.neuron_idx)
            top = keep.pole_max(cols["base"][:, j], ident.pole)
            row = {}
            for name, side in sides.items():
                ratio = keep.pole_max(cols[name][:, j], ident.pole) / top if top > 0 else 1.0
                with side.entered() as runner:
                    row[name] = keep.readings(runner, record, ratio or 1.0)
            out[ident.slug] = row
    return out


def _flips(readings: dict[str, dict]) -> dict:
    """Whether the pole flips on each side, and whether the changed side's flip counts."""
    base = readings["base"]["passes"]
    flips = {k: base and not v["passes"] for k, v in readings.items() if k != "base"}
    floors = [v for k, v in flips.items() if k.startswith("floor")]
    return {"flips": flips, "counts": flips["changed"] and not any(floors)}


def _moved(stats: dict, columns: dict, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Which neurons of ``layer`` move above both floors, and above the self floor alone."""
    drift = {k: (1 - s.pearson(layer)).nan_to_num(0.0) for k, s in stats.items()}
    change = {k: c[layer] for k, c in columns.items()}

    def above(names: list[str]) -> torch.Tensor:
        floor_drift = torch.stack([drift[k] for k in names]).max(0).values
        floor_change = torch.stack([change[k] for k in names]).max(0).values
        return (drift["changed"] > floor_drift + TOL) | (change["changed"] > floor_change + TOL)

    floors = [k for k in drift if k != "changed"]
    return above(floors), above([k for k in floors if k == "floor:self"] or floors)


def footprint(stats: dict, columns: dict, n_layers: int, direct: set, known: set) -> list[dict]:
    """The footprint's rows: each neuron that moves above both floors, studied or not.

    A neuron that the patch edits directly needs only to move above the self floor. The
    random floor edits the same weights by the same amount.
    """
    rows = []
    for layer in range(n_layers):
        moved, moved_at_all = _moved(stats, columns, layer)
        for neuron in range(moved.shape[0]):
            is_direct = (layer, None) in direct or (layer, neuron) in direct
            if not (bool(moved[neuron]) or (is_direct and bool(moved_at_all[neuron]))):
                continue
            r = stats["changed"].pearson(layer)[neuron]
            out_change = columns["changed"][layer][neuron]
            rows.append({"table": "footprint", "layer": layer, "neuron": neuron,
                         "studied": (layer, neuron) in known, "edited": is_direct,
                         "drift": round(float(1 - r), 6) if not torch.isnan(r) else None,
                         "output_change": round(float(out_change), 6)})  # fmt: skip
    return rows


def verdict(flipped: list[str], unstudied: list[dict]) -> str:
    """The report's verdict (module docstring)."""
    if flipped:
        return "interference"
    return "footprint not yet studied" if unstudied else "no known interference"


def coverage(base_stats, records: list, n_layers: int) -> dict:
    """The share of poles that stand VERIFIED, and the share of activation mass they carry."""
    total = sum(float(base_stats.sums[la]["pos"].sum() + base_stats.sums[la]["neg"].sum())
                for la in range(n_layers))  # fmt: skip
    poles = sum(2 * base_stats.sums[la]["pos"].shape[0] for la in range(n_layers))
    mass = 0.0
    for r in records:
        ident = r.submission.identifier
        side = "pos" if ident.pole == "positive" else "neg"
        mass += float(base_stats.sums[ident.layer][side][ident.neuron_idx])
    return {"verified_poles": len(records), "poles": poles,
            "pole_share": round(len(records) / poles, 6) if poles else 0.0,
            "mass_share": round(mass / total, 6) if total else 0.0}  # fmt: skip


__all__ = ["behaviour", "coverage", "edited", "footprint", "studied", "units", "verdict"]

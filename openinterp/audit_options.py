"""Options that ``open-interpretability audit`` and ``scan`` share, and how unit options resolve.

The shared ones are the scan cache and ``--selection``.

``--selection`` says how a chosen unit was selected, as its provenance records it.

A ``--random`` draw records ``random`` and counts towards the registry statistics. A
unit given by layer and index records ``manual``, unless ``--selection`` says otherwise.
The planner's effect-weighted explore draws (issue #3486) pass ``attribution``. They are
drawn in proportion to an attribution-patching estimate, so they must not count as random.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

scan_cache_option = click.option(
    "--scan-cache/--no-scan-cache",
    "use_scan_cache",
    default=True,
    show_default=True,
    help="Replay a cached scan of the layer, or store this one "
    "(see `open-interpretability scan-cache`).",
)


def open_scan_cache(enabled: bool):
    """The default layer-scan cache, or ``None`` when ``enabled`` is false."""
    from openinterp.harness.layer_cache import LayerCache

    return LayerCache.default() if enabled else None


CHOSEN = ("manual", "attribution")  # selections a unit given by layer and index may record

selection_option = click.option(
    "--selection",
    type=click.Choice(CHOSEN),
    help="How a unit given by layer and index was chosen (default manual). The planner's "
    "effect-weighted draws pass attribution. Ignored with --random.",
)


def selection(random_pick: bool, chosen: str | None) -> str:
    """The ``provenance.selection`` of a scan."""
    return "random" if random_pick else chosen or "manual"


def read_superposition(path: Path, unit: str):
    """(unit, model, layer, profile) from a superposition's definition file."""
    from pydantic import ValidationError

    from openinterp.core.profiles import SuperpositionProfile

    if unit != "neuron":
        raise click.UsageError("--superposition is a unit of its own; don't combine it.")
    data = json.loads(path.read_text(encoding="utf-8"))
    try:
        model, layer = data.pop("model"), data.pop("layer")
        return "superposition", model, layer, SuperpositionProfile.model_validate(data)
    except (KeyError, ValidationError) as exc:
        raise click.BadParameter(str(exc), param_hint="--superposition") from exc


def resolve_unit(unit, neuron, pole, head, feature, dictionary) -> str:
    """The unit ``--unit`` names, or the one the other options imply."""
    if unit is None:
        unit = "head" if head is not None else "neuron"
        unit = "feature" if feature is not None or dictionary is not None else unit
    if unit != "neuron" and (neuron is not None or pole is not None):
        raise click.UsageError(f"--neuron and --pole are for neurons; a {unit} has neither.")
    if head is not None and unit != "head":
        raise click.UsageError("--head needs --unit head (or no --unit).")
    if unit != "feature" and (feature is not None or dictionary is not None):
        raise click.UsageError("--feature and --dictionary need --unit feature (or no --unit).")
    return unit

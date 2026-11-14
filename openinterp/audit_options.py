"""Options that ``openinterp audit`` and ``scan`` share: the scan cache and ``--selection``.

``--selection`` says how a chosen unit was selected, as its provenance records it.

A ``--random`` draw records ``random`` and counts towards the registry statistics. A
unit given by layer and index records ``manual``, unless ``--selection`` says otherwise.
The planner's effect-weighted explore draws (issue #3486) pass ``attribution``. They are
drawn in proportion to an attribution-patching estimate, so they must not count as random.
"""

from __future__ import annotations

import click

scan_cache_option = click.option(
    "--scan-cache/--no-scan-cache",
    "use_scan_cache",
    default=True,
    show_default=True,
    help="Replay a cached scan of the layer, or store this one (see `openinterp scan-cache`).",
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

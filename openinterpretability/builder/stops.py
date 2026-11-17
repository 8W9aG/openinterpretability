"""What the site shows of the scans that stopped: random draws and stops on chosen units.

A scan stops on a unit pole that is dead or default-on. A random draw keeps its stop in
``wiki/draws/``, and a chosen unit in ``wiki/stops/`` with a page. The unit's page lists
each stop with its experiments, the outcome in place of a verdict (#3297).
"""

from __future__ import annotations

from collections.abc import Sequence

from openinterpretability.builder.tree import unit_href, unit_side
from openinterpretability.builder.wiki import stop_page
from openinterpretability.core.draws import ChosenStop, UnauditableDraw, draw_path
from openinterpretability.core.paths import DEFAULT_WIKI_DIR


def stops_by_unit(stops: Sequence[UnauditableDraw]) -> dict[str, list[dict]]:
    """A view of each stop, by the page of its unit, newest first.

    ``stops`` are the random draws and the stops of chosen units together. A view's
    ``page`` is the stop's wiki page, or the draw's record, since a draw has no page.
    """
    by_unit: dict[str, list[dict]] = {}
    for stop in sorted(stops, key=lambda s: s.timestamp, reverse=True):
        chosen = isinstance(stop, ChosenStop)
        page = stop_page(DEFAULT_WIKI_DIR, stop) if chosen else draw_path(DEFAULT_WIKI_DIR, stop)
        view = {
            "stop": stop,
            "side": unit_side(stop.identifier),
            "role": "chosen" if chosen else "random draw",
            # The site doesn't publish wiki pages, so the link goes to the repository.
            "page": page.as_posix(),
        }
        by_unit.setdefault(unit_href(stop.identifier), []).append(view)
    return by_unit

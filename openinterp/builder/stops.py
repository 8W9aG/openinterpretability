"""What the site shows of the scans that stopped on chosen units (``wiki/stops/``)."""

from __future__ import annotations

from collections.abc import Sequence

from openinterp.builder.tree import unit_href, unit_side
from openinterp.builder.wiki import stop_label, stop_page
from openinterp.core.draws import ChosenStop


def stops_by_unit(stops: Sequence[ChosenStop]) -> dict[str, list[dict]]:
    """A view of each stop, by the page of its unit, newest first."""
    by_unit: dict[str, list[dict]] = {}
    for stop in sorted(stops, key=lambda s: s.timestamp, reverse=True):
        view = {
            "stop": stop,
            "side": unit_side(stop.identifier),
            "label": stop_label(stop),
            # The site doesn't publish wiki pages, so the link goes to the repository.
            "page": stop_page("wiki", stop).as_posix(),
        }
        by_unit.setdefault(unit_href(stop.identifier), []).append(view)
    return by_unit


def pageless_stops(by_unit: dict[str, list[dict]], studied: set) -> list[dict]:
    """The views of stops on units with no page, newest first."""
    views = [view for href, unit in by_unit.items() if href not in studied for view in unit]
    return sorted(views, key=lambda view: view["stop"].timestamp, reverse=True)

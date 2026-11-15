"""A model's circuit pages: the index, a page per circuit and per hub, and the redirects.

:mod:`openinterp.builder.circuits` says how circuits are named. Each unit with a page
gets a URL under ``circuits/`` and under ``hubs/``. One of the two may be a real page.
Every other one redirects to the page the unit appears on now. A unit in no circuit and
no hub redirects to its own page. So a URL from any earlier build keeps working.
"""

from __future__ import annotations

from openinterp.builder.circuits import HUB_RULE, Circuit, Hub, unit_key, unit_pages
from openinterp.builder.hrefs import circuit_href, circuits_href, hub_href


def render_circuits(render, model, found: list[Circuit], found_hubs: list[Hub]) -> dict[str, str]:
    """Render ``model``'s circuit pages, and return each unit's circuit or hub page by href."""
    slug = model.slug
    pages = unit_pages(slug, found, found_hubs)
    if not (found or found_hubs):
        return pages
    circuits = [(c, circuit_href(slug, unit_key(c.root))) for c in found]
    hubs = [(hub, hub_href(slug, unit_key(hub.unit))) for hub in found_hubs]
    n_units = sum(len(c.units) for c in found)
    view = {"model": model, "found": circuits, "hubs": hubs, "n_units": n_units, **HUB_RULE}
    render(circuits_href(slug), "circuits.html", **view)
    for c, href in circuits:
        render(href, "circuit.html", model=model, c=c)
    for hub, href in hubs:
        render(href, "hub.html", model=model, hub=hub)
    real = {href for _, href in circuits + hubs}
    for layer in model.layers:
        for unit in layer.units:
            key = unit_key(unit)
            for href in (circuit_href(slug, key), hub_href(slug, key)):
                if href not in real:
                    render(href, "redirect.html", target=pages.get(unit.href, unit.href))
    return pages

"""A model's circuit pages: the index, a page per circuit and per hub, and the redirects.

:mod:`openinterpretability.builder.circuits` says how circuits are named. Each unit with a page
gets a URL under ``circuits/`` and under ``hubs/``. One of the two may be a real page.
Every other one redirects to the page the unit appears on now. A unit in no circuit and
no hub redirects to its own page. So a URL from any earlier build keeps working.
"""

from __future__ import annotations

from openinterpretability.builder.circuit_units import CircuitData, circuit_views
from openinterpretability.builder.circuits import HUB_RULE, Circuit, Hub, unit_key, unit_pages
from openinterpretability.builder.hrefs import circuit_href, circuits_href, hub_href


def render_circuits(render, model, found: list[Circuit], found_hubs: list[Hub]) -> dict[str, str]:
    """Render ``model``'s circuit pages, and return each unit's circuit or hub page by href.

    The registry circuits (:mod:`openinterpretability.builder.circuit_units`) come from the
    site context's ``circuit_data``, keyed by model slug.
    """
    slug = model.slug
    pages = unit_pages(slug, found, found_hubs)
    data = render.common.get("circuit_data", {}).get(slug, CircuitData())
    registry = circuit_views(model, data, [e for c in found for e in c.edges])
    if not (found or found_hubs or registry):
        return pages
    circuits = [(c, circuit_href(slug, unit_key(c.root))) for c in found]
    hubs = [(hub, hub_href(slug, unit_key(hub.unit))) for hub in found_hubs]
    n_units = sum(len(c.units) for c in found)
    view = {"model": model, "found": circuits, "hubs": hubs, "n_units": n_units, **HUB_RULE}
    render(circuits_href(slug), "circuits.html", registry=registry, **view)
    for c, href in circuits:
        render(href, "circuit.html", model=model, c=c)
    for rc in registry:
        render(rc.href, "circuit_unit.html", model=model, rc=rc)
    _render_hubs(render, model, hubs, registry, data.norms)
    real = {href for _, href in circuits + hubs}
    for layer in model.layers:
        for unit in layer.units:
            key = unit_key(unit)
            for href in (circuit_href(slug, key), hub_href(slug, key)):
                if href not in real:
                    render(href, "redirect.html", target=pages.get(unit.href, unit.href))
    return pages


def _render_hubs(render, model, hubs, registry, norms) -> None:
    """Each hub's page: its center's standing, its artefact facts and its circuits."""
    from openinterpretability.hubs import hub_entities

    entities = hub_entities([hub for hub, _ in hubs], norms)
    for (hub, href), entity in zip(hubs, entities, strict=True):
        near = {u.href for u in hub.neighbours}
        joins = [rc for rc in registry if near & {m.href for m in rc.members if m.href}]
        render(href, "hub.html", model=model, hub=hub, entity=entity, joins=joins)

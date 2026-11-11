"""The dictionary latents aligned with a neuron's poles, as its page shows them."""

from __future__ import annotations

from openinterp.builder.tree import feature_href
from openinterp.crossref import neuronpedia_url


def aligned_latents(model, neuron, alignment: dict, studied: set) -> list[dict]:
    """Return each pole's aligned dictionary latents for a neuron page.

    Each latent links to its page here (if audited) and on Neuronpedia (if hosted there).
    """
    if not neuron.is_neuron:
        return []
    poles = []
    for pole in neuron.sides:
        rows = alignment.get(f"{model.slug}/L{neuron.layer}/N{neuron.idx}-{pole}", [])
        views = []
        for row in rows:
            sides = []
            for name in ("writes", "reads"):
                side = getattr(row, name)
                if side is None:
                    continue
                matches = []
                for m in side.matches:
                    href = feature_href(model.slug, neuron.layer, row.dictionary, m.feature_idx)
                    matches.append(
                        {
                            "m": m,
                            "href": href if href in studied else None,
                            "neuronpedia": neuronpedia_url(row, m.feature_idx),
                            "above": m.cosine > side.chance,
                        }
                    )
                sides.append({"name": name, "side": side, "matches": matches})
            views.append({"row": row, "sides": sides})
        if views:
            poles.append({"pole": pole, "rows": views})
    return poles

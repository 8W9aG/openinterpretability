"""Find prior art for a unit after the proposer answers (`open-interpretability audit`).

A dictionary latent that Neuronpedia hosts is looked up directly. A neuron pole is
aligned with each dictionary preset that Neuronpedia hosts at its layer
(:func:`openinterp.harness.crossref.align_layer`). Rows from ``open-interpretability crossref`` are
used where they exist. The latents above chance on each side are then looked up, the best
``TOP`` first.

Lookups go through the cached, polite fetch of :mod:`openinterp.crossref`. A failed
lookup or alignment is left out, so prior art never stops an audit.
``$OPENINTERP_PRIOR_ART=0`` turns the readout off.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from openinterp.core.fence import fence
from openinterp.core.prior_art import PriorArt

TOP = 3  # latents per side of a pole
ENV = "OPENINTERP_PRIOR_ART"


def enabled() -> bool:
    """False when ``$OPENINTERP_PRIOR_ART`` turns the readout off."""
    return os.environ.get(ENV, "1").lower() not in ("", "0", "off", "false")


def find_prior_art(submission, runner, *, rows: dict | None = None, fetch: Callable | None = None):
    """Return the unit's prior art; ``None`` for a unit kind with none, or when turned off.

    ``rows`` are crossref rows by slug (:func:`openinterp.crossref.load_rows`). ``fetch``
    returns a latent's explanations (default :func:`openinterp.crossref.latent_explanations`).
    """
    from openinterp.crossref import latent_explanations

    if not enabled():
        return None
    fetch = fetch or latent_explanations
    ident = submission.identifier
    if ident.kind == "dictionary_feature":
        entry = _entry(fetch, ident.dictionary, ident.layer, ident.feature_idx, "self", None)
        return [entry] if entry else []
    if ident.kind != "mlp_neuron_pole":
        return None
    found = []
    for name, side, matches in _aligned(ident, runner, rows):
        for feature_idx, cosine in matches:
            entry = _entry(fetch, name, ident.layer, feature_idx, side, cosine)
            if entry:
                found.append(entry)
    return found


def _entry(fetch, dictionary: str, layer: int, feature_idx: int, relation: str, cosine):
    """The latent's first machine-written explanation, else its first; ``None`` if none."""
    from openinterp.crossref import latent_url, machine_explanation

    url = latent_url(dictionary, layer, feature_idx)
    if url is None:
        return None
    explanations = fetch(dictionary, layer, feature_idx) or []
    best = machine_explanation(explanations) or next(iter(explanations), None)
    if best is None:
        return None
    return PriorArt(
        dictionary=dictionary,
        feature_idx=feature_idx,
        relation=relation,
        cosine=cosine,
        description=best.description,
        explainer=best.explainer,
        url=url,
    )


def _aligned(ident, runner, rows: dict | None):
    """Yield ``(dictionary, side, [(feature_idx, cosine)])`` for each hosted dictionary."""
    from openinterp.crossref import dictionaries_for, load_rows
    from openinterp.harness.dictionaries import DICTIONARIES

    rows = load_rows() if rows is None else rows
    known = {r.dictionary: r for r in rows.get(ident.slug, [])}
    for name in dictionaries_for(ident.model_name, ident.layer):
        if not DICTIONARIES[name].neuronpedia_source(ident.layer):
            continue
        sides = _sides(known[name]) if name in known else _align(runner, name, ident)
        for side, (matches, chance) in sides.items():
            above = [m for m in matches if abs(m[1]) > chance]
            yield name, side, sorted(above, key=lambda m: -abs(m[1]))[:TOP]


def _sides(row) -> dict:
    """``{side: ([(feature_idx, cosine)], chance)}`` of a stored crossref row."""
    out = {}
    for side in ("writes", "reads"):
        s = getattr(row, side)
        if s is not None:
            out[side] = ([(m.feature_idx, m.cosine) for m in s.matches], s.chance)
    return out


def _align(runner, name: str, ident) -> dict:
    """Align the pole with dictionary ``name`` now; empty if that fails."""
    from openinterp.core.exceptions import ArchitectureNotSupportedError
    from openinterp.harness.crossref import align_layer
    from openinterp.harness.dictionaries import dictionary_spec

    try:
        found = align_layer(
            runner, dictionary_spec(name), ident.layer, [(ident.neuron_idx, ident.pole)]
        )
    except (ArchitectureNotSupportedError, OSError, ValueError):
        return {}
    write, read = found[(ident.neuron_idx, ident.pole)]
    out = {"writes": ([(m.feature_idx, m.cosine) for m in write.matches], write.chance)}
    if read is not None:
        out["reads"] = ([(m.feature_idx, m.cosine) for m in read.matches], read.chance)
    return out


def format_for_reviewer(prior_art: list[PriorArt] | None) -> str:
    """The prior art as a block of the reviewer's prompt; empty if there is none."""
    if not prior_art:
        return ""
    lines = []
    for p in prior_art:
        where = (
            "this latent itself"
            if p.relation == "self"
            else f"{p.dictionary} latent {p.feature_idx}, which {p.relation} along this pole "
            f"(cosine {p.cosine:+.2f})"
        )
        lines.append(f"- {where}: {p.description} [{p.source}, {p.explainer}]")
    return (
        "Prior art: published explanations of this unit or of latents aligned with it. They "
        "are other people's claims, not measurements, and the proposer did not see them. "
        "Where one disagrees with the hypothesis, write probes that tell them apart.\n"
        + fence("\n".join(lines))
        + "\n"
    )

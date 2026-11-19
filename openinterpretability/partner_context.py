"""What the registry knows about a pole's co-activating partners, for the proposer (#3240).

The scan finds the poles in the layer that co-activate with the audited pole
(:class:`~openinterpretability.core.partners.CorrelatedNeuron`). The proposer prompt lists
them with their lift and ρ (:func:`openinterpretability.agents.proposer.format_correlated`).
:func:`partner_context` adds what the registry knows about each one. A studied partner
gets its standing, with errata applied, and the explanation the standing rests on. A
VERIFIED partner hints at what this pole shares with it, or at what it must be told apart
from. An unstudied partner gets its strongest tokens in the scan's uniform sample.

It reads only the partners' folders of the archive, as
:func:`openinterpretability.sisters.sister_context` does.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from openinterpretability.core.fence import fence

INTRO = (
    "The co-activating poles above, as this registry knows them. A studied one has its "
    "standing and the explanation it rests on. A VERIFIED partner hints at what this pole "
    "shares with it, or at what your hypothesis must tell it apart from. An unstudied one "
    "has its strongest tokens in the scan's uniform sample. Treat them as leads: ground the "
    "hypothesis in this pole's own evidence."
)


def partner_slugs(identifier, correlated: Sequence) -> dict[str, tuple[str, list[str]]]:
    """``{slug: (label, top tokens)}`` of each partner pole of ``identifier``."""
    from openinterpretability.core.identifiers import NeuronIdentifier

    out = {}
    for c in correlated:
        partner = NeuronIdentifier(model_name=identifier.model_name, layer=identifier.layer,
                                   neuron_idx=c.neuron_idx, pole=c.pole)  # fmt: skip
        label = f"N{c.neuron_idx}{'+' if c.pole == 'positive' else '-'}"
        out[partner.slug] = (label, list(c.top_tokens))
    return out


def format_partners(partners: dict, findings: dict[str, tuple[str, str]]) -> str:
    """The section on the partners, in the order of ``partners``, or "" if it says nothing."""
    lines = []
    for slug, (label, tokens) in partners.items():
        if slug in findings:
            status, explanation = findings[slug]
            lines.append(f"- {label}: {status}: {explanation}")
        elif tokens:
            lines.append(f"- {label}: not studied; top tokens: " + ", ".join(map(repr, tokens)))
    if not lines:
        return ""
    return f"\n{INTRO}\n" + fence("\n".join(lines)) + "\n"


def partner_context(identifier, correlated: Sequence, wiki_dir: Path | None) -> str:
    """The proposer's section on the pole's partners, or "".

    With no wiki (``wiki_dir`` None or missing), every partner counts as unstudied.
    """
    from openinterpretability.core.errata import load_errata
    from openinterpretability.sisters import _records_of, pole_findings

    if identifier.kind != "mlp_neuron_pole" or not correlated:
        return ""
    partners = partner_slugs(identifier, correlated)
    findings: dict[str, tuple[str, str]] = {}
    if wiki_dir is not None and Path(wiki_dir).exists():
        findings = pole_findings(_records_of(wiki_dir, partners), load_errata(wiki_dir))
    return format_partners(partners, findings)

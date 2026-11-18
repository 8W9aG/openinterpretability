"""The evidence for the narrated levels of ``explain``, and the check of their citations (#3743).

:func:`evidence` gathers level 1 of an explanation. That is the answer, the explained and
blocked shares, and prompt coverage. It is also the top units and groups with their shares,
ablation checks and statuses, the paths, and the next experiments. A unit's explanation
goes in only when it is VERIFIED. It is fenced, because another agent wrote it
(:mod:`openinterpretability.core.fence`). The narrator
(:mod:`openinterpretability.agents.narrator`) reads this and nothing else.

:func:`check_citations` lists the units that a narrative cites but the evidence doesn't hold,
so a reader can see a claim with no support.
"""

from __future__ import annotations

import re

from openinterpretability.core.fence import fence

# A unit as the text cites it: L3/N7+, L3/N7-, L3/N7−, L2/H10, L20/S0123456789ab, or C4.
CITATION = re.compile(r"\bL\d+/(?:N\d+[+\-−]?|H\d+|S[0-9a-f]{12})|\bC\d+\b")


def _row(name: str, attribution: float, total: float, status: str, explanation) -> dict:
    """One unit or group: its share, its attribution and status, and a VERIFIED explanation."""
    row = {"unit": name, "share": round(abs(attribution) / total, 4),
           "attribution": round(attribution, 4), "status": status}  # fmt: skip
    if status == "VERIFIED" and explanation:
        row["explanation"] = fence(explanation)  # another agent's text: data, not instructions
    return row


NOTES = (
    "Shares are of first-order |attribution| on this prompt. A unit's role is known only when "
    "its status is VERIFIED. A unit with a hub is a hub's center: wired to many studied units, "
    "it can be an artefact, so say so with its facts."
)


def evidence(result, coverage: dict, plan, *, experiments: int = 10,
             answer: dict | None = None) -> dict:  # fmt: skip
    """Level 1 of the explanation, as the narrator reads it (module docstring)."""
    from openinterpretability.explain import PoleAttribution

    total = result.total or 1.0
    units = []
    for u in result.poles:
        row = _row(u.label, u.attribution, total, u.status, u.explanation)
        row["ablation_change"] = None if u.ablation_change is None else round(u.ablation_change, 4)
        if getattr(u, "hub", None):
            row["hub"] = u.hub  # the unit is a hub's center (#4472)
        units.append(row)
    groups = []
    for g in result.groups[:10]:
        row = _row(g.name, g.attribution, total, g.status, g.explanation)
        groups.append(row | {"kind": g.kind, "members": g.members})
    nxt = [{"unit": e.target, "kind": e.draw, "status": e.status, "coverage_gain": e.gain}
           for e in [*plan.experiments, *plan.groups][:experiments]]  # fmt: skip
    return {
        "prompt": fence(result.prompt),
        "answer": result.token,
        "explained_share": round(result.verified_share, 4),
        "unexplained_share": round(1 - result.verified_share, 4),
        "blocked_share": round(plan.blocked, 4),
        "prompt_coverage": coverage.get("coverage"),
        "random_coverage": coverage.get("random"),  # matched by kind and layer (#4777)
        "coverage_caution": caution(coverage, result.verified_share),
        "heads_only": coverage.get("heads_only"),
        "poles_only": coverage.get("poles_only"),
        "minimal_set": [
            PoleAttribution(*u, 0.0, 0.0).label for u in coverage.get("minimal_units", [])
        ],
        "verified_units_ablated_together": result.joint,
        "units": units,
        "groups": groups,
        "paths": result.edges[:20],
        "next_experiments": nxt,
        "answer_experiment": _answer(answer),
        "notes": NOTES,
    }


def _answer(answer: dict | None) -> dict:
    """The newest answer claim on the prompt (#4778), its text fenced: another agent wrote it."""
    if answer is None or answer.get("newest") is None:
        return {"status": "unstudied", "attempts": 0}
    newest = dict(answer["newest"])
    newest["claim"] = fence(newest["claim"])
    newest["roles"] = {unit: fence(role) for unit, role in newest["roles"].items()}
    return newest | {"attempts": answer["attempts"]}


def caution(coverage: dict, explained_share: float) -> bool:
    """Whether coverage here shows little (``ExplanationRecord.coverage_caution``, #4777)."""
    from openinterpretability.core.explanation_records import CAUTION_MARGIN, CAUTION_SHARE

    share, base = coverage.get("coverage"), coverage.get("random")
    if share is None:
        return False
    return (base is not None and share - base < CAUTION_MARGIN) or explained_share < CAUTION_SHARE


def cited(text: str) -> set[str]:
    """The units that ``text`` cites, with the minus sign written one way."""
    return {m.replace("−", "-") for m in CITATION.findall(text)}


def check_citations(narrative, ev: dict) -> list[str]:
    """The units that the narrative cites but the evidence doesn't hold, sorted."""
    known = {str(row["unit"]).replace("−", "-") for key in ("units", "groups", "next_experiments")
             for row in ev[key]}  # fmt: skip
    known |= set(ev.get("answer_experiment", {}).get("route", []))
    known |= {u.replace("−", "-") for u in ev.get("minimal_set", [])}  # the evidence lists them
    for path in ev["paths"]:
        known |= {path["source"].replace("−", "-"), path["target"].replace("−", "-")}
    text = " ".join([narrative.paragraph, *(s.title + " " + s.text for s in narrative.sections)])
    # A pole cited without its sign still names a known neuron.
    neurons = {k.rstrip("+-") for k in known}
    return sorted(c for c in cited(text) if c not in known and c not in neurons)

"""``sister`` draws: VERIFIED poles' explanations tested in other models.

``open-interpretability next`` makes these draws.
:func:`openinterp.planner.plan_next` adds these to the follow-up candidates. The weights
are in docs/PROTOCOL.md (Sister study).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from openinterp.planner import (
    BASE_WEIGHTS,
    SKILLS,
    Candidate,
    _decay,
    claim_key,
    configured_windows,
)

# A match of a VERIFIED pole that isn't its sister can be drawn too. Its weight is this
# share of a sister's weight, times its correlation. Any link can be tested. But the
# draws lean heavily towards the sisters, where the explanation is likeliest to carry over.
LINK_WEIGHT = 0.05
# So can any pole of another configured model, found by the search or not. For each
# VERIFIED pole and model, one draw picks a random pole there. Its weight is this share
# of a sister's weight.
# It is also the control for the matched draws.
UNMATCHED_WEIGHT = 0.01


def link_affinity(row, match) -> float:
    """Return how strongly to draw a study of ``match``.

    1 for the pole's sister, else :data:`LINK_WEIGHT` × its correlation. Never zero, as
    for every weight here.
    """
    return 1.0 if row.sister is match else LINK_WEIGHT * max(match.pearson, 0.01)


def sister_candidates(
    states, claims, now, pool: Sequence[str], root: Path, seed: int = 0
) -> list[Candidate]:
    """Return the ``sister`` draws for the models of ``pool``.

    A sister draw audits a pole of another model with a VERIFIED pole's explanation,
    unchanged. The draws are every unstudied match that ``open-interpretability sisters`` found
    (read from ``root``), weighted by :func:`link_affinity`. Each model also gets a
    random pole (:data:`UNMATCHED_WEIGHT`, seeded from ``seed``). Only models in the pool
    are drawn, so CI can reproduce the record.
    """
    from openinterp.sisters import DEFAULT_PATH, load_rows

    rows, windows = load_rows(root / DEFAULT_PATH), configured_windows(root / "pyproject.toml")
    out = []
    for slug, state in states.items():
        ident = state.record.submission.identifier
        if state.record.verdict.status != "VERIFIED" or ident.kind != "mlp_neuron_pole":
            continue
        for row in rows.get(slug, []):
            for m in row.matches if row.other in pool else []:
                if m.slug(row.other) not in states:
                    out.append(_sister(state, row, m, windows.get(row.other), claims, now))
        for other in pool:
            if other != ident.model_name:
                out.append(_unmatched(state, other, windows.get(other), seed, claims, now))
    return out


def _unmatched(state, other: str, window: int | None, seed: int, claims, now) -> Candidate:
    """The draw that audits a random pole of ``other`` with ``state``'s explanation."""
    import hashlib

    digest = hashlib.sha256(f"{seed} {state.slug} {other}".encode()).digest()
    audit = f"--model {other} --random --seed {int.from_bytes(digest[:4])}"
    if window:
        audit += f" --window-tokens {window}"
    audit += f" --sister-of {state.record_path.as_posix()}"
    factor, notes = _decay(state, claims, now)
    why = (
        f"a random pole of {other}, matched by no search, with {state.slug}'s explanation: "
        "the control for its matches"
    )
    verify = "open-interpretability verify --audit-file <new submission> --corpus <scan corpus>"
    return Candidate(
        kind="sister",
        model=other,
        weight=BASE_WEIGHTS["sister"] * UNMATCHED_WEIGHT * factor,
        rationale="; ".join([why, *notes]),
        commands=[f"open-interpretability audit {audit}", verify],
        skills=SKILLS["sister"],
        record=state.record_path.as_posix(),
        audit=audit,
    )


def _sister(state, row, m, window: int | None, claims, now) -> Candidate:
    """The draw that audits ``m`` with ``state``'s explanation."""
    target = m.slug(row.other)
    factor, notes = _decay(state, claims, now)
    if any(claim_key(target) in c for c in claims):
        factor, notes = factor * 0.05, [*notes, "an open PR already claims the match"]
    audit = f"--model {row.other} --layer {m.layer} --neuron {m.neuron_idx} --pole {m.pole}"
    if window:
        audit += f" --window-tokens {window}"
    audit += f" --sister-of {state.record_path.as_posix()}"
    link = "sister" if row.sister is m else "correlated match (not a sister)"
    why = f"{link} of {state.slug} (r {m.pearson:+.2f}): does its explanation carry over?"
    verify = "open-interpretability verify --audit-file <new submission> --corpus <scan corpus>"
    return Candidate(
        kind="sister",
        model=row.other,
        weight=BASE_WEIGHTS["sister"] * link_affinity(row, m) * factor,
        rationale="; ".join([why, *notes]),
        commands=[f"open-interpretability audit {audit}", verify],
        slug=target,
        skills=SKILLS["sister"],
        layer=m.layer,
        record=state.record_path.as_posix(),
        audit=audit,
    )

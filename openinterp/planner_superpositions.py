"""``superposition`` draws: audits of the superpositions defined in ``data/superpositions/``.

A superposition is a sparse direction over one layer's MLP neurons (issue #3215).
``open-interpretability superpositions`` writes the definitions, one per line (issue #3686).
:func:`openinterp.planner.plan_next` adds a draw for each definition that has no
experiment and no claim (issue #3687).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from openinterp.core.paths import DEFAULT_EXCHANGE_DIR, DEFAULT_WIKI_DIR
from openinterp.planner import BASE_WEIGHTS, SKILLS, Candidate, claim_key, configured_windows

DEFINITIONS_DIR = Path("data/superpositions")
SHOW_COMMAND = "open-interpretability superpositions show"
# A superposition draw has the weight of one sister draw (the full sister weight, not
# LINK_WEIGHT or UNMATCHED_WEIGHT). Both test a ready unit that no experiment has studied yet.
SUPERPOSITION_WEIGHT = BASE_WEIGHTS["sister"]


def load_definitions(path: Path) -> list[dict]:
    """Return the definitions in the JSONL file ``path``, one per non-empty line."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def definition_identifier(definition: dict):
    """Return the :class:`SuperpositionIdentifier` of one definition."""
    from openinterp.core.identifiers import SuperpositionIdentifier
    from openinterp.core.profiles import SuperpositionProfile, superposition_id

    fields = {k: v for k, v in definition.items() if k not in ("model", "layer")}
    profile = SuperpositionProfile.model_validate(fields)
    return SuperpositionIdentifier(
        model_name=definition["model"],
        layer=definition["layer"],
        superposition_id=superposition_id(profile.members),
    )


def default_definitions(model: str) -> Path:
    """Return the file that holds ``model``'s definitions, relative to the repo root."""
    from openinterp.core.identifiers import model_slug

    return DEFINITIONS_DIR / f"{model_slug(model)}.jsonl"


def definition_file(slug: str) -> str:
    """Return the path where a draw of the superposition ``slug`` writes its definition."""
    return f"{DEFAULT_EXCHANGE_DIR}/{claim_key(slug)}/superposition.json"


def superposition_candidates(states, claims: Sequence[str], root: Path) -> list[Candidate]:
    """Return a ``superposition`` draw for each unstudied definition under ``root``.

    The definitions are in every ``root/data/superpositions/*.jsonl``. A definition is
    left out when ``states`` has an experiment on it, or when an open PR in ``claims``
    claims it. It is also left out when a scan of it stopped (``wiki/stops/``): it is
    dead or default-on, and no audit can follow. A missing folder gives no draws.
    """
    stopped = _stopped(root / DEFAULT_WIKI_DIR)
    folder = root / DEFINITIONS_DIR
    files = sorted(folder.glob("*.jsonl")) if folder.is_dir() else []
    windows = configured_windows(root / "pyproject.toml")
    out = []
    for path in files:
        for definition in load_definitions(path):
            ident = definition_identifier(definition)
            if ident.slug in states or ident.slug in stopped:
                continue
            if any(claim_key(ident.slug) in c for c in claims):
                continue
            source = path.relative_to(root)
            out.append(_candidate(ident, definition, source, windows.get(ident.model_name)))
    return out


def _stopped(wiki_dir: Path) -> set[str]:
    """The slugs of the chosen units whose scans stopped."""
    from openinterp.core.draws import load_stops

    return {s.identifier.slug for s in load_stops(wiki_dir)} if wiki_dir.is_dir() else set()


def _candidate(ident, definition: dict, source: Path, window: int | None) -> Candidate:
    """The draw that audits the superposition ``ident``, defined in the file ``source``."""
    target = definition_file(ident.slug)
    show = f"{SHOW_COMMAND} {ident.superposition_id} --model {ident.model_name} --out {target}"
    if source != default_definitions(ident.model_name):
        show += f" --definitions {source.as_posix()}"
    audit = f"--superposition {target}"
    if window:
        audit += f" --window-tokens {window}"
    members = len(definition["members"])
    why = (
        f"an unstudied superposition of {members} neurons on layer {ident.layer} "
        f"({definition['method']}, from {definition['source']})"
    )
    verify = "open-interpretability verify --audit-file <new submission> --corpus <scan corpus>"
    return Candidate(
        kind="superposition",
        model=ident.model_name,
        weight=SUPERPOSITION_WEIGHT,
        rationale=why,
        commands=[show, f"open-interpretability audit {audit}", verify],
        slug=ident.slug,
        skills=SKILLS["superposition"],
        layer=ident.layer,
        audit=audit,
    )

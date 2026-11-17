"""``circuit`` draws: an experiment on a circuit with none, or on one that has moved (#4286).

A circuit is a cluster of a model's wiring graph. Its id survives as the registry grows
(``data/circuits/<model>.jsonl``, :mod:`openinterpretability.circuit_ids`). An experiment
on it tests a task hypothesis (:mod:`openinterpretability.core.circuit_records`). Its
record keeps the circuit's members as they were (``cluster_members``).

- A circuit with **no experiment** gets a draw of weight :data:`CIRCUIT_WEIGHT`.
- A **stale** circuit has members that changed since its newest experiment. Its draw is
  weighted by the share of members that changed, over all of them, old and new.

A circuit whose newest experiment matches its members gets no draw. Neither does a hub,
which has no experiments of its own (``docs/circuits-and-hubs.md``). The agent writes the
task (prompt pairs) and the reviewer writes the probes. ``open-interpretability loop``
can't, because its agents are API calls, so it draws again over a circuit draw.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from openinterpretability.core.paths import DEFAULT_EXCHANGE_DIR, DEFAULT_WIKI_DIR
from openinterpretability.planner import SKILLS, Candidate, claim_key
from openinterpretability.planner_claims import claimed

CIRCUIT_WEIGHT = 1.0  # like an explanation draw: occasional


def newest_record(wiki_dir: Path, model_slug: str, circuit_id: str):
    """The circuit's newest record, or None when it has none."""
    from openinterpretability.core.circuit_records import CIRCUITS_WIKI, CircuitRecord

    folder = wiki_dir / CIRCUITS_WIKI / model_slug / circuit_id
    newest = max(folder.glob("*.json"), default=None) if folder.is_dir() else None
    if newest is None:
        return None
    return newest, CircuitRecord.model_validate_json(newest.read_text(encoding="utf-8"))


def members_of(hrefs) -> frozenset[str]:
    """``L4/N2999``-style members, from unit pages or labels alike."""
    from openinterpretability.harness.circuit_measures import Member

    return frozenset(str(Member.parse(h)) for h in hrefs)


def commands(model: str, circuit_id: str, key: str) -> list[str]:
    """The propose and review commands of a circuit draw, with its working files."""
    out = f"{DEFAULT_EXCHANGE_DIR}/{key}"
    return [
        f"open-interpretability circuits propose --model {model} --circuit {circuit_id} "
        f"--task {out}/task.json --explanation '<what the members compute>' "
        f"--agent-model <proposer model> --out {out}/hypothesis.json",
        f"open-interpretability circuits review --hypothesis {out}/hypothesis.json "
        f"--probes {out}/probes.json",
    ]


def circuit_candidates(claims: Sequence[str], root: Path) -> list[Candidate]:
    """A draw for each circuit with no experiment, or whose members moved (module docstring)."""
    from openinterpretability import circuit_ids

    folder = root / circuit_ids.CIRCUITS_DIR
    out = []
    for path in sorted(folder.glob("*.jsonl")) if folder.is_dir() else []:
        model_slug = path.stem
        circuits, _ = circuit_ids.load(path)
        for circuit_id, hrefs in sorted(circuits.items()):
            found = _candidate(root, model_slug, circuit_id, members_of(hrefs))
            if found is not None and not claimed(claim_key(found.slug), claims):
                out.append(found)
    return out


def _candidate(root: Path, model_slug: str, circuit_id: str, now: frozenset[str]):
    """The draw for one circuit, or None when its newest experiment is current."""
    newest = newest_record(root / DEFAULT_WIKI_DIR, model_slug, circuit_id)
    if newest is None:
        weight, why, record, model = CIRCUIT_WEIGHT, "a circuit with no experiment", None, None
    else:
        path, rec = newest
        then = members_of(rec.hypothesis.cluster_members)
        changed = now ^ then
        if not changed:
            return None
        share = len(changed) / len(now | then)
        weight = CIRCUIT_WEIGHT * share
        why = (f"stale: {len(changed)} of {len(now | then)} members changed since its newest "
               f"experiment ({rec.verdict.status})")  # fmt: skip
        record, model = path.as_posix(), rec.hypothesis.model_name
    model = model or _model_name(root, model_slug)
    slug = f"{model_slug}/circuits/{circuit_id}"
    return Candidate(
        kind="circuit", model=model, weight=weight, rationale=why,
        commands=commands(model, circuit_id, claim_key(slug)), slug=slug,
        skills=SKILLS["circuit"], record=record,
    )  # fmt: skip


def _model_name(root: Path, model_slug: str) -> str:
    """The configured model whose slug is ``model_slug``, else the slug with ``--`` as ``/``."""
    from openinterpretability.core.identifiers import model_slug as slug_of
    from openinterpretability.planner import configured_models

    for model in configured_models(root / "pyproject.toml"):
        if slug_of(model) == model_slug:
            return model
    return model_slug.replace("--", "/")

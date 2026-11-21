"""``explanation`` draws, and the unit draws that recorded explanations ask for (issue #3744).

An explanation draw explains one answer and records it
(:mod:`openinterpretability.core.explanation_records`). Its prompt comes from
``data/explanation-prompts.jsonl``, a small committed list of classic prompts, paired
with each configured model:

- a **new** explanation for a model and prompt that has none, one candidate for each draw;
- a **refresh** of an explanation that is stale, because a unit that it cites has a
  different status now.

A recorded explanation also lists the units that carry its answer. Each pole or head of them
that is still unstudied gets an ``explore`` candidate: an audit chosen by attribution
(``--selection attribution``). A studied unit already has the planner's own follow-ups. So
the loop drives each explained answer's units toward VERIFIED, one normal experiment at a
time, as ``explain --run`` does for one answer.
"""

from __future__ import annotations

import json
import random
import shlex
from collections.abc import Sequence
from pathlib import Path

from openinterpretability.core.paths import DEFAULT_WIKI_DIR
from openinterpretability.planner import (
    SKILLS,
    Candidate,
    _audit_options,
    claim_key,
    configured_windows,
)
from openinterpretability.planner_claims import claimed

PROMPTS = Path("data/explanation-prompts.jsonl")
EXPLANATION_WEIGHT = 1.0  # a new explanation or a refresh: occasional, like an opposite pole
UNIT_WEIGHT = 1.5  # an unstudied unit that an explanation needs


def load_prompts(root: Path) -> list[dict]:
    """The prompts under ``root``, one per non-empty line; none when the file is missing."""
    path = root / PROMPTS
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def explanation_slug(model: str, prompt: str, position: int = -1) -> str:
    """``<model slug>/explanations/<prompt id>``: what an explanation draw claims."""
    from openinterpretability.core.explanation_records import prompt_id
    from openinterpretability.core.identifiers import model_slug

    return f"{model_slug(model)}/explanations/{prompt_id(model, position, prompt)}"


def _explain(model: str, prompt: str, why: str) -> Candidate:
    # Every value is quoted, and the prompt comes after `--`. So each value stays one
    # argument, whatever it holds (#4397).
    command = f"open-interpretability explain --model {shlex.quote(model)} -- {shlex.quote(prompt)}"
    return Candidate(kind="explanation", model=model, weight=EXPLANATION_WEIGHT, rationale=why,
                     commands=[command], slug=explanation_slug(model, prompt),
                     skills=SKILLS["explanation"])  # fmt: skip


def explanation_candidates(states, claims: Sequence[str], root: Path, models: Sequence[str],
                           seed: int) -> list[Candidate]:  # fmt: skip
    """New, stale and unit candidates (module docstring), leaving out claimed ones."""
    from openinterpretability.builder.explanation_pages import load_views

    views = load_views(root / DEFAULT_WIKI_DIR, states)
    done = {(v.record.model_name, v.record.prompt) for v in views}
    out = [_explain(v.record.model_name, v.record.prompt,
                    f"stale: {len(v.changed)} cited units changed status since it was written")
           for v in views if v.stale]  # fmt: skip
    fresh = [
        (m, p["prompt"]) for m in models for p in load_prompts(root) if (m, p["prompt"]) not in done
    ]
    if fresh:
        model, prompt = random.Random(seed).choice(fresh)
        out.append(_explain(model, prompt, "a prompt that no explanation covers yet"))
    out += unit_candidates(views, states, configured_windows(root / "pyproject.toml"))
    return [c for c in out if not claimed(claim_key(c.slug), claims)]


def unit_candidates(views, states, windows: dict) -> list[Candidate]:
    """An attribution audit for each unstudied pole or head that an explanation cites."""
    from openinterpretability.builder.explanation_pages import UNIT_PAGE
    from openinterpretability.explain_groups import HEAD
    from openinterpretability.explain_plan import audit_command

    seen, out = set(), []
    for view in views:
        model = view.record.model_name
        for u in view.record.units:
            if not u.slug or u.slug in states or u.slug in seen or u.status == "stopped":
                continue
            seen.add(u.slug)
            found = UNIT_PAGE.match(u.slug)
            if found is None:  # pragma: no cover - ExplainedUnit.slug is validated by UNIT_SLUG
                continue
            index = int(found["unit"][1:])
            pole = HEAD if found["unit"][0] == "H" else found["pole"]
            if pole is None:  # pragma: no cover - UNIT_SLUG requires a neuron's pole
                continue
            audit = audit_command(model, (int(found["layer"]), index, pole), windows.get(model))
            why = (f"unstudied, and the explanation of {view.record.prompt[:60]!r} needs it "
                   f"({u.share:.1%} of the attribution)")  # fmt: skip
            out.append(Candidate(
                kind="explore", model=model, weight=UNIT_WEIGHT, rationale=why, commands=[audit],
                slug=u.slug, skills=SKILLS["explore"], layer=int(found["layer"]),
                audit=_audit_options([audit]),
            ))  # fmt: skip
    return out

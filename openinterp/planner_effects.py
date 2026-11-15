"""Explore draws weighted by a pole's effect on the loss (issue #3486).

On GPT-2, VERIFIED poles bring back no more of a layer's loss than random poles do
(``docs/analyses/registry-coverage.md``). So the registry may favour poles that are easy
to explain over poles that matter. So a share of explore draws (``[tool.openinterp]
effect_share``) picks a pole in proportion to its estimated effect. The effects come from
the model's table in ``data/pole-effects/`` (``open-interpretability pole-effects``). These
draws record ``selection: attribution``, so they never count towards the registry
statistics, which use only random draws. A model with no table always draws uniformly.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

EFFECTS_DIR = Path("data") / "pole-effects"
DEFAULT_EFFECT_SHARE = 0.3


def table_path(root: Path, model: str) -> Path:
    """Where ``model``'s table of pole effects is kept, under the repository ``root``."""
    return root / EFFECTS_DIR / f"{model.replace('/', '--')}.json"


def effect_share(pyproject: Path) -> float:
    """``[tool.openinterp] effect_share``: the share of explore draws weighted by effect."""
    from openinterp.planner import _config

    return float(_config(pyproject).get("effect_share", DEFAULT_EFFECT_SHARE))


def unit_slug(model: str, row: dict) -> str:
    """The slug of the pole that a table row names."""
    from openinterp.core.identifiers import NeuronIdentifier

    return NeuronIdentifier(
        model_name=model, layer=row["layer"], neuron_idx=row["neuron"], pole=row["pole"]
    ).slug


def weighted_pole(rng: random.Random, model: str, rows: list[dict], taken: set[str]):
    """A row drawn in proportion to its effect, among untaken poles that raise the loss.

    ``taken`` holds studied slugs and the claim keys of open PRs.
    """
    from openinterp.planner import claim_key

    slugs = {r_id: unit_slug(model, r) for r_id, r in enumerate(rows)}
    free = [
        r
        for r_id, r in enumerate(rows)
        if r["effect"] > 0 and not {slugs[r_id], claim_key(slugs[r_id])} & taken
    ]
    if not free:
        return None
    return rng.choices(free, weights=[r["effect"] for r in free], k=1)[0]


def effect_draw(rng: random.Random, model: str, pyproject: Path, taken: set[str], window):
    """An explore Candidate weighted by effect, or None for a uniform draw.

    It takes one draw from ``rng`` for the share, and one more if it weights.
    """
    from openinterp.planner import SKILLS, Candidate, _audit_options

    path = table_path(pyproject.parent, model)
    if rng.random() >= effect_share(pyproject) or not path.exists():
        return None
    row = weighted_pole(rng, model, json.loads(path.read_text())["poles"], taken)
    if row is None:
        return None
    audit = (
        f"open-interpretability audit --model {model} --layer {row['layer']} "
        f"--neuron {row['neuron']} --pole {row['pole']} --selection attribution"
    ) + (f" --window-tokens {window}" if window else "")
    return Candidate(
        kind="explore",
        model=model,
        weight=0.0,
        rationale=f"new pole drawn by its estimated loss effect ({row['effect']:.2e}); "
        "selection attribution, so it doesn't count towards registry statistics",
        commands=[audit],
        slug=unit_slug(model, row),
        skills=SKILLS["explore"],
        layer=row["layer"],
        audit=_audit_options([audit]),
    )

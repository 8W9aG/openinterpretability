"""Why the greedy coverage plan stalled, and what a joint plan needs (#7061).

It plans ``explain`` for a seeded sample of 8 copy-family prompts on pythia-70m: 6 train and
2 retain prompts. The family is ``docs/analyses/copy-family.jsonl`` (#5232). No LLM is called.
For each prompt it compares four plans of unit experiments:

- ``attribution``: the plan before #7061. It tries the first 120 candidates and picks at most 60.
- ``lazy``: :func:`openinterpretability.explain_joint.greedy`, the plan now.
- ``full``: the same greedy step, which scores every candidate at each pick.
- ``prune``: every planned unit kept, then units dropped while the coverage holds.

Then it plans jointly over the first 2, 4 and 8 prompts. Each row goes to the JSONL file.

    uv run python scripts/coverage_planner.py --device cuda \\
        --out papers/coverage-planner/results.jsonl
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = "EleutherAI/pythia-70m"
FAMILY = Path("docs/analyses/copy-family.jsonl")
SEED = 7061
TRAIN, RETAIN = 6, 2  # prompts of each split in the sample
OLD_POOL, OLD_STEPS = 120, 60  # the plan before #7061 (explain_coverage, protocol of #4558)
TARGETS = (1.0, 0.9, 0.8)
JOINT = (2, 4, 8)  # the first k prompts of the sample, planned jointly
JOINT_TARGET = 0.9


def sample(rows: list[dict], seed: int = SEED) -> list[dict]:
    """``TRAIN`` wrong-copy train prompts and ``RETAIN`` right-copy retain prompts, seeded.

    Args:
        rows: The family's prompt rows.
        seed: The seed of the draw.

    Examples:
        >>> rows = [{"kind": "wrong_copy", "split": "train", "prompt": str(i)} for i in range(9)]
        >>> rows += [{"kind": "right_copy", "split": "retain", "prompt": str(i)} for i in range(3)]
        >>> [r["kind"] for r in sample(rows)].count("right_copy")
        2
    """
    rng = random.Random(seed)
    train = [r for r in rows if r["kind"] == "wrong_copy" and r["split"] == "train"]
    retain = [r for r in rows if r["kind"] == "right_copy" and r["split"] == "retain"]
    return rng.sample(train, TRAIN) + rng.sample(retain, RETAIN)


def trajectory(now: float, gains: list[float]) -> list[float]:
    """The coverage after each pick: ``now`` plus the running sum of the gains.

    Args:
        now: The coverage before any pick.
        gains: Each pick's gain.

    Examples:
        >>> trajectory(0.1, [0.2, 0.3])
        [0.1, 0.3, 0.6]
    """
    out = [round(now, 4)]
    for gain in gains:
        out.append(round(out[-1] + gain, 4))
    return out


def first_at(path: list[float], target: float) -> int | None:
    """The number of picks after which ``path`` first reaches ``target``, or None.

    A value within ``explain_joint.TOLERANCE`` of the target reaches it, as in the plan.

    Args:
        path: The coverage after 0, 1, 2 … picks.
        target: The coverage to reach.

    Examples:
        >>> first_at([0.1, 0.5, 0.9], 0.8), first_at([0.1], 0.8)
        (2, None)
    """
    from openinterpretability.explain_joint import TOLERANCE

    return next((i for i, v in enumerate(path) if v >= target - TOLERANCE), None)


def attribution_plan(sets, candidates: list) -> list[float]:
    """The plan before #7061: greedy over the first ``OLD_POOL`` candidates, ``OLD_STEPS`` picks.

    Args:
        sets: The prompt's batched coverage measure.
        candidates: The planned experiments, in the order the old plan received them.
    """
    from openinterpretability.explain_joint import capped

    units, left = [], list(candidates[:OLD_POOL])
    path = [capped(sets.raw([[]])[0])]
    while path[-1] < 1.0 and left and len(path) <= OLD_STEPS:
        found = [capped(v) for v in sets.raw([[*units, *e.units] for e in left])]
        best = max(range(len(left)), key=lambda i: found[i])
        if found[best] <= path[-1]:
            break  # no candidate in the window raises coverage
        units += left.pop(best).units
        path.append(found[best])
    return [round(v, 4) for v in path]


def _slots(units: list) -> list[tuple[int, int, int]]:
    """Each unit's place in the masks of ``KeptSets.masks``: (mask, layer, index).

    Args:
        units: The units.

    Examples:
        >>> _slots([(0, 1, "positive"), (2, 3, "negative"), (1, 4, "head")])
        [(0, 0, 1), (1, 2, 3), (2, 1, 4)]
    """
    from openinterpretability.explain_groups import HEAD

    return [(2 if pole == HEAD else 0 if pole == "positive" else 1, layer, index)
            for layer, index, pole in units]  # fmt: skip


def prune(sets, candidates: list, targets=TARGETS, batch: int = 256) -> dict:
    """The experiments left when units are dropped from the full set while coverage holds.

    The first pass measures each experiment's drop when it alone is removed. The second
    removes them in order of the smallest drop. It keeps a removal only while the coverage
    stays at the target. Both passes edit masks, so no pass rebuilds the full set.

    Args:
        sets: The prompt's batched coverage measure.
        candidates: The planned experiments.
        targets: The coverages to hold.
        batch: The experiments left out in one batch of the first pass.
    """
    from openinterpretability.explain_joint import capped

    whole = sets.masks([[u for e in candidates for u in e.units]])
    full = sets.raw_masks(whole)[0]
    slots = [_slots(e.units) for e in candidates]
    drops = []
    for start in range(0, len(candidates), batch):
        block = range(start, min(start + batch, len(candidates)))
        masks = tuple(m.expand(-1, len(block), -1).clone() for m in whole)
        for row, i in enumerate(block):  # each row leaves one experiment out
            for kind, layer, index in slots[i]:
                masks[kind][layer, row, index] = False
        drops += [full - capped(v) for v in sets.raw_masks(masks)]
    order = sorted(range(len(candidates)), key=lambda i: drops[i])
    out = {"full": round(full, 4)}
    for target in targets:
        masks, kept = tuple(m.clone() for m in whole), len(candidates)
        for i in order:  # one forward pass per experiment, in order of the smallest drop
            was = [(kind, layer, index, bool(masks[kind][layer, 0, index]))
                   for kind, layer, index in slots[i]]  # fmt: skip
            for kind, layer, index, _ in was:
                masks[kind][layer, 0, index] = False
            if capped(sets.raw_masks(masks)[0]) >= target:
                kept -= 1
            else:  # the drop costs too much coverage: put the units back
                for kind, layer, index, value in was:
                    masks[kind][layer, 0, index] = value
        out[str(target)] = kept
    return out


def plan_prompt(runner, means, row: dict) -> tuple[dict, object, list]:  # long-function: four plans
    """Analyse one prompt as ``explain`` does, and compare the four plans on it.

    Args:
        runner: The model's runner.
        means: The unit means.
        row: The prompt's row of the family.

    Returns:
        The prompt's row of results, its batched measure and its planned experiments.
    """
    from openinterpretability import explain_coverage as ec
    from openinterpretability.cli_explain import Ask, _analyse, _answer
    from openinterpretability.explain_joint import greedy

    seen = {}
    original = ec.coverage_plan

    def spy(*args, **kwargs):  # keep the candidates in the order that the plan receives them
        """Record the plan's arguments, then make the plan.

        Args:
            args: The plan's positional arguments.
            kwargs: The plan's keyword arguments.

        Returns:
            The plan.
        """
        seen["args"], seen["order"] = args, list(args[6].experiments)
        return original(*args, **kwargs)

    ec.coverage_plan = spy
    try:
        ask = Ask(MODEL, row["prompt"], _answer(runner, None), -1, 20, 8, 1.0, 10)
        _analyse(runner, means, ask, Path("wiki"))
    finally:
        ec.coverage_plan = original
    _, prompt, _, token, position, order, planned, _ = seen["args"]
    explained = [u for u, _, _, done in order if done]
    sets = ec.scorer(runner, means, prompt, token, position, explained)
    candidates = seen["order"]  # not planned.experiments, which the new plan reorders
    out = {"table": "prompt", "prompt": prompt, "kind": row["kind"], "experiments": len(candidates),
           "explained": len(explained)}  # fmt: skip
    timed = time.time()
    out["attribution"] = {"path": attribution_plan(sets, candidates)}
    for name, pool in (("lazy", None), ("full", len(candidates))):
        start = time.time()
        found = greedy([sets], candidates, 1.0, **({"pool": pool} if pool else {}))
        out[name] = {"path": trajectory(found.now[0], found.gains), "scans": found.scans,
                     "seconds": round(time.time() - start),
                     "chosen": [e.target for e in found.chosen]}  # fmt: skip
    out["prune"] = prune(sets, candidates)
    out["seconds"] = round(time.time() - timed)
    for name in ("attribution", "lazy", "full"):
        out[name] |= {f"to_{t}": first_at(out[name]["path"], t) for t in TARGETS}
    return out, sets, candidates


def joint_rows(measures: list, plans: list, rows: list[dict]) -> list[dict]:
    """Plan the first k prompts jointly, for each k in ``JOINT``.

    Args:
        measures: Each prompt's batched coverage measure.
        plans: Each prompt's planned experiments.
        rows: Each prompt's row of results, with its own lazy plan.
    """
    from openinterpretability.explain_joint import candidates_of, greedy

    out = []
    for k in JOINT:
        start = time.time()
        candidates = candidates_of([SimpleNamespace(experiments=p) for p in plans[:k]])
        found = greedy(measures[:k], candidates, JOINT_TARGET)
        own = [r["lazy"]["chosen"][: r["lazy"][f"to_{JOINT_TARGET}"] or len(r["lazy"]["chosen"])]
               for r in rows[:k]]  # fmt: skip
        union = {t for picks in own for t in picks}
        out.append({"table": "joint", "prompts": k, "target": JOINT_TARGET,
                    "picks": len(found.chosen), "reached": found.reached,
                    "picked": [round(v, 4) for v in found.picked],
                    "separate_union": len(union), "separate_sum": sum(map(len, own)),
                    "shared": sum(sum(t in picks for picks in own) > 1 for t in union),
                    "scans": found.scans, "seconds": round(time.time() - start)})  # fmt: skip
    return out


@click.command()
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cpu", show_default=True)
def main(out: Path, device: str) -> None:  # pragma: no cover - loads the model
    """Plan the sample's prompts four ways, then jointly, and write the rows to OUT.

    Args:
        out: The JSONL file.
        device: The model's device.
    """
    from openinterpretability.cli import _load_runner
    from openinterpretability.evidence import read_corpus
    from openinterpretability.explain import unit_means

    family = [json.loads(line) for line in FAMILY.read_text(encoding="utf-8").splitlines()]
    chosen = sample([r for r in family if r.get("table") == "prompt"])
    runner = _load_runner(MODEL, device, "float32")
    means = unit_means(runner, MODEL, read_corpus(Path("corpus.jsonl"))[0], 20_000)
    rows, measures, plans = [], [], []  # the joint plans reuse each prompt's measure and plan
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in chosen:
            found, sets, candidates = plan_prompt(runner, means, row)
            rows.append(found)
            measures.append(sets)
            plans.append(candidates)
            f.write(json.dumps(found) + "\n")
            f.flush()
            click.echo(json.dumps({k: found[k] for k in ("prompt", "seconds")}))
        for joint in joint_rows(measures, plans, rows):
            f.write(json.dumps(joint) + "\n")
            f.flush()
            click.echo(json.dumps(joint))


if __name__ == "__main__":
    main()

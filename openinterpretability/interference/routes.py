"""Route level: whether each recorded route still holds on a changed model (issue #5230).

A circuit can break although its units keep. So the report re-runs the routes the registry
records, on every side:

- **Answer routes.** The newest record of each claim that stands VERIFIED is reviewed again
  with its own claim, probes, policy and seed (``answer_review.review``). No record is written.
  A route *flips* when it is VERIFIED on the base and not on a side.
- **Explanations.** Each explained prompt's best record has its causal coverage measured
  again (``explain_coverage.prompt_coverage``), with the record's own explained units. A
  coverage *drop* is the base's coverage less the side's.

Each side gets corpus means of its own, cached under the side's key, so a patched model
never reuses the base's means.
"""

from __future__ import annotations

from pathlib import Path

TOL = 1e-6


def answer_records(wiki_dir: Path, model: str) -> list:
    """The newest record of each answer claim of ``model`` that stands VERIFIED."""
    from openinterpretability.core.answer_records import ANSWERS_WIKI, load
    from openinterpretability.core.identifiers import model_slug

    root = Path(wiki_dir) / ANSWERS_WIKI / model_slug(model)
    out = []
    for folder in sorted(root.iterdir()) if root.is_dir() else []:
        newest: dict[int, object] = {}
        for record in load(Path(wiki_dir), model, folder.name):
            newest[record.claim.attempt] = record
        out += [r for r in newest.values() if r.verdict.status == "VERIFIED"]
    return out


def explanation_records(wiki_dir: Path, model: str) -> list:
    """The best explanation record of each prompt of ``model`` that keeps its inputs."""
    from openinterpretability.core.explanation_records import best, load_all

    out = []
    for records in load_all(Path(wiki_dir)).values():
        mine = [r for _, r in records if r.model_name == model and r.inputs is not None]
        if (chosen := best(mine)) is not None:
            out.append(chosen)
    return out


def side_means(sides: dict, key: str, texts: list[str], tokens: int) -> dict:
    """Each side's unit means over ``texts``, cached under ``key`` and the side's name."""
    from openinterpretability.explain import unit_means

    out = {}
    for name, side in sides.items():
        with side.entered() as runner:
            out[name] = unit_means(runner, f"{key}+{name}:{side.name}", texts, tokens)
    return out


def answer_routes(sides: dict, records: list, means: dict) -> list[dict]:
    """One row per VERIFIED answer route: its status on each side, and its counted flip."""
    from openinterpretability.answer_review import review

    rows = []
    for record in records:
        status = {}
        for name, side in sides.items():
            seed = record.verdict.measures.null_seed or 0
            with side.entered() as runner:
                again = review(runner, means[name], record.claim, record.probes,
                               record.verdict.policy, seed=seed)  # fmt: skip
            failed = [g for g, ok in again.verdict.gates.items() if not ok]
            status[name] = {"status": again.verdict.status, "failed": failed}
        held = status["base"]["status"] == "VERIFIED"
        flips = {k: held and v["status"] != "VERIFIED" for k, v in status.items() if k != "base"}
        floors = [v for k, v in flips.items() if k.startswith("floor")]
        rows.append({"table": "route", "kind": "answer", "id": record.id,
                     "prompt": record.claim.prompt, "status": status, "flips": flips,
                     "counts": flips["changed"] and not any(floors)})  # fmt: skip
    return rows


def explanation_routes(sides: dict, records: list, means: dict) -> list[dict]:
    """One row per explained prompt: its coverage on each side, and its counted drop."""
    from openinterpretability.core.answer_records import parse_unit
    from openinterpretability.explain_coverage import prompt_coverage

    rows = []
    for record in records:
        units = [parse_unit(label) for label in record.inputs.explained]
        order = [(unit, 0.0, "VERIFIED", True) for unit in units]
        coverage = {}
        for name, side in sides.items():
            token, at = record.inputs.token_id, record.position
            with side.entered() as runner:
                found = prompt_coverage(runner, record.prompt, means[name], token, at, order,
                                        draws=0)  # fmt: skip
            coverage[name] = found["coverage"]
        drop = {k: (coverage["base"] or 0) - (v or 0) for k, v in coverage.items() if k != "base"}
        floor = max([v for k, v in drop.items() if k.startswith("floor")] or [0.0])
        rows.append({"table": "route", "kind": "explanation", "prompt": record.prompt,
                     "coverage": coverage, "drop": drop,
                     "counts": drop["changed"] > max(floor, 0.0) + TOL})  # fmt: skip
    return rows

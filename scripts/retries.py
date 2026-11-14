"""Account for retries (issue #3204, docs/analyses/retries.md).

Each follow-up on a unit is another chance to reach VERIFIED. This script counts, per
native neuron pole and attention head, the experiments before its first VERIFIED. It gives
three VERIFIED shares (per record, per unit, per attempt), and a union bound on false
discoveries that charges each claim for its tries. It also compares the held-out detection
of units VERIFIED at the first attempt with that of units VERIFIED later.

    uv run python scripts/retries.py [--out docs/analyses/retries.json]

It reads the records and errata as the site does, so attempts, re-measurements and
set-aside experiments match the site's.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from null_calibration import wilson  # noqa: E402
from verified_vs_detection import bootstrap, mean, normalise  # noqa: E402

from openinterp.core.paths import DEFAULT_WIKI_DIR, experiments_dir  # noqa: E402


def units(wiki_dir: Path = DEFAULT_WIKI_DIR) -> list[tuple[list, object]]:
    """Each native pole's or head's experiments, oldest first, without set-aside ones.

    Each comes with the pole's standing, which the site's false-discovery number reads.
    """
    from openinterp.builder.compiler import load_records
    from openinterp.builder.standing import native_standings
    from openinterp.builder.tree import build_tree
    from openinterp.core.errata import load_errata

    models = build_tree(load_records(experiments_dir(wiki_dir)), load_errata(wiki_dir))
    out = []
    for node, side, standing in native_standings(models):
        pole = [e for e in node.experiments if e.side == side and not e.set_aside]
        if pole:
            out.append((sorted(pole, key=lambda e: e.record.verdict.timestamp), standing))
    return out


def attempts(pole: list) -> dict:
    """How many experiments, all and independent, until the first VERIFIED (or in all)."""
    first = next((i for i, e in enumerate(pole) if e.record.verdict.status == "VERIFIED"), None)
    tried = pole if first is None else pole[: first + 1]
    return {
        "verified": first is not None,
        "all": len(tried),
        "independent": sum(e.independent for e in tried),
        "first": None if first is None else pole[first],
    }


def selection(pole: list) -> str:
    """How the unit was first chosen: its first experiment's provenance."""
    provenance = pole[0].record.submission.provenance
    return provenance.selection if provenance else "unknown"


def shares(rows: list[dict], records: int, verified_records: int) -> dict:
    """VERIFIED per record, per unit and per attempt (units ever VERIFIED / attempts spent)."""
    ever = sum(r["verified"] for r in rows)
    spent = sum(r["all"] for r in rows)
    spent_independent = sum(r["independent"] for r in rows)
    return {
        "per_record": wilson(verified_records, records),
        "per_unit": wilson(ever, len(rows)),
        "per_attempt": wilson(ever, spent),
        "per_independent_attempt": wilson(ever, spent_independent),
        "units": len(rows), "units_verified": ever, "records": records,
        "verified_records": verified_records, "attempts": spent,
        "independent_attempts": spent_independent,
    }  # fmt: skip


def union_bound(claims: list[tuple[float, int]]) -> dict:
    """Expected false discoveries: today's sum of 1 − P, and min(1, k·(1 − P)) for k tries."""
    today = sum(1 - p for p, _ in claims)
    bound = sum(min(1.0, k * (1 - p)) for p, k in claims)
    n = len(claims)
    return {
        "claims": n,
        "expected_false": round(today, 2),
        "rate": round(today / n, 4) if n else None,
        "union_bound_false": round(bound, 2),
        "union_bound_rate": round(bound / n, 4) if n else None,
    }


def first_verified_scores(rows: list[dict]) -> list[dict]:
    """The held-out scores of each unit's first VERIFIED record, tagged first try or later."""
    out = []
    for r in rows:
        scores = r["first"].record.verdict.explanation_scores if r["first"] else None
        if scores is None or scores.detection is None:
            continue
        out.append({
            "status": "VERIFIED" if r["independent"] == 1 else "LATER",
            "judge": normalise(scores.judge_model),
            "detection": scores.detection.balanced_accuracy,
            "fuzzing": scores.fuzzing.balanced_accuracy if scores.fuzzing else None,
        })  # fmt: skip
    return out


def first_vs_later(scored: list[dict], kind: str) -> dict:
    """First-try minus later, within judge and pooled, with a bootstrap interval."""
    from verified_vs_detection import within_judge

    first = [r[kind] for r in scored if r["status"] == "VERIFIED" and r[kind] is not None]
    later = [r[kind] for r in scored if r["status"] == "LATER" and r[kind] is not None]
    return {
        "n_first": len(first), "n_later": len(later),
        "mean_first": mean(first), "mean_later": mean(later),
        "within_judge": within_judge(scored, kind),
        "within_judge_ci": bootstrap(scored, lambda d: within_judge(d, kind)),
    }  # fmt: skip


def report(units_: list[tuple[list, object]]) -> dict:
    """Every number the write-up uses."""
    poles = [p for p, _ in units_]
    rows = [attempts(p) | {"pole": p, "standing": s} for p, s in units_]
    records = sum(len(p) for p in poles)
    verified_records = sum(e.record.verdict.status == "VERIFIED" for p in poles for e in p)
    verified = [r for r in rows if r["verified"]]
    by_selection: dict[str, list[dict]] = defaultdict(list)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_selection[selection(r["pole"])].append(r)
        by_model[r["pole"][0].record.submission.identifier.model_name].append(r)

    def counts(group, key):
        return dict(sorted(Counter(r[key] for r in group if r["verified"]).items()))

    # The site's claims: current VERIFIED standings, each at its weakest agreeing review
    # (tree.false_discovery). Each is charged for its independent tries to the first VERIFIED.
    claims = [
        (min(e.sig.p_useful for e in r["standing"].readings), r["independent"])
        for r in verified
        if r["standing"].verified
    ]
    scored = first_verified_scores(rows)
    return {
        "shares": shares(rows, records, verified_records),
        "attempts_to_first_verified": {"all": counts(rows, "all"),
                                       "independent": counts(rows, "independent")},
        "by_selection": {k: shares(v, sum(len(r["pole"]) for r in v),
                                   sum(e.record.verdict.status == "VERIFIED"
                                       for r in v for e in r["pole"])) | {
                             "independent_to_first": counts(v, "independent")}
                         for k, v in sorted(by_selection.items())},
        "by_model": {k: {"units": len(v), "verified": sum(r["verified"] for r in v),
                         "independent_to_first": counts(v, "independent")}
                     for k, v in sorted(by_model.items())},
        "false_discoveries": union_bound(claims),
        "held_out": {kind: first_vs_later(scored, kind) for kind in ("detection", "fuzzing")},
    }  # fmt: skip


def main() -> None:
    """Compute the report and write it as JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, default=Path("docs/analyses/retries.json"))
    ap.add_argument("--wiki-dir", type=Path, default=DEFAULT_WIKI_DIR)
    args = ap.parse_args()
    result = report(units(args.wiki_dir))
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("shares", "false_discoveries", "held_out")}, indent=1))


if __name__ == "__main__":
    main()

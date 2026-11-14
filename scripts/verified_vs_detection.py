"""Does VERIFIED predict held-out detection? (issue #3239, docs/analyses/verified-vs-detection.md).

Reads every neuron-pole record with explanation scores, and compares detection and
fuzzing balanced accuracy between VERIFIED and other records. The judge model changes
the scores a lot, so the main number is the difference within each judge, pooled over
judges. Bootstrap intervals resample records within each judge.

    uv run python scripts/verified_vs_detection.py [--out docs/analyses/verified-vs-detection.json]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from openinterp.core.paths import DEFAULT_WIKI_DIR, experiments_dir
from openinterp.core.schemas import AuditRecord
from openinterp.harness.explanation_scores import auroc

KINDS = ("detection", "fuzzing")
MIN_PER_ARM = 10  # a judge needs this many records in each arm to enter the pooled number
BOOTSTRAP = 2000


def normalise(model: str | None) -> str:
    """One name per model: drop `external/`, context tags such as `[1m]`, and `-contributor`."""
    name = re.sub(r"\[.*?\]", "", (model or "unknown").removeprefix("external/"))
    return name.removesuffix("-contributor").replace("1-3", "1.3")


def rows(wiki_dir: Path = DEFAULT_WIKI_DIR) -> list[dict]:
    """One row per scored neuron-pole record."""
    out = []
    for path in sorted(experiments_dir(wiki_dir).rglob("*.json")):
        record = AuditRecord.model_validate_json(path.read_text())
        verdict, ident = record.verdict, record.submission.identifier
        scores = verdict.explanation_scores if verdict else None
        if ident.kind != "mlp_neuron_pole" or scores is None or scores.detection is None:
            continue
        out.append({
            "slug": ident.slug, "time": verdict.timestamp.isoformat(), "status": verdict.status,
            "model": ident.model_name, "judge": normalise(scores.judge_model),
            "reviewer": normalise(verdict.reviewer_agent_model),
            "protocol": verdict.policy.protocol_version if verdict.policy else "unknown",
            "detection": scores.detection.balanced_accuracy,
            "fuzzing": scores.fuzzing.balanced_accuracy if scores.fuzzing else None,
        })  # fmt: skip
    return out


def latest_per_unit(data: list[dict]) -> list[dict]:
    """The newest record of each unit, since a unit's records are not independent."""
    best: dict[str, dict] = {}
    for row in data:
        if row["slug"] not in best or row["time"] > best[row["slug"]]["time"]:
            best[row["slug"]] = row
    return list(best.values())


def mean(xs: list[float]) -> float | None:
    """The mean, or ``None`` for no values."""
    return sum(xs) / len(xs) if xs else None


def arms(data: list[dict], kind: str) -> tuple[list[float], list[float]]:
    """(VERIFIED scores, other scores) of one kind, leaving out missing ones."""
    ok = [r[kind] for r in data if r["status"] == "VERIFIED" and r[kind] is not None]
    other = [r[kind] for r in data if r["status"] != "VERIFIED" and r[kind] is not None]
    return ok, other


def within_judge(data: list[dict], kind: str) -> float | None:
    """VERIFIED minus other, per judge, pooled with weights n_v·n_o/(n_v+n_o)."""
    by_judge: dict[str, list[dict]] = defaultdict(list)
    for row in data:
        by_judge[row["judge"]].append(row)
    total = weight = 0.0
    for group in by_judge.values():
        ok, other = arms(group, kind)
        if len(ok) >= MIN_PER_ARM and len(other) >= MIN_PER_ARM:
            w = len(ok) * len(other) / (len(ok) + len(other))
            total += w * (mean(ok) - mean(other))
            weight += w
    return total / weight if weight else None


def bootstrap(data: list[dict], stat, seed: int = 0) -> tuple[float, float] | None:
    """A 95% interval for ``stat(data)``, resampling records within each judge."""
    by_judge: dict[str, list[dict]] = defaultdict(list)
    for row in data:
        by_judge[row["judge"]].append(row)
    rng, values = random.Random(seed), []
    for _ in range(BOOTSTRAP):
        sample = [rng.choice(g) for g in by_judge.values() for _ in g]
        value = stat(sample)
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def summary(data: list[dict], kind: str) -> dict:
    """Means, the raw and within-judge differences with intervals, and the AUROC."""
    ok, other = arms(data, kind)
    labelled = [(r[kind], r["status"] == "VERIFIED") for r in data if r[kind] is not None]
    return {
        "n_verified": len(ok), "n_other": len(other),
        "mean_verified": mean(ok), "mean_other": mean(other),
        "difference": (mean(ok) - mean(other)) if ok and other else None,
        "within_judge": within_judge(data, kind),
        "within_judge_ci": bootstrap(data, lambda d: within_judge(d, kind)),
        "auroc": auroc([s for s, _ in labelled], [y for _, y in labelled]),
    }  # fmt: skip


def strata(data: list[dict], key: str, kind: str) -> dict[str, dict]:
    """``summary`` without intervals for each value of ``key`` (model, judge, …)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in data:
        groups[row[key]].append(row)
    out = {}
    for value, group in sorted(groups.items()):
        ok, other = arms(group, kind)
        out[value] = {"n_verified": len(ok), "n_other": len(other), "mean_verified": mean(ok),
                      "mean_other": mean(other),
                      "difference": (mean(ok) - mean(other)) if ok and other else None}  # fmt: skip
    return out


def status_mean(data: list[dict], kind: str, status: str) -> float | None:
    """The mean score of one kind over the records with ``status``."""
    return mean([r[kind] for r in data if r["status"] == status and r[kind] is not None])


def report(data: list[dict]) -> dict:
    """Every number the write-up uses."""
    out: dict = {"records": len(data)}
    statuses = ("VERIFIED", "NEEDS_REVISION", "REJECTED")
    keys = ("model", "judge", "reviewer", "protocol")
    for kind in KINDS:
        out[kind] = {
            "all_records": summary(data, kind),
            "latest_per_unit": summary(latest_per_unit(data), kind),
            "by_status": {s: status_mean(data, kind, s) for s in statuses},
            **{f"by_{k}": strata(data, k, kind) for k in keys},
        }
    return out


def main() -> None:
    """Compute the report and write it as JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, default=Path("docs/analyses/verified-vs-detection.json"))
    ap.add_argument("--wiki-dir", type=Path, default=DEFAULT_WIKI_DIR)
    args = ap.parse_args()
    result = report(rows(args.wiki_dir))
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    for kind in KINDS:
        s = result[kind]["all_records"]
        print(f"{kind}: VERIFIED {s['mean_verified']:.3f} (n={s['n_verified']}) vs other "
              f"{s['mean_other']:.3f} (n={s['n_other']}); within judge {s['within_judge']:+.3f} "
              f"{s['within_judge_ci']}; AUROC {s['auroc']:.3f}")  # fmt: skip


if __name__ == "__main__":
    main()

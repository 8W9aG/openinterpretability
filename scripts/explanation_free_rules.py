"""Explanation-free rules on the registry's stored scoring windows (issue #3388).

"Scoring Explanations Without Reading Them" (arXiv 2610.37676) finds that a rule that
never reads the explanation can score evaluation items. It asks that each score be
reported beside the best such rule. This script applies such rules to the detection
and fuzzing windows that every scored record stores, and prints their balanced accuracy
next to the judges'. Each rule's cut is chosen on the same windows, so its score is an
upper bound, as the paper's post hoc rules are.

    uv run python scripts/explanation_free_rules.py
        [--out docs/analyses/explanation-free-rules.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openinterp.core.paths import DEFAULT_WIKI_DIR, experiments_dir
from openinterp.core.schemas import AuditRecord


def balanced(pairs: list[tuple[bool, bool]]) -> float | None:
    """Balanced accuracy of (predicted, label) pairs, or ``None`` with one class."""
    pos = [p for p, y in pairs if y]
    neg = [not p for p, y in pairs if not y]
    if not pos or not neg:
        return None
    return (sum(pos) / len(pos) + sum(neg) / len(neg)) / 2


def contiguous(marks: list[int]) -> bool:
    """True when the marked tokens form one unbroken run."""
    return bool(marks) and max(marks) - min(marks) + 1 == len(marks)


FUZZING_RULES = {
    "marks_at_least_4": lambda w: len(w.highlighted) >= 4,
    "marks_contiguous": lambda w: contiguous(sorted(w.highlighted)),
    "contiguous_and_at_least_2": lambda w: (
        len(w.highlighted) >= 2 and contiguous(sorted(w.highlighted))
    ),
}


def best_length_rule(windows) -> float | None:
    """The best "active if the window has at least n tokens" rule, n chosen on the windows."""
    pairs = [(len(w.tokens), w.label) for w in windows]
    scores = [balanced([(n >= cut, y) for n, y in pairs]) for cut in sorted({n for n, _ in pairs})]
    scores = [s for s in scores if s is not None]
    return max(scores) if scores else None


def pooled(records) -> dict:
    """Each rule's balanced accuracy over all records' windows, with the judges' for comparison."""
    det = [w for r in records for w in r.detection_windows]
    fuz = [w for r in records for w in r.fuzzing_windows]
    out = {
        "records": len(records),
        "detection_windows": len(det),
        "fuzzing_windows": len(fuz),
        "judge_detection": balanced([(w.judged, w.label) for w in det if w.judged is not None]),
        "judge_fuzzing": balanced([(w.judged, w.label) for w in fuz if w.judged is not None]),
        "detection_length_rule": best_length_rule(det),
    }
    for name, rule in FUZZING_RULES.items():
        out[f"fuzzing_{name}"] = balanced([(rule(w), w.label) for w in fuz])
    return out


def scored_records(wiki_dir: Path = DEFAULT_WIKI_DIR) -> list:
    """The explanation scores of every neuron-pole record that has them."""
    out = []
    for path in sorted(experiments_dir(wiki_dir).rglob("*.json")):
        record = AuditRecord.model_validate_json(path.read_text())
        scores = record.verdict.explanation_scores if record.verdict else None
        if record.submission.identifier.kind == "mlp_neuron_pole" and scores and scores.detection:
            out.append(scores)
    return out


def main() -> None:
    """Compute the rules' scores and write them as JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", type=Path, default=Path("docs/analyses/explanation-free-rules.json"))
    args = ap.parse_args()
    result = pooled(scored_records())
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    for key, value in result.items():
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()

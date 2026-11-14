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
from openinterp.harness.explanation_scores import _balanced, explanation_free_baseline


def pooled(records) -> dict:
    """The rules' balanced accuracy over all records' windows, beside the judges'.

    The rules are ``explanation_free_baseline``, the same ones each new score records.
    """
    det = [w for r in records for w in r.detection_windows]
    fuz = [w for r in records for w in r.fuzzing_windows]
    return {
        "records": len(records),
        "detection_windows": len(det),
        "fuzzing_windows": len(fuz),
        "judge_detection": _balanced([(w.judged, w.label) for w in det if w.judged is not None]),
        "judge_fuzzing": _balanced([(w.judged, w.label) for w in fuz if w.judged is not None]),
        **explanation_free_baseline(det, fuz),
    }


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

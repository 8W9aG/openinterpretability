"""The statistics of the copy-gate study (scripts/copy_gate.py, #5233)."""

import importlib.util
import json
import sys
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "copy_gate.py"
spec = importlib.util.spec_from_file_location("copy_gate", SCRIPT)
gate = importlib.util.module_from_spec(spec)
sys.modules["copy_gate"] = gate
spec.loader.exec_module(gate)


def test_auroc_counts_ties_half_and_has_an_interval():
    assert gate.auroc([3.0, 4.0], [1.0, 2.0]) == 1.0
    assert gate.auroc([1.0], [1.0]) == 0.5
    low, high = gate.interval([0.9, 0.8, 0.7, 0.2], [0.1, 0.3, 0.6, 0.4])
    assert 0.0 <= low <= gate.auroc([0.9, 0.8, 0.7, 0.2], [0.1, 0.3, 0.6, 0.4]) <= high <= 1.0


def test_the_threshold_lets_through_at_most_its_share_of_negatives():
    neg = [float(i) for i in range(100)]
    cut = gate.threshold(neg, 0.05)
    assert sum(n > cut for n in neg) <= 5
    assert gate._recall_at([200.0, 50.0], neg) == 0.5


def test_out_of_fold_scores_come_from_probes_that_never_saw_the_row():
    g = torch.Generator().manual_seed(0)
    x = torch.cat([torch.randn(40, 3, generator=g) + 2, torch.randn(40, 3, generator=g) - 2])
    y = torch.cat([torch.ones(40), torch.zeros(40)])
    oof = gate._out_of_fold("logistic", x, y)
    assert gate.auroc(oof[y == 1].tolist(), oof[y == 0].tolist()) > 0.95  # separable data
    noise = torch.randn(80, 3, generator=g)
    oof = gate._out_of_fold("logistic", noise, y)
    assert gate.auroc(oof[y == 1].tolist(), oof[y == 0].tolist()) < 0.75  # nothing to learn


def test_the_recorded_operating_point_matches_the_analysis():
    rows = [
        json.loads(line) for line in (SCRIPT.parents[1] / "docs/analyses/copy-gate.jsonl").open()
    ]
    best = next(r for r in rows if r.get("probe") == "logistic_corpus" and r.get("layer") == 5)
    assert (best["recall"], best["right_fpr"], best["corpus_fpr"]) == (0.5843, 0.0515, 0.0)

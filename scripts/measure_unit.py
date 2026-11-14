"""Measure texts on one unit of an exchange directory, as a fraction of the unit's scale.

    uv run python scripts/measure_unit.py <unit dir> "<text>" ["<text>" ...]

It prints each text's peak and each token's value. The proposers of a blinded study
(issues #3202 and #3201) measure with it, so the command they run names no arm or study.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Each blinded study keeps its units under exchange/<study>/ (p3202: the untrained null,
# p3201: ground-truth calibration). The directory picks the study's runner.
STUDIES = {"p3201": "ground_truth", "p3202": "untrained_null"}

if __name__ == "__main__":
    study = next((m for d, m in STUDIES.items() if d in Path(sys.argv[1]).resolve().parts), None)
    if study is None:
        sys.exit(f"{sys.argv[1]} is not a unit directory of a blinded study")
    __import__(study).cli(["strengths", *sys.argv[1:]])

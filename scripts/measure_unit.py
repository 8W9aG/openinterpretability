"""Measure texts on one unit of an exchange directory, as a fraction of the unit's scale.

    uv run python scripts/measure_unit.py <unit dir> "<text>" ["<text>" ...]

It prints each text's peak and each token's value. The proposers of a blinded study
(issue #3202) measure with it, so the command they run names no arm.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from untrained_null import cli  # noqa: E402

if __name__ == "__main__":
    cli(["strengths", *sys.argv[1:]])

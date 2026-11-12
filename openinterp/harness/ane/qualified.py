"""The models whose scans may run on the ANE: each one passed ``scripts/ane_qualify.py``.

The ANE computes in fp16, and Core ML may run on the CPU the parts of a model that
the ANE can't hold. A model of a supported architecture is not enough. So a model is
listed here only after a qualification run on real text passed every check. The checks
are in ``scripts/ane_qualify.py`` and the results in ``docs/analyses/ane-qualification.md``.
A model that is not listed scans on torch. To add one, run the script, and add the
model with the layers and date of its passing run.
"""

from __future__ import annotations

# Model name -> the layers that its qualification run checked, and the run's date.
QUALIFIED: dict[str, tuple[tuple[int, ...], str]] = {}


def qualified(model_name: str) -> bool:
    """True if scans of ``model_name`` may run on the ANE."""
    return model_name in QUALIFIED

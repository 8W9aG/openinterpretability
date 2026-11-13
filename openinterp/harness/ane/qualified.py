"""The models whose scans may run on the ANE: each one passed ``scripts/ane_qualify.py``.

The ANE computes in fp16, and Core ML may run on the CPU the parts of a model that
the ANE can't hold. A model of a supported architecture is not enough. So a model is
listed here only after a qualification run on real text passed every check. The checks
are in ``scripts/ane_qualify.py`` and the results in ``docs/analyses/ane-qualification.md``.
A model that is not listed scans on torch. To add one, run the script, and add the
model with the layers and date of its passing run.

A model can be qualified only up to a layer. Then a scan of a deeper layer runs on
torch. Qwen2.5-1.5B's last layer, 27, failed the correlation check, and torch's own
fp16 failed it by the same amount. So fp16 is not precise enough there, on any device.
"""

from __future__ import annotations

# Model name -> the layers that its qualification run checked, and the run's date.
QUALIFIED: dict[str, tuple[tuple[int, ...], str]] = {
    "openai-community/gpt2": ((5, 11), "2026-11-12"),  # with 512-token windows
    "Qwen/Qwen2.5-0.5B": ((12, 20), "2026-11-12"),
    "Qwen/Qwen2.5-1.5B": ((8, 20), "2026-11-12"),  # layer 27 failed: see DEEPEST
}

# Model name -> the deepest layer that may scan on the ANE. A model not here: every layer.
DEEPEST: dict[str, int] = {
    "Qwen/Qwen2.5-1.5B": 20,
}


def qualified(model_name: str) -> bool:
    """True if scans of ``model_name`` may run on the ANE, for some layers."""
    return model_name in QUALIFIED


def too_deep(model_name: str, layer: int) -> str | None:
    """Why a scan of ``layer`` may not run on the ANE, or None if it may."""
    deepest = DEEPEST.get(model_name)
    if deepest is not None and layer > deepest:
        return f"{model_name} is qualified for the ANE only to layer {deepest}"
    return None

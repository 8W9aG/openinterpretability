"""The environment a scan ran in, recorded with each submission."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from openinterp.core.schemas import RunEnvironment
from openinterp.harness.memory import is_streamed
from openinterp.harness.moe import experts_implementation

if TYPE_CHECKING:
    from openinterp.harness.hooks import NeuronHookRunner


def run_environment(runner: NeuronHookRunner) -> RunEnvironment:
    """The device, weights dtype, model revision, library versions and streaming of ``runner``."""
    from openinterp import __version__

    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    config = getattr(runner.model, "config", None)
    return RunEnvironment(
        device=str(runner.device),
        dtype=str(next(runner.model.parameters()).dtype).removeprefix("torch."),
        model_revision=getattr(runner, "model_revision", None)
        or getattr(config, "_commit_hash", None),
        openinterp_version=__version__,
        torch_version=torch.__version__,
        transformers_version=transformers_version,
        streamed=True if is_streamed(runner.model) else None,
        experts_implementation=experts_implementation(runner.model),
    )

"""What every loaded model gets before it runs: inference mode, aligned weights, a fused GELU."""

from __future__ import annotations

from torch import nn

from openinterp.harness.cpu_speed import fuse_gelu


def prepare_model(model: nn.Module) -> None:
    """Make ``model`` ready to measure, in place.

    * Nothing here trains the model. Gradients are only taken for inputs and activations.
      A weight that requires a gradient makes autograd keep more. On a streamed
      Qwen2.5-1.5B, integrated gradients then peaked at 3.7 GB, not 1.8 GB.
    * Misaligned weights are copied (:func:`align_parameters`).
    * GPT-2's GELU runs as one kernel (:func:`openinterp.harness.cpu_speed.fuse_gelu`).
    """
    model.eval().requires_grad_(False)
    align_parameters(model)
    fuse_gelu(model)


def align_parameters(model: nn.Module, alignment: int = 64) -> None:
    """Copy parameters that sit at a misaligned address into fresh memory.

    Weights memory-mapped from a safetensors file start where the file puts them.
    GPT-2's weights are not even 4-byte aligned. A matrix-vector product on them (as the
    vocabulary lens does) dies with SIGBUS in Apple's BLAS. Shared (tied) parameters are
    one object, so they stay tied.
    """
    for param in model.parameters():
        if param.data_ptr() % alignment:
            param.data = param.data.clone()

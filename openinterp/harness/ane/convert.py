"""Convert a :class:`~openinterp.harness.ane.forward.ToLayer` to a compiled Core ML model.

This is the only module that imports coremltools, which the project installs only on
Apple-silicon Macs. It imports coremltools when a function runs, so the rest of the
package works without it.
"""

from __future__ import annotations

import collections
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

MIN_MACOS = "macOS15"  # the deployment target: Core ML's fp16 ops for the ANE


def available() -> bool:
    """True if coremltools can be imported."""
    try:
        import coremltools  # noqa: F401
    except ImportError:
        return False
    return True


def version() -> str | None:
    """The coremltools version, or None if it is not installed."""
    try:
        import coremltools
    except ImportError:
        return None
    return coremltools.__version__


def compile_to(net: torch.nn.Module, sample: torch.Tensor, dest: Path) -> None:
    """Trace ``net`` on ``sample``, convert it to fp16, and compile it to ``dest``.

    ``sample`` is token ids ``[rows, seq]`` (the input ``ids``) or the residual stream
    ``[rows, seq, d_model]`` (the input ``resid``, in fp16). ``dest`` is a ``.mlmodelc``
    directory that does not exist yet.
    """
    import coremltools as ct

    with torch.no_grad():
        traced = torch.jit.trace(net.eval(), sample, check_trace=False)
    if sample.is_floating_point():
        spec = ct.TensorType(name="resid", shape=sample.shape, dtype=np.float16)
    else:
        spec = ct.TensorType(name="ids", shape=sample.shape, dtype=np.int32)
    model = ct.convert(
        traced,
        inputs=[spec],
        outputs=[ct.TensorType(name="acts")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=getattr(ct.target, MIN_MACOS),
    )
    work = Path(tempfile.mkdtemp(prefix="openinterp-coreml-"))
    try:
        package = work / "to_layer.mlpackage"
        model.save(str(package))
        ct.utils.compile_model(str(package), str(dest))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def load(path: Path):
    """The compiled model at ``path``, set to run on the CPU and the ANE."""
    import coremltools as ct

    return ct.models.CompiledMLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)


# The last input arrays given to each Core ML model. When a model is idle for about half
# a second, Core ML resets its execution stream on a background thread and releases the
# last input. coremltools 9.0 then frees that array without the interpreter's lock, and
# the process ends with a segmentation fault. An array that is still referenced here is
# not freed. Only the latest inputs of each model can be released that way, so a few are
# kept. Token ids are 8 KB a batch, and a chunk's residual input is 6 MB (Qwen2.5-1.5B).
KEEP_INPUTS = 4
_INPUTS: dict[int, collections.deque] = collections.defaultdict(
    lambda: collections.deque(maxlen=KEEP_INPUTS)
)


def predict(model, inputs: torch.Tensor) -> torch.Tensor:
    """Run ``model`` on token ids or the residual stream; return its output in fp32.

    The output is ``[rows, seq, width]``: activations, or the residual stream.
    """
    if inputs.is_floating_point():
        name, array = "resid", inputs.numpy().astype(np.float16)
    else:
        name, array = "ids", inputs.numpy().astype(np.int32)
    _INPUTS[id(model)].append(array)  # see _INPUTS
    out = model.predict({name: array})["acts"]
    return torch.as_tensor(np.asarray(out, dtype=np.float32)).reshape(*inputs.shape[:2], -1)

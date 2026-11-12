"""Convert a :class:`~openinterp.harness.ane.forward.ToLayer` to a compiled Core ML model.

This is the only module that imports coremltools, which the project installs only on
Apple-silicon Macs. It imports coremltools when a function runs, so the rest of the
package works without it.
"""

from __future__ import annotations

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


def compile_to(net: torch.nn.Module, rows: int, seq: int, dest: Path) -> None:
    """Trace ``net`` on ``[rows, seq]`` token ids, convert it to fp16, and compile it to ``dest``.

    ``dest`` is a ``.mlmodelc`` directory that does not exist yet.
    """
    import coremltools as ct

    ids = torch.zeros((rows, seq), dtype=torch.long)
    with torch.no_grad():
        traced = torch.jit.trace(net.eval(), ids, check_trace=False)
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="ids", shape=ids.shape, dtype=np.int32)],
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


def predict(model, ids: torch.Tensor) -> torch.Tensor:
    """Run ``model`` on ``ids`` ``[rows, seq]``; return fp32 activations ``[rows, seq, width]``."""
    out = model.predict({"ids": ids.numpy().astype(np.int32)})["acts"]
    return torch.as_tensor(np.asarray(out, dtype=np.float32)).reshape(*ids.shape, -1)

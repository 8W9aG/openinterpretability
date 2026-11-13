"""Where a measurement ran: the device, the hardware, the versions and how the device was chosen."""

from __future__ import annotations

from pydantic import Field

from openinterp.core.strict import StrictModel


class RunEnvironment(StrictModel):
    """Where a measurement ran. Reproducing a record pins the model revision and dtype.

    The rest explains drift that another setup can cause. That is the device and its
    hardware, the operating system, the library versions, and where the scan's forward
    pass ran. ``device_choice`` says how the device was chosen. For example, the planner
    can send a job to the CPU because the GPU is busy. A record holds no hostname or
    user name.
    """

    device: str
    dtype: str = Field(description="Dtype of the loaded weights, e.g. float32.")
    model_revision: str | None = Field(default=None, description="Hugging Face commit hash.")
    openinterp_version: str
    torch_version: str
    transformers_version: str | None = None
    streamed: bool | None = Field(default=None, description="Layers read from disk per pass.")
    experts_implementation: str | None = Field(default=None, description="MoE experts forward.")
    scan_backend: str | None = Field(default=None, description="Scan's pass, if not on device.")
    device_name: str | None = Field(
        default=None, description="The hardware: the GPU's name, or the chip or CPU's name."
    )
    os: str | None = Field(default=None, description="Operating system, e.g. macOS-15.6.")
    device_choice: str | None = Field(
        default=None, description="How the device was chosen: forced, planned, or a fallback."
    )
    coremltools_version: str | None = Field(
        default=None, description="Core ML tools, if the scan ran on the ANE."
    )

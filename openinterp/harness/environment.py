"""The environment a scan ran in, recorded with each submission."""

from __future__ import annotations

import os
import platform
import subprocess
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from openinterp.core.schemas import RunEnvironment
from openinterp.harness import devices
from openinterp.harness.memory import is_streamed
from openinterp.harness.moe import experts_implementation

if TYPE_CHECKING:
    from openinterp.harness.hooks import NeuronHookRunner


# A hosted environment, by the environment variable that it sets. The first one set wins.
HOSTS = (
    ("COLAB_RELEASE_TAG", "colab"),
    ("KAGGLE_KERNEL_RUN_TYPE", "kaggle"),
    ("GITHUB_ACTIONS", "github-actions"),
    ("CI", "ci"),
)
PRODUCT_ENV = "OPENINTERP_AGENT_PRODUCT"


def host() -> str | None:
    """The hosted environment the run is in, or None on a local machine."""
    return next((name for var, name in HOSTS if os.environ.get(var)), None)


def agent_product() -> str | None:
    """The agent product that drove the run.

    ``$OPENINTERP_AGENT_PRODUCT`` wins when set. Else ``claude-code`` when Claude Code
    runs the command (it sets ``CLAUDECODE=1``). Else None.
    """
    if named := os.environ.get(PRODUCT_ENV, "").strip():
        return named
    return "claude-code" if os.environ.get("CLAUDECODE") == "1" else None


def run_environment(runner: NeuronHookRunner) -> RunEnvironment:
    """Where ``runner`` measures: its device and hardware, versions, and how it was chosen."""
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
        device_name=device_name(runner.device),
        os=platform.platform(terse=True),
        device_choice=devices.choice(),
        cpu_threads=torch.get_num_threads() if runner.device.type == "cpu" else None,
        host=host(),
        agent_product=agent_product(),
    )


def device_name(device: torch.device) -> str | None:
    """The hardware behind ``device``: a CUDA GPU's name, else the chip or CPU's name."""
    if device.type == "cuda":
        with suppress(Exception):
            return torch.cuda.get_device_name(device)
    if platform.system() == "Darwin":  # one chip serves the CPU and MPS
        with suppress(OSError, subprocess.SubprocessError):
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            )
            return out.stdout.strip() or None
    with suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None

"""The --memory option offers exactly the harness's memory modes (#3164)."""

from typing import get_args

from openinterpretability.cli import cli
from openinterpretability.harness.memory import MemoryMode


def test_the_memory_choices_are_the_memory_modes():
    # cli_memory spells the modes out, so that the CLI loads without torch.
    audit = cli.commands["audit"]
    (option,) = [p for p in audit.params if p.name == "memory"]
    assert tuple(option.type.choices) == get_args(MemoryMode)

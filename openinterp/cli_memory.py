"""The ``--memory`` option: load a model whole, or stream it from disk.

It is its own module so that the CLI can import it without importing torch.
"""

from __future__ import annotations

import click


def _request_memory(_ctx, _param, value: str) -> str:
    from openinterp.harness.memory import request

    request(value)
    return value


memory_option = click.option(
    "--memory",
    type=click.Choice(["auto", "normal", "stream"]),
    default="auto",
    show_default=True,
    expose_value=False,
    callback=_request_memory,
    help="stream: keep the decoder layers on disk and read them each pass, for a model "
    "larger than RAM; much slower. auto streams only when the model doesn't fit in free "
    "RAM and swap. See docs/experiments.md, Memory.",
)

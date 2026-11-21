"""``open-interpretability mcp``: the registry search as a read-only MCP tool (#3242).

The server speaks MCP over stdio, so an agent's client starts it as a subprocess. It has one
tool, ``query_registry``, which calls :func:`openinterpretability.registry_query.search`. It
reads the registry parquet and changes nothing. It needs the ``mcp`` extra:
``pip install 'open-interpretability[mcp]'``.

To add it to Claude Code: ``claude mcp add open-interpretability -- open-interpretability mcp``.
Set ``OPENINTERPRETABILITY_REGISTRY`` to search a local copy of the parquet instead.
"""

from __future__ import annotations

import click

from openinterpretability.cli import cli
from openinterpretability.registry_query import search


def query_registry(
    model: str | None = None,
    layer: int | None = None,
    kind: str | None = None,
    status: str | None = None,
    text: str | None = None,
    token: str | None = None,
    all_records: bool = False,
    limit: int = 20,
) -> list[dict]:
    """Search the Open Interpretability registry of explained units in open-weight models.

    Each result is one experiment on one unit: a neuron pole or an attention head. It has the
    model, layer and unit, the unit's standing (VERIFIED, NEEDS_REVISION, REJECTED or
    CONTESTED), the explanation and the tokens the unit fires on. Every filter is optional.

    Args:
        model: Model name, such as "openai-community/gpt2" or "Qwen/Qwen2.5-0.5B".
        layer: Layer number.
        kind: Unit kind: "mlp_neuron_pole" or "attn_head".
        status: The unit's standing, such as "VERIFIED".
        text: Text that the explanation contains, in any case.
        token: A token that the unit fires on, exactly as stored, such as " the".
        all_records: True for every experiment, not only each unit's newest.
        limit: The most results, from 1 to 500.

    Returns:
        The matching experiments.

    A limit out of range is refused before any file is read.

    >>> query_registry(limit=0)
    Traceback (most recent call last):
    ValueError: limit must be between 1 and 500, not 0
    """
    return search(None, model=model, layer=layer, kind=kind, status=status, text=text,
                  token=token, all_records=all_records, limit=limit)  # fmt: skip


def server():
    """The MCP server, with the read-only ``query_registry`` tool.

    Returns:
        An ``MCPServer`` that is not running yet.

    >>> server().name
    'open-interpretability'
    """
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations

    app = MCPServer("open-interpretability",
                    instructions="Search the Open Interpretability registry.")  # fmt: skip
    readonly = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=True)
    app.tool(annotations=readonly)(query_registry)
    return app


@cli.command("mcp")
def _mcp_command():
    """Serve the registry search as a read-only MCP tool over stdio (needs the mcp extra)."""
    try:
        app = server()
    except ModuleNotFoundError as e:
        raise click.ClickException(
            "The MCP server needs the mcp extra: pip install 'open-interpretability[mcp]'"
        ) from e
    app.run("stdio")

"""``open-interpretability model-check``: can a model join the registry? (#3370)."""

from __future__ import annotations

import click

from openinterpretability.cli import _load_runner, cli


@cli.command(name="model-check")
@click.argument("model_name")
@click.option("--device", default="cpu", show_default=True, help="cpu, mps, cuda or auto.")
@click.option("--revision", default=None, help="A Hub commit; the default branch if not given.")
def _model_check(model_name, device, revision):
    """Check that MODEL_NAME's layout, a smoke run and its config are ready.

    Run it before a model joins `[tool.openinterpretability] models`. Each line is PASS,
    WARN or FAIL. It exits 1 when any check fails. See docs/adding-a-model.md.

    \f
    Args:
        model_name: The model's Hub name.
        device: Where to load it.
        revision: The Hub commit to load.
    """
    from openinterpretability.model_check import check

    found = check(_load_runner(model_name, device, "float32", revision), model_name)
    for finding in found:
        click.echo(finding.line())
    if any(f.status == "fail" for f in found):
        raise SystemExit(1)

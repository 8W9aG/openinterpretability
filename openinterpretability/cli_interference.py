"""``open-interpretability interference``: what a model change does to its explanations (#5230).

The changed model is the base with a patch applied (``--patch``, #5229). It can also be
another model of the same architecture (``--changed``), such as an instruct model. The
report re-tests every VERIFIED pole of the base and reads every neuron's activation and
output. It also measures behaviour. Each measure runs against two noise floors
(``openinterpretability/interference``).
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from openinterpretability.cli import _load_runner, cli, wiki_dir_option, with_model_options
from openinterpretability.cli_patch import patch_options, patch_target


def _sides(base, changed: str | None, target, device: str, dtype: str):
    """The sides of the report, and the neurons that a patch edits directly."""
    from openinterpretability.interference.floors import Patched, Perturbed, Plain, random_patch
    from openinterpretability.interference.report import edited
    from openinterpretability.patches.history import replay

    sides = {"base": Plain(base, "base"), "floor:self": Plain(base, "base again")}
    if target is not None:
        ops = replay(target.patch.id, target.lookup)
        noise, root = random_patch(ops)
        sides["changed"] = Patched(base, ops, f"patch {target.patch.id[:12]}")
        sides["floor:random"] = Patched(base, noise, "random patch", root)
        return sides, edited(ops)
    other = _load_runner(changed, device, dtype)
    sides["changed"] = Plain(other, changed)
    sides["floor:random"] = Perturbed(base, other, "random change")
    return sides, set()


@cli.command(name="interference")
@click.option("--model", "model_name", required=True, help="The base model.")
@click.option("--changed", default=None, help="Another model of the same architecture.")
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True,
              help="The JSONL rows: summary, VERIFIED poles, answers, footprint.")  # fmt: skip
@click.option("--corpus", default="corpus.jsonl", show_default=True,
              type=click.Path(path_type=Path))  # fmt: skip
@click.option("--tokens", default=20_000, show_default=True, help="Corpus tokens in the sample.")
@patch_options
@wiki_dir_option()
@with_model_options
def interference_command(model_name, changed, out, corpus, tokens, patch_ref, evidence_out,
                         wiki_dir, device, dtype):  # fmt: skip
    """Report how a patch or another model changes what the registry explains of MODEL.

    Give exactly one of ``--changed`` and ``--patch``. With ``--patch`` and
    ``--evidence-out``, the summary is also written for ``patch seal --evidence keep=...``.
    """
    from openinterpretability.evidence import read_corpus
    from openinterpretability.interference import behaviour, keep, report, routes
    from openinterpretability.interference.run import Inputs, compact, run

    if (changed is None) == (patch_ref is None):
        raise click.UsageError("give exactly one of --changed and --patch")
    target = patch_target(patch_ref, evidence_out, wiki_dir, model_name) if patch_ref else None
    base = _load_runner(model_name, device, dtype)
    sides, direct = _sides(base, changed, target, device, dtype)
    wiki = Path(wiki_dir)
    given = Inputs(
        records=keep.verified_records(wiki, model_name),
        batches=keep.sample_windows(base.tokenizer, tokens, corpus),
        prompts=behaviour.explained_prompts(wiki, model_name), n_layers=base.n_layers,
        known=report.studied(wiki, model_name), direct=direct,
        answers=routes.answer_records(wiki, model_name),
        explanations=routes.explanation_records(wiki, model_name),
        texts=read_corpus(corpus)[0], means_tokens=tokens, key=model_name,
    )  # fmt: skip
    rows = compact(run(sides, given))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    if evidence_out is not None:
        Path(evidence_out).write_text(json.dumps(rows[0], indent=1) + "\n", encoding="utf-8")
    click.echo(json.dumps(rows[0]))

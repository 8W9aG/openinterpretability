"""``open-interpretability patch``: fix patches, their history and branches (issue #5229).

A patch is a commit-like record of changes to a base model (``openinterpretability/patches``).
These commands write, read and check patches in ``wiki/patches/<model>/``. A fix's own checks
run with ``--patch`` on ``answers review`` and ``explain``, and ``seal`` folds their outputs
into the patch's evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from openinterpretability.cli import _load_runner, cli, wiki_dir_option, with_model_options

model_option = click.option("--model", "model_name", required=True, help="The base model.")


def patch_options(fn):
    """``--patch`` and ``--evidence-out``: run a fix's check on the patched model (#5229)."""
    fn = click.option(
        "--evidence-out", type=click.Path(dir_okay=False, path_type=Path), default=None,
        help="--patch: write the result here, for `patch seal --evidence`.",
    )(fn)  # fmt: skip
    return click.option(
        "--patch", "patch_ref", default=None,
        help="Run on the base model with this patch applied (a stored id or a draft file). "
        "The result goes to --evidence-out, never to the registry.",
    )(fn)  # fmt: skip


def patch_target(patch_ref: str | None, evidence_out: Path | None, wiki_dir, model_name):
    """The patch that ``--patch`` names, or None. ``--patch`` needs ``--evidence-out``."""
    if patch_ref is None:
        return None
    if evidence_out is None:
        raise click.UsageError("--patch writes its result to --evidence-out; give one")
    from openinterpretability.patches.schema import PatchError
    from openinterpretability.patches.session import resolve

    try:
        return resolve(patch_ref, Path(wiki_dir), model_name)
    except PatchError as error:
        raise click.UsageError(str(error)) from error


@cli.group(name="patch")
def patch_group() -> None:
    """Fix patches: write, check, branch, merge, revert and rebase them (#5229)."""


def _store(wiki_dir: Path, model_name: str):
    from openinterpretability.patches.store import PatchStore

    return PatchStore(wiki_dir, f"{model_name}@any")


def _fail(error: Exception) -> click.ClickException:
    return click.ClickException(str(error))


@patch_group.command(name="seal")
@click.argument("draft", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--evidence", "evidence", multiple=True, metavar="NAME=FILE",
              help="A check's output, kept under NAME in the evidence (repeatable).")  # fmt: skip
@click.option("--verdict", type=click.Choice(["fixed", "not fixed"]), default=None)
@wiki_dir_option()
def seal_command(draft: Path, evidence, verdict, wiki_dir) -> None:
    """Seal DRAFT with its evidence, write it to the store, sign it and print its id."""
    from openinterpretability.patches.schema import Patch, PatchError
    from openinterpretability.patches.store import PatchStore

    patch = Patch.model_validate_json(draft.read_text(encoding="utf-8"))
    found = dict(patch.evidence)
    for item in evidence:
        name, _, file = item.partition("=")
        if not name or not file:
            raise click.UsageError(f"--evidence takes NAME=FILE, not {item!r}")
        found[name] = json.loads(Path(file).read_text(encoding="utf-8"))
    if verdict:
        found["verdict"] = verdict
    sealed = patch.model_copy(update={"id": "", "evidence": found}).sealed()
    try:
        path = PatchStore(wiki_dir, sealed.base).write(sealed)
    except PatchError as error:
        raise _fail(error) from error
    click.echo(f"{sealed.id} → {path}")


@patch_group.command(name="show")
@click.argument("ref")
@model_option
@wiki_dir_option()
def show_command(ref: str, model_name: str, wiki_dir: Path) -> None:
    """Print the patch REF (an id or a unique prefix) as JSON."""
    click.echo(_store(wiki_dir, model_name).get(ref).model_dump_json(indent=2))


@patch_group.command(name="log")
@click.argument("ref")
@model_option
@wiki_dir_option()
def log_command(ref: str, model_name: str, wiki_dir: Path) -> None:
    """List REF and its ancestors, newest first."""
    from openinterpretability.patches.history import log

    store = _store(wiki_dir, model_name)
    for patch in log(store.get(ref).id, store.get):
        what = patch.message.get("failure") or patch.message.get("text") or ""
        click.echo(f"{patch.id[:12]}  {patch.verdict or '-':9}  {len(patch.ops)} ops  {what}")


@patch_group.command(name="diff")
@click.argument("a")
@click.argument("b")
@model_option
@wiki_dir_option()
def diff_command(a: str, b: str, model_name: str, wiki_dir: Path) -> None:
    """The ops that only A applies, only B applies, and both with other parameters."""
    from openinterpretability.patches.history import diff

    store = _store(wiki_dir, model_name)
    click.echo(json.dumps(diff(store.get(a).id, store.get(b).id, store.get), indent=1))


@patch_group.command(name="branches")
@model_option
@wiki_dir_option()
def branches_command(model_name: str, wiki_dir: Path) -> None:
    """List the base model's branches and the patches they point at."""
    for name, head in _store(wiki_dir, model_name).branches().items():
        click.echo(f"{name}\t{head[:12]}")


@patch_group.command(name="branch")
@click.argument("name")
@click.argument("ref")
@click.option("--create", is_flag=True, help="Create the branch; without it, move it.")
@model_option
@wiki_dir_option()
def branch_command(name: str, ref: str, create: bool, model_name: str, wiki_dir: Path) -> None:
    """Point branch NAME at REF. main only moves to a patch whose verdict is fixed."""
    from openinterpretability.patches.schema import PatchError

    store = _store(wiki_dir, model_name)
    try:
        store.move(name, store.get(ref).id, create=create)
    except PatchError as error:
        raise _fail(error) from error


@patch_group.command(name="delete-branch")
@click.argument("name")
@model_option
@wiki_dir_option()
def delete_branch_command(name: str, model_name: str, wiki_dir: Path) -> None:
    """Delete branch NAME. Its patches stay."""
    _store(wiki_dir, model_name).delete(name)


def _write(patch, wiki_dir: Path) -> None:
    from openinterpretability.patches.store import PatchStore

    path = PatchStore(wiki_dir, patch.base).write(patch)
    click.echo(f"{patch.id} → {path}")


@patch_group.command(name="revert")
@click.argument("ref")
@click.argument("ops", nargs=-1, required=True)
@click.option("--message", required=True, help="Why the ops are reverted.")
@model_option
@wiki_dir_option()
def revert_command(ref, ops, message, model_name, wiki_dir) -> None:
    """A patch on REF that drops OPS (each <patch-id>:<op-id>). It needs its own evidence."""
    from openinterpretability.patches.history import revert
    from openinterpretability.patches.schema import PatchError

    store = _store(wiki_dir, model_name)
    try:
        _write(revert(store.get(ref).id, ops, {"text": message}, store.get), wiki_dir)
    except PatchError as error:
        raise _fail(error) from error


@patch_group.command(name="merge")
@click.argument("refs", nargs=-1, required=True)
@click.option("--resolution", type=click.Choice(["additive"]), default=None,
              help="How parameter conflicts are resolved.")  # fmt: skip
@click.option(
    "--order",
    "order",
    multiple=True,
    help="Ops (<patch-id>:<op-id>) in the order they apply at a shared hook point.",
)
@click.option("--message", required=True)
@model_option
@wiki_dir_option()
def merge_command(refs, resolution, order, message, model_name, wiki_dir) -> None:
    """Merge REFS. Conflicts block it until --resolution or --order resolves them."""
    from openinterpretability.patches.history import merge
    from openinterpretability.patches.schema import Op, PatchError

    store = _store(wiki_dir, model_name)
    ops = [Op(op_id="order", type="order", targets=list(order))] if order else []
    text = {"text": message} | ({"resolution": resolution} if resolution else {})
    try:
        _write(merge([store.get(r).id for r in refs], ops, text, store.get), wiki_dir)
    except PatchError as error:
        raise _fail(error) from error


@patch_group.command(name="rebase")
@click.argument("refs", nargs=-1, required=True)
@click.option("--onto", default=None, help="The new parent; by default the base itself.")
@click.option("--base", "new_base", required=True, help="The new base, <model>@<revision>.")
@model_option
@wiki_dir_option()
def rebase_command(refs, onto, new_base, model_name, wiki_dir) -> None:
    """Copy REFS (oldest first) onto a new parent or base. Their evidence must be run again."""
    from openinterpretability.patches.history import rebase

    store = _store(wiki_dir, model_name)
    parent = store.get(onto).id if onto else None
    for patch in rebase([store.get(r).id for r in refs], parent, new_base, store.get):
        _write(patch, wiki_dir)


@patch_group.command(name="check")
@click.argument("ref")
@model_option
@click.option("--corpus", default="corpus.jsonl", show_default=True,
              type=click.Path(path_type=Path))  # fmt: skip
@click.option("--texts", default=8, show_default=True, help="Corpus texts in the sample.")
@click.option("--strength", default=0.0, show_default=True,
              help="0: every output scale at zero. 1: ops that start as the identity.")  # fmt: skip
@wiki_dir_option()
@with_model_options
def check_command(ref, model_name, corpus, texts, strength, wiki_dir, device, dtype) -> None:
    """The zero-start check: REF at --strength must keep the base model's logits."""
    import torch

    from openinterpretability.evidence import read_corpus
    from openinterpretability.patches.apply import ZeroStartError, zero_start_check
    from openinterpretability.patches.history import replay
    from openinterpretability.patches.session import resolve

    target = resolve(ref, wiki_dir, model_name)
    runner = _load_runner(model_name, device, dtype)
    sample = read_corpus(corpus)[0][:texts]
    worst = 0.0
    for text in sample:
        ids = torch.tensor([runner.tokenizer.encode(text)[:128]], device=runner.device)
        ops = replay(target.patch.id, target.lookup)
        try:
            worst = max(worst, zero_start_check(runner.model, ops, ids, strength))
        except ZeroStartError as error:
            raise _fail(error) from error
    click.echo(f"zero-start check passed on {len(sample)} texts: largest |Δ logit| {worst:.3g}")

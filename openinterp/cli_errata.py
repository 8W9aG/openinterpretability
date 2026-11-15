"""``open-interpretability erratum draft``: write an erratum's line from a rule (issue #3271).

A maintainer gives the bug, the fix commit and the records it covers: a rule, listed
records, or both. The command lists every record that the erratum covers, so that the
maintainer checks a list and not a hand-written JSON line. It checks the draft as
``open-interpretability check`` will check the file (:mod:`openinterp.errata_checks`).
Then it prints the line. ``--write`` appends the line to ``wiki/errata.jsonl``. Signing
stays a separate step, ``open-interpretability sign wiki/errata.jsonl``, after the
maintainer reads the list.
"""

from __future__ import annotations

import json
import subprocess
from datetime import date as Date
from pathlib import Path

import click

from openinterp.cli import cli, wiki_dir_option

KINDS = ("mlp_neuron_pole", "attn_head", "dictionary_feature")


@cli.group()
def erratum() -> None:
    """Draft errata for signed records (docs/PROTOCOL.md §5)."""


def _edit(text: str) -> dict:
    """``PATH=JSON`` as an edit; a value that isn't JSON is a string."""
    path, sep, value = text.partition("=")
    if not sep:
        raise click.BadParameter(f"{text!r} is not PATH=VALUE", param_hint="--edit")
    try:
        return {"path": path, "value": json.loads(value)}
    except json.JSONDecodeError:
        return {"path": path, "value": value}


def _rule(**fields) -> dict | None:
    """The rule's fields that were given, or ``None`` when none was."""
    rule = {k: (list(v) if isinstance(v, tuple) else v) for k, v in fields.items()}
    rule = {k: v for k, v in rule.items() if v not in (None, [])}
    return rule or None


def resolve_commit(ref: str) -> str | None:
    """The full hash of the commit that ``ref`` names here, or ``None`` if there is none."""
    done = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    return done.stdout.strip() if done.returncode == 0 else None


def draft_erratum(
    wiki_dir: Path, fields: dict, resolve=resolve_commit
) -> tuple[object, list[tuple[str, object]]]:
    """The erratum that ``fields`` describe, and the records it covers as ``(ref, record)``.

    Raises ``click.ClickException`` for each way ``open-interpretability check`` would
    refuse it. These are an id already used, a fix commit that doesn't exist, and a listed
    record that isn't in the archive. Also no record covered, or an edit that doesn't apply.
    ``fields["fixed_in"]`` can be any ref, such as a branch or ``HEAD``. The erratum holds
    the hash of its commit.
    """
    from pydantic import ValidationError

    from openinterp.builder.wiki import load_archive_paths
    from openinterp.core.errata import Erratum, apply_errata, load_errata, record_ref

    commit = resolve(fields["fixed_in"])
    if commit is None:
        raise click.ClickException(f"--fixed-in {fields['fixed_in']} is not a commit here")
    try:
        draft = Erratum.model_validate_json(json.dumps(fields | {"fixed_in": commit}))
    except ValidationError as exc:
        raise click.ClickException(f"the erratum is not valid: {exc}") from exc
    if draft.id in {e.id for e in load_errata(wiki_dir)}:
        raise click.ClickException(f"erratum id {draft.id!r} is already in the errata file")
    archive = {record_ref(r): r for _, r in load_archive_paths(wiki_dir)}
    missing = [ref for ref in draft.records if ref not in archive]
    if missing:
        raise click.ClickException(f"listed records not in the archive: {', '.join(missing)}")
    covered = [(ref, r) for ref, r in sorted(archive.items()) if draft.covers(r)]
    if not covered:
        raise click.ClickException("the erratum covers no record")
    for ref, record in covered:
        try:
            apply_errata(record, [draft])
        except ValueError as exc:
            raise click.ClickException(f"{ref}: {exc}") from exc
    return draft, covered


@erratum.command("draft")
@wiki_dir_option()
@click.option("--id", "erratum_id", required=True, help="Unique id: lower case and hyphens.")
@click.option("--title", required=True, help="One line for the banner.")
@click.option("--bug", required=True, help="What was wrong, and what it did to records.")
@click.option("--fixed-in", required=True, help="The commit that fixed the bug.")
@click.option("--date", "day", default=None, help="The erratum's date (default today).")
@click.option("--record", "records", multiple=True, help="A record path in wiki/experiments/.")
@click.option("--model", "models", multiple=True, help="Rule: a model name.")
@click.option("--kind", "kinds", multiple=True, type=click.Choice(KINDS), help="Rule: unit kind.")
@click.option("--protocol-version", "protocol_versions", multiple=True, help="Rule: version.")
@click.option("--harness-version", "harness_versions", multiple=True, help="Rule: version.")
@click.option("--reviewed-from", help="Rule: first verdict time that matches (ISO, with zone).")
@click.option("--reviewed-before", help="Rule: verdict time from which records don't match.")
@click.option("--edit", "edits", multiple=True, help="PATH=JSON: a corrected field.")
@click.option("--set-aside", is_flag=True, help="The records count toward no status.")
@click.option("--write", is_flag=True, help="Append the line to wiki/errata.jsonl.")
def draft(wiki_dir, erratum_id, title, bug, fixed_in, day, records, edits, set_aside, write,
          **rule):  # fmt: skip
    """List the records that an erratum covers, check it, and print its line.

    Give a rule (--model, --kind, --protocol-version, --harness-version, --reviewed-from,
    --reviewed-before), --record paths, or both. Give --edit corrections or --set-aside.
    Read the list, then sign the file with `open-interpretability sign wiki/errata.jsonl`.
    """
    from openinterp.core.errata import errata_path

    fields = {
        "id": erratum_id, "date": day or Date.today().isoformat(), "title": title, "bug": bug,
        "fixed_in": fixed_in, "edits": [_edit(e) for e in edits], "set_aside": set_aside,
        "records": list(records), "rule": _rule(**rule),
    }  # fmt: skip
    found, covered = draft_erratum(wiki_dir, fields)
    for ref, record in covered:
        verdict = record.verdict
        status, when = (verdict.status, verdict.timestamp.isoformat()) if verdict else ("-", "-")
        click.echo(f"  {ref}  {status}  {when}")
    click.echo(f"The erratum covers {len(covered)} records.")
    line = found.model_dump_json(exclude_defaults=True)
    click.echo(line)
    if write:
        path = errata_path(wiki_dir)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        click.echo(f"Appended to {path}. Read the list, then: open-interpretability sign {path}")

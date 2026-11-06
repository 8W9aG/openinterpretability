"""Commands that vouch for records: ``reproduce`` and ``sign``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import click

from openinterp.cli import cli, wiki_dir_option
from openinterp.core.paths import experiments_dir


@cli.command()
@click.argument("records", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--base-ref",
    help="Reproduce every record added or changed since this git ref (e.g. origin/main).",
)
@click.option(
    "--sample",
    type=click.IntRange(min=1),
    help="Reproduce this many records of the registry, chosen in rotation by --date. Implies "
    "--include-signed, and compares only the measurements of older protocols' records.",
)
@click.option(
    "--date",
    "sample_date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="The day that selects the --sample records (default: today, UTC).",
)
@click.option(
    "--include-signed",
    is_flag=True,
    help="Reproduce signed records too. A signature that doesn't verify still fails.",
)
@wiki_dir_option()
@click.option(
    "--atol",
    default=1e-3,
    show_default=True,
    type=click.FloatRange(min=0),
    help="Absolute tolerance, as a fraction of the neuron's max activation.",
)
@click.option("--rtol", default=1e-2, show_default=True, type=click.FloatRange(min=0))
@click.option(
    "--device",
    default="auto",
    show_default=True,
    help="cpu, cuda, mps or auto (CUDA, else MPS, else CPU).",
)
@click.option(
    "--corpus",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The scans' corpus: re-draw each record's corpus sample from it and check it "
    "holds the top examples. Records scanned on another corpus get a warning.",
)
@click.option(
    "--allow-model",
    "allowed_models",
    multiple=True,
    help="Only load these Hugging Face models (repeatable); records naming any other model "
    "fail without a download. Default: any model.",
)
@click.option(
    "--allowed-signers",
    default=Path(".github/allowed_signers"),
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Records signed by a key listed here skip reproduction. With --base-ref the file "
    "is read from that ref, so a branch can't add its own key.",
)
def reproduce(
    records,
    base_ref,
    sample,
    sample_date,
    include_signed,
    wiki_dir,
    atol,
    rtol,
    device,
    corpus,
    allowed_models,
    allowed_signers,
):
    """Re-run experiment records against their pinned model and compare (no LLM calls).

    Each record's scan evidence, counterfactual deltas and review are measured again
    with the model revision and dtype it pins, under its own review policy. Exits 1 if
    any number differs beyond tolerance or any pass/fail, gate or status differs.
    """
    import os

    from openinterp.cli_evidence import read_corpus
    from openinterp.reproduce import Tolerance

    paths = _chosen_records(records, base_ref, sample, sample_date, wiki_dir)
    if not paths:
        click.echo("No records to reproduce.")
        return
    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    scan_corpus = None
    if corpus is not None:
        texts, _, corpus_sha, _ = read_corpus(corpus)
        scan_corpus = (texts, corpus_sha)
    reproducer = _Reproducer(
        signers=_allowed_signers(allowed_signers, base_ref),
        tolerance=Tolerance(atol=atol, rtol=rtol),
        scan_corpus=scan_corpus,
        allowed_models=allowed_models,
        device=device,
        include_signed=include_signed or sample is not None,
        measurements_only=sample is not None,
    )
    results: list[tuple[Path, list[str], list[str]]] = []
    for path in paths:
        status, problems, warnings = reproducer.check(path)
        results.append((path, problems, warnings))
        _report(path, status, problems, warnings, in_actions)
    failed = sum(bool(problems) for _, problems, _ in results)
    summary = f"{len(paths) - failed}/{len(paths)} record(s) reproduced."
    click.echo(summary)
    if in_actions and os.environ.get("GITHUB_STEP_SUMMARY"):
        _step_summary(os.environ["GITHUB_STEP_SUMMARY"], summary, results)
    if failed:
        raise SystemExit(1)


def _chosen_records(records, base_ref, sample, sample_date, wiki_dir) -> list[Path]:
    """Return the named records, then the branch's changes, then the day's sample."""
    from datetime import UTC, datetime

    from openinterp.checks import changed_records
    from openinterp.reproduce import rotating_sample

    paths = list(records)
    if base_ref:
        paths += [p for p in changed_records(wiki_dir, base_ref) if p not in paths]
    if sample:
        day = (sample_date or datetime.now(UTC)).date()
        registry = sorted(experiments_dir(wiki_dir).rglob("*.json"))
        paths += [p for p in rotating_sample(registry, sample, day) if p not in paths]
    return paths


@dataclass
class _Reproducer:
    """Checks one record at a time, loading each pinned model once."""

    signers: object
    tolerance: object
    scan_corpus: tuple[list[str], str] | None
    allowed_models: tuple[str, ...]
    device: str
    include_signed: bool = False
    measurements_only: bool = False  # for records from another protocol version
    runners: dict[tuple, object] = field(default_factory=dict)

    def check(self, path: Path) -> tuple[str, list[str], list[str]]:
        """(status, mismatches, warnings) for the record at ``path``."""
        from pydantic import ValidationError

        from openinterp.core.schemas import AuditRecord
        from openinterp.reproduce import reproduce_record
        from openinterp.signing import SignatureError, verify_record

        try:
            signer = verify_record(path, self.signers)
        except SignatureError as exc:
            return "FAILED", [str(exc)], []
        if signer and not self.include_signed:
            return f"SIGNED by {signer} (reproduction skipped)", [], []
        try:
            record = AuditRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            return "FAILED", ["invalid experiment record"], []
        reason = self.refusal(record)
        if reason:
            return "FAILED", [reason], []
        runner = self.runner(record)
        report = reproduce_record(
            record,
            runner,
            self.tolerance,
            path,
            self.scan_corpus,
            measurements_only=self.measurements_only,
        )
        return ("FAILED" if report.mismatches else "REPRODUCED"), report.mismatches, report.warnings

    def refusal(self, record) -> str | None:
        """Why ``record`` can't be reproduced here, or ``None``."""
        from openinterp.reproduce import check_protocol

        model, allowed = record.submission.identifier.model_name, self.allowed_models
        reason = None if self.measurements_only else check_protocol(record)
        if allowed and model not in allowed:
            reason = f"model {model} is not on the allow-list ({', '.join(allowed)})"
        elif allowed and not _preset_dictionary(record):
            reason = "its dictionary is not a preset's (openinterp/harness/dictionaries.py)"
        return reason

    def runner(self, record):
        from openinterp.cli import _load_runner
        from openinterp.harness.units import attn_implementation
        from openinterp.reproduce import pinned_environment

        model = record.submission.identifier.model_name
        revision, dtype = pinned_environment(record)
        attn = attn_implementation(record.submission.identifier)
        key = (model, revision, dtype, attn)
        if key not in self.runners:
            self.runners[key] = _load_runner(model, self.device, dtype, revision, attn)
        return self.runners[key]


def _report(path: Path, status: str, problems, warnings, in_actions: bool) -> None:
    """One record's result, as GitHub Actions annotations when running there."""
    click.echo(f"{status}: {path}")
    for message in problems:
        click.echo(f"::error file={path}::{message}" if in_actions else f"  mismatch: {message}")
    for message in warnings:
        click.echo(f"::warning file={path}::{message}" if in_actions else f"  warning: {message}")


def _step_summary(path: str, summary: str, results) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"### Reproduction\n\n{summary}\n\n")
        for record, problems, _ in results:
            fh.write(f"- {'❌' if problems else '✅'} `{record}`\n")
            fh.writelines(f"  - {message}\n" for message in problems)


def _preset_dictionary(record) -> bool:
    """Check that a feature record's dictionary is on the allow-list.

    A feature record pins the weights of its dictionary's preset. So the allow-list also
    limits which dictionaries are downloaded. Other records have none to check.
    """
    from openinterp.harness.dictionaries import DICTIONARIES

    ident, profile = record.submission.identifier, record.submission.dictionary_profile
    if ident.kind != "dictionary_feature" or profile is None:
        return True
    spec = DICTIONARIES.get(ident.dictionary)
    return spec is not None and (spec.repo, spec.revision) == (profile.repo, profile.revision)


def _allowed_signers(path: Path, base_ref: str | None) -> str:
    """The allowed-signers file's content: from ``base_ref`` if given, else from disk."""
    import subprocess

    if base_ref:
        shown = subprocess.run(
            ["git", "show", f"{base_ref}:{path.as_posix()}"], capture_output=True, text=True
        )
        return shown.stdout if shown.returncode == 0 else ""
    return path.read_text(encoding="utf-8") if path.exists() else ""


@cli.command()
@click.argument(
    "records", nargs=-1, required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--key",
    default=Path("~/.ssh/openinterp_signing"),
    show_default=True,
    envvar="OPENINTERP_SIGNING_KEY",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Private SSH key to sign with (or $OPENINTERP_SIGNING_KEY); its public key must be "
    "in --allowed-signers. Use a key kept for signing, not your login key.",
)
@click.option(
    "--allowed-signers",
    default=Path(".github/allowed_signers"),
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
def sign(records, key, allowed_signers):
    """Vouch for experiment records with your SSH key, so CI skips reproducing them.

    Writes <id>.json.sig next to each record; the record itself is not changed. The
    verdict still comes from the review protocol: a signature only says you vouch that
    the record is what it claims to be. Commit the .sig files with the records.
    """
    from openinterp.signing import SignatureError, sign_record, verify_record

    signers = _allowed_signers(allowed_signers, None)
    for path in records:
        sig = sign_record(path, key.expanduser())
        try:
            signer = verify_record(path, signers)
        except SignatureError:
            click.echo(
                f"Wrote {sig}, but {key}.pub is not in {allowed_signers}, so CI will reject "
                "it. Add it with:\n"
                f'  echo "<your-name> namespaces=\\"openinterp-record\\" $(cat {key}.pub)" '
                f">> {allowed_signers}",
                err=True,
            )
            raise SystemExit(1) from None
        click.echo(f"Signed {path} as {signer}: {sig}")

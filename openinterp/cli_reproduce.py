"""Commands that vouch for records: ``reproduce``, ``sign`` and ``signatures``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import click

from openinterp.cli import cli, wiki_dir_option
from openinterp.cli_memory import memory_option
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
    "--time-budget",
    type=click.FloatRange(min=0),
    help="Minutes after which no new record is started. The records not reached are listed "
    "and don't fail the run. Leave room for the slowest record within any outer time limit.",
)
@click.option(
    "--include-signed",
    is_flag=True,
    help="Reproduce signed records too. A signature that doesn't verify still fails. A "
    "signature that landed outside its key's validity window is reproduced anyway.",
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
@memory_option
@click.option(
    "--device",
    default="auto",
    show_default=True,
    help="cpu, cuda, mps or auto (CUDA, else MPS, else CPU).",
)
@click.option(
    "--corpus",
    "corpora",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="A scan corpus (repeatable): re-draw the corpus sample of each record that "
    "scanned it, matched by SHA-256, and check it holds the top examples. The built-in "
    "corpus is always known. Other records get a warning.",
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
    time_budget,
    include_signed,
    wiki_dir,
    atol,
    rtol,
    device,
    corpora,
    allowed_models,
    allowed_signers,
):
    """Re-run experiment records against their pinned model and compare (no LLM calls).

    Each record's scan evidence, counterfactual deltas and review are measured again
    with the model revision and dtype it pins, under its own review policy. Exits 1 if
    any number differs beyond tolerance or any pass/fail, gate or status differs.
    """
    import os

    from openinterp.core.errata import load_errata
    from openinterp.reproduce import Tolerance
    from openinterp.signing import allowed_signers_text, landing_times

    paths = _chosen_records(records, base_ref, sample, sample_date, wiki_dir)
    if not paths:
        click.echo("No records to reproduce.")
        return
    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    reproducer = _Reproducer(
        signers=allowed_signers_text(allowed_signers, base_ref),
        landed=landing_times(base_ref or "HEAD"),
        tolerance=Tolerance(atol=atol, rtol=rtol),
        scan_corpora=scan_corpora(corpora),
        corpus_given=bool(corpora),
        allowed_models=allowed_models,
        device=device,
        include_signed=include_signed or sample is not None,
        measurements_only=sample is not None,
        errata=load_errata(wiki_dir),
    )
    results, unreached = _check_all(reproducer, paths, in_actions, time_budget)
    failed = sum(bool(problems) for _, problems, _ in results)
    summary = f"{len(results) - failed}/{len(results)} record(s) reproduced."
    if unreached:
        summary += f" {len(unreached)} not reached in the {time_budget:g}-minute time budget."
    click.echo(summary)
    if in_actions and os.environ.get("GITHUB_STEP_SUMMARY"):
        _step_summary(os.environ["GITHUB_STEP_SUMMARY"], summary, results, unreached)
    if failed:
        raise SystemExit(1)


def _check_all(reproducer, paths, in_actions: bool, time_budget: float | None):
    """Check each record in turn. Return the results, and the records not started.

    No record starts once ``time_budget`` minutes have passed.
    """
    import time

    start = time.monotonic()
    results: list[tuple[Path, list[str], list[str]]] = []
    for n, path in enumerate(paths):
        if time_budget is not None and time.monotonic() - start >= time_budget * 60:
            for later in paths[n:]:
                click.echo(f"NOT REACHED: {later}")
            return results, paths[n:]
        status, problems, warnings = reproducer.check(path)
        results.append((path, problems, warnings))
        _report(path, status, problems, warnings, in_actions)
    return results, []


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
    landed: dict  # record path -> when its signature landed (signing.landing_times)
    tolerance: object
    scan_corpora: dict[str, list[str]]  # documents by the SHA-256 of their corpus file
    allowed_models: tuple[str, ...]
    device: str
    corpus_given: bool = False  # --corpus given: warn for records it doesn't cover
    include_signed: bool = False
    measurements_only: bool = False  # for records from another protocol version
    errata: list = field(default_factory=list)  # wiki/errata.jsonl (core/errata.py)
    runners: dict[tuple, object] = field(default_factory=dict)

    def check(self, path: Path) -> tuple[str, list[str], list[str]]:
        """(status, mismatches, warnings) for the record at ``path``."""
        from pydantic import ValidationError

        from openinterp.core.schemas import AuditRecord
        from openinterp.reproduce import reproduce_record
        from openinterp.signing import SignatureError, SignatureOutsideWindow, verify_record

        notes = []
        try:
            signer = verify_record(path, self.signers, self.landed.get(path.resolve()))
        except SignatureOutsideWindow as exc:
            signer, notes = None, [f"{exc}, so it is reproduced"]
        except SignatureError as exc:
            return "FAILED", [str(exc)], []
        if signer and not self.include_signed:
            return f"SIGNED by {signer} (reproduction skipped)", [], []
        try:
            record = AuditRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            return "FAILED", ["invalid experiment record"], []
        record, aside = self.corrected(record, notes)
        if aside:
            return f"SET ASIDE by erratum {aside} (reproduction skipped)", [], notes
        reason = self.refusal(record)
        if reason:
            return "FAILED", [reason], notes
        runner = self.runner(record)
        report = reproduce_record(
            record,
            runner,
            self.tolerance,
            path,
            self.scan_corpus(record, path, notes),
            measurements_only=self.measurements_only,
        )
        status = "FAILED" if report.mismatches else "REPRODUCED"
        return status, report.mismatches, notes + report.warnings

    def scan_corpus(self, record, path: Path, notes: list[str]):
        """(documents, SHA-256) of the corpus that ``record`` scanned, if it is known here."""
        prov = record.submission.provenance
        if prov is None or prov.corpus_sample is None:
            return None
        if prov.corpus_sha256 in self.scan_corpora:
            return self.scan_corpora[prov.corpus_sha256], prov.corpus_sha256
        if self.corpus_given:
            notes.append(
                "no corpus given is the scan's, so its sample was not checked. "
                + _rebuild_hint(record, path)
            )
        return None

    def corrected(self, record, notes: list[str]):
        """``record`` with its errata's edits, and the id of an erratum that sets it aside.

        A harness fix can change what a record reproduces to. Its erratum's edits give the
        corrected values, so those are the values to check.
        """
        from openinterp.core.errata import apply_errata, covering

        found = covering(self.errata, record)
        aside = next((x.id for x in found if x.set_aside), None)
        if aside or not found:
            return record, aside
        notes.append(f"checked with the edits of erratum {', '.join(x.id for x in found)}")
        return apply_errata(record, found)[0], None

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
        from openinterp.half_precision import reproduction_dtype
        from openinterp.harness.units import attn_implementation
        from openinterp.reproduce import pinned_environment

        model = record.submission.identifier.model_name
        revision, pinned = pinned_environment(record)
        dtype = reproduction_dtype(pinned)
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


def _step_summary(path: str, summary: str, results, unreached=()) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"### Reproduction\n\n{summary}\n\n")
        for record, problems, _ in results:
            fh.write(f"- {'❌' if problems else '✅'} `{record}`\n")
            fh.writelines(f"  - {message}\n" for message in problems)
        fh.writelines(f"- ⏭️ `{record}` (not reached)\n" for record in unreached)


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
    from openinterp.signing import (
        SignatureError,
        SignatureOutsideWindow,
        allowed_signers_text,
        sign_record,
        verify_record,
    )

    signers = allowed_signers_text(allowed_signers)
    for path in records:
        sig = sign_record(path, key.expanduser())
        try:
            signer = verify_record(path, signers)
        except SignatureOutsideWindow as exc:
            sig.unlink()
            click.echo(f"Did not sign {path}: {exc}.", err=True)
            raise SystemExit(1) from None
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


@cli.command()
@click.argument("records", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--ref",
    default="HEAD",
    show_default=True,
    help="Check each signature at the time it landed on this ref's first-parent history.",
)
@click.option(
    "--allowed-signers",
    default=Path(".github/allowed_signers"),
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--paths-only", is_flag=True, help="Print only the paths, one per line.")
@wiki_dir_option()
def signatures(records, ref, allowed_signers, paths_only, wiki_dir):
    """List signed records whose signature CI must not trust (default: every record).

    Each signature is checked at the time it landed on --ref, against the validity
    window of its key in --allowed-signers. After you revoke a key with valid-before
    (SECURITY.md), use it to find the records signed in its exposure window. Then
    reproduce them: open-interpretability reproduce --include-signed \\
    $(open-interpretability signatures --paths-only).
    """
    from openinterp.signing import (
        SignatureError,
        allowed_signers_text,
        landing_times,
        signature_path,
        verify_record,
    )

    paths = list(records) or sorted(experiments_dir(wiki_dir).rglob("*.json"))
    paths = [p for p in paths if signature_path(p).exists()]
    signers, landed = allowed_signers_text(allowed_signers), landing_times(ref)
    untrusted = 0
    for path in paths:
        at = landed.get(path.resolve())
        try:
            verify_record(path, signers, at)
        except SignatureError as exc:
            untrusted += 1
            when = "not landed" if at is None else f"landed {at:%Y-%m-%d %H:%M}Z"
            click.echo(path if paths_only else f"{path} ({when}): {exc}")
    if not paths_only:
        click.echo(f"{untrusted} of {len(paths)} signed record(s) not trusted.")


def scan_corpora(paths) -> dict[str, list[str]]:
    """Documents by the SHA-256 of their corpus file: the built-in corpus and ``paths``."""
    from openinterp.cli_evidence import read_corpus

    corpora = {}
    for path in (None, *paths):
        texts, _, sha, _ = read_corpus(path)
        corpora[sha] = texts
    return corpora


def _rebuild_hint(record, path: Path) -> str:
    """How to get the corpus that ``record`` scanned, from its source or the committed table."""
    from openinterp.corpus_rebuild import scanned_source

    try:
        scanned_source(record.submission)
    except ValueError as error:
        return f"Its corpus can't be rebuilt: {error}."
    return f"Rebuild it with `open-interpretability corpus --same-as {path} --out <file>`."

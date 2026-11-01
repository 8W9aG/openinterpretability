"""Registry-maintenance commands: ``wiki``, ``check``, ``stats`` and ``next``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import click

from openinterp.cli import cli, wiki_dir_option
from openinterp.core.paths import experiments_dir
from openinterp.planner import DEFAULT_EPSILON


@cli.command()
@click.option(
    "--import-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Import loose audit records (*.json) found here as experiments.",
)
@wiki_dir_option()
def wiki(import_dir, wiki_dir):
    """Rebuild the wiki indexes, importing loose records as experiments if asked."""
    from openinterp.builder.compiler import load_records
    from openinterp.builder.wiki import backfill

    written = backfill(load_records(import_dir) if import_dir else [], wiki_dir)
    for page in written:
        click.echo(f"Wiki: {page}")
    click.echo(f"{len(written)} new experiments; indexes rebuilt in {wiki_dir}")


@cli.command()
@wiki_dir_option()
@click.option(
    "--base-ref",
    help="Git ref to diff against (e.g. origin/main) to enforce write-once wiki pages.",
)
def check(wiki_dir, base_ref):
    """Check that every posted experiment has a valid record and wiki page (CI gate)."""
    import os

    from openinterp.checks import check_registry

    report = check_registry(wiki_dir, base_ref=base_ref)
    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    for problem in report.problems:
        if in_actions:
            click.echo(f"::error file={problem.path}::{problem.message}")
        else:
            click.echo(f"error: {problem}", err=True)

    summary = (
        f"Checked {report.records} experiment records and {report.pages} experiment pages"
        + (f" and {report.draws} draw records" if report.draws else "")
        + ": "
        + ("all consistent." if report.ok else f"{len(report.problems)} problem(s).")
    )
    click.echo(summary)
    if in_actions and os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(f"### Experiment check\n\n{summary}\n\n")
            fh.writelines(f"- `{p.path}`: {p.message}\n" for p in report.problems)
    if not report.ok:
        raise SystemExit(1)


def registry_stats(records) -> str:
    """State of the registry: experiments, units, models, VERIFIED and self-reviewed shares,
    and the protocol versions their verdicts were computed under."""
    from collections import Counter

    def share(n: int, of: int) -> str:
        return f"{n}/{of} ({100 * n / of:.0f}%)" if of else "0/0"

    def summary(rs) -> str:
        units = {r.submission.identifier.slug for r in rs}
        reviewed = [r for r in rs if r.verdict is not None]
        verified = sum(r.is_verified for r in rs)
        self_reviewed = sum(r.self_reviewed for r in reviewed)
        return (
            f"{len(rs)} experiments on {len(units)} units; VERIFIED {share(verified, len(rs))}, "
            f"self-reviewed {share(self_reviewed, len(reviewed))}"
        )

    by_model: dict[str, list] = {}
    for r in records:
        by_model.setdefault(r.submission.identifier.model_name, []).append(r)
    versions = Counter(
        (r.verdict.policy.protocol_version if r.verdict.policy else "< 0.2")
        for r in records
        if r.verdict is not None
    )
    lines = [f"Registry: {summary(records)}, {len(by_model)} models"]
    lines += [f"  {model}: {summary(rs)}" for model, rs in sorted(by_model.items())]
    lines.append(
        "Protocol versions: "
        + (", ".join(f"{v} ×{n}" for v, n in sorted(versions.items(), key=_newest)) or "none")
    )
    return "\n".join(lines)


def _newest(item) -> tuple:
    version = item[0]
    return (1,) if version == "< 0.2" else (0, *(-int(p) for p in version.split(".")))


@cli.command()
@wiki_dir_option()
def stats(wiki_dir):
    """Print the state of the registry: experiments, units, models, VERIFIED share,
    self-reviewed share and the protocol versions in use."""
    from openinterp.builder.compiler import load_records

    click.echo(registry_stats(load_records(experiments_dir(wiki_dir))))


@cli.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--full", is_flag=True, help="Don't cut probes, examples or the Discussion.")
def show(path, full):
    """Summarise a record, submission, scan evidence (ev.json) or a unit's directory.

    Read this instead of opening the JSON: verdict, failed gates and probes, scores and
    each page's Discussion, or a scan's corpus, peak and top examples."""
    from openinterp.show import show as summarise

    try:
        click.echo(summarise(path, full))
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="PATH") from exc


@cli.command(name="next")
@wiki_dir_option()
@click.option("--seed", type=int, help="Reproduce a previous choice. Default: fresh randomness.")
@click.option(
    "--epsilon",
    default=DEFAULT_EPSILON,
    show_default=True,
    type=click.FloatRange(0, 1),
    help="Probability of exploring a uniformly random new neuron.",
)
@click.option(
    "--temperature",
    default=1.0,
    show_default=True,
    type=click.FloatRange(min=0.05),
    help="Higher flattens the weights (more exploration among follow-ups).",
)
@click.option(
    "--model",
    "models",
    multiple=True,
    help="Model(s) to explore. Default: [tool.openinterp] models in pyproject.toml.",
)
@click.option(
    "--claims/--no-claims",
    default=True,
    show_default=True,
    help="Down-weight neurons claimed by open PRs (needs `gh`).",
)
@click.option("--top", default=0, show_default=True, help="Also list the N heaviest candidates.")
@click.option("--json", "as_json", is_flag=True, help="Print the full plan as JSON.")
@click.option(
    "--steps",
    is_flag=True,
    help="Also print every step of the draw as exact commands: claim, history, run, post.",
)
@click.option(
    "--corpus", default="corpus.jsonl", show_default=True, help="Scan corpus, for --steps."
)
@click.option(
    "--proposer",
    help="For --steps: a LiteLLM model for `audit`. Default: the neuron-proposer subagent.",
)
@click.option(
    "--reviewer",
    help="For --steps: a LiteLLM model for `verify`. Default: the neuron-reviewer and "
    "explanation-judge subagents.",
)
def next_experiment(
    wiki_dir,
    seed,
    epsilon,
    temperature,
    models,
    claims,
    top,
    as_json,
    steps,
    corpus,
    proposer,
    reviewer,
):
    """Pick the next experiment (randomised; balances exploration and follow-ups)."""
    from openinterp.planner import open_pr_claims, plan_next

    plan = plan_next(
        wiki_dir,
        seed=seed,
        epsilon=epsilon,
        temperature=temperature,
        models=models,
        claims=open_pr_claims() if claims else (),
    )
    if as_json:
        click.echo(plan.to_json())
        return
    c = plan.choice
    click.echo(f"Next experiment: {c.kind}" + (f" on {c.slug}" if c.slug else f" ({c.model})"))
    click.echo(f"Why: {c.rationale}")
    for command in c.commands:
        click.echo(f"  $ {command}")
    if any(cmd.startswith("openinterp audit") and "--neuron" in cmd for cmd in c.commands):
        _scan_cache_note(c)
    if c.skills:
        click.echo(f"Skills: {', '.join(c.skills)}")
    click.echo(
        f"(seed {plan.seed}; {len(plan.candidates)} follow-up candidates; "
        f"{'explored' if plan.explored else 'exploited'})"
    )
    _list_candidates(plan, top)
    if steps:
        from openinterp.draw_steps import render
        from openinterp.draw_steps import steps as draw_steps

        click.echo("\nSteps:")
        click.echo(render(draw_steps(plan, corpus, proposer, reviewer)))


def _scan_cache_note(c) -> None:
    from openinterp.harness.layer_cache import LayerCache

    cache = LayerCache.default()
    if cache is not None and c.layer in cache.cached_layers(c.model):
        click.echo(
            f"Scan cache: layer {c.layer} is cached, so the audit replays its scan "
            "(when the corpus options match) instead of running the model."
        )


def _list_candidates(plan, top: int) -> None:
    """The ``top`` heaviest follow-ups by share of the weight, and the explore draws."""
    total = sum(x.weight for x in plan.candidates) or 1.0
    for x in plan.candidates[:top]:
        click.echo(f"  {x.weight / total:6.1%}  {x.kind:18s} {x.slug}  — {x.rationale}")
    if top and len(plan.explore_models) > 1:
        click.echo("Explore draws by model:")
        for model, p in sorted(plan.explore_models.items(), key=lambda kv: -kv[1]):
            click.echo(f"  {p:6.1%}  {model}")


@cli.command()
@click.argument("records", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--base-ref",
    help="Reproduce every record added or changed since this git ref (e.g. origin/main).",
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
    records, base_ref, wiki_dir, atol, rtol, device, corpus, allowed_models, allowed_signers
):
    """Re-run experiment records against their pinned model and compare (no LLM calls).

    Each record's scan evidence, counterfactual deltas and review are measured again
    with the model revision and dtype it pins, under its own review policy. Exits 1 if
    any number differs beyond tolerance or any pass/fail, gate or status differs.
    """
    import os

    from openinterp.checks import changed_records
    from openinterp.cli_evidence import read_corpus
    from openinterp.reproduce import Tolerance

    paths = list(records)
    if base_ref:
        paths += [p for p in changed_records(wiki_dir, base_ref) if p not in paths]
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


@dataclass
class _Reproducer:
    """Checks one record at a time, loading each pinned model once."""

    signers: object
    tolerance: object
    scan_corpus: tuple[list[str], str] | None
    allowed_models: tuple[str, ...]
    device: str
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
        if signer:
            return f"SIGNED by {signer} (reproduction skipped)", [], []
        try:
            record = AuditRecord.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError:
            return "FAILED", ["invalid experiment record"], []
        reason = self.refusal(record)
        if reason:
            return "FAILED", [reason], []
        runner = self.runner(record)
        report = reproduce_record(record, runner, self.tolerance, path, self.scan_corpus)
        return ("FAILED" if report.mismatches else "REPRODUCED"), report.mismatches, report.warnings

    def refusal(self, record) -> str | None:
        """Why ``record`` can't be reproduced here, or ``None``."""
        from openinterp.reproduce import check_protocol

        model, allowed = record.submission.identifier.model_name, self.allowed_models
        reason = check_protocol(record)
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
    """A feature record pins the weights of its dictionary's preset, so the allow-list also
    limits which dictionaries are downloaded. Other records have none to check."""
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

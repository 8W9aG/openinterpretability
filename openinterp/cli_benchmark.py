"""``openinterp benchmark``: freeze, measure, seal and score the explainer benchmark.

See docs/benchmark.md. :mod:`openinterp.benchmark` holds the logic that needs no model.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from openinterp.cli import cli, recorded_model_option, wiki_option, with_model_options
from openinterp.core.agent_names import EXTERNAL_PREFIX

PRIVATE_SUFFIX = ".private.jsonl"  # .gitignore keeps these files out of the repository

bench_option = click.option(
    "--bench",
    "bench_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="A frozen version, e.g. benchmark/v1.",
)


@cli.group()
def benchmark() -> None:
    """The public benchmark for explanation tools (docs/benchmark.md)."""


@benchmark.command()
@wiki_option
@click.option("--version", "version", required=True, help="Version name, e.g. v1. Salts the split.")
@click.option("--out", "out_root", default=Path("benchmark"), show_default=True, type=Path)
@click.option("--force", is_flag=True, help="Replace an existing, unpublished version.")
def freeze(wiki_dir, version, out_root, force):
    """Build a version from the registry's records: units, split and dev probes."""
    from openinterp import benchmark as bench
    from openinterp.builder.wiki import load_archive

    out_dir = out_root / version
    if (out_dir / bench.MANIFEST).exists() and not force:
        raise click.ClickException(f"{out_dir} exists; versions are frozen (use a new version)")
    frozen = bench.freeze(load_archive(wiki_dir), version)
    bench.write(frozen, out_dir)
    m = frozen.manifest
    click.echo(
        f"Wrote {out_dir}: {m['units']['dev']} dev units ({m['dev_probes']} probes), "
        f"{m['units']['test']} test units awaiting fresh probes"
    )


@benchmark.command()
@bench_option
@wiki_option
@click.option("--unit", "slug", required=True, help="A test unit's slug.")
@click.option(
    "--probes",
    "probes_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="ProbeSet JSON, e.g. a reviewer's answer to `openinterp review --emit-prompt`.",
)
@click.option(
    "--labels",
    "labels_path",
    required=True,
    type=Path,
    help=f"The private labels file (name ending {PRIVATE_SUFFIX}). Rows are appended.",
)
@with_model_options
def measure(bench_dir, wiki_dir, slug, probes_file, labels_path, device, dtype, **_):
    """Measure fresh probes for one test unit, and append their labels to a private file."""
    from openinterp import benchmark as bench
    from openinterp.agents.review_measurements import measure_probes, settle_first_token_floor
    from openinterp.agents.reviewer import ProbeSet
    from openinterp.cli import _load_unit_runner
    from openinterp.core.paths import experiments_dir
    from openinterp.core.schemas import AuditRecord, ReviewPolicy

    if not labels_path.name.endswith(PRIVATE_SUFFIX):
        raise click.ClickException(f"--labels must end with {PRIVATE_SUFFIX}, which git ignores")
    unit = bench.read_units(bench_dir).get(slug)
    if unit is None or unit.split != "test":
        raise click.ClickException(f"{slug} is not a test unit of {bench_dir}")
    path = experiments_dir(wiki_dir) / slug / f"{unit.records[-1]}.json"
    record = AuditRecord.model_validate_json(path.read_text(encoding="utf-8"))
    probes = ProbeSet.model_validate_json(probes_file.read_text()).probes
    runner = _load_unit_runner(record.submission.identifier, device, dtype, unit.model_revision)
    policy = record.verdict.policy or ReviewPolicy()
    _, measured_unit, scale = settle_first_token_floor(record.submission, runner, policy)
    threshold = policy.activation_threshold * scale
    results, _, _ = measure_probes(runner, measured_unit, probes, scale, threshold)
    rows = [
        {"unit": slug, "text": r.prompt, "active": r.passed == r.expected_active, "kind": r.kind}
        for r in results
        if r.measured
    ]
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with labels_path.open("a", encoding="utf-8") as f:
        f.writelines(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in rows)
    active = sum(r["active"] for r in rows)
    click.echo(f"{slug}: appended {len(rows)} probes ({active} active) to {labels_path}")


@benchmark.command()
@bench_option
@click.option(
    "--labels", "labels_path", required=True, type=click.Path(exists=True, path_type=Path)
)
def seal(bench_dir, labels_path):
    """Sort the private test labels and publish their SHA-256 in the manifest."""
    from openinterp import benchmark as bench

    try:
        manifest = bench.seal(bench_dir, labels_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Sealed {manifest['test_probes']} test probes: {manifest['test_labels_sha256']}")


@benchmark.command()
@bench_option
@click.option("--split", type=click.Choice(["dev", "test"]), default="dev", show_default=True)
@click.option(
    "--explanations",
    required=True,
    help="JSONL of {unit, explanation}, or `registry` for the registry's own explanations.",
)
@click.option(
    "--labels",
    "labels_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="The private test labels; must match the manifest's hash.",
)
@click.option(
    "--judge-model",
    required=True,
    callback=recorded_model_option,
    help="LiteLLM model string, or external/<name> to answer prompt files.",
)
@click.option(
    "--out",
    "out_path",
    required=True,
    type=Path,
    help="Per-unit results (JSONL). Units already in it are not judged again. "
    "The summary goes next to it as <name>.summary.json.",
)
@click.option("--limit", type=int, help="Score only the first N units, for a cheap trial.")
@click.option(
    "--prompts-dir",
    type=click.Path(file_okay=False, path_type=Path),
    help="With an external/<name> judge: write the prompt of every unit without an answer "
    "to this directory at once, and score the units whose answers are there. Run the "
    "command again when the answers are written.",
)
def score(bench_dir, split, explanations, labels_path, judge_model, out_path, limit, prompts_dir):
    """Apply each explanation to its unit's probes with a judge, and score the labels.

    With ``--prompts-dir``, many judges answer at once, one prompt file each. The summary
    is written only when every explained unit has an answer.
    """
    from openinterp import benchmark as bench

    if prompts_dir and not judge_model.startswith(EXTERNAL_PREFIX):
        raise click.UsageError(f"--prompts-dir needs an {EXTERNAL_PREFIX}<name> judge model")
    try:
        probes = bench.load_labels(bench_dir, split, labels_path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    units = [u for u in bench.read_units(bench_dir).values() if u.split == split]
    units = sorted(units, key=lambda u: u.unit)[:limit]
    probes = {u.unit: probes[u.unit] for u in units}
    explained = _explanations(bench_dir, split, explanations)
    done = {r["unit"]: r for r in bench.read_jsonl(out_path)} if out_path.exists() else {}
    judge = _file_judge(prompts_dir) if prompts_dir else _judge(judge_model)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:

        def write(row):
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            click.echo(f"{row['unit']}: {row['balanced_accuracy']:.3f}", err=True)

        rows = bench.score(probes, explained, judge, done=done, on_row=write)
    pending = len({u.unit for u in units} & set(explained)) - len(rows)
    if pending:
        click.echo(
            f"{pending} unit(s) wait for a judge's answer, one prompt each in {prompts_dir}. "
            "Run this command again when they are answered."
        )
        return
    meta = {"split": split, "explanations": explanations, "judge_model": judge_model}
    _write_summary(bench_dir, out_path, rows, units, {**meta, "limit": limit})


def _write_summary(bench_dir: Path, out_path: Path, rows, units, meta: dict) -> None:
    """Write ``<out>.summary.json`` next to ``out_path``, and print the split's score."""
    from openinterp import benchmark as bench

    version = bench.read_manifest(bench_dir)["version"]
    summary = {"benchmark": version, **meta, **bench.summarize(rows, units)}
    summary_path = out_path.with_name(out_path.name.removesuffix(".jsonl") + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    click.echo(
        f"{meta['split']}: mean balanced accuracy {summary['mean_balanced_accuracy']} over "
        f"{summary['units']} units ({summary['explained']} explained), 95% CI {summary['ci95']}"
    )


def _explanations(bench_dir: Path, split: str, source: str) -> dict[str, str]:
    from openinterp import benchmark as bench

    files = {"dev": bench.DEV_EXPLANATIONS, "test": bench.TEST_EXPLANATIONS}
    rows = bench.read_jsonl(bench_dir / files[split] if source == "registry" else Path(source))
    return {r["unit"]: r["explanation"] for r in rows}


def _judge(model: str):
    """``judge(unit, explanation, texts)``: which texts the explanation says make it fire."""
    from openinterp.agents.judge import ExplanationJudge, TaskWindow, answered

    agent = ExplanationJudge(model)

    def judge(unit: str, explanation: str, texts: list[str]) -> list[bool]:
        windows = [TaskWindow(text=t, tokens=[t]) for t in texts]
        answer = agent.detect(explanation, windows)
        return answered(answer.active, len(texts), "detection")

    return judge


def _file_judge(directory: Path):
    """A judge that reads each unit's answer from ``directory``, or asks for it there.

    A unit's prompt is ``<unit>.prompt.md``, with ``--`` for each ``/`` of the slug. It is
    the prompt that an ``external/<name>`` judge writes, and it names the answer file,
    ``<unit>.json``. The judge returns ``None`` until that answer exists. A bad answer is
    moved to ``<unit>.rejected.json`` with the reason in ``<unit>.error.txt``. Then the
    unit waits for a new answer.
    """
    from pydantic import ValidationError

    from openinterp.agents.judge import DetectionAnswer, ExplanationJudge, TaskWindow, answered

    directory.mkdir(parents=True, exist_ok=True)

    def judge(unit: str, explanation: str, texts: list[str]) -> list[bool] | None:
        name = unit.replace("/", "--")
        answer_path, error_path = directory / f"{name}.json", directory / f"{name}.error.txt"
        if answer_path.exists():
            try:
                answer = DetectionAnswer.model_validate_json(answer_path.read_text())
                result = answered(answer.active, len(texts), "detection")
            except (ValidationError, ValueError) as exc:
                error_path.write_text(f"{exc}\n", encoding="utf-8")
                answer_path.replace(directory / f"{name}.rejected.json")
            else:
                error_path.unlink(missing_ok=True)
                return result
        windows = [TaskWindow(text=t, tokens=[t]) for t in texts]
        prompt = ExplanationJudge.detection_prompt(explanation, windows)
        text = ExplanationJudge.render_prompt(DetectionAnswer, prompt, answer_path)
        (directory / f"{name}.prompt.md").write_text(text, encoding="utf-8")
        return None

    return judge

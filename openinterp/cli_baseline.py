"""``openinterp patchscope-baseline``: the Patchscope readout as a blind proposer baseline.

See :mod:`openinterp.baseline` for the rule and the comparison.
"""

from __future__ import annotations

from pathlib import Path

import click

from openinterp.cli import cli, with_model_options


@cli.command("patchscope-baseline")
@click.option("--wiki-dir", default="wiki", show_default=True, type=click.Path(path_type=Path))
@click.option(
    "--work-dir",
    default="exchange/patchscope-baseline",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Readouts, judge prompts and their answers.",
)
@click.option(
    "--judge-model",
    default="external/claude-haiku-5-5",
    show_default=True,
    help="Write judge prompts for the experiments this judge scored, and record their "
    "answers as its own: the pair is only fair with the same judge.",
)
@click.option(
    "--out",
    default="docs/analyses/patchscope-baseline",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Writes <out>.jsonl (one row per experiment) and <out>.md (the comparison).",
)
@with_model_options
def patchscope_baseline(wiki_dir, work_dir, judge_model, out, device, dtype):
    """Score each unit's Patchscope readout, turned into an explanation by a fixed rule,
    on the held-out windows its agent's explanation was scored on.

    \b
    Run it once to read every unit and write judge prompts under --work-dir, answer each
    prompt with its own explanation-judge, then run it again to score the answers. Each
    run writes whatever can be compared so far. The embedding score uses the encoder the
    record's own scores used, so it needs no judge.
    """
    from openinterp.agents.judge import DetectionAnswer, FuzzingAnswer, answered
    from openinterp.baseline import (
        DEV_SLUGS,
        METRICS,
        BaselineRow,
        blind_explanation,
        blind_reading,
        metric,
        report,
        rescored,
    )
    from openinterp.builder.wiki import experiment_id, load_archive_paths
    from openinterp.cli_score import _sibling, _write_task
    from openinterp.harness.explanation_scores import embedding_similarities

    records = [
        r
        for _, r in load_archive_paths(wiki_dir)
        if r.verdict
        and r.verdict.explanation_scores
        and r.submission.identifier.kind != "attn_head"
    ]
    click.echo(f"{len(records)} experiments have explanation scores.", err=True)
    readouts = _readouts(records, work_dir / "readouts", device, dtype)
    rows, pending = [], 0
    for record in records:
        sub, scores = record.submission, record.verdict.explanation_scores
        eid, slug = experiment_id(record), sub.identifier.slug
        readout = readouts[slug]
        explanation = blind_explanation(readout)
        agent = {name: metric(scores, name) for name in METRICS}
        row = dict(
            experiment_id=eid,
            slug=slug,
            status=record.verdict.status,
            selection=sub.provenance.selection if sub.provenance else None,
            dev=slug in DEV_SLUGS,
            reading=(blind_reading(readout) or readout.readings[0]).text,
            agent_explanation=scores.explanation,
            baseline_explanation=explanation,
            judge_model=scores.judge_model,
            agent=agent,
            baseline=dict.fromkeys(METRICS),
        )
        if explanation is None:
            rows.append(BaselineRow(**row))
            continue
        sims = None
        if scores.embedding_model and agent["embedding"] is not None:
            texts = [w.text for w in scores.detection_windows]
            sims = embedding_similarities(scores.embedding_model, explanation, texts)
        detection = fuzzing = None
        task_path = work_dir / "judge" / slug / f"{eid}.task.json"
        if scores.judge_model == judge_model:
            det_path = _sibling(task_path, ".detection.json")
            fuz_path = _sibling(task_path, ".fuzzing.json")
            need_fuzzing = bool(scores.fuzzing_windows)  # no prompt without fuzzing windows
            if det_path.exists() and (fuz_path.exists() or not need_fuzzing):
                det = _answer(DetectionAnswer, det_path).active
                detection = answered(det, len(scores.detection_windows), "detection")
                fuz = None
                if need_fuzzing:
                    fuz = _answer(FuzzingAnswer, fuz_path).correct
                    fuzzing = answered(fuz, len(scores.fuzzing_windows), "fuzzing")
                row |= {"baseline_judge_model": judge_model, "detection_answer": det}
                row |= {"fuzzing_answer": fuz}
            else:
                pending += 1
                if not task_path.exists():
                    _write_task(_task(record, explanation), task_path)
        base = rescored(
            scores,
            explanation,
            judge_model=row.get("baseline_judge_model"),
            detection=detection,
            fuzzing=fuzzing,
            similarities=sims,
        )
        row["baseline"] = {name: metric(base, name) for name in METRICS}
        rows.append(BaselineRow(**row))

    out.parent.mkdir(parents=True, exist_ok=True)
    jsonl = out.with_suffix(".jsonl")
    jsonl.write_text("".join(r.model_dump_json() + "\n" for r in rows))
    table = report(rows)
    out.with_suffix(".md").write_text(table)
    click.echo(table)
    if pending:
        click.echo(f"{pending} experiments await judge answers under {work_dir / 'judge'}.")
    click.echo(f"Wrote {jsonl} and {out.with_suffix('.md')}")


def _answer(model, path: Path):
    try:
        return model.model_validate_json(path.read_text())
    except ValueError as exc:
        raise click.ClickException(f"{path} is not a valid answer; judge it again: {exc}") from exc


def _readouts(records, directory: Path, device: str, dtype: str) -> dict:
    """Each unit's readout, read once under its record's pinned revision and cached."""
    from openinterp.cli import _load_unit_runner
    from openinterp.core.patchscope import PatchscopeReadout
    from openinterp.harness.patchscope import patchscope, write_direction
    from openinterp.harness.units import submission_unit
    from openinterp.reproduce import pinned_environment

    runners, readouts = {}, {}
    for record in records:
        sub = record.submission
        slug = sub.identifier.slug
        path = directory / f"{slug}.json"
        if slug in readouts:
            continue
        if sub.patchscope is not None:
            readouts[slug] = sub.patchscope
            continue
        if path.exists():
            readouts[slug] = PatchscopeReadout.model_validate_json(path.read_text())
            continue
        revision, pinned = pinned_environment(record)
        key = (sub.identifier.model_name, revision, pinned)
        if key not in runners:
            runners[key] = _load_unit_runner(sub.identifier, device, pinned, revision)
        runner = runners[key]
        click.echo(f"Reading {sub.identifier} through a Patchscope…", err=True)
        readout = patchscope(
            runner, sub.identifier.layer, write_direction(runner, submission_unit(sub))
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(readout.model_dump_json(indent=2))
        readouts[slug] = readout
    return readouts


def _task(record, explanation: str):
    """A judge task on the record's own held-out windows, with ``explanation`` instead."""
    from openinterp.agents.judge import ScoringTask, TaskWindow

    sub, scores = record.submission, record.verdict.explanation_scores

    def windows(ws):
        return [TaskWindow(text=w.text, tokens=w.tokens, highlighted=w.highlighted) for w in ws]

    return ScoringTask(
        identifier=sub.identifier,
        head_profile=sub.head_profile,
        dictionary_profile=sub.dictionary_profile,
        explanation=explanation,
        threshold=scores.threshold,
        window_tokens=scores.window_tokens,
        quantiles=scores.quantiles,
        quantile_edges=scores.quantile_edges,
        seed=scores.seed,
        corpus=scores.corpus,
        corpus_sha256=scores.corpus_sha256,
        candidate_windows=scores.candidate_windows,
        active_windows=scores.active_windows,
        detection=windows(scores.detection_windows),
        fuzzing=windows(scores.fuzzing_windows),
    )

"""``openinterp score``: detection, fuzzing and embedding scores for an explanation."""

from __future__ import annotations

import json
import math
import secrets
from pathlib import Path

import click

from openinterp.cli import cli, with_model_options
from openinterp.harness.explanation_scores import DEFAULT_EMBEDDING_MODEL


def _sibling(task_path: Path, suffix: str) -> Path:
    """``score.task.json`` -> ``score<suffix>``."""
    return task_path.with_name(task_path.name.removesuffix(".json").removesuffix(".task") + suffix)


def _load_submission(path: Path):
    from openinterp.core.schemas import AuditRecord, NeuronAuditSubmission

    raw = path.read_text()
    if "submission" in json.loads(raw):
        return AuditRecord.model_validate_json(raw).submission
    return NeuronAuditSubmission.model_validate_json(raw)


def _threshold(submission) -> float:
    from openinterp.core.schemas import ReviewPolicy

    if not submission.max_activation:
        raise click.ClickException("the submission has no max_activation; rescan it first.")
    return ReviewPolicy().activation_threshold * submission.max_activation


@cli.command()
@click.option(
    "--audit-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Submission or record whose explanation to score. Required unless --task.",
)
@click.option(
    "--corpus",
    type=click.Path(exists=True, path_type=Path),
    help="Where held-out windows come from; the scan's top examples are left out. "
    "Use the scan's corpus, so the windows are text the proposer could have seen but didn't.",
)
@click.option("--seed", type=int, help="Seed for drawing windows (recorded).")
@click.option(
    "--judge-model",
    default=None,
    help="Judge for detection and fuzzing: a LiteLLM model string, or external/<name>. "
    "With --task, the name recorded for whoever answered.",
)
@click.option(
    "--embedding-model",
    default=DEFAULT_EMBEDDING_MODEL,
    show_default=True,
    help="Hugging Face sentence encoder for the embedding score; '' to skip it.",
)
@click.option(
    "--emit-task",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Draw the windows, write them here (e.g. score.task.json) with the judge's two "
    "prompts next to them (score.detection.prompt.md, score.fuzzing.prompt.md), and exit.",
)
@click.option(
    "--task",
    "task_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="A task from --emit-task, answered in score.detection.json and score.fuzzing.json "
    "next to it. No LLM is called.",
)
@click.option("--window-tokens", default=32, show_default=True, type=click.IntRange(4, 512))
@click.option("--quantiles", default=5, show_default=True, type=click.IntRange(1, 20))
@click.option(
    "--per-quantile",
    default=5,
    show_default=True,
    type=click.IntRange(1, 50),
    help="Active windows per activation bin, for each of detection and fuzzing.",
)
@click.option("--max-windows", default=4000, show_default=True, type=click.IntRange(1))
@click.option(
    "--out",
    type=click.Path(dir_okay=False, path_type=Path),
    help="ExplanationScores JSON (default with --task: score.scores.json next to it).",
)
@with_model_options
def score(
    audit_file,
    corpus,
    seed,
    judge_model,
    embedding_model,
    emit_task,
    task_file,
    window_tokens,
    quantiles,
    per_quantile,
    max_windows,
    out,
    device,
    dtype,
):
    """Score an explanation on held-out text: detection, fuzzing and embedding.

    \b
    Reported, not gated (docs/PROTOCOL.md §4), but every review records them: pass the
    result to `openinterp review … --explanation-scores <file>`. In one step the judge is
    an LLM:
      openinterp score --audit-file sub.json --corpus corpus.jsonl --judge-model M --out s.json
    In two steps any agent can judge, without an API key:
      openinterp score --audit-file sub.json --corpus corpus.jsonl --emit-task score.task.json
      (answer score.detection.prompt.md and score.fuzzing.prompt.md)
      openinterp score --task score.task.json --judge-model external/<model that answered>
    """
    from openinterp.agents.base import DEFAULT_MODEL
    from openinterp.agents.judge import (
        DetectionAnswer,
        FuzzingAnswer,
        ScoringTask,
    )
    from openinterp.cli import _load_runner  # looked up per call, so tests can swap it

    if task_file is not None:
        if audit_file is not None or emit_task is not None:
            raise click.UsageError("--task scores an emitted task; drop --audit-file/--emit-task.")
        if not judge_model:
            raise click.UsageError("--judge-model is required with --task: name who answered.")
        task = ScoringTask.model_validate_json(task_file.read_text())
        detection = DetectionAnswer.model_validate_json(
            _sibling(task_file, ".detection.json").read_text()
        )
        fuzzing = (  # no fuzzing prompt when no held-out window was active
            FuzzingAnswer.model_validate_json(_sibling(task_file, ".fuzzing.json").read_text())
            if task.fuzzing
            else FuzzingAnswer(correct=[])
        )
        out = out or _sibling(task_file, ".scores.json")
        runner = _load_runner(task.identifier.model_name, device, dtype)
        scores = _score_task(task, runner, detection, fuzzing, judge_model, embedding_model)
    else:
        if audit_file is None:
            raise click.UsageError("--audit-file is required unless --task is given.")
        if emit_task is None and out is None:
            raise click.UsageError("--out is required when scoring in one step.")
        submission = _load_submission(audit_file)
        runner = _load_runner(submission.identifier.model_name, device, dtype)
        sampling = (seed, window_tokens, quantiles, per_quantile, max_windows)
        if emit_task is not None:
            _write_task(_draw_task(runner, submission, corpus, *sampling), emit_task)
            return
        scores = score_submission(
            runner, submission, corpus, judge_model or DEFAULT_MODEL, embedding_model, *sampling
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(scores.model_dump_json(indent=2))
    click.echo(_summary(scores))
    click.echo(f"Wrote {out}")


def score_submission(
    runner,
    submission,
    corpus: Path | None,
    judge_model: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    seed: int | None = None,
    window_tokens: int = 32,
    quantiles: int = 5,
    per_quantile: int = 5,
    max_windows: int = 4000,
):
    """Draw held-out windows and score the explanation with an LLM judge, in one step."""
    from openinterp.agents.judge import ExplanationJudge, FuzzingAnswer

    sampling = (seed, window_tokens, quantiles, per_quantile, max_windows)
    task = _draw_task(runner, submission, corpus, *sampling)
    judge = ExplanationJudge(judge_model)
    click.echo(f"Asking {judge.model} to judge {len(task.detection)} windows…", err=True)
    detection = judge.detect(task.explanation, task.detection)
    fuzzing = (
        judge.fuzz(task.explanation, task.fuzzing) if task.fuzzing else FuzzingAnswer(correct=[])
    )
    return _score_task(task, runner, detection, fuzzing, judge.model, embedding_model)


def _draw_task(runner, submission, corpus, seed, window_tokens, quantiles, per_quantile, cap):
    from openinterp.agents.judge import ScoringTask, TaskWindow
    from openinterp.cli_evidence import read_corpus
    from openinterp.harness.explanation_scores import sample_heldout

    texts, corpus_name, corpus_sha, _ = read_corpus(corpus)
    seed = secrets.randbits(32) if seed is None else seed
    threshold = _threshold(submission)
    click.echo(f"Measuring held-out {window_tokens}-token windows (seed {seed})…", err=True)
    try:
        sample = sample_heldout(
            runner,
            texts,
            submission.identifier,
            threshold,
            seed=seed,
            exclude=[e.text for e in submission.top_examples],
            window_tokens=window_tokens,
            max_windows=cap,
            quantiles=quantiles,
            per_quantile=per_quantile,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    def windows(ws):
        return [
            TaskWindow(text=w.text, tokens=w.tokens, highlighted=list(w.highlighted)) for w in ws
        ]

    return ScoringTask(
        identifier=submission.identifier,
        explanation=submission.natural_language_explanation,
        threshold=threshold,
        window_tokens=window_tokens,
        quantiles=quantiles,
        quantile_edges=sample.quantile_edges,
        seed=seed,
        corpus=corpus_name,
        corpus_sha256=corpus_sha,
        candidate_windows=sample.candidate_windows,
        active_windows=sample.active_windows,
        detection=windows(sample.detection),
        fuzzing=windows(sample.fuzzing),
    )


def _write_task(task, path: Path) -> None:
    from openinterp.agents.judge import DetectionAnswer, ExplanationJudge, FuzzingAnswer

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(task.model_dump_json(indent=2))
    click.echo(f"Wrote {path} ({len(task.detection)} detection, {len(task.fuzzing)} fuzzing)")
    if not task.active_windows:
        click.echo("No held-out window is active: detection scores false positives only.")
    for kind, model, prompt in [
        ("detection", DetectionAnswer, ExplanationJudge.detection_prompt),
        ("fuzzing", FuzzingAnswer, ExplanationJudge.fuzzing_prompt),
    ]:
        if not getattr(task, kind):
            continue
        prompt_path, answer = _sibling(path, f".{kind}.prompt.md"), _sibling(path, f".{kind}.json")
        text = prompt(task.explanation, getattr(task, kind))
        prompt_path.write_text(ExplanationJudge.render_prompt(model, text, answer))
        click.echo(f"Judge prompt: {prompt_path} (answer in {answer})")


def _score_task(task, runner, detection, fuzzing, judge_model, embedding_model):
    from openinterp.agents.judge import answered
    from openinterp.core.explanation import ExplanationScores, ScoredWindow
    from openinterp.harness.explanation_scores import embedding_similarities, remeasure

    try:
        det = answered(detection.active, len(task.detection), "detection")
        fuz = answered(fuzzing.correct, len(task.fuzzing), "fuzzing")
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    sims = [None] * len(task.detection)
    if embedding_model:
        click.echo(f"Embedding the explanation and windows with {embedding_model}…", err=True)
        sims = embedding_similarities(
            embedding_model, task.explanation, [w.text for w in task.detection]
        )

    def unmeasured(windows, answers, similarities):
        return [
            ScoredWindow(
                text=w.text,
                tokens=w.tokens,
                activations=[],
                highlighted=w.highlighted,
                label=False,  # re-derived from a fresh measurement below
                judged=a,
                similarity=s,
            )
            for w, a, s in zip(windows, answers, similarities, strict=True)
        ]

    scores = ExplanationScores(
        slug=task.identifier.slug,
        explanation=task.explanation,
        threshold=task.threshold,
        window_tokens=task.window_tokens,
        quantiles=task.quantiles,
        quantile_edges=task.quantile_edges,
        seed=task.seed,
        corpus=task.corpus,
        corpus_sha256=task.corpus_sha256,
        candidate_windows=task.candidate_windows,
        active_windows=task.active_windows,
        judge_model=judge_model,
        embedding_model=embedding_model or None,
        detection_windows=unmeasured(task.detection, det, sims),
        fuzzing_windows=unmeasured(task.fuzzing, fuz, [None] * len(task.fuzzing)),
    )
    return remeasure(scores, runner, task.identifier)


def _summary(scores) -> str:
    def judged(s):
        if s is None:
            return "—"
        return (
            f"{s.balanced_accuracy:.2f} (TPR {_num(s.true_positive_rate)}, "
            f"FPR {_num(s.false_positive_rate)})"
        )

    return (
        f"Detection {judged(scores.detection)} · fuzzing {judged(scores.fuzzing)} · "
        f"embedding AUROC {_num(scores.embedding_auroc)} · "
        f"{scores.active_windows}/{scores.candidate_windows} held-out windows active"
    )


def _num(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def load_scores(path: Path | None, submission):
    """The explanation scores in ``path``, checked against ``submission``, or ``None``."""
    from openinterp.core.explanation import ExplanationScores

    if path is None:
        return None
    scores = ExplanationScores.model_validate_json(path.read_text())
    if scores.slug != submission.identifier.slug:
        raise click.UsageError(f"{path} scores {scores.slug}, not {submission.identifier.slug}.")
    if scores.explanation != submission.natural_language_explanation:
        raise click.UsageError(f"{path} scores a different explanation; re-run openinterp score.")
    if not math.isclose(scores.threshold, _threshold(submission), rel_tol=1e-9):
        raise click.UsageError(f"{path} used another activation threshold; re-run it.")
    return scores

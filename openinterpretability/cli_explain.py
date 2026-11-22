"""Attribute one answer of a model to its labelled neuron poles and attention heads.

The command is ``open-interpretability explain``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import click

from openinterpretability import explain_plan
from openinterpretability.cli import _load_runner, cli, wiki_dir_option, with_model_options
from openinterpretability.cli_patch import patch_options, patch_target
from openinterpretability.explain_backend import narrator
from openinterpretability.explain_output import (
    _as_json,
    _print_coverage,
    _print_narrative,
    _print_plan,
    reached,
)
from openinterpretability.explain_print import print_result
from openinterpretability.explain_reuse import look_first
from openinterpretability.explain_run import run_plan
from openinterpretability.explain_stopped import lines as stopped_lines
from openinterpretability.harness.progress import Stages
from openinterpretability.hubs import mark_hubs
from openinterpretability.patches.session import patched


@cli.command(name="explain")
@click.argument("prompt")
@click.option("--model", "model_name", default="Qwen/Qwen2.5-0.5B", show_default=True)
@click.option("--token", help="The answer token to explain. Default: the model's top token.")
@click.option("--position", default=-1, show_default=True, help="The position to explain.")
@click.option("--top", default=20, show_default=True, help="Units to list.")
@click.option(
    "--edges", default=8, show_default=True, help="Top units whose incoming paths are traced."
)
@click.option(
    "--corpus", default=None, type=click.Path(path_type=Path),
    help="The corpus of each unit's mean. Default: the registry's pile-10k corpus "
    "(explain_corpus.EXPLAIN_CORPUS), ./corpus.jsonl if it holds it, else built once into "
    "the cache (#5829).",
)  # fmt: skip
@click.option(
    "--mean-tokens",
    default=20_000,
    show_default=True,
    help="Tokens of the corpus for each neuron's mean, the attribution's baseline (cached).",
)
@click.option(
    "--target", default=None, type=click.FloatRange(0, 1),
    help="The causal coverage to aim at: the share of the answer's log-probability that the "
    "explained units bring back, the rest mean-ablated (#4476). Default: 1.",
)  # fmt: skip
@click.option(
    "--experiments", "show", default=10, show_default=True, help="Planned experiments to list."
)
@click.option(
    "--level", type=click.Choice(["mechanism", "long", "paragraph"]), default="mechanism",
    show_default=True, help="mechanism: the evidence. long and paragraph: an LLM narrates it.",
)  # fmt: skip
@click.option("--narrator-model", default=None, help="The narrator's model (agents' default).")
@click.option(
    "--run", "run", default=None, type=click.IntRange(0), is_flag=False, flag_value=0,
    help="Run the planned unit and superposition experiments through the experiment loop, by "
    "coverage gain, until the explained share reaches --target or no work is left (#4473). "
    "--run N stops after N experiments too. Without it, only list them.",
)  # fmt: skip
@click.option("--proposer", default="", help="--run: the LiteLLM proposer model.")
@click.option("--reviewer", default="", help="--run: the LiteLLM reviewer model.")
@click.option("--provider", default="", help="--run: a provider profile (docs/providers.md).")
@click.option(
    "--max-attempts", default=5, show_default=True, type=click.IntRange(1),
    help="--run: runs per unit before giving up on it.",
)  # fmt: skip
@click.option(
    "--record", "write_record", is_flag=True,
    help="Write the explanation to wiki/explanations/, for its page on the site (#3744).",
)  # fmt: skip
@click.option(
    "--local", is_flag=True,
    help="--run, --complete: run in this checkout with no PR and no GitHub call. The records "
    "stay in --wiki-dir, and the record is written there too. Share them later in one PR with "
    "`open-interpretability explain-share` (#5827).",
)  # fmt: skip
@click.option(
    "--land/--no-land", "land", default=True, show_default=True,
    help="--record: land the record as its own PR, which auto-merge.yml merges on green "
    "(#4475, #5826). --no-land writes it to --wiki-dir only.",
)  # fmt: skip
@click.option(
    "--complete", is_flag=True,
    help="From the prompt to a published explanation in one command (#4476): --run to "
    "--target, --level long and --record, landed as a PR.",
)  # fmt: skip
@click.option("--json", "as_json", is_flag=True, help="Print the result as JSON.")
@click.option(
    "-y", "--yes", is_flag=True,
    help="Don't ask before calling LLMs (--run, --complete, a narrated --level). With no "
    "terminal to ask on, it is needed (#5830).",
)  # fmt: skip
@click.option(
    "--quiet", is_flag=True,
    help="No stage lines or progress bars (#5830). --json implies it. Warnings still print.",
)  # fmt: skip
@patch_options
@wiki_dir_option()
@with_model_options
def explain_command(  # long-function: one argument per option, then one line per step
    prompt, model_name, token, position, top, edges, corpus, mean_tokens, target, show, level,
    narrator_model, run, proposer, reviewer, provider, max_attempts, write_record, local, land,
    complete, as_json, yes, quiet, patch_ref, evidence_out,
    wiki_dir, device, dtype,
):  # fmt: skip
    """Explain the model's answer to PROMPT as fully as the registry allows.

    It attributes the answer's logit to every MLP neuron pole and attention head, against
    each unit's mean on the corpus. It labels the top units from the registry. It reports
    the share of the attribution that VERIFIED units hold, and checks it by mean-ablating
    them. It lists the superpositions and circuits that carry the most attribution, and
    traces the paths between the top units. Then it lists the experiments that would
    explain the rest: for each pole, head, superposition and circuit that isn't VERIFIED.

    With ``--patch``, it explains the patched model's answer as a fix's evidence, written to
    ``--evidence-out``. Nothing goes to the registry, so ``--run``, ``--record`` and
    ``--complete`` are refused (#5229).
    """
    fix = patch_target(patch_ref, evidence_out, wiki_dir, model_name)
    if fix is not None and (complete or write_record or run is not None):
        raise click.UsageError("--patch writes evidence only: drop --run, --record and --complete")
    runner = _load_runner(model_name, device, dtype) if fix else None
    with patched(runner.model if runner else None, fix), _quiet(quiet or as_json):
        _explain(
            prompt, model_name, token, position, top, edges, corpus, mean_tokens, target, show,
            level, narrator_model, run, proposer, reviewer, provider, max_attempts,
            write_record, land and not local, complete, as_json or fix is not None, wiki_dir,
            device, dtype, fix, evidence_out, local, yes,
        )  # fmt: skip


@contextmanager
def _quiet(on: bool) -> Iterator[None]:
    """Turn the stage lines and progress bars off while the block runs, when ``on`` (#5830).

    Args:
        on: ``--quiet``.

    Yields:
        Nothing. The quiet setting is put back as it was when the block ends.
    """
    from openinterpretability.harness.progress import QUIET_ENV

    before = os.environ.get(QUIET_ENV)
    if on:
        os.environ[QUIET_ENV] = "1"
    try:
        yield
    finally:
        if before is None:
            os.environ.pop(QUIET_ENV, None)
        else:
            os.environ[QUIET_ENV] = before


def _default_corpus() -> Path:
    """The explain corpus (:func:`.explain_corpus.default_corpus`), as a click error if it fails.

    Returns:
        The corpus file.

    Raises:
        click.ClickException: The cache is off, or the build gives other bytes.
    """
    from openinterpretability.explain_corpus import default_corpus

    try:
        return default_corpus(lambda line: click.echo(line, err=True))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _explain(  # long-function: the body of explain_command, one line per step
    prompt, model_name, token, position, top, edges, corpus, mean_tokens, target, show, level,
    narrator_model, run, proposer, reviewer, provider, max_attempts, write_record, land, complete,
    as_json, wiki_dir, device, dtype, fix=None, evidence_out=None, local=False, yes=True,
):  # fmt: skip
    """Run ``explain`` once, on the model as it stands (patched or not)."""
    from openinterpretability.evidence import read_corpus
    from openinterpretability.explain import unit_means

    if complete:  # run to the target, narrate in full, and land the record (#4476)
        run, level, write_record, land = 0 if run is None else run, "long", True, not local
    reuse = look_first(Path(wiki_dir), model_name, prompt, position, target, level,
                       write_record or run is not None)  # fmt: skip
    if reuse.action == "done" and fix is None:
        return
    if reuse.action == "level":
        run = None  # an earlier explanation already reaches the target: only narrate (#4503)
    runner = _load_runner(model_name, device, dtype)
    stages = Stages(6 + 2 * (run is not None) + (level != "mechanism") + bool(write_record))
    stages.next("Reading the corpus")
    texts, _, corpus_sha, corpus_source = read_corpus(corpus or _default_corpus())
    stages.next("Measuring each unit's mean on the corpus, the attribution's baseline")
    means = unit_means(runner, model_name + (fix.key if fix else ""), texts, mean_tokens)
    ask = Ask(model_name, prompt, _answer(runner, token), position, top, edges, target, show)
    found = _analyse(runner, means, ask, Path(wiki_dir), stages)
    if run is not None or level != "mechanism":  # an LLM is about to be called (#5830)
        _ask_to_spend(found, run, max_attempts, level, Path(wiki_dir), model_name, yes)
    home = Path(wiki_dir).resolve()  # the record goes here, though --run moves to its worktree
    ran, code = None, 0
    if run is not None:  # run, then explain the answer as the registry stands after it (#4474)
        agents = (proposer, reviewer, provider)
        stages.next("Running the plan's experiments")
        ran, code, pending = run_plan(found, ask, agents, max_attempts, run, Path(wiki_dir),
                                      measure=(runner, means), local=local)  # fmt: skip
        stages.next("Explaining the answer again, as the registry stands after the run")
        with pending.applied():  # its open PRs' records count, marked pending (#5826)
            found = _analyse(runner, means, ask, Path(wiki_dir))
        ran = ran.model_copy(update={"share_after": found.coverage.get("coverage") or 0.0})
    result, planned, coverage = found.result, found.planned, found.coverage
    told = {}
    if level != "mechanism":
        stages.next("Narrating the explanation")
        narrator_model = narrator(narrator_model, lambda line: click.echo(line, err=True))
        told = _narrate(result, coverage, planned, narrator_model, found.answer)
    if as_json:
        out = _as_json(result, planned, show) | {
            "coverage": coverage,
            "coverage_plan": found.reach,
            "answer_experiment": found.answer,
            "stopped": [{k: v for k, v in r.items() if k != "unit"} for r in found.stopped],
        }
        text = json.dumps(out | told | {"run": ran and ran.model_dump()}, indent=1, default=str)
        click.echo(text)
        if evidence_out is not None:  # a fix's evidence (#5229)
            Path(evidence_out).parent.mkdir(parents=True, exist_ok=True)
            Path(evidence_out).write_text(text + "\n", encoding="utf-8")
    elif level != "paragraph":
        print_result(result)
        _print_coverage(coverage)
        _print_plan(planned, show, found.reach)
        click.echo("\n".join(stopped_lines(found.stopped, ask.model)))
        click.echo("\n".join(explain_plan.answer_lines(found.answer)))
    if not as_json:
        _print_narrative(told, level)
    if write_record:
        from openinterpretability.explain_figure_data import figure_data

        stages.next("Writing the record")
        from openinterpretability.reproduce_explanation import inputs_of

        figures = figure_data(runner, means, result, coverage, planned, found.groups)
        record = _record(model_name, result, coverage, planned, told, ran, figures,
                         found.reach["target"])  # fmt: skip
        inputs = inputs_of(runner, ask, result, planned, (corpus_sha, corpus_source), mean_tokens)
        record = record.model_copy(update=reuse.fields | {"inputs": inputs})
        _keep(record, home, land)
    if not as_json:
        click.echo(reached(result.verified_share, found.reach["target"], coverage, ran))
    if code:
        raise SystemExit(code)


@dataclass(frozen=True)
class Ask:
    """What to explain: the model, the prompt and its answer, and how much to show."""

    model: str
    prompt: str
    token: int | None  # the answer's token id, or None for the model's top token
    position: int
    top: int
    edges: int
    target: float | None
    show: int


def _analyse(  # long-function: one line per step of the analysis
    runner, means, ask: Ask, wiki_dir: Path, stages=None,
) -> SimpleNamespace:  # fmt: skip
    """Attribute, label, mark hubs, measure the coverage and plan, from the registry now.

    ``explain --run`` calls it twice: to plan the run, and after it (#4474). ``stages``
    (:class:`~openinterpretability.harness.progress.Stages`) says each of its four stages;
    the second call passes none.
    """
    say = stages.next if stages is not None else (lambda _: None)
    from openinterpretability import explain_groups
    from openinterpretability.explain import explain, registry_labels
    from openinterpretability.explain_coverage import (
        POOL,
        add_gains,
        coverage_plan,
        prompt_coverage,
    )
    from openinterpretability.explain_stopped import rank as rank_stopped
    from openinterpretability.planner import configured_windows, load_states

    model = ask.model
    say("Attributing the answer to every unit, labelling and checking them")
    states = load_states(wiki_dir)  # read once: labels and the plan both use it
    labels = registry_labels(wiki_dir, model, states)
    groups = explain_groups.superpositions(model) + explain_groups.circuits(model)
    result = explain(
        runner, model, ask.prompt, means, labels, token=ask.token, position=ask.position,
        top=ask.top, edges=ask.edges, groups=groups,
        group_labels=explain_groups.circuit_labels(wiki_dir, model),
    )  # fmt: skip
    mark_hubs(result, model, states)  # a top unit that is a hub's center, and its facts
    planned = explain_plan.plan(  # every unit experiment: the coverage target picks how many
        result, states, model, target=explain_plan.TARGET,
        window=configured_windows().get(model), groups=groups,
    )  # fmt: skip
    args = (runner, ask.prompt, means, result.token_id, ask.position, result.order)
    say("Measuring the prompt coverage, against random units")
    coverage = prompt_coverage(*args)
    say("Ranking the planned experiments by the coverage each gains")
    add_gains(*args, planned, pool=max(POOL, ask.show))  # rank the experiments by coverage gained
    stopped = rank_stopped(*args, planned, wiki_dir, model)  # the blocked units, by gain (#4951)
    target = explain_plan.TARGET if ask.target is None else ask.target
    say("Planning the fewest experiments that reach the target")
    reach = coverage_plan(*args, planned, target)  # the target is causal coverage (#4476)
    dependence = None
    if coverage.get("minimal_units"):  # does the all-unit cut give the gates a baseline (#5967)
        from openinterpretability.answer_review import dependence as measure_dependence

        say("Checking that ablating every unit cuts the answer more than random sets do")
        dependence = measure_dependence(
            runner, means, ask.prompt, result.token_id, ask.position, coverage
        )
    answer = explain_plan.answer_step(
        wiki_dir, model, ask.prompt, ask.position, result.token, coverage, dependence
    )  # how the units make the answer (#4778)
    return SimpleNamespace(result=result, planned=planned, coverage=coverage, groups=groups,
                           reach=reach, answer=answer, stopped=stopped)  # fmt: skip


def _ask_to_spend(found, run, attempts: int, level: str, wiki_dir: Path, model: str,
                  yes: bool) -> None:  # fmt: skip
    """Say what the LLM steps will run and may cost, and ask to go on (#5830).

    Args:
        found: The analysis, whose plan says how many experiments the run may take.
        run: ``--run``: None for no run, 0 for no cap, else the cap.
        attempts: ``--max-attempts``.
        level: ``--level``; a narrated level calls the narrator.
        wiki_dir: The registry, for the estimate.
        model: The model's Hub name.
        yes: ``--yes``.
    """
    from openinterpretability.explain_confirm import NARRATED, confirm, estimate, warning

    experiments = 0
    if run is not None:
        needed = found.reach.get("experiments")
        experiments = len(found.planned.experiments) if needed is None else needed
        experiments = min(experiments, run) if run else experiments
    narrate = level in NARRATED
    confirm(warning(experiments, attempts, narrate, estimate(wiki_dir, model)), yes)


def _answer(runner, token: str | None) -> int | None:
    """The answer token's id, or None for the model's top token."""
    if token is None:
        return None
    ids = runner.tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise click.BadParameter(f"{token!r} is {len(ids)} tokens, not one", param_hint="--token")
    return ids[0]


def _narrate(result, coverage, planned, model: str | None, answer: dict | None = None) -> dict:
    """The narrator's long form and paragraph, the evidence it read, and its checks (#3743)."""
    from openinterpretability.agents.base import DEFAULT_MODEL
    from openinterpretability.agents.narrator import Narrator
    from openinterpretability.core import costs
    from openinterpretability.explain_narrate import check_citations, evidence

    ev = evidence(result, coverage, planned, answer=answer)
    narrator = Narrator(model or DEFAULT_MODEL)
    costs.reset()  # so that the costs below are the narrator's alone
    narrative = narrator.narrate(ev)
    return {
        "evidence": ev, "narrative": narrative.model_dump(), "narrator_model": narrator.model,
        "citations_not_in_evidence": check_citations(narrative, ev),
        "narrator_costs": costs.measured().model_dump(mode="json"),
    }  # fmt: skip


def _keep(record, home: Path, land: bool) -> None:
    """Land ``record`` as its own PR, or with ``--no-land`` write it under ``home`` (#4475).

    The record is written under ``home`` before it lands, so a landing that fails loses
    nothing, the narration included (#5965). Once it lands, the copy goes, so a pull of the
    merged PR meets no untracked file.
    """
    from openinterpretability.experiment_loop import LoopError
    from openinterpretability.explain_land import land_record, write_local

    path = write_local(record, home)
    if not land:
        click.echo(f"Explanation → {path}")
        return
    try:
        url, merged = land_record(record)
    except (LoopError, click.ClickException) as exc:
        raise click.ClickException(
            f"the record didn't land ({exc}). It is kept at {path}: land it later with "
            f"`open-interpretability explain-share {path}`."
        ) from exc
    path.unlink(missing_ok=True)
    click.echo(f"Explanation → {url} ({'merged' if merged else 'left open'})")


def _record(model: str, result, coverage: dict, planned, told: dict, run=None, figures=None,
            target: float | None = None):  # fmt: skip
    """The explanation record (core/explanation_records.py), as the registry stands."""
    from datetime import UTC, datetime

    from openinterpretability import __version__
    from openinterpretability.core.explanation_records import ExplanationRecord, Narration

    units = _explained_units(model, result)
    narration = None
    if told:
        n = told["narrative"]
        narration = Narration(
            sections=n["sections"], paragraph=n["paragraph"],
            narrator_model=told["narrator_model"],
            citations_not_in_evidence=told["citations_not_in_evidence"],
        )  # fmt: skip
    return ExplanationRecord(
        model_name=model, prompt=result.prompt, position=result.position, answer=result.token,
        explained_share=result.verified_share, blocked_share=planned.blocked,
        prompt_coverage=coverage.get("coverage"), random_coverage=coverage.get("random"),
        raw_coverage=coverage.get("raw"),
        units=units, narration=narration, planned_experiments=len(planned.experiments),
        package_version=__version__, written=datetime.now(UTC), run=run, figures=figures,
        target=target, coverage_detail=ExplanationRecord.detail(coverage),
    )  # fmt: skip


def _explained_units(model: str, result) -> list:
    """The record's mechanism: the top units, then the top superpositions and circuits."""
    from openinterpretability.core.explanation_records import ExplainedUnit
    from openinterpretability.explain import HEAD, _slug

    total = result.total or 1.0
    units = []
    for u in result.poles:
        kind = "head" if u.pole == HEAD else "pole"
        units.append(ExplainedUnit(
            unit=u.label, slug=_slug(model, u.key), kind=kind, share=abs(u.attribution) / total,
            attribution=u.attribution, status=u.status, explanation=u.explanation, hub=u.hub,
        ))  # fmt: skip
    for g in result.groups[:10]:
        units.append(ExplainedUnit(
            unit=g.name, kind=g.kind, share=abs(g.attribution) / total,
            attribution=g.attribution, status=g.status, explanation=g.explanation,
        ))  # fmt: skip
    return units


@cli.command(name="explain-share")
@click.argument("record_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def _explain_share(record_file):
    """Share an `explain --local` run: one PR with RECORD_FILE and its local experiments.

    It opens the PR ready, and leaves it for auto-merge.yml or a maintainer. It never waits
    for CI. See docs/experiments.md and #5827.

    \f
    Args:
        record_file: The explanation record that `explain --local` wrote.
    """
    from openinterpretability.explain_share import share

    try:
        url, merged = share(record_file.resolve())
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Shared: {url} ({'merged' if merged else 'open'})")

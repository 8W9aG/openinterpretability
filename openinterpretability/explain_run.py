"""``explain --run``: run an answer's plan through the experiment loop (#4473, #4474).

The run takes the plan's unit and superposition experiments by coverage gain, and runs
them until the explained share reaches the target. When the loop ends, its worktree is
reset to ``origin/main``. :func:`run_plan` returns what the run did, so the command can
explain the answer again, as the registry stands after the run, and record that.
"""

from __future__ import annotations

from pathlib import Path

import click

from openinterpretability.loop_pending import PendingRecords


def run_plan(found, ask, agents: tuple[str, str, str], max_attempts: int, limit: int,
         wiki_dir: Path, measure=(None, None)):  # fmt: skip
    """Run the plan through the experiment loop, and return what it did.

    It returns the run's summary, its exit code and the records of its open PRs
    (:class:`~openinterpretability.loop_pending.PendingRecords`).

    Units and superpositions run by coverage gain until the causal coverage reaches the
    target (#4473, #4476). ``measure`` is the ``(runner, means)`` that the coverage meter
    runs, or ``(None, None)`` to count the share of |attribution| instead. Circuits wait
    for an agent to write their task (``explanation_targets.circuit_notes``). When the loop
    ends, its worktree is reset to ``origin/main``. Its PRs land without waiting for CI, so
    the caller measures with the pending records in place (#5826).
    """
    from openinterpretability.cli_loop import _api_model, _models, run_loop
    from openinterpretability.explanation_targets import (
        CoverageMeter,
        ShareMeter,
        TargetedDraws,
        by_gain,
        circuit_notes,
        targets,
    )
    from openinterpretability.planner import configured_windows

    for model_name in agents[:2]:
        if model_name:
            _api_model(None, None, model_name)  # an external model has nobody to answer it
    proposer, reviewer = _models(*agents)
    result, planned, model = found.result, found.planned, ask.model
    ordered = [*planned.experiments, *by_gain(planned.groups)]  # units in greedy order (#4558)
    chosen = targets(ordered, model, found.groups)
    target = _reachable(found.reach, planned, circuit_notes(ordered))
    click.echo(f"Running {len(chosen)} experiments, by coverage gain, until the coverage "
               f"reaches {target:.0%}: " + ", ".join(t.label for t in chosen))  # fmt: skip
    carrying = {(g.kind, g.name) for g in result.groups}  # the groups that carry the answer
    groups = [g for g in found.groups if (g.kind, g.name) in carrying]
    runner, means = measure
    meter = (ShareMeter(model, result.order, groups, wiki_dir) if runner is None else
             CoverageMeter(runner, means, ask.prompt, result.token_id, ask.position, model,
                           result.order, groups, wiki_dir))  # fmt: skip
    window = configured_windows().get(model)
    draws = TargetedDraws(chosen, reviewer, max_attempts, meter=meter, target=target,
                          limit=limit, window=window)  # fmt: skip
    out: dict = {}
    code = run_loop(proposer, reviewer, draws, out=out)
    loop = out.get("loop")
    ran = _ran(loop, found.coverage.get("coverage") or 0.0, code)
    pending = loop.options.pending if loop is not None else PendingRecords()
    return ran.model_copy(update={"shares": draws.shares}), code, pending


def _ran(loop, before: float, code: int):
    """The run's summary for the record (#4474). Its share after is set once re-measured."""
    from openinterpretability.core.explanation_records import ExplainRun
    from openinterpretability.experiment_loop import reset

    if loop is not None:
        reset()  # the loop's worktree, at origin/main: the merged records are there
    results = loop.results if loop is not None else []
    stopped = (loop.finished if loop is not None else None) or (
        f"the loop stopped with exit code {code}: failures or a rate limit" if code else "done"
    )
    return ExplainRun(
        share_before=before, share_after=before, stopped=stopped,
        landed=[f"{r.slug} {r.verdict} {r.url}" for r in results if r.merged],
        left_open=[r.url for r in results if r.url and not r.merged],
        pending=[f"{r.slug} {r.verdict} {r.url}" for r in results if r.url and not r.merged],
    )  # fmt: skip


def _reachable(reach: dict, planned, circuits: list[str]) -> float:
    """The coverage the run aims at, said with what blocks the rest (#4473, #4476).

    ``reach`` is :func:`~openinterpretability.explain_coverage.coverage_plan`'s estimate.
    The run reaches at most the coverage with every planned unit kept. Blocked units can't
    rise, and circuits wait for an agent.
    """
    click.echo(f"Coverage now: {reach['now']:.1%}. With every planned unit explained: "
               f"{reach['reach']:.1%}. Blocked: {planned.blocked:.1%} of the attribution "
               f"(stopped units, or no follow-up).")  # fmt: skip
    for note in circuits:
        click.echo(f"Not run: circuit {note}")
    if reach["experiments"] is not None:
        return reach["target"]
    click.echo(f"The target {reach['target']:.0%} is out of reach, so the run stops at "
               f"{reach['reach']:.1%}, the most it can explain.")  # fmt: skip
    return reach["reach"]

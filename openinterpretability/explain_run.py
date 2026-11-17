"""``explain --run``: run an answer's plan through the experiment loop (#4473, #4474).

The run takes the plan's unit and superposition experiments by coverage gain, and runs
them until the explained share reaches the target. When the loop ends, its worktree is
reset to ``origin/main``. :func:`run_plan` returns what the run did, so the command can
explain the answer again, as the registry stands after the run, and record that.
"""

from __future__ import annotations

from pathlib import Path

import click


def run_plan(found, ask, agents: tuple[str, str, str], max_attempts: int, limit: int,
         wiki_dir: Path):  # fmt: skip
    """Run the plan through the experiment loop, and return what it did and its exit code.

    Units and superpositions run by coverage gain until the explained share reaches the
    target (#4473). Circuits wait for an agent to write their task
    (:func:`~openinterpretability.explanation_targets.circuit_notes`). When the loop ends,
    its worktree is reset to ``origin/main``, so the records that it landed are there (#4474).
    """
    from openinterpretability.cli_loop import _api_model, _models, run_loop
    from openinterpretability.explanation_targets import (
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
    ordered = by_gain([*planned.experiments, *planned.groups])
    chosen = targets(ordered, model, found.groups)
    target = _reachable(planned, circuit_notes(ordered))
    click.echo(f"Running {len(chosen)} experiments, by coverage gain, until the explained share "
               f"reaches {target:.0%}: " + ", ".join(t.label for t in chosen))  # fmt: skip
    carrying = {(g.kind, g.name) for g in result.groups}  # the groups that carry the answer
    groups = [g for g in found.groups if (g.kind, g.name) in carrying]
    meter = ShareMeter(model, result.order, groups, wiki_dir)
    window = configured_windows().get(model)
    draws = TargetedDraws(chosen, reviewer, max_attempts, meter=meter, target=target,
                          limit=limit, window=window)  # fmt: skip
    out: dict = {}
    code = run_loop(proposer, reviewer, draws, out=out)
    ran = _ran(out.get("loop"), result.verified_share, code)
    return ran.model_copy(update={"shares": draws.shares}), code


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
    )  # fmt: skip


def _reachable(planned, circuits: list[str]) -> float:
    """The target the run aims at, said with what blocks the rest (#4473).

    The run reaches at most the share now plus what the planned unit experiments add.
    Blocked units can't rise, and circuits wait for an agent.
    """
    click.echo(f"Explained now: {planned.explained:.1%}. Blocked: {planned.blocked:.1%} "
               f"(stopped units, or no follow-up). The planned experiments reach "
               f"{planned.reaches:.1%}.")  # fmt: skip
    for note in circuits:
        click.echo(f"Not run: circuit {note}")
    if planned.target <= planned.reaches:
        return planned.target
    click.echo(f"The target {planned.target:.0%} is out of reach, so the run stops at "
               f"{planned.reaches:.1%}, the most it can explain.")  # fmt: skip
    return planned.reaches

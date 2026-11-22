"""One experiment of ``explain --run`` in a headless Claude Code session (#5831).

With no LiteLLM model and ``claude`` on ``PATH`` (:func:`.explain_backend.backend`), each
draw of the run is done by one ``claude -p`` session. The loop still draws, by coverage
gain, and still measures the coverage in-process between sessions. The session gets the
draw's steps, with the proposer and reviewer as Claude Code subagents, and the
land-experiment rules. It ends its reply with a ``RESULT:`` line, which becomes the
iteration's :class:`~openinterpretability.experiment_loop.Result`. As for any landing, it
never waits for CI (#5826).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from openinterpretability import experiment_loop as el
from openinterpretability.claude_sessions import Session, result_line, session_text

SUBAGENTS = "external/claude-code"  # the proposer and reviewer: the session's subagents


@dataclass(frozen=True)
class SessionRun:
    """The session that runs each draw, and the folder for its logs, results and ledgers."""

    session: Session
    log_dir: Path


def session_prompt(slug: str, kind: str, steps_text: str) -> str:
    """What one session does: this draw's steps, landed without waiting for CI.

    Args:
        slug: The unit's slug.
        kind: The draw's kind, for example ``explore``.
        steps_text: The draw's steps (:func:`.draw_steps.render`).

    Returns:
        The prompt.

    Examples:
        >>> "RESULT: <slug>" in session_prompt("m/L1/N2-positive", "explore", "1. Claim")
        True
    """
    return f"""Run exactly one experiment end to end and post it, following the land-experiment
skill (which builds on choose-next-experiment), then stop. You are in an unattended run of
`open-interpretability explain --run`: don't ask questions, and note any call you make in
the PR description.

- The draw is fixed: a {kind} draw on {slug}. Don't run `open-interpretability next`. Run
  these steps, in order:

{steps_text}

- The proposer and the reviewer are Claude Code subagents: use the neuron-proposer,
  neuron-reviewer and explanation-judge subagents, as the steps say. Record the model IDs
  that they report.
- Post the result whatever the verdict. Then mark the PR ready and stop. Don't wait for CI
  and don't merge: auto-merge.yml merges it once CI passes (#5826).
- Finish on a clean working tree. End your reply with one line:
  RESULT: <slug> <verdict> <PR url> <merged|open>   (or RESULT: none <reason>)
"""


def session_iteration(options: el.Options, plan_next, claims) -> el.Result:
    """Draw as :func:`.experiment_loop.iteration` does, and run the draw in one session.

    Args:
        options: The loop's options, whose ``sessions`` says how to start a session.
        plan_next: Draws the next experiment.
        claims: What open PRs claim.

    Returns:
        The result that the session's ``RESULT:`` line gives.

    Raises:
        el.LoopError: The draw names no unit, ``claude`` failed, or the session ran nothing.
    """
    from openinterpretability.draw_steps import render

    el.reset()
    with options.pending.applied():  # the open PRs' records count too (#5826)
        plan = el.draw(options, plan_next, claims)
    slug = el.draw_slug(plan.choice)
    if slug is None:
        raise el.LoopError(
            f"the unit of a {plan.choice.kind} draw on {plan.choice.model} is unknown"
        )
    corpus = options.corpus or "corpus.jsonl"
    # None for each agent: the steps name the subagents that answer each prompt.
    steps = el.steps(plan, corpus, None, None, options.corpus is None)
    el.say(f"drew {plan.choice.kind} {slug} on {plan.choice.model}: a Claude Code session runs it")
    run = options.sessions
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")  # one per session, however quick
    run.log_dir.mkdir(parents=True, exist_ok=True)
    result, log = run.log_dir / f"{stamp}.json", run.log_dir / f"{stamp}.log"
    prompt = session_prompt(slug, plan.choice.kind, render(steps))
    code = run.session.run(prompt, Path.cwd(), result, log, run.log_dir / f"{stamp}.costs.jsonl")
    if code:
        raise el.LoopError(f"claude exited {code} (a usage limit?); see {log}")
    return _result(result_line(session_text(result)), log, options)


def _result(line: str | None, log: Path, options: el.Options) -> el.Result:
    """The iteration's result from the session's ``RESULT:`` line; its PR's records kept.

    Args:
        line: The line, without its prefix, or None.
        log: The session's log, for the error.
        options: The loop's options, whose pending records keep the PR's files.

    Returns:
        The result.

    Raises:
        el.LoopError: The session gave no line, or ran no experiment.
    """
    parts = (line or "").split()
    if len(parts) < 4 or parts[0] == "none":
        raise el.LoopError(f"the session ran no experiment ({line or 'no RESULT line'}); see {log}")
    slug, verdict, url, state = parts[:4]
    options.pending.keep()  # the session's branch is checked out: keep what its PR adds
    return el.Result(slug, verdict, url, state == "merged")

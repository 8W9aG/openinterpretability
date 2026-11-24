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

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from openinterpretability import experiment_loop as el
from openinterpretability.claude_sessions import Session, result_line, session_text

SUBAGENTS = "external/claude-code"  # the proposer and reviewer: the session's subagents
# A prefix rule in ALLOWED_TOOLS matches only a command that starts with it, and an
# unattended session denies the rest (#6145).
COMMANDS = """- Run each step's command as given, one command per Bash call. Don't add `cd`,
  `&&`, `;`, pipes or `env` prefixes: you already run in the worktree. An unattended run
  denies a command of any other shape, and the run stops (#6145)."""


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
{COMMANDS}
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
    started = datetime.now(UTC)
    code = run.session.run(prompt, Path.cwd(), result, log, run.log_dir / f"{stamp}.costs.jsonl")
    if code:
        raise el.LoopError(f"claude exited {code} (a usage limit?); see {log}")
    return _result(result_line(session_text(result)), log, options, slug, started)


class NothingRan(el.LoopError):
    """A session ran no experiment, so the run gives back the draw's attempt (#6066)."""

    def __init__(self, message: str, slug: str) -> None:
        """Keep the reason, and the drawn unit whose attempt the run gives back.

        Args:
            message: Why the session counts as having run nothing.
            slug: The drawn unit's slug.
        """
        super().__init__(message)
        self.nothing_ran = slug


def landed_pr(slug: str, since: datetime, prs=None) -> tuple[str, bool] | None:
    """A PR on ``slug``'s unit opened since ``since``, when a session printed no result line.

    A session can land its record and still end without a ``RESULT:`` line (#6066, #6069).

    Args:
        slug: The unit's slug.
        since: When the session started.
        prs: ``gh pr list`` rows, for a test; read from GitHub when None.

    Returns:
        The PR's URL and whether it merged, or None.

    Examples:
        >>> rows = [{"headRefName": "experiment/m-L1-N2-positive-revise-1124010203",
        ...          "createdAt": "2026-11-24T01:02:30Z", "url": "u", "state": "MERGED"}]
        >>> start = datetime(2026, 11, 24, 1, 2, tzinfo=UTC)
        >>> landed_pr("m/L1/N2-positive", start, rows), landed_pr("m/L1/N3-positive", start, rows)
        (('u', True), None)
    """
    from openinterpretability.planner import claim_key

    if prs is None:
        listed = ["gh", "pr", "list", "--state", "all", "--limit", "100",
                  "--json", "headRefName,createdAt,url,state"]  # fmt: skip
        done = subprocess.run(listed, capture_output=True, text=True, check=False)
        prs = json.loads(done.stdout or "[]") if done.returncode == 0 else []
    head = f"experiment/{claim_key(slug)}"
    for pr in prs:
        opened = datetime.fromisoformat(pr["createdAt"])  # Python 3.11 reads the Z
        if pr["headRefName"].startswith(head) and opened >= since:
            return pr["url"], pr["state"] == "MERGED"
    return None


def _git(*args: str) -> str:
    """The output of one ``git`` command, or "" when it fails.

    Args:
        *args: The command's arguments, after ``git``.

    Returns:
        Its output.
    """
    done = subprocess.run(["git", *args], capture_output=True, text=True)  # a failure is ""
    return done.stdout if done.returncode == 0 else ""


def stranded_branch(slug: str, since: datetime, git=_git) -> tuple[str, bool] | None:
    """A branch on ``slug``'s unit that a session pushed since ``since`` with no PR (#6145).

    Args:
        slug: The unit's slug.
        since: When the session started.
        git: Runs one ``git`` command and returns its output.

    Returns:
        The branch, and whether it adds a record, or None.

    Examples:
        >>> out = {"ls-remote": "abc refs/heads/experiment/m-L1-N2-positive",
        ...        "log": "2026-11-24T01:05:00+00:00", "diff": "wiki/experiments/r.json"}
        >>> git = lambda *args: out.get(args[0], "")
        >>> stranded_branch("m/L1/N2-positive", datetime(2026, 11, 24, 1, 0, tzinfo=UTC), git)
        ('experiment/m-L1-N2-positive', True)
    """
    from openinterpretability.planner import claim_key

    listed = git("ls-remote", "--heads", "origin", f"refs/heads/experiment/{claim_key(slug)}*")
    for row in listed.splitlines():
        branch = row.split()[-1].removeprefix("refs/heads/")
        git("fetch", "--quiet", "origin", branch)
        stamp = git("log", "-1", "--format=%cI", "FETCH_HEAD").strip()
        if not stamp or datetime.fromisoformat(stamp) < since:
            continue  # an older branch: not this session's
        added = git("diff", "--name-only", "--diff-filter=A", "origin/main...FETCH_HEAD").split()
        record = any(p.startswith("wiki/experiments/") and p.endswith(".json") for p in added)
        return branch, record
    return None


def _nothing_landed(slug: str, since: datetime | None, why: str, log: Path, strand) -> None:
    """Raise for a session that opened no PR, naming any branch that it left (#6145).

    Args:
        slug: The drawn unit.
        since: When the session started.
        why: What the session printed.
        log: The session's log.
        strand: Finds a branch that it pushed with no PR (:func:`stranded_branch`).

    Raises:
        el.LoopError: It pushed a record with no PR. The experiment ran, so the attempt counts.
        NothingRan: It ran nothing. A branch with only its claim is named, to delete.
    """
    left = strand(slug, since) if slug and since else None
    if left and left[1]:
        raise el.LoopError(f"the session pushed a record on {left[0]} but opened no PR: "
                           f"open it with `gh pr create --head {left[0]}`; see {log}")  # fmt: skip
    note = (
        f"; it left {left[0]} with only its claim: `git push origin --delete {left[0]}`"
        if left
        else ""
    )
    raise NothingRan(f"the session ran no experiment ({why}){note}; see {log}", slug)


def _result(line: str | None, log: Path, options: el.Options, slug: str,
            since: datetime | None, find=landed_pr,
            strand=stranded_branch) -> el.Result:  # fmt: skip
    """The iteration's result from the session's ``RESULT:`` line; its PR's records kept.

    Args:
        line: The line, without its prefix, or None.
        log: The session's log, for the error.
        options: The loop's options, whose pending records keep the PR's files.
        slug: The drawn unit, to find a PR that the session landed without a line.
        since: When the session started.
        find: Finds that PR (:func:`landed_pr`).
        strand: Finds a branch that the session pushed with no PR (:func:`stranded_branch`).

    Returns:
        The result.

    Raises:
        NothingRan: The session gave no line, or ran no experiment, and opened no PR.
        el.LoopError: The session pushed a record but opened no PR.
    """
    parts = (line or "").split()  # pragma: no mutate, equivalent: any one word is under 4 parts
    if len(parts) < 4 or parts[0] == "none":
        found = find(slug, since) if slug and since else None
        if found is None:
            _nothing_landed(slug, since, line or "no RESULT line", log, strand)
        options.pending.keep()
        return el.Result(slug, "unknown", found[0], found[1])
    slug, verdict, url, state = parts[:4]
    options.pending.keep()  # the session's branch is checked out: keep what its PR adds
    return el.Result(slug, verdict, url, state == "merged")

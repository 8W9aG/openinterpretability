"""``open-interpretability loop``: run experiments back to back with API agents (issue #3246).

Each iteration resets a detached worktree to ``origin/main`` and draws with the planner.
It claims the draw as ``choose-next-experiment`` does, then runs the draw's steps
(``draw_steps.steps``, the commands ``open-interpretability next --steps`` prints). Then it
posts the result, and lands it as the ``land-experiment`` skill does.
``open-interpretability`` commands run in this process, so a model loads once
(``runner_cache``). git and gh run as subprocesses.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from openinterpretability.core.paths import DEFAULT_WIKI_DIR, draws_dir, experiments_dir, stops_dir
from openinterpretability.core.record_kinds import reproducible
from openinterpretability.draw_steps import REVIEW_KINDS, VERBS, draw_slug, steps
from openinterpretability.loop_pending import PendingRecords
from openinterpretability.planner import claim_key
from openinterpretability.until_verified import decide, keep_going, next_turn

MERGE_PERMISSIONS = ("ADMIN", "MAINTAIN", "WRITE")
OPENINTERPRETABILITY = ("uv", "run", "open-interpretability")


class LoopError(Exception):
    """One iteration failed. Its message is the reason that the loop gives."""


AGENT_ONLY_KINDS = ("circuit",)  # draws whose steps need an agent to write a file
AGENT_ONLY_REDRAWS = 5  # draws before the loop gives up on finding one it can run


class Finished(Exception):
    """The draws have nothing left to run: the loop ends cleanly. The message says why."""


@dataclass(frozen=True)
class Options:
    """What every iteration of a loop shares."""

    proposer: str
    reviewer: str
    epsilon: float = 0.7
    corpus: str | None = None  # None: each new scan builds a fresh corpus from its seed
    until_verified: bool | None = False  # None: a seeded chance per draw (until_verified.py)
    max_turns: int = 3  # an until-verified draw's turns, at most (until_verified.MAX_TURNS)
    # The records of the loop's open PRs, which draws and measures count (#5826).
    pending: PendingRecords = field(default_factory=PendingRecords)
    # explain --local: run in the user's checkout, with no claim, push or PR (#5827).
    local: bool = False


@dataclass(frozen=True)
class Result:
    """The outcome of one iteration, as the RESULT line reports it."""

    slug: str | None
    verdict: str
    url: str | None = None
    merged: bool = False
    record: str | None = None  # a local run's record, relative to the checkout (#5827)

    def line(self) -> str:
        """``RESULT: <slug> <verdict> <PR url> <merged|open>``, or ``RESULT: none <why>``.

        A local run's line is ``RESULT: <slug> <verdict> local <record>``.
        """
        if self.slug is None:
            return f"RESULT: none {self.verdict}"
        if self.record is not None:
            return f"RESULT: {self.slug} {self.verdict} local {self.record}"
        return (
            f"RESULT: {self.slug} {self.verdict} {self.url} {'merged' if self.merged else 'open'}"
        )


def sh(args: list[str], check: bool = True, timeout: float | None = None) -> str:
    """Run ``args`` in the current directory and return its stdout; echo both streams."""
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise LoopError(f"`{' '.join(args[:3])}` timed out") from exc
    sys.stdout.write(f"$ {shlex.join(args)}\n{done.stdout}{done.stderr}")
    if check and done.returncode != 0:
        tail = (done.stderr or done.stdout).strip().splitlines()[-1:] or ["no output"]
        raise LoopError(f"`{' '.join(args[:3])}` failed: {tail[0]}")
    return done.stdout.strip()


def openinterpretability(args: list[str]) -> None:
    """Run one ``open-interpretability`` command in this process.

    A non-zero exit raises LoopError.
    """
    from openinterpretability.cli import cli

    sys.stdout.write(f"$ openinterpretability {shlex.join(args)}\n")
    try:
        cli.main(args=args, prog_name="open-interpretability", standalone_mode=False)
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise LoopError(f"`openinterpretability {args[0]}` exited {exc.code}") from exc


def run_command(command: str) -> None:
    """Run one step's command: ``uv run open-interpretability …`` in-process, others in a shell."""
    args = shlex.split(command)
    if tuple(args[:3]) == OPENINTERPRETABILITY:
        openinterpretability(args[3:])
    else:
        sh(args)


def reset() -> None:
    """Detach the current worktree at a fresh ``origin/main`` and drop old experiment branches."""
    sh(["git", "fetch", "--quiet", "origin", "main"])
    sh(["git", "checkout", "--quiet", "--force", "--detach", "origin/main"])
    sh(["git", "clean", "-fdq"])
    branches = sh(["git", "for-each-ref", "--format=%(refname:short)", "refs/heads/experiment/"])
    for branch in branches.split():
        sh(["git", "branch", "-D", branch])


def proposer_of(record_path: str | None) -> str | None:
    """The proposer model of the record at ``record_path``, if there is one."""
    from openinterpretability.core.schemas import AuditRecord

    if not record_path or not Path(record_path).exists():
        return None
    record = AuditRecord.model_validate_json(Path(record_path).read_text())
    return record.submission.submitting_agent_model


def draw(options: Options, plan_next, claims):
    """Draw with the planner. A review the reviewer can't do independently is drawn again once.

    ``plan_next`` and ``claims`` are passed in so that tests can replace them.
    """
    for _ in range(AGENT_ONLY_REDRAWS):
        plan = plan_next(DEFAULT_WIKI_DIR, epsilon=options.epsilon, claims=claims())
        c = plan.choice
        if c.kind in AGENT_ONLY_KINDS:
            continue  # its task is written by an agent, which this loop doesn't have (#4286)
        if c.kind not in REVIEW_KINDS or proposer_of(c.record) != options.reviewer:
            return plan
    raise LoopError(f"no draw that {options.reviewer} can run, in {AGENT_ONLY_REDRAWS} tries")


def pr_url() -> str:
    """The URL of the pull request for the branch that is checked out."""
    return sh(["gh", "pr", "view", "--json", "url", "-q", ".url"])


def new_files(prefix: Path) -> list[Path]:
    """Files under ``prefix`` that git does not track yet."""
    out = sh(["git", "ls-files", "--others", "--exclude-standard", str(prefix)])
    return [Path(line) for line in out.splitlines() if line]


def verdict_of(kind: str, key: str, new=None) -> tuple[str, str]:
    """The new record's verdict and the commit message that posts it.

    A scan that stopped on its unit writes a draw or stop record and no experiment, so
    its verdict is STOPPED. ``new`` gives the new files under a folder: by default
    :func:`new_files`, which asks git. A local run passes its own (#5827).
    """
    files = new or new_files
    from openinterpretability.core.explanation_records import EXPLANATIONS_WIKI
    from openinterpretability.core.schemas import AuditRecord

    if kind == "circuit":  # a circuit draw writes a circuit record (core/circuit_records.py)
        from openinterpretability.core.circuit_records import CIRCUITS_WIKI, CircuitRecord

        found = [p for p in files(DEFAULT_WIKI_DIR / CIRCUITS_WIKI) if p.suffix == ".json"]
        if not found:
            raise LoopError("the draw's steps wrote no circuit record")
        status = CircuitRecord.model_validate_json(Path(found[0]).read_text()).verdict.status
        return status, f"{VERBS[kind]} {key} ({status})"
    if kind == "explanation":  # an explanation draw writes an explanation record, no verdict
        if not files(DEFAULT_WIKI_DIR / EXPLANATIONS_WIKI):
            raise LoopError("the draw's steps wrote no explanation")
        return "EXPLAINED", f"{VERBS[kind]} {key}"
    records = [p for p in files(experiments_dir(DEFAULT_WIKI_DIR)) if p.suffix == ".json"]
    if not records:
        if files(draws_dir(DEFAULT_WIKI_DIR)) or files(stops_dir(DEFAULT_WIKI_DIR)):
            return "STOPPED", f"{VERBS[kind]} {key} (STOPPED)"
        raise LoopError("the draw's steps wrote no record")
    verdict = AuditRecord.model_validate_json(records[0].read_text()).verdict
    accuracy = verdict.metrics.balanced_accuracy if verdict.metrics else None
    shown = "n/a" if accuracy is None else f"{accuracy:.2f}"
    return verdict.status, f"{VERBS[kind]} {key} ({verdict.status}, {shown})"


def post(kind: str, key: str) -> str:
    """Write the pages, check the registry, commit ``wiki/``, push and mark the PR ready."""
    openinterpretability(["wiki"])
    openinterpretability(["check", "--base-ref", "origin/main"])
    status, message = verdict_of(kind, key)
    sh(["git", "add", str(DEFAULT_WIKI_DIR)])
    sh(["git", "commit", "--quiet", "-m", message])
    sh(["git", "push", "--quiet"])
    sh(["gh", "pr", "ready"])
    return status


def merge_blockers(view: dict, permission: str) -> list[str]:
    """Why the auto-merge workflow won't merge the PR described by ``view``, if at all.

    ``view`` is ``gh pr view --json files``. The rules are ``auto-merge.yml``'s: the
    maintainer's PR may change anything in ``wiki/``, and anyone else's only records that CI
    reproduces. A token below ``WRITE`` means that CI waits for a maintainer's approval.
    """
    files = [f["path"] for f in view.get("files", [])]
    wiki = f"{DEFAULT_WIKI_DIR}/"
    reasons = [f"it changes {p}, outside {wiki}" for p in files if not p.startswith(wiki)]
    if permission != "ADMIN":  # anyone but the maintainer: reproducible records only (#4792)
        reasons += [f"{p} is not a record that CI reproduces" for p in files
                    if p.startswith(wiki) and not reproducible(p)]  # fmt: skip
    if permission not in MERGE_PERMISSIONS:
        reasons.append(f"CI needs a maintainer's approval (the token's permission is "
                       f"{permission or 'unknown'})")  # fmt: skip
    return reasons


_CONSOLE: list = []  # the loop's own stdout, while an iteration writes to its log


def say(text: str) -> None:
    """Print one progress line of the iteration, at once (#3142).

    Every stage prints one, so a slow iteration doesn't look like a stuck one. The line
    goes to the iteration's log and to the console that runs the loop. The ``RESULT``
    line at the end is separate, and its form is unchanged.
    """
    line = f"loop: {text}"
    print(line, flush=True)
    for console in _CONSOLE:
        if console is not sys.stdout:
            print(line, file=console, flush=True)


def land(url: str) -> bool:
    """Hand the PR over to whoever merges it, without waiting for CI (#5826). True if merged.

    The loop never waits for the checks and never merges. ``auto-merge.yml`` merges a PR
    that it qualifies once CI passes, and a maintainer takes any other. A PR that is still
    a draft is marked ready, so the workflow sees it.
    """
    import json

    view = json.loads(sh(["gh", "pr", "view", url, "--json", "state,isDraft,files"]))
    if view.get("state") == "MERGED":  # the auto-merge workflow got there first
        return True
    if view.get("isDraft"):
        sh(["gh", "pr", "ready", url])
    permission = sh(["gh", "repo", "view", "--json", "viewerPermission", "-q", ".viewerPermission"])
    if reasons := merge_blockers(view, permission):
        sh(["gh", "pr", "comment", url, "--body",
            "Left for a maintainer: " + "; ".join(reasons) + "."])  # fmt: skip
        say("left for a maintainer: " + "; ".join(reasons))
    else:
        say("left for auto-merge, once CI passes")
    return False


def delete_branch() -> None:
    """Delete the checked-out branch on origin, so that a later draw can claim it again."""
    branch = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    sh(["git", "push", "--quiet", "origin", "--delete", branch], check=False)


def iteration(options: Options, plan_next, claims) -> Result:
    """Run one experiment: reset, draw, claim, run, post and land, or a local one (#5827)."""
    if options.local:
        from openinterpretability.loop_local import local_iteration

        return local_iteration(options, plan_next, claims)
    reset()
    with options.pending.applied():  # the open PRs' records count too (#5826)
        plan = draw(options, plan_next, claims)
    slug = draw_slug(plan.choice)
    if slug is None:
        raise LoopError(f"the unit of a {plan.choice.kind} draw on {plan.choice.model} is unknown")
    corpus = options.corpus or "corpus.jsonl"
    claim, *work, _ = steps(
        plan, corpus, options.proposer, options.reviewer, options.corpus is None
    )
    say(f"drew {plan.choice.kind} {slug} on {plan.choice.model}")
    for command in claim.commands:
        run_command(command)
    url = pr_url()
    say(f"claimed: {url}")
    try:
        status = run_work(work, plan.choice.kind, claim_key(slug))
        if decide(plan.seed, options.until_verified):
            status = more_turns(options, plan.choice.kind, slug, status)
    except Exception as exc:
        note = f"Abandoned by `open-interpretability loop`: {exc}"
        sh(["gh", "pr", "close", url, "--comment", note], check=False)
        delete_branch()
        raise
    merged = land(url)
    options.pending.keep()
    return Result(slug, status, url, merged)


def more_turns(options: Options, kind: str, slug: str, status: str) -> str:
    """Run an until-verified draw's later turns on its branch (#4485); return the last verdict.

    Each turn is the unit's superseding follow-up, which builds on the turn before. The
    sequence ends on a terminal verdict, on a unit with no such follow-up, or after
    ``options.max_turns``. The PR lands once, after the last turn.
    """
    turn = 1
    while keep_going(kind, status, turn, options.max_turns):
        plan = next_turn(slug, DEFAULT_WIKI_DIR)
        if plan is None:
            say(f"no superseding follow-up for {slug}: the draw ends at turn {turn}")
            break
        turn, kind = turn + 1, plan.choice.kind
        say(f"turn {turn}/{options.max_turns}: {kind} after {status}")
        corpus = options.corpus or "corpus.jsonl"
        _, *work, _ = steps(plan, corpus, options.proposer, options.reviewer,
                            options.corpus is None)  # fmt: skip
        status = run_work(work, kind, claim_key(slug))
    return status


def run_steps(work) -> None:
    """Run every step after the claim. A scan that stops on its unit ends the steps there.

    Args:
        work: The draw's steps, from its first work step to its last.
    """
    from openinterpretability.evidence import Unauditable

    try:
        for i, step in enumerate(work, 1):
            say(f"step {i}/{len(work)}: {step.title}")
            for command in step.commands:
                run_command(command)
    except Unauditable as exc:
        sys.stdout.write(f"The scan stopped on the unit: {exc}\n")


def run_work(work, kind: str, key: str) -> str:
    """Run every step after the claim, then post. A scan that stops on its unit posts at once."""
    run_steps(work)
    say("posting the result")
    return post(kind, key)


BACKOFF_MINUTES = (1, 5, 15)  # waits after a provider's rate or usage limit; then stop
_LIMIT_WORDS = ("rate limit", "ratelimit", "usage limit", "429", "overloaded", "quota")


def is_rate_limit(exc: BaseException) -> bool:
    """True if ``exc``, or an exception it was raised from, is a provider's rate or usage limit."""
    seen: BaseException | None = exc
    while seen is not None:
        text = f"{type(seen).__name__} {seen}".lower()
        if any(word in text for word in _LIMIT_WORDS):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def stop_file() -> Path:
    """The kill switch that ``scripts/experiment-loop.sh`` also honours."""
    import os

    from openinterpretability.core.legacy_names import cache_base

    return Path(os.environ.get("OPENINTERPRETABILITY_STOP_FILE") or cache_base() / "stop")


@dataclass
class Loop:
    """The loop's counters, and the pieces that tests replace."""

    options: Options
    log_dir: Path
    max_experiments: int = 0  # 0: until stopped
    max_failures: int = 3
    plan_next: object = None
    claims: object = None
    sleep: object = None
    done: int = 0
    failures: int = 0
    backoffs: int = 0
    results: list = field(default_factory=list)  # each iteration's Result, for the caller
    finished: str | None = None  # why the draws ended, when they did (Finished)

    def run(self) -> int:
        """Run until the limit, the stop file or too many failures; return the exit code."""
        while not self.max_experiments or self.done < self.max_experiments:
            if stop_file().exists():
                print(f"Stopping: {stop_file()} exists.")
                return 0
            self._wait_while_paused()
            if (code := self._one()) is not None:
                return code
        return 0

    def _one(self) -> int | None:
        """Run one iteration and count it. Return an exit code when the loop must stop."""
        import contextlib
        from datetime import UTC, datetime

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log = self.log_dir / f"{stamp}.log"
        _CONSOLE[:] = [sys.stdout]
        try:
            with (
                open(log, "w") as out,
                contextlib.redirect_stdout(out),
                contextlib.redirect_stderr(out),
            ):
                result = iteration(self.options, self.plan_next, self.claims)
        except Finished as exc:  # a targeted run (explain --run) has no target left
            self.finished = str(exc)
            print(f"RESULT: none finished: {exc}", flush=True)
            return 0
        except Exception as exc:  # any failure of one experiment; the loop decides what next
            return self._failed(exc, log)
        finally:
            _CONSOLE.clear()
        self.done, self.failures, self.backoffs = self.done + 1, 0, 0
        self.results.append(result)
        print(result.line(), flush=True)
        return None

    def _failed(self, exc: Exception, log: Path) -> int | None:
        """Back off on a rate limit, count any other failure, and say when to stop."""
        if is_rate_limit(exc):
            if self.backoffs == len(BACKOFF_MINUTES):
                print(f"RESULT: none rate limited after {self.backoffs} waits ({log})")
                return 1
            minutes = BACKOFF_MINUTES[self.backoffs]
            self.backoffs += 1
            print(f"Rate limited; waiting {minutes} min ({log})", flush=True)
            self.sleep(minutes * 60)
            return None
        self.failures += 1
        print(f"RESULT: none {exc} ({log})", flush=True)
        return 1 if self.failures >= self.max_failures else None

    def _wait_while_paused(self) -> None:
        """Wait, a minute at a time, while ``open-interpretability pause`` is in force."""
        from openinterpretability.core import pause

        said = False
        while (paused := pause.read()) is not None and not stop_file().exists():
            if not said:
                print(paused.describe(), flush=True)
                said = True
            self.sleep(60)

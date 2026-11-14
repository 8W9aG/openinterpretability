"""``openinterp loop``: run experiments back to back with API agents (issue #3246).

Each iteration resets a detached worktree to ``origin/main`` and draws with the planner.
It claims the draw as ``choose-next-experiment`` does, then runs the draw's steps
(``draw_steps.steps``, the commands ``openinterp next --steps`` prints). Then it posts the
result, and lands it as the ``land-experiment`` skill does. ``openinterp`` commands run in
this process, so a model loads once (``runner_cache``). git and gh run as subprocesses.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from openinterp.core.paths import DEFAULT_WIKI_DIR, draws_dir, experiments_dir, stops_dir
from openinterp.draw_steps import REVIEW_KINDS, VERBS, draw_slug, steps
from openinterp.planner import claim_key

MERGE_PERMISSIONS = ("ADMIN", "MAINTAIN", "WRITE")
CHECKS_TIMEOUT = 45 * 60  # seconds; the land-experiment skill gives up after 45 minutes
OPENINTERP = ("uv", "run", "openinterp")


class LoopError(Exception):
    """One iteration failed. Its message is the reason that the loop gives."""


@dataclass(frozen=True)
class Options:
    """What every iteration of a loop shares."""

    proposer: str
    reviewer: str
    epsilon: float = 0.7
    corpus: str | None = None  # None: each new scan builds a fresh corpus from its seed


@dataclass(frozen=True)
class Result:
    """The outcome of one iteration, as the RESULT line reports it."""

    slug: str | None
    verdict: str
    url: str | None = None
    merged: bool = False

    def line(self) -> str:
        """``RESULT: <slug> <verdict> <PR url> <merged|open>``, or ``RESULT: none <why>``."""
        if self.slug is None:
            return f"RESULT: none {self.verdict}"
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


def openinterp(args: list[str]) -> None:
    """Run one ``openinterp`` command in this process. A non-zero exit raises LoopError."""
    from openinterp.cli import cli

    sys.stdout.write(f"$ openinterp {shlex.join(args)}\n")
    try:
        cli.main(args=args, prog_name="openinterp", standalone_mode=False)
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise LoopError(f"`openinterp {args[0]}` exited {exc.code}") from exc


def run_command(command: str) -> None:
    """Run one command of a step: ``uv run openinterp …`` in-process, anything else in a shell."""
    args = shlex.split(command)
    if tuple(args[:3]) == OPENINTERP:
        openinterp(args[3:])
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
    from openinterp.core.schemas import AuditRecord

    if not record_path or not Path(record_path).exists():
        return None
    record = AuditRecord.model_validate_json(Path(record_path).read_text())
    return record.submission.submitting_agent_model


def draw(options: Options, plan_next, claims):
    """Draw with the planner. A review the reviewer can't do independently is drawn again once.

    ``plan_next`` and ``claims`` are passed in so that tests can replace them.
    """
    for _ in range(2):
        plan = plan_next(DEFAULT_WIKI_DIR, epsilon=options.epsilon, claims=claims())
        c = plan.choice
        if c.kind not in REVIEW_KINDS or proposer_of(c.record) != options.reviewer:
            return plan
    raise LoopError(f"twice drew a review of {options.reviewer}'s own record")


def pr_url() -> str:
    """The URL of the pull request for the branch that is checked out."""
    return sh(["gh", "pr", "view", "--json", "url", "-q", ".url"])


def new_files(prefix: Path) -> list[Path]:
    """Files under ``prefix`` that git does not track yet."""
    out = sh(["git", "ls-files", "--others", "--exclude-standard", str(prefix)])
    return [Path(line) for line in out.splitlines() if line]


def verdict_of(kind: str, key: str) -> tuple[str, str]:
    """The new record's verdict and the commit message that posts it.

    A scan that stopped on its unit writes a draw or stop record and no experiment, so
    its verdict is STOPPED.
    """
    from openinterp.core.schemas import AuditRecord

    records = [p for p in new_files(experiments_dir(DEFAULT_WIKI_DIR)) if p.suffix == ".json"]
    if not records:
        if new_files(draws_dir(DEFAULT_WIKI_DIR)) or new_files(stops_dir(DEFAULT_WIKI_DIR)):
            return "STOPPED", f"{VERBS[kind]} {key} (STOPPED)"
        raise LoopError("the draw's steps wrote no record")
    verdict = AuditRecord.model_validate_json(records[0].read_text()).verdict
    accuracy = verdict.metrics.balanced_accuracy if verdict.metrics else None
    shown = "n/a" if accuracy is None else f"{accuracy:.2f}"
    return verdict.status, f"{VERBS[kind]} {key} ({verdict.status}, {shown})"


def post(kind: str, key: str) -> str:
    """Write the pages, check the registry, commit ``wiki/``, push and mark the PR ready."""
    openinterp(["wiki"])
    openinterp(["check", "--base-ref", "origin/main"])
    status, message = verdict_of(kind, key)
    sh(["git", "add", str(DEFAULT_WIKI_DIR)])
    sh(["git", "commit", "--quiet", "-m", message])
    sh(["git", "push", "--quiet"])
    sh(["gh", "pr", "ready"])
    return status


def merge_blockers(view: dict, permission: str) -> list[str]:
    """Why the PR described by ``view`` (``gh pr view --json``) may not be merged, if at all."""
    files = [f["path"] for f in view.get("files", [])]
    wiki = f"{DEFAULT_WIKI_DIR}/"
    reasons = [f"it changes {p}, outside {wiki}" for p in files if not p.startswith(wiki)]
    if permission not in MERGE_PERMISSIONS:
        reasons.append(f"the token's permission is {permission or 'unknown'}")
    if view.get("isDraft"):
        reasons.append("it is a draft")
    if view.get("mergeable") != "MERGEABLE":
        reasons.append(f"GitHub says it is {view.get('mergeable', 'not mergeable')}")
    return reasons


def land(url: str) -> bool:
    """Wait for the checks, then merge the PR if every merge condition holds. True if merged."""
    import json

    try:
        sh(
            ["gh", "pr", "checks", url, "--watch", "--fail-fast", "--interval", "30"],
            timeout=CHECKS_TIMEOUT,
        )
    except LoopError as exc:
        sh(["gh", "pr", "comment", url, "--body", f"Left open: the checks didn't pass ({exc})."])
        return False
    fields = "state,isDraft,mergeable,files,headRefOid"
    view = json.loads(sh(["gh", "pr", "view", url, "--json", fields]))
    if view.get("state") == "MERGED":  # the auto-merge workflow got there first
        return True
    permission = sh(["gh", "repo", "view", "--json", "viewerPermission", "-q", ".viewerPermission"])
    if reasons := merge_blockers(view, permission):
        sh(["gh", "pr", "comment", url, "--body", "Left open: " + "; ".join(reasons) + "."])
        return False
    sh(["gh", "pr", "merge", url, "--squash", "--match-head-commit", view["headRefOid"]])
    delete_branch()
    return True


def delete_branch() -> None:
    """Delete the checked-out branch on origin, so that a later draw can claim it again."""
    branch = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    sh(["git", "push", "--quiet", "origin", "--delete", branch], check=False)


def iteration(options: Options, plan_next, claims) -> Result:
    """Run one experiment end to end: reset, draw, claim, run, post and land."""
    reset()
    plan = draw(options, plan_next, claims)
    slug = draw_slug(plan.choice)
    if slug is None:
        raise LoopError(f"the unit of a {plan.choice.kind} draw on {plan.choice.model} is unknown")
    corpus = options.corpus or "corpus.jsonl"
    claim, *work, _ = steps(
        plan, corpus, options.proposer, options.reviewer, options.corpus is None
    )
    for command in claim.commands:
        run_command(command)
    url = pr_url()
    try:
        status = run_work(work, plan.choice.kind, claim_key(slug))
    except Exception as exc:
        sh(
            ["gh", "pr", "close", url, "--comment", f"Abandoned by `openinterp loop`: {exc}"],
            check=False,
        )
        delete_branch()
        raise
    return Result(slug, status, url, land(url))


def run_work(work, kind: str, key: str) -> str:
    """Run every step after the claim, then post. A scan that stops on its unit posts at once."""
    from openinterp.evidence import Unauditable

    try:
        for step in work:
            for command in step.commands:
                run_command(command)
    except Unauditable as exc:
        sys.stdout.write(f"The scan stopped on the unit: {exc}\n")
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

    cache = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(os.environ.get("OPENINTERP_STOP_FILE") or Path(cache) / "openinterp" / "stop")


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
        try:
            with (
                open(log, "w") as out,
                contextlib.redirect_stdout(out),
                contextlib.redirect_stderr(out),
            ):
                result = iteration(self.options, self.plan_next, self.claims)
        except Exception as exc:  # any failure of one experiment; the loop decides what next
            return self._failed(exc, log)
        self.done, self.failures, self.backoffs = self.done + 1, 0, 0
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
        """Wait, a minute at a time, while ``openinterp pause`` is in force."""
        from openinterp.core import pause

        said = False
        while (paused := pause.read()) is not None and not stop_file().exists():
            if not said:
                print(paused.describe(), flush=True)
                said = True
            self.sleep(60)

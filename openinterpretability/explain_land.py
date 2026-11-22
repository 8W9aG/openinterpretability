"""``explain --record``: land the explanation record as its own PR (#4475).

The record is landed as an experiment is, with the experiment loop's code
(:mod:`openinterpretability.experiment_loop`). In the loop's worktree, at a fresh
``origin/main``, it goes on its own branch. ``open-interpretability check`` passes, and a
ready PR is opened. :func:`~openinterpretability.experiment_loop.land` waits for the checks
and merges it when the merge rules allow. They allow a PR that changes only ``wiki/``.

A second explanation of the same prompt and model adds a newer file under the same prompt
id. The site shows the newest record of each prompt, so its page is updated, not doubled.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from openinterpretability.core.explanation_records import ExplanationRecord, record_path
from openinterpretability.core.identifiers import model_slug
from openinterpretability.core.paths import DEFAULT_WIKI_DIR


def branch_name(record: ExplanationRecord, now: datetime | None = None) -> str:
    """``explanation/<model slug>-<prompt id>-<UTC time>``: one branch per landing."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"explanation/{model_slug(record.model_name)}-{record.prompt_id}-{stamp}"


def pr_body(record: ExplanationRecord) -> str:
    """What the PR says: the shares, the coverage and the experiments the run landed."""
    lines = [
        f"An explanation of {record.model_name}'s answer {record.answer!r} to:",
        "",
        f"> {record.prompt}",
        "",
        f"- **Explained share:** {record.explained_share:.1%} of the answer's attribution "
        f"is in VERIFIED units, or in units of VERIFIED superpositions and circuits.",
    ]
    if record.prompt_coverage is not None:
        random = "—" if record.random_coverage is None else f"{record.random_coverage:.1%}"
        lines.append(f"- **Prompt coverage:** {record.prompt_coverage:.1%}, against "
                     f"{random} for random units of the same count.")  # fmt: skip
    if record.run is not None:
        run = record.run
        lines.append(f"- **The run:** from {run.share_before:.1%} to {run.share_after:.1%}. "
                     f"It stopped because {run.stopped}.")  # fmt: skip
        lines += [f"  - landed: {landed}" for landed in run.landed]
        lines += [f"  - left open: {url}" for url in run.left_open]
        lines += [f"  - pending, counted before it merges: {p}" for p in run.pending]
        lines += [f"  - shared from a local run: {x}" for x in run.local]
    lines += ["", "Written by `open-interpretability explain --record` (#4475)."]
    return "\n".join(lines)


def land_record(record: ExplanationRecord) -> tuple[str, bool]:
    """Land ``record`` as its own PR. Return the PR's URL, and whether it merged."""
    from openinterpretability.cli_loop import loop_worktree
    from openinterpretability.experiment_loop import (
        land,
        openinterpretability,
        pr_url,
        reset,
        say,
        sh,
    )

    os.chdir(loop_worktree()[1])
    reset()
    branch = branch_name(record)
    sh(["git", "switch", "--quiet", "-c", branch])
    path = record_path(DEFAULT_WIKI_DIR, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=1) + "\n", encoding="utf-8")
    openinterpretability(["check", "--base-ref", "origin/main"])
    title = f"Explanation: {record.model_name} · {record.prompt_id}"
    sh(["git", "add", str(path)])
    sh(["git", "commit", "--quiet", "-m", title])
    sh(["git", "push", "--quiet", "-u", "origin", branch])
    sh(["gh", "pr", "create", "--title", title, "--body", pr_body(record)])
    url = pr_url()
    say(f"explanation PR: {url}")
    return url, land(url)


def write_local(record: ExplanationRecord, wiki_dir: Path) -> Path:
    """Write ``record`` under ``wiki_dir``, with no PR (``--no-land``)."""
    path = record_path(Path(wiki_dir), record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return path

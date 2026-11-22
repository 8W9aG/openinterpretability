"""One experiment of an ``explain --local`` run: no claim, push or PR (#5827).

:func:`~openinterpretability.experiment_loop.iteration` hands a local loop's iterations to
:func:`local_iteration`. It calls the loop's own steps through the
:mod:`~openinterpretability.experiment_loop` module, so the two paths share them.
"""

from __future__ import annotations

from pathlib import Path

from openinterpretability import experiment_loop as el


def wiki_files() -> set[Path]:
    """Every file under ``wiki/``, so that a local run can tell which files its steps wrote.

    Returns:
        The files' paths, relative to the checkout.

    Examples:
        >>> import os, tempfile
        >>> here = os.getcwd()
        >>> os.chdir(tempfile.mkdtemp())
        >>> Path("wiki/a").mkdir(parents=True)
        >>> _ = Path("wiki/a/r.json").write_text("{}")
        >>> wiki_files()
        {PosixPath('wiki/a/r.json')}
        >>> os.chdir(here)
    """
    return {p for p in el.DEFAULT_WIKI_DIR.rglob("*") if p.is_file()}


def local_iteration(options: el.Options, plan_next, claims) -> el.Result:
    """Run one experiment in the user's checkout, with no claim, commit, push or PR (#5827).

    The draw and the work steps are those of :func:`.experiment_loop.iteration`. So the
    record has the same inputs, and CI can reproduce it once it is shared
    (``explain-share``). The claim (branch, push, draft PR) and the post (commit, push,
    ready) are left out. The record is a file that the steps added under ``wiki/``, so an
    earlier local record is never taken for it.

    Args:
        options: The loop's options.
        plan_next: Draws the next experiment.
        claims: What open PRs claim; a local loop passes none.

    Returns:
        The experiment's result, with the path of its record.

    Raises:
        el.LoopError: The draw names no unit.
    """
    plan = el.draw(options, plan_next, claims)
    slug = el.draw_slug(plan.choice)
    if slug is None:
        raise el.LoopError(
            f"the unit of a {plan.choice.kind} draw on {plan.choice.model} is unknown"
        )
    corpus = options.corpus or "corpus.jsonl"
    # The work steps only: the claim and the post both call git and gh.
    _, *work, _ = el.steps(plan, corpus, options.proposer, options.reviewer, options.corpus is None)
    el.say(f"drew {plan.choice.kind} {slug} on {plan.choice.model}, locally")
    before = wiki_files()  # so that only this draw's files count as its record
    el.run_steps(work)
    added = sorted(wiki_files() - before)

    def new(prefix: Path) -> list[Path]:
        """The files that the steps added under ``prefix``.

        Args:
            prefix: A folder of the wiki.

        Returns:
            The new files there.
        """
        return [p for p in added if p.is_relative_to(prefix)]

    status, _ = el.verdict_of(plan.choice.kind, el.claim_key(slug), new)
    record = next((p for p in added if p.suffix == ".json"), None)
    el.say(f"kept locally: {record}")
    return el.Result(slug, status, record=str(record) if record else None)

"""Who made each experiment that records no contributor: the author of the PR that merged it.

A record's ``contributor`` (:mod:`openinterpretability.core.contributors`) names its proposer when
it was given. Older records name nobody. For them, the site credits the GitHub user who opened
the pull request that merged the record.

1. Git history gives the pull request: the first-parent commit on ``main`` that added the
   record has the PR's number in its subject. A merge commit has "Merge pull request #N",
   and a squash commit ends with "(#N)".
2. GitHub's API gives the PR's author (``user.login``). Each login is kept in a cache
   (``~/.cache/openinterpretability/pr-authors.json``, ``$OPENINTERPRETABILITY_PR_AUTHORS_CACHE``),
   so a build asks only for new PRs. Without a network, PRs that the cache doesn't know name nobody.

A login is a public handle, never an email. Bots name nobody. Nothing is written to a
record: the build reads this again each time.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

from labinacan.site import github
from openinterpretability.core.cache_dirs import cache_root
from openinterpretability.core.contributors import contributor_problem
from openinterpretability.core.paths import DEFAULT_WIKI_DIR, experiments_dir

CACHE_ENV = "OPENINTERPRETABILITY_PR_AUTHORS_CACHE"
PR_NUMBER = re.compile(r"^Merge pull request #(\d+) |\(#(\d+)\)$")


def merged_prs(wiki_dir: Path | str = DEFAULT_WIKI_DIR, ref: str = "HEAD") -> dict[str, int]:
    """The PR that added each record to ``ref``, by the record's path under ``experiments/``.

    A record added by a commit with no PR number, or not on ``ref`` yet, is absent. So is
    every record in a shallow clone, whose history is incomplete.
    """

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout

    if git("rev-parse", "--is-shallow-repository").strip() != "false":
        return {}
    root = experiments_dir(wiki_dir).as_posix().rstrip("/") + "/"
    log = git(
        "log", "--first-parent", "--diff-filter=A", "--name-only", "--format=@%s", ref,
        "--", root,
    )  # fmt: skip
    prs: dict[str, int] = {}
    number = None
    for line in log.splitlines():
        if line.startswith("@"):
            found = PR_NUMBER.search(line[1:])
            number = int(found.group(1) or found.group(2)) if found else None
        elif line.endswith(".json") and line.startswith(root) and number is not None:
            prs.setdefault(line.removeprefix(root), number)
    return prs


def pr_authors(numbers: Iterable[int], repo: str) -> dict[int, str]:
    """The login of each PR in ``numbers`` that a person opened, from the cache or GitHub.

    ``repo`` is ``owner/name``. A PR that a bot opened, or that GitHub can't give, is absent.
    labinacan reads them (:func:`labinacan.site.github.pr_authors`), with this repo's cache
    and its rule for handles (#6628).
    """
    valid = lambda login: contributor_problem(login) is None  # noqa: E731
    return github.pr_authors(numbers, repo, cache=_cache_path(), get=_get, valid=valid)


def load_users(repo_url: str, wiki_dir: Path | str = DEFAULT_WIKI_DIR) -> dict[str, str]:
    """The GitHub user who opened the PR that merged each record, by record path.

    The paths are relative to ``experiments/``, as
    :func:`openinterpretability.core.errata.record_ref` gives them. ``repo_url`` is the project's
    GitHub URL.
    """
    repo = repo_url.removeprefix("https://github.com/").strip("/")
    prs = merged_prs(wiki_dir)
    authors = pr_authors(set(prs.values()), repo) if prs else {}
    return {ref: authors[n] for ref, n in prs.items() if n in authors}


def _get(url: str) -> list[dict]:
    """GitHub's JSON at ``url``, sent with this repo's User-Agent. Tests replace it."""
    return github.get_json(url, "openinterpretability")


def _cache_path() -> Path | None:
    root = cache_root(CACHE_ENV, "pr-authors")
    return None if root is None else root / "pr-authors.json"

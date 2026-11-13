"""Who made each experiment that records no contributor: the author of the PR that merged it.

A record's ``contributor`` (:mod:`openinterp.core.contributors`) names its proposer when it
was given. Older records name nobody. For them, the site credits the GitHub user who opened
the pull request that merged the record.

1. Git history gives the pull request: the first-parent commit on ``main`` that added the
   record has the PR's number in its subject. A merge commit has "Merge pull request #N",
   and a squash commit ends with "(#N)".
2. GitHub's API gives the PR's author (``user.login``). Each login is kept in a cache
   (``~/.cache/openinterp/pr-authors.json``, ``$OPENINTERP_PR_AUTHORS_CACHE``), so a build
   asks only for new PRs. Without a network, PRs that the cache doesn't know name nobody.

A login is a public handle, never an email. Bots name nobody. Nothing is written to a
record: the build reads this again each time.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import urllib.request
from collections.abc import Iterable
from pathlib import Path

from openinterp.core.cache_dirs import cache_root
from openinterp.core.contributors import contributor_problem
from openinterp.core.paths import DEFAULT_WIKI_DIR, experiments_dir

CACHE_ENV = "OPENINTERP_PR_AUTHORS_CACHE"
API = "https://api.github.com"
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
    """
    path = _cache_path()
    cache = _read_cache(path)
    missing = {n for n in numbers if str(n) not in cache}
    if missing:
        cache |= _fetch(repo, missing)
        _write_cache(path, cache)
    return {n: cache[str(n)] for n in numbers if cache.get(str(n))}


def load_users(repo_url: str, wiki_dir: Path | str = DEFAULT_WIKI_DIR) -> dict[str, str]:
    """The GitHub user who opened the PR that merged each record, by record path.

    The paths are relative to ``experiments/``, as :func:`openinterp.core.errata.record_ref`
    gives them. ``repo_url`` is the project's GitHub URL.
    """
    repo = repo_url.removeprefix("https://github.com/").strip("/")
    prs = merged_prs(wiki_dir)
    authors = pr_authors(set(prs.values()), repo) if prs else {}
    return {ref: authors[n] for ref, n in prs.items() if n in authors}


def _fetch(repo: str, wanted: set[int]) -> dict[str, str]:
    """Logins of the PRs in ``wanted``, newest first, a page of 100 at a time.

    A bot's PR, or a login that isn't a valid handle, maps to "". So the cache knows it,
    and doesn't ask again. Stops when it has every PR in ``wanted``, or at the last page.
    """
    found: dict[str, str] = {}
    page = 1
    while wanted - {int(n) for n in found}:
        query = f"state=closed&sort=created&direction=desc&per_page=100&page={page}"
        try:
            pulls = _get(f"{API}/repos/{repo}/pulls?{query}")
        except OSError as exc:
            print(f"warning: can't read PR authors from GitHub ({exc}); some users are unknown")
            break
        if not pulls:
            break
        for pull in pulls:
            user = pull.get("user") or {}
            login = user.get("login", "")
            human = user.get("type") == "User" and contributor_problem(login) is None
            found[str(pull["number"])] = login if human else ""
        page += 1
    return found


def _get(url: str) -> list[dict]:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "openinterp"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)  # noqa: S310  (https API URL)
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return json.load(response)


def _cache_path() -> Path | None:
    root = cache_root(CACHE_ENV, "pr-authors")
    return None if root is None else root / "pr-authors.json"


def _read_cache(path: Path | None) -> dict[str, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path else {}
    except (OSError, ValueError):
        return {}


def _write_cache(path: Path | None, cache: dict[str, str]) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, sort_keys=True), encoding="utf-8")

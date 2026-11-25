"""What a site build asks GitHub: open pull requests, and who opened merged ones (#6628).

``open_prs`` lists a repository's open PRs with the ``gh`` CLI. ``pr_authors`` reads PR
authors from GitHub's API, and keeps them in a cache file, so a build asks only for new PRs.
Without ``gh`` or a network, each gives what it has: no PRs, or the cached authors.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

API = "https://api.github.com"
# The header names. urllib's Request sets their case itself, so only their spelling matters.
ACCEPT, USER_AGENT, AUTHORIZATION = "Accept", "User-Agent", "Authorization"


def open_prs(fields: str, limit: int = 300) -> list[dict]:
    """The open PRs of the current repository, as ``gh pr list --json`` gives them.

    Args:
        fields: The comma-separated fields to ask for, such as ``number,url``.
        limit: The most PRs to list.

    Returns:
        One dict per PR, or an empty list without ``gh`` or when it fails.
    """
    if not shutil.which("gh"):
        return []
    try:
        out = subprocess.run(
            ["gh", "pr", "list", "--state", "open", "--limit", str(limit), "--json", fields],
            check=True, capture_output=True, text=True, timeout=60,
        ).stdout  # fmt: skip
    except (subprocess.SubprocessError, OSError):
        return []
    return json.loads(out)


def get_json(url: str, agent: str) -> list[dict]:
    """GitHub's JSON at ``url``, with the token in ``$GITHUB_TOKEN`` or ``$GH_TOKEN`` if set.

    Args:
        url: The API URL.
        agent: The User-Agent to send, such as the project's name.

    Returns:
        The decoded JSON.
    """
    headers = {ACCEPT: "application/vnd.github+json", USER_AGENT: agent}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers[AUTHORIZATION] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)  # noqa: S310  (https API URL)
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return json.load(response)


def pr_authors(
    numbers: Iterable[int],
    repo: str,
    *,
    cache: Path | None,
    get: Callable[[str], list[dict]],
    valid: Callable[[str], bool],
) -> dict[int, str]:
    """The login of each PR in ``numbers`` that a person opened, from the cache or GitHub.

    A PR that a bot opened, or that GitHub can't give, is absent.

    Args:
        numbers: The PRs' numbers.
        repo: The repository, as ``owner/name``.
        cache: The cache file, or None to keep nothing.
        get: Reads one API URL (:func:`get_json`, with the project's User-Agent).
        valid: Whether a login is a handle that the project may show.

    Returns:
        Each PR's author's login, by its number.

    Examples:
        >>> pulls = [{"number": 7, "user": {"login": "ann", "type": "User"}}]
        >>> pr_authors({7}, "o/r", cache=None, get=lambda url: pulls, valid=bool)
        {7: 'ann'}
    """
    cached = _read_cache(cache)
    missing = {n for n in numbers if str(n) not in cached}
    if missing:
        cached |= _fetch(repo, missing, get, valid)
        _write_cache(cache, cached)
    return {n: cached[str(n)] for n in numbers if cached.get(str(n))}


def _fetch(repo: str, wanted: set[int], get, valid) -> dict[str, str]:
    """Logins of the PRs in ``wanted``, newest first, a page of 100 at a time.

    A bot's PR, or a login that isn't a valid handle, maps to "". So the cache knows it,
    and doesn't ask again. Stops when it has every PR in ``wanted``, or at the last page.

    Args:
        repo: The repository, as ``owner/name``.
        wanted: The PRs' numbers.
        get: Reads one API URL.
        valid: Whether a login is a handle that the project may show.

    Returns:
        Each PR's login, or "", by its number as a string.
    """
    found: dict[str, str] = {}
    page = 1
    while wanted - {int(n) for n in found}:
        query = f"state=closed&sort=created&direction=desc&per_page=100&page={page}"
        try:
            pulls = get(f"{API}/repos/{repo}/pulls?{query}")
        except OSError as exc:
            print(f"warning: can't read PR authors from GitHub ({exc}); some users are unknown")
            break
        if not pulls:
            break
        for pull in pulls:
            user = pull.get("user") or {}
            login = user.get("login", "")
            human = user.get("type") == "User" and valid(login)
            found[str(pull["number"])] = login if human else ""
        page += 1
    return found


def _read_cache(path: Path | None) -> dict[str, str]:
    """The cached logins, or none when there is no file or it can't be read.

    Args:
        path: The cache file, or None.

    Returns:
        Each PR's login by its number as a string.
    """
    try:
        return json.loads(path.read_bytes()) if path else {}
    except (OSError, ValueError):
        return {}


def _write_cache(path: Path | None, cache: dict[str, str]) -> None:
    """Write ``cache`` to ``path``, when there is a path.

    Args:
        path: The cache file, or None to keep nothing.
        cache: Each PR's login by its number as a string.
    """
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(json.dumps(cache, sort_keys=True).encode())

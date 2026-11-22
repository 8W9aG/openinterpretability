"""Each paper's and post's link to its X post (#5840).

``social post --record`` writes the posts that a run made. Each row holds the entry's id,
which is its page's URL, and the id of its first post. ``social link`` then writes each
post's URL into its folder, as ``x_url`` in ``media/social.json``. That file holds the
drawn figures' hashes already. The deploy opens a PR with those files, and the page shows
the link.

The entry's id doesn't change, and the social state holds it. So the deploy that follows
the link PR doesn't post the entry again.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

STATUS = "https://x.com/i/status/"  # a post's URL with no handle: X sends it to the account
# A page's URL ends in research/<slug>/ or posts/<slug>/; drafts are never posted.
_PAGE = re.compile(r"/(research|posts)/([\w-]+)/(?:index\.html)?$")
FOLDERS = {"research": "papers", "posts": "posts"}


def post_url(post_id: str) -> str:
    """The URL of the post with id ``post_id``.

    Args:
        post_id: The id that X gave the post.

    Returns:
        Its URL.

    Examples:
        >>> post_url("123")
        'https://x.com/i/status/123'
    """
    return STATUS + post_id


def folder_of(entry_id: str, root: Path = Path(".")) -> Path | None:
    """The repository folder of the paper or post whose page is ``entry_id``.

    Args:
        entry_id: The feed entry's id: the page's URL.
        root: The repository's root.

    Returns:
        ``papers/<slug>`` or ``posts/<slug>`` under ``root``, or None for another page.

    Examples:
        >>> folder_of("https://open-interpretability.com/research/a-paper/index.html")
        PosixPath('papers/a-paper')
        >>> folder_of("https://open-interpretability.com/models/x/index.html") is None
        True
    """
    found = _PAGE.search(entry_id)
    return root / FOLDERS[found.group(1)] / found.group(2) if found else None


def write_records(path: Path, posted: list[tuple[str, str]]) -> None:
    """Write this run's posts to ``path``, for the job that opens the link PR.

    Args:
        path: The JSON file.
        posted: ``(entry id, first post id)`` of each entry posted.

    Examples:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "posted.json"
        >>> write_records(path, [("https://x.org/research/p/", "9")])
        >>> json.loads(path.read_text())[0]["url"]
        'https://x.com/i/status/9'
    """
    rows = [{"entry": entry, "post": post, "url": post_url(post)} for entry, post in posted]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")


def link(posted: Path, root: Path = Path(".")) -> list[Path]:
    """Set ``x_url`` in the ``media/social.json`` of each folder that ``posted`` names.

    The file's other keys, the drawn figures' hashes, stay as they are.

    Args:
        posted: The JSON file that ``social post --record`` wrote.
        root: The repository's root.

    Returns:
        The files written. An entry whose folder isn't in the repository is skipped.

    Examples:
        >>> import tempfile
        >>> root = Path(tempfile.mkdtemp())
        >>> (root / "papers" / "p").mkdir(parents=True)
        >>> write_records(root / "posted.json", [("https://x.org/research/p/index.html", "9")])
        >>> [p.relative_to(root).as_posix() for p in link(root / "posted.json", root)]
        ['papers/p/media/social.json']
        >>> json.loads((root / "papers/p/media/social.json").read_text())
        {'x_url': 'https://x.com/i/status/9'}
    """
    written = []
    for row in json.loads(posted.read_text(encoding="utf-8")):
        folder = folder_of(row["entry"], root)
        if folder is None or not folder.is_dir():
            continue
        path = folder / "media" / "social.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        data["x_url"] = row["url"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        written.append(path)
    return written

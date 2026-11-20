"""The project's public links, read from ``pyproject.toml``'s metadata.

``[project.urls]`` holds the site's URL (``Homepage``), the repo's (``Repository``) and the X
account's (``X``).
The builder and ``open-interpretability deploy`` read them here, so each URL is written down once.
"""

from __future__ import annotations

from email.utils import parseaddr
from importlib.metadata import metadata

from openinterpretability import DISTRIBUTION


def project_links() -> dict[str, str]:
    """The site and repo URLs, the contact email and the project's one-line description."""
    meta = metadata(DISTRIBUTION)
    urls = dict(u.split(", ", 1) for u in meta.get_all("Project-URL") or [])
    return {
        "site_url": urls["Homepage"],
        "repo_url": urls["Repository"],
        "contact_email": parseaddr(meta["Author-email"])[1],
        "description": meta["Summary"].split(": ", 1)[-1],  # for social cards (#3289)
        "x_url": urls["X"],
        "x_handle": x_handle(urls["X"]),  # twitter:site (#5389)
    }


def x_handle(url: str) -> str:
    """The X handle in an account's URL, so that the handle is not written a second time.

    Args:
        url: The account's URL.

    Returns:
        ``@`` and the URL's last path part.

    >>> x_handle("https://x.com/example")
    '@example'
    >>> x_handle("https://x.com/example/")
    '@example'
    """
    return "@" + url.rstrip("/").rsplit("/", 1)[-1]


def site_url() -> str:
    """The public URL of the site's root, with a final ``/``."""
    return project_links()["site_url"]

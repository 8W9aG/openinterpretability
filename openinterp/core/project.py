"""The project's public links, read from ``pyproject.toml``'s metadata.

``[project.urls]`` holds the site's URL (``Homepage``) and the repo's (``Repository``).
The builder and ``open-interpretability deploy`` read them here, so each URL is written down once.
"""

from __future__ import annotations

from email.utils import parseaddr
from importlib.metadata import metadata

from openinterp import DISTRIBUTION


def project_links() -> dict:
    """The site and repo URLs and the contact email."""
    meta = metadata(DISTRIBUTION)
    urls = dict(u.split(", ", 1) for u in meta.get_all("Project-URL") or [])
    return {
        "site_url": urls["Homepage"],
        "repo_url": urls["Repository"],
        "contact_email": parseaddr(meta["Author-email"])[1],
    }


def site_url() -> str:
    """The public URL of the site's root, with a final ``/``."""
    return project_links()["site_url"]

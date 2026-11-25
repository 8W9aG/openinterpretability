"""The project's public links, read by ``labinacan/config.py`` from ``pyproject.toml``.

``[project.urls]`` holds the site's URL (``Homepage``), the repo's (``Repository``) and the X
account's (``X``). ``[project] maintainers`` holds the contact email, because PyPI accepts
only http(s) URLs in ``[project.urls]`` (#6434). The reader is generic, so it lives in
``labinacan/`` (#6629). This module keeps the names that the project imports.
"""

from __future__ import annotations

from labinacan.config import contact_email, project_links, site_url, x_handle

__all__ = ["contact_email", "project_links", "site_url", "x_handle"]

"""The human contributor behind an experiment's proposal or review, beside the agent model.

A contributor is a public handle such as a GitHub username, never an email, and is only
recorded when given: ``--contributor`` or ``$OPENINTERP_CONTRIBUTOR``. Git config is never
read, so nobody is named without choosing to be."""

from __future__ import annotations

import re

CONTRIBUTOR_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,38}$"
CONTRIBUTOR_ENV = "OPENINTERP_CONTRIBUTOR"


def contributor_problem(handle: str) -> str | None:
    """Why ``handle`` can't be recorded as a contributor, or None if it can."""
    if "@" in handle:
        return "a contributor is a public handle (e.g. a GitHub username), never an email"
    if not re.fullmatch(CONTRIBUTOR_PATTERN, handle):
        return (
            "a contributor is 1-39 letters, digits, '.', '_' or '-', starting with a letter "
            "or digit"
        )
    return None

"""Which model revisions CI may load to reproduce a record.

A record names the model commit it ran on, and ``reproduce`` loads that commit. On the
Hugging Face Hub anyone can open a pull request on a public model, and its weights live
under refs like ``refs/pr/1``. So a record must not choose any revision it likes. CI
reproduces a record of an allow-listed model only at a full 40-character commit SHA. That
commit must be in the history of the model's ``main`` branch.
"""

from __future__ import annotations

import re
from functools import cache

SHA = re.compile(r"^[0-9a-f]{40}$")


@cache
def main_commits(model: str) -> frozenset[str]:
    """Every commit SHA in the history of ``model``'s ``main`` branch on the Hub."""
    from huggingface_hub import HfApi

    return frozenset(c.commit_id for c in HfApi().list_repo_commits(model, revision="main"))


def unapproved_revision(model: str, revision: str | None) -> str | None:
    """Why CI must not load ``revision`` of ``model``, or None when it may.

    No revision means ``main`` itself, which is allowed.
    """
    if revision is None:
        return None
    if not SHA.match(revision):
        return f"its model revision {revision!r} is not a full commit SHA"
    if revision not in main_commits(model):
        return f"its model revision {revision[:12]} is not on {model}'s main branch"
    return None

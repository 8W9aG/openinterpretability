"""Every GitHub Action in a workflow is pinned to a full commit SHA (.github/dependabot.yml).

A tag such as ``@v7`` can be moved to other code. A job that holds secrets would then
run that code. Dependabot updates a pinned SHA and its version comment.
"""

import re
from pathlib import Path

WORKFLOWS = Path(__file__).parents[1] / ".github" / "workflows"
USES = re.compile(r"^\s*-?\s*uses:\s*(\S+)", re.M)
PINNED = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")


def unpinned(text: str) -> list[str]:
    """The ``uses:`` values in a workflow that are not a remote action at a full SHA.

    A local action (``./path``) and a Docker image (``docker://``) are not checked.
    """
    return [
        u
        for u in USES.findall(text)
        if not u.startswith(("./", "docker://")) and not PINNED.match(u)
    ]


def test_unpinned_finds_a_tag_and_passes_a_sha():
    text = "steps:\n  - uses: actions/checkout@v7\n  - uses: a/b@" + "0" * 40 + "  # v1\n"
    assert unpinned(text) == ["actions/checkout@v7"]


def test_every_workflow_pins_its_actions():
    found = {
        f.name: bad for f in sorted(WORKFLOWS.glob("*.yml")) if (bad := unpinned(f.read_text()))
    }
    assert not found, f"pin these actions to a full commit SHA: {found}"

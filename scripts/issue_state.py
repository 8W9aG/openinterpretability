"""Set the Linear workflow state of GitHub issues: Todo, In Progress, In Review, Done, Released.

GitHub issues have no "in progress" state, so the state lives in Linear. Linear's GitHub
integration links each issue, and this script finds the Linear issue from that link.

    python3 scripts/issue_state.py 3371 "In Progress"
    python3 scripts/issue_state.py 3371 3372 "In Review"
    python3 scripts/issue_state.py --check 3371
    python3 scripts/issue_state.py --release v0.0.20 [--dry-run]

An agent may work only on an eligible issue. It refuses "In Progress" for any other.
An issue is eligible when one of these holds:

- the maintainer opened it.
- Dependabot or CodeQL (code scanning) opened it.
- the nightly reproduction workflow opened it, with the title that the workflow sets.
- the maintainer gave it a thumbs up.

An issue labelled `proposed` is never eligible, whoever opened it. The maintainer uses
the label to hold an issue back, and removes it to release the issue.

`--check` prints whether each issue is eligible, and why. It exits 1 if one is not.
`--release <tag>` moves every closed issue in milestone ``<tag>`` to Released. The
`linear-released` workflow runs it when that release is published.
Setting a state needs LINEAR_API_KEY. The check needs the GitHub CLI (`gh`). The
`work-an-issue` skill says when to call this script.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = "8W9aG/openinterpretability"
TEAM_KEY = "OPE"
LINEAR_API = "https://api.linear.app/graphql"
DONE = "Done"
# The state of an issue that shipped in a published release.
RELEASED = "Released"
STATES = ("Todo", "In Progress", "In Review", DONE, RELEASED)
MAINTAINER = "8W9aG"
# Bots whose issues an agent may work on. CodeQL files code-scanning issues as the
# GitHub Advanced Security app.
TRUSTED_BOTS = ("dependabot[bot]", "github-advanced-security[bot]")
NIGHTLY_BOT = "github-actions[bot]"
# The maintainer holds an issue back with this label.
PROPOSED = "proposed"
NIGHTLY_WORKFLOW = Path(__file__).parents[1] / ".github/workflows/reproduce-nightly.yml"


def nightly_title() -> str:
    """The title of the issues that the nightly reproduction opens (its TITLE env)."""
    match = re.search(r"^\s*TITLE:\s*(.+?)\s*$", NIGHTLY_WORKFLOW.read_text(), re.M)
    if match is None:
        raise RuntimeError(f"no TITLE in {NIGHTLY_WORKFLOW}")
    return match.group(1).strip("\"'")


def gh_api(path: str):
    """GET ``path`` from GitHub's REST API with the GitHub CLI, as JSON."""
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def eligibility(number: str) -> tuple[bool, str]:
    """Return whether an agent may work on issue ``number``, and the reason."""
    issue = gh_api(f"repos/{REPO}/issues/{number}")
    author = issue["user"]["login"]
    if PROPOSED in {label["name"] for label in issue.get("labels", [])}:
        return False, f"labelled {PROPOSED!r}: the maintainer has not triaged it"
    if author == MAINTAINER:
        return True, f"opened by {MAINTAINER}"
    if author in TRUSTED_BOTS:
        return True, f"opened by {author}"
    if author == NIGHTLY_BOT and issue["title"] == nightly_title():
        return True, "opened by the nightly reproduction"
    if issue["reactions"]["+1"]:
        reactions = gh_api(f"repos/{REPO}/issues/{number}/reactions?per_page=100")
        if any(r["content"] == "+1" and r["user"]["login"] == MAINTAINER for r in reactions):
            return True, f"opened by {author}, thumbs up from {MAINTAINER}"
    return False, f"opened by {author}, with no thumbs up from {MAINTAINER}"


def linear(query: str, variables: dict) -> dict:
    """Run one query on Linear's GraphQL API and return its data."""
    body = json.dumps({"query": query, "variables": variables}).encode()
    headers = {"Content-Type": "application/json", "Authorization": os.environ["LINEAR_API_KEY"]}
    with urllib.request.urlopen(urllib.request.Request(LINEAR_API, body, headers)) as r:
        out = json.load(r)
    if "errors" in out:
        raise RuntimeError(out["errors"])
    return out["data"]


def state_id(name: str) -> str:
    """Return the ID of the team's workflow state called ``name``."""
    query = (
        "query($k: String!) { workflowStates(filter: {team: {key: {eq: $k}}}) "
        "{ nodes { id name } } }"
    )
    nodes = linear(query, {"k": TEAM_KEY})["workflowStates"]["nodes"]
    return next(n["id"] for n in nodes if n["name"] == name)


def set_state(number: str, state: str) -> str:
    """Move the Linear issue linked to GitHub issue ``number`` to ``state``."""
    url = f"https://github.com/{REPO}/issues/{number}"
    query = (
        "query($u: String!) { attachmentsForURL(url: $u) { nodes { issue { id identifier } } } }"
    )
    nodes = linear(query, {"u": url})["attachmentsForURL"]["nodes"]
    if not nodes:
        return f"#{number}: not in Linear yet"
    issue = nodes[0]["issue"]
    mutation = (
        "mutation($id: String!, $s: String!) "
        "{ issueUpdate(id: $id, input: {stateId: $s}) { success } }"
    )
    linear(mutation, {"id": issue["id"], "s": state_id(state)})
    return f"#{number} {issue['identifier']}: {state}"


def check(numbers: list[str]) -> bool:
    """Print each issue's eligibility. Return True when every one is eligible."""
    ok = True
    for number in numbers:
        allowed, reason = eligibility(number)
        ok &= allowed
        print(f"#{number}: {'eligible' if allowed else 'NOT eligible'} ({reason})")
    return ok


def milestone_issues(tag: str) -> list[tuple[str, str]]:
    """``(number, state)`` of every issue in milestone ``tag``, open or closed."""
    out = subprocess.run(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            REPO,
            "--milestone",
            tag,
            "--state",
            "all",
            "--limit",
            "1000",
            "--json",
            "number,state",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return [(str(i["number"]), i["state"]) for i in json.loads(out.stdout)]


def release(tag: str, dry_run: bool = False) -> bool:
    """Move the closed issues of milestone ``tag`` to Released. Return False if one failed."""
    ok = True
    for number, state in milestone_issues(tag):
        if state != "CLOSED":
            print(f"#{number}: still open, left as it is")
        elif dry_run:
            print(f"#{number}: would move to {RELEASED}")
        else:
            try:
                print(set_state(number, RELEASED))
            except (RuntimeError, OSError) as exc:
                ok = False
                print(f"#{number}: failed: {exc}")
    return ok


def main() -> None:
    """Check eligibility, or set a state. "In Progress" needs every issue to be eligible."""
    args = [a.lstrip("#") for a in sys.argv[1:]]
    if args[:1] == ["--check"]:
        sys.exit(0 if check(args[1:]) else 1)
    if args[:1] == ["--release"] and len(args) >= 2:
        sys.exit(0 if release(args[1], "--dry-run" in args[2:]) else 1)
    *numbers, state = args or [""]
    if not numbers or state not in STATES:
        sys.exit(
            f"usage: issue_state.py <number>... <{'|'.join(STATES)}> | --check <number>..."
            " | --release <tag> [--dry-run]"
        )
    if state == "In Progress" and not check(numbers):
        sys.exit("Refused: work only on eligible issues (see the docstring).")
    for number in numbers:
        print(set_state(number, state))


if __name__ == "__main__":
    main()

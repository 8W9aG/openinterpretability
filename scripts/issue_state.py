"""Set the Linear workflow state of GitHub issues: Todo, In Progress, In Review, Done, Released.

GitHub issues have no "in progress" state, so the state lives in Linear. Linear's GitHub
integration links each issue, and this script finds the Linear issue from that link.

    python3 scripts/issue_state.py 3371 "In Progress"
    python3 scripts/issue_state.py 3371 3372 "In Review"
    python3 scripts/issue_state.py --check 3371
    python3 scripts/issue_state.py --muse-check 5300
    python3 scripts/issue_state.py --release v0.0.20 [--dry-run]
    python3 scripts/issue_state.py --blocked-by 3372 3371

An agent may work only on an eligible issue. It refuses "In Progress" for any other.
An issue is eligible when one of these holds:

- the maintainer opened it.
- Dependabot or CodeQL (code scanning) opened it.
- the nightly reproduction workflow opened it, with the title that the workflow sets.
- the maintainer gave it a thumbs up.

An issue labelled `proposed` is never eligible, whoever opened it. The maintainer uses
the label to hold an issue back, and removes it to release the issue.

`--check` prints whether each issue is eligible, and why. It exits 1 if one is not.
`--muse-check` does the same for the `muse` label, with a stricter rule. Only an issue
that the maintainer opened, or gave a thumbs up, may have it. Bots and workflows don't count.
The `muse-label` workflow removes the label from any other issue (docs/muse.md).
`--release <tag>` moves every closed issue in milestone ``<tag>`` to Released. The
`linear-released` workflow runs it when that release is published.
`--blocked-by <issue> <blocker>` records that ``<issue>`` can't start until ``<blocker>``
closes (#3842). It adds Linear's "blocks" relation and GitHub's native dependency at once.
`scripts/linear_sync.py` keeps the two the same after that.
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
# An issue for Muse (docs/muse.md) to act on, such as posting a paper to alphaXiv (#5259).
# The next-issue loop never takes one.
MUSE = "muse"
WORKFLOWS = Path(__file__).parents[1] / ".github/workflows"
# Workflows that open an issue when they fail, each under the fixed titles in its TITLE envs.
AUTOMATIC_WORKFLOWS = (WORKFLOWS / "reproduce-nightly.yml", WORKFLOWS / "deploy.yml")


def workflow_titles(workflow: Path) -> set[str]:
    """The titles of the issues that ``workflow`` opens: every TITLE env in it.

    ``deploy.yml`` has two: one for a failed deploy, one for a failed post to X (#5528).

    Args:
        workflow: The workflow file.

    Returns:
        Each title, without quotes.

    Raises:
        RuntimeError: When the file has no TITLE env.
    """
    found = re.findall(r"^\s*TITLE:\s*(.+?)\s*$", workflow.read_text(), re.M)
    if not found:
        raise RuntimeError(f"no TITLE in {workflow}")
    return {title.strip("\"'") for title in found}


def automatic_titles() -> set[str]:
    """The titles of the issues that this repository's workflows open when they fail."""
    return set().union(*(workflow_titles(w) for w in AUTOMATIC_WORKFLOWS))


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
    if author == NIGHTLY_BOT and issue["title"] in automatic_titles():
        return True, f"opened by a workflow ({issue['title']})"
    return approval(number, issue)


def approval(number: str, issue: dict) -> tuple[bool, str]:
    """Whether the maintainer gave issue ``number`` a thumbs up that no later edit undid.

    Args:
        number: The issue's number.
        issue: The issue, as the GitHub API gives it.

    Returns:
        Whether the thumbs up stands, and the reason.
    """
    author = issue["user"]["login"]
    if issue["reactions"]["+1"]:
        reactions = gh_api(f"repos/{REPO}/issues/{number}/reactions?per_page=100")
        approved = [
            r["created_at"]
            for r in reactions
            if r["content"] == "+1" and r["user"]["login"] == MAINTAINER
        ]
        if approved:
            later = [
                who for when, who in edits(number) if when > max(approved) and who != MAINTAINER
            ]
            if later:
                return False, (
                    f"opened by {author}, edited by {later[-1]} after {MAINTAINER}'s thumbs up:"
                    " the maintainer must approve it again (remove the thumbs up, then add it)"
                )
            return True, f"opened by {author}, thumbs up from {MAINTAINER}"
    return False, f"opened by {author}, with no thumbs up from {MAINTAINER}"


def muse_eligibility(number: str) -> tuple[bool, str]:
    """Return whether issue ``number`` may carry the ``muse`` label, and the reason.

    Muse acts on every issue with the label. So only two kinds of issue may have it: the
    maintainer's own issues, and the ones with the maintainer's thumbs up. Bots don't count.

    Args:
        number: The issue's number.

    Returns:
        Whether the issue may have the label, and the reason.
    """
    issue = gh_api(f"repos/{REPO}/issues/{number}")
    if issue["user"]["login"] == MAINTAINER:
        return True, f"opened by {MAINTAINER}"
    return approval(number, issue)


EDITS_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      userContentEdits(last: 100) { nodes { editedAt editor { login } } }
      timelineItems(itemTypes: [RENAMED_TITLE_EVENT], last: 100) {
        nodes { ... on RenamedTitleEvent { createdAt actor { login } } }
      }
    }
  }
}
"""


def edits(number: str) -> list[tuple[str, str]]:
    """Each change of the issue's body or title, as ``(ISO time, login)``, oldest first.

    A thumbs up approves the text that the maintainer read. :func:`eligibility` refuses
    an issue that someone else changed after it. A deleted account has the login "".
    """
    owner, name = REPO.split("/")
    args = ["gh", "api", "graphql", "-f", f"query={EDITS_QUERY}", "-f", f"owner={owner}"]
    args += ["-f", f"name={name}", "-F", f"number={number}"]
    out = subprocess.run(args, capture_output=True, text=True, check=True)
    issue = json.loads(out.stdout)["data"]["repository"]["issue"]
    found = [
        (n["editedAt"], (n.get("editor") or {}).get("login", ""))
        for n in issue["userContentEdits"]["nodes"]
    ]
    found += [
        (n["createdAt"], (n.get("actor") or {}).get("login", ""))
        for n in issue["timelineItems"]["nodes"]
    ]
    return sorted(found)


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
    issue = linear_issue(number)
    if issue is None:
        return f"#{number}: not in Linear yet"
    mutation = (
        "mutation($id: String!, $s: String!) "
        "{ issueUpdate(id: $id, input: {stateId: $s}) { success } }"
    )
    linear(mutation, {"id": issue["id"], "s": state_id(state)})
    return f"#{number} {issue['identifier']}: {state}"


def linear_issue(number: str) -> dict | None:
    """The Linear issue (``id``, ``identifier``) linked to GitHub issue ``number``, or None."""
    url = f"https://github.com/{REPO}/issues/{number}"
    query = (
        "query($u: String!) { attachmentsForURL(url: $u) { nodes { issue { id identifier } } } }"
    )
    nodes = linear(query, {"u": url})["attachmentsForURL"]["nodes"]
    return nodes[0]["issue"] if nodes else None


RELATE = """mutation($a: String!, $b: String!) { issueRelationCreate(
  input: {issueId: $a, relatedIssueId: $b, type: blocks}) { success } }"""
ADD_BLOCKED_BY = """mutation($i: ID!, $b: ID!) {
  addBlockedBy(input: {issueId: $i, blockingIssueId: $b}) { clientMutationId } }"""


def node_id(number: str) -> str:
    """The GraphQL node id of GitHub issue ``number``."""
    out = subprocess.run(["gh", "issue", "view", number, "--repo", REPO, "--json", "id"],
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
    return json.loads(out)["id"]


def block(number: str, blocker: str) -> str:
    """Record that ``number`` is blocked by ``blocker``, in Linear and on GitHub."""
    issue, by = linear_issue(number), linear_issue(blocker)
    if issue and by:
        linear(RELATE, {"a": by["id"], "b": issue["id"]})
    query = f"query={ADD_BLOCKED_BY}"
    ids = [f"i={node_id(number)}", f"b={node_id(blocker)}"]
    subprocess.run(["gh", "api", "graphql", "-f", query, "-f", ids[0], "-f", ids[1]],
                   capture_output=True, text=True, check=True)  # fmt: skip
    where = "Linear and GitHub" if issue and by else "GitHub (not in Linear yet)"
    return f"#{number} is blocked by #{blocker}, in {where}"


def check(numbers: list[str], rule=eligibility) -> bool:
    """Print each issue's eligibility under ``rule``. Return True when every one is eligible."""
    ok = True
    for number in numbers:
        allowed, reason = rule(number)
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
    # The flags first, each with its own rule. A plain list of numbers then sets a state.
    if args[:1] == ["--check"]:
        sys.exit(0 if check(args[1:]) else 1)
    if args[:1] == ["--muse-check"]:
        sys.exit(0 if check(args[1:], muse_eligibility) else 1)
    if args[:1] == ["--blocked-by"] and len(args) == 3:
        print(block(args[1], args[2]))
        return
    if args[:1] == ["--release"] and len(args) >= 2:
        sys.exit(0 if release(args[1], "--dry-run" in args[2:]) else 1)
    *numbers, state = args or [""]
    if not numbers or state not in STATES:
        sys.exit(
            f"usage: issue_state.py <number>... <{'|'.join(STATES)}> | --check <number>..."
            " | --muse-check <number>..."
            " | --release <tag> [--dry-run] | --blocked-by <issue> <blocker>"
        )
    if state == "In Progress" and not check(numbers):
        sys.exit("Refused: work only on eligible issues (see the docstring).")
    for number in numbers:
        print(set_state(number, state))


if __name__ == "__main__":
    main()

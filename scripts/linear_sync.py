"""Keep each issue's priority and workstream the same in Linear and on GitHub.

Linear's GitHub integration syncs the title, body, state, labels and comments of
each issue. It does not sync Linear's priority field or Linear projects. This
script syncs those two fields:

- Linear is the source of truth. Its priority becomes the issue's
  `priority: <name>` label, and the Priority field on the GitHub Roadmap project.
  Its project becomes the Roadmap's Workstream field.
- When Linear has no priority, a `priority:` label on GitHub sets it. When Linear
  has no project, the Roadmap's Workstream sets it. So an agent that files an
  issue with `gh` can give both.

The workflow `.github/workflows/linear-sync.yml` runs it every hour. Run it by
hand with `--dry-run` to print the changes without making them.

Environment: LINEAR_API_KEY, GH_TOKEN (labels), ROADMAP_TOKEN (the Roadmap project;
without it, the script skips the Roadmap).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import urllib.request
from dataclasses import dataclass, field

REPO = "8W9aG/openinterpretability"
ROADMAP_OWNER = "8W9aG"
ROADMAP_NUMBER = 2
LINEAR_API = "https://api.linear.app/graphql"
PRIORITY_LABEL = "priority: "
# Linear's priority numbers. 0 means no priority.
PRIORITIES = {1: "Urgent", 2: "High", 3: "Medium", 4: "Low"}


@dataclass
class Linked:
    """One issue as Linear and GitHub each see it."""

    url: str
    linear_id: str
    linear_priority: int
    linear_project: str | None
    gh_labels: list[str]
    roadmap_item: str | None = None
    roadmap_priority: str | None = None
    roadmap_workstream: str | None = None


@dataclass
class Actions:
    """The changes that make one issue the same in both places."""

    linear_priority: int | None = None
    linear_project: str | None = None
    add_labels: list[str] = field(default_factory=list)
    remove_labels: list[str] = field(default_factory=list)
    add_to_roadmap: bool = False
    roadmap_priority: str | None = None
    roadmap_workstream: str | None = None

    def empty(self) -> bool:
        """Return True when the issue needs no change."""
        return self == Actions()


def label_priority(labels: list[str]) -> int:
    """Return the Linear priority that the issue's `priority:` label names, or 0."""
    by_name = {v.lower(): k for k, v in PRIORITIES.items()}
    for lab in labels:
        if lab.startswith(PRIORITY_LABEL):
            return by_name.get(lab[len(PRIORITY_LABEL) :].strip().lower(), 0)
    return 0


def plan(issue: Linked, workstreams: set[str]) -> Actions:
    """Decide the changes for one issue. Linear wins when both sides have a value."""
    act = Actions()
    priority = issue.linear_priority or label_priority(issue.gh_labels)
    if priority and not issue.linear_priority:
        act.linear_priority = priority
    project = issue.linear_project or issue.roadmap_workstream
    if project and not issue.linear_project:
        act.linear_project = project
    if priority:
        want = PRIORITY_LABEL + PRIORITIES[priority].lower()
        stale = [lab for lab in issue.gh_labels if lab.startswith(PRIORITY_LABEL) and lab != want]
        act.remove_labels = stale
        if want not in issue.gh_labels:
            act.add_labels = [want]
    act.add_to_roadmap = issue.roadmap_item is None
    if priority and issue.roadmap_priority != PRIORITIES[priority]:
        act.roadmap_priority = PRIORITIES[priority]
    if project in workstreams and issue.roadmap_workstream != project:
        act.roadmap_workstream = project
    return act


def linear(query: str, variables: dict | None = None) -> dict:
    """Run one query on Linear's GraphQL API and return its data."""
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    headers = {"Content-Type": "application/json", "Authorization": os.environ["LINEAR_API_KEY"]}
    with urllib.request.urlopen(urllib.request.Request(LINEAR_API, body, headers)) as r:
        out = json.load(r)
    if "errors" in out:
        raise RuntimeError(out["errors"])
    return out["data"]


def gh(*args: str, token: str | None = None) -> str:
    """Run the GitHub CLI and return its output. A token, if given, replaces GH_TOKEN."""
    env = dict(os.environ, GH_TOKEN=token) if token else None
    r = subprocess.run(["gh", *args], capture_output=True, text=True, check=True, env=env)
    return r.stdout


def github_graphql(query: str, token: str | None, **variables: str) -> dict:
    """Run one query on GitHub's GraphQL API and return its data."""
    args = ["api", "graphql", "-f", f"query={query}"]
    for k, v in variables.items():
        args += ["-f", f"{k}={v}"]
    return json.loads(gh(*args, token=token))["data"]


LINEAR_ISSUES = """query($after: String) { issues(first: 100, after: $after) {
  pageInfo { hasNextPage endCursor }
  nodes { id priority project { name } attachments { nodes { url sourceType } } } } }"""


def linear_issues() -> dict[str, tuple[str, int, str | None]]:
    """Map each GitHub issue URL to its Linear issue's ID, priority and project."""
    out, after = {}, None
    while True:
        page = linear(LINEAR_ISSUES, {"after": after})["issues"]
        for n in page["nodes"]:
            for a in n["attachments"]["nodes"]:
                if a["sourceType"] == "github" and f"github.com/{REPO}/issues/" in a["url"]:
                    out[a["url"]] = (n["id"], n["priority"], (n["project"] or {}).get("name"))
        if not page["pageInfo"]["hasNextPage"]:
            return out
        after = page["pageInfo"]["endCursor"]


def github_labels() -> dict[str, list[str]]:
    """Map each GitHub issue URL to its label names."""
    rows = json.loads(
        gh(
            "issue",
            "list",
            "--repo",
            REPO,
            "--state",
            "all",
            "--limit",
            "5000",
            "--json",
            "url,labels",
        )
    )
    return {r["url"]: [lab["name"] for lab in r["labels"] or []] for r in rows}


ROADMAP = """query($owner: String!, $number: Int!, $after: String) { user(login: $owner) {
  projectV2(number: $number) { id fields(first: 50) { nodes {
    ... on ProjectV2SingleSelectField { id name options { id name } } } }
  items(first: 100, after: $after) { pageInfo { hasNextPage endCursor } nodes { id
    content { ... on Issue { url } }
    fieldValues(first: 20) { nodes { ... on ProjectV2ItemFieldSingleSelectValue {
      name field { ... on ProjectV2SingleSelectField { name } } } } } } } } } }"""


def roadmap(token: str) -> tuple[dict, dict[str, dict]]:
    """Return the Roadmap (ID and single-select fields) and its items by issue URL."""
    items, after, meta = {}, "", {}
    while True:
        args = [
            "api",
            "graphql",
            "-f",
            f"query={ROADMAP}",
            "-f",
            f"owner={ROADMAP_OWNER}",
            "-F",
            f"number={ROADMAP_NUMBER}",
        ] + (["-f", f"after={after}"] if after else [])
        proj = json.loads(gh(*args, token=token))["data"]["user"]["projectV2"]
        fields = {f["name"]: f for f in proj["fields"]["nodes"] if f}
        meta = {"id": proj["id"], "fields": fields}
        for n in proj["items"]["nodes"]:
            url = (n["content"] or {}).get("url")
            vals = {v["field"]["name"]: v["name"] for v in n["fieldValues"]["nodes"] if v}
            if url:
                items[url] = {"id": n["id"], **vals}
        page = proj["items"]["pageInfo"]
        if not page["hasNextPage"]:
            return meta, items
        after = page["endCursor"]


def collect(token: str | None) -> tuple[list[Linked], dict]:
    """Join each Linear issue to its GitHub labels and Roadmap item."""
    labels = github_labels()
    meta, items = roadmap(token) if token else ({}, {})
    out = []
    for url, (lid, prio, proj) in linear_issues().items():
        item = items.get(url, {})
        out.append(
            Linked(
                url,
                lid,
                prio,
                proj,
                labels.get(url, []),
                item.get("id"),
                item.get("Priority"),
                item.get("Workstream"),
            )
        )
    return out, meta


def apply_linear(issue: Linked, act: Actions, projects: dict[str, str]) -> None:
    """Set the Linear priority and project that the plan names."""
    update: dict = {}
    if act.linear_priority:
        update["priority"] = act.linear_priority
    if act.linear_project and act.linear_project in projects:
        update["projectId"] = projects[act.linear_project]
    if update:
        linear(UPDATE_ISSUE, {"id": issue.linear_id, "i": update})


def apply_labels(issue: Linked, act: Actions) -> None:
    """Add and remove the GitHub labels that the plan names."""
    args = []
    for lab in act.add_labels:
        args += ["--add-label", lab]
    for lab in act.remove_labels:
        args += ["--remove-label", lab]
    if args:
        gh("issue", "edit", issue.url, "--repo", REPO, *args)


UPDATE_ISSUE = """mutation($id: String!, $i: IssueUpdateInput!) {
  issueUpdate(id: $id, input: $i) { success } }"""
SET_FIELD = """mutation($p: ID!, $i: ID!, $f: ID!, $o: String!) { updateProjectV2ItemFieldValue(
  input: {projectId: $p, itemId: $i, fieldId: $f, value: {singleSelectOptionId: $o}}) {
  clientMutationId } }"""
ADD_ITEM = """mutation($p: ID!, $c: ID!) {
  addProjectV2ItemById(input: {projectId: $p, contentId: $c}) { item { id } } }"""


def apply_roadmap(issue: Linked, act: Actions, meta: dict, token: str) -> None:
    """Add the issue to the Roadmap if needed, then set its Priority and Workstream."""
    item = issue.roadmap_item
    if act.add_to_roadmap:
        node = json.loads(gh("issue", "view", issue.url, "--repo", REPO, "--json", "id"))["id"]
        added = github_graphql(ADD_ITEM, token, p=meta["id"], c=node)
        item = added["addProjectV2ItemById"]["item"]["id"]
    for name, value in (("Priority", act.roadmap_priority), ("Workstream", act.roadmap_workstream)):
        if value:
            f = meta["fields"][name]
            option = next(o["id"] for o in f["options"] if o["name"] == value)
            github_graphql(SET_FIELD, token, p=meta["id"], i=item, f=f["id"], o=option)


def main() -> None:
    """Plan and apply the changes for every issue that Linear links to GitHub."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--dry-run", action="store_true", help="print the changes and make none")
    dry = ap.parse_args().dry_run
    token = os.environ.get("ROADMAP_TOKEN")
    issues, meta = collect(token)
    options = meta.get("fields", {}).get("Workstream", {}).get("options", [])
    workstreams = {o["name"] for o in options}
    nodes = linear("{ projects(first: 250) { nodes { id name } } }")["projects"]["nodes"]
    projects = {p["name"]: p["id"] for p in nodes}
    for name in sorted(set(projects) - workstreams) if token else []:
        print(f"warning: no Roadmap Workstream option for Linear project {name!r}. Add it by hand.")
    changed = 0
    for issue in issues:
        act = plan(issue, workstreams)
        if not token:
            act.add_to_roadmap, act.roadmap_priority, act.roadmap_workstream = False, None, None
        if act.empty():
            continue
        changed += 1
        print(issue.url, {k: v for k, v in vars(act).items() if v})
        if not dry:
            apply_linear(issue, act, projects)
            apply_labels(issue, act)
            if token:
                apply_roadmap(issue, act, meta, token)
    print(f"{len(issues)} linked issues, {changed} {'to change' if dry else 'changed'}")


if __name__ == "__main__":
    main()

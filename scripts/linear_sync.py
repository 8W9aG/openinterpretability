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

It also syncs "blocked by" relations (#3842). Linear's integration doesn't sync them,
so Linear's "blocks" relations become GitHub's native issue dependencies. The rule for each
blocked issue is the one for priority. When Linear names any blockers for it, GitHub gets
exactly those. When Linear names none, GitHub's blockers are added to Linear, so a
dependency added with `gh` alone reaches Linear.

It also corrects one state. When a GitHub issue closes, Linear's integration can
move its Linear issue to Released, not Done. An issue is Released only when its
milestone is a published release (the `linear-released` workflow). So the script
moves an issue in Released back to Done when its milestone is not a published
release.

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
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import issue_state  # noqa: E402

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


def released_early(milestone: str | None, published: set[str]) -> bool:
    """Return True when an issue in Released has no milestone that is a published release."""
    return milestone not in published


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


RELEASED_ISSUES = """query($k: String!, $s: String!, $after: String) { issues(first: 100,
  after: $after, filter: {team: {key: {eq: $k}}, state: {name: {eq: $s}}}) {
  pageInfo { hasNextPage endCursor }
  nodes { id identifier attachments { nodes { url sourceType } } } } }"""


def released_issues() -> list[tuple[str, str, str]]:
    """Return the ID, identifier and GitHub issue URL of each linked Linear issue in Released."""
    out, after = [], None
    variables = {"k": issue_state.TEAM_KEY, "s": issue_state.RELEASED}
    while True:
        page = linear(RELEASED_ISSUES, {**variables, "after": after})["issues"]
        for n in page["nodes"]:
            for a in n["attachments"]["nodes"]:
                if a["sourceType"] == "github" and f"github.com/{REPO}/issues/" in a["url"]:
                    out.append((n["id"], n["identifier"], a["url"]))
                    break
        if not page["pageInfo"]["hasNextPage"]:
            return out
        after = page["pageInfo"]["endCursor"]


def published_releases() -> set[str]:
    """Return the tags of the repository's published releases. Drafts are not published."""
    rows = json.loads(
        gh("release", "list", "--repo", REPO, "--limit", "1000", "--json", "tagName,isDraft")
    )
    return {r["tagName"] for r in rows if not r["isDraft"]}


def github_milestones() -> dict[str, str | None]:
    """Map each GitHub issue URL to the title of its milestone, or None."""
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
            "url,milestone",
        )
    )
    return {r["url"]: (r["milestone"] or {}).get("title") for r in rows}


def unrelease(dry: bool) -> int:
    """Move each issue in Released to Done when its milestone is not a published release.

    Return the number of issues to move (dry run) or moved.
    """
    issues = released_issues()
    if not issues:
        return 0
    published, milestones = published_releases(), github_milestones()
    done = None if dry else issue_state.state_id(issue_state.DONE)
    moved = 0
    for linear_id, identifier, url in issues:
        milestone = milestones.get(url)
        if not released_early(milestone, published):
            continue
        moved += 1
        print(f"{url} {identifier}: milestone {milestone!r} is not released, so move to Done")
        if not dry:
            linear(UPDATE_ISSUE, {"id": linear_id, "i": {"stateId": done}})
    return moved


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


LINEAR_RELATIONS = """query($after: String) { issues(first: 100, after: $after) {
  pageInfo { hasNextPage endCursor }
  nodes { id attachments { nodes { url sourceType } }
    relations { nodes { type relatedIssue { id } } } } } }"""
GITHUB_DEPENDENCIES = """query($after: String) { repository(owner: "8W9aG",
  name: "openinterpretability") { issues(first: 100, after: $after) {
  pageInfo { hasNextPage endCursor }
  nodes { id url blockedBy(first: 50) { nodes { url } } } } } }"""
RELATE = """mutation($a: String!, $b: String!) { issueRelationCreate(
  input: {issueId: $a, relatedIssueId: $b, type: blocks}) { success } }"""
ADD_BLOCKED_BY = """mutation($i: ID!, $b: ID!) {
  addBlockedBy(input: {issueId: $i, blockingIssueId: $b}) { clientMutationId } }"""
REMOVE_BLOCKED_BY = """mutation($i: ID!, $b: ID!) {
  removeBlockedBy(input: {issueId: $i, blockingIssueId: $b}) { clientMutationId } }"""
Blockers = dict[str, set[str]]  # blocked issue's URL -> its blockers' URLs


def linear_relations() -> tuple[dict[str, str], Blockers]:
    """Each linked issue's Linear id by URL, and Linear's blockers of each issue."""
    ids, blocks, after = {}, [], None
    while True:
        page = linear(LINEAR_RELATIONS, {"after": after})["issues"]
        for n in page["nodes"]:
            mine = f"github.com/{REPO}/issues/"
            urls = [a["url"] for a in n["attachments"]["nodes"]
                    if a["sourceType"] == "github" and mine in a["url"]]  # fmt: skip
            if urls:
                ids[urls[0]] = n["id"]
            blocks += [(n["id"], r["relatedIssue"]["id"]) for r in n["relations"]["nodes"]
                       if r["type"] == "blocks" and r["relatedIssue"]]  # fmt: skip
        if not page["pageInfo"]["hasNextPage"]:
            break
        after = page["pageInfo"]["endCursor"]
    url_of = {lid: url for url, lid in ids.items()}
    out: Blockers = {}
    for blocker, blocked in blocks:
        if blocker in url_of and blocked in url_of:
            out.setdefault(url_of[blocked], set()).add(url_of[blocker])
    return ids, out


def github_dependencies() -> tuple[dict[str, str], Blockers]:
    """Each issue's GitHub node id by URL, and GitHub's blockers of each issue."""
    ids, out, after = {}, {}, ""
    while True:
        args = ["api", "graphql", "-f", f"query={GITHUB_DEPENDENCIES}"]
        page = json.loads(gh(*args, *(["-f", f"after={after}"] if after else [])))
        issues = page["data"]["repository"]["issues"]
        for n in issues["nodes"]:
            ids[n["url"]] = n["id"]
            if n["blockedBy"]["nodes"]:
                out[n["url"]] = {b["url"] for b in n["blockedBy"]["nodes"]}
        if not issues["pageInfo"]["hasNextPage"]:
            return ids, out
        after = issues["pageInfo"]["endCursor"]


def plan_relations(linear_side: Blockers, github_side: Blockers) -> list[tuple[str, str, str]]:
    """The changes, each (action, blocked URL, blocker URL), that make both sides the same.

    Linear wins for an issue it names blockers for. For any other issue, GitHub's blockers
    are copied to Linear.
    """
    changes = []
    for blocked in sorted(linear_side.keys() | github_side.keys()):
        want, have = linear_side.get(blocked, set()), github_side.get(blocked, set())
        if want:
            changes += [("github_add", blocked, b) for b in sorted(want - have)]
            changes += [("github_remove", blocked, b) for b in sorted(have - want)]
        else:
            changes += [("linear_add", blocked, b) for b in sorted(have)]
    return changes


def sync_relations(dry: bool) -> int:
    """Make the "blocked by" relations the same in Linear and on GitHub. Return the count."""
    linear_ids, linear_side = linear_relations()
    node_ids, github_side = github_dependencies()
    linked = set(linear_ids)
    github_side = {k: v & linked for k, v in github_side.items() if k in linked and v & linked}
    changes = plan_relations(linear_side, github_side)
    for action, blocked, blocker in changes:
        print(f"{blocked}: {action.replace('_', ' ')} blocked by {blocker}")
        if dry:
            continue
        if action == "linear_add":
            linear(RELATE, {"a": linear_ids[blocker], "b": linear_ids[blocked]})
        else:
            query = ADD_BLOCKED_BY if action == "github_add" else REMOVE_BLOCKED_BY
            github_graphql(query, None, i=node_ids[blocked], b=node_ids[blocker])
    return len(changes)


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
    early = unrelease(dry)
    print(f"{early} issues in Released before their release, {'to move' if dry else 'moved'}")
    related = sync_relations(dry)
    print(f"{related} blocked-by relations {'to change' if dry else 'changed'}")


if __name__ == "__main__":
    main()

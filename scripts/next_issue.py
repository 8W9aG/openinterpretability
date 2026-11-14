"""Pick the next issue to work: from the next release, or pulled into it when it has none.

The next release is the oldest open milestone. Its open issues are candidates when
nobody has started them (Linear state Todo or Backlog) and an agent may work on them
(``issue_state.py --check``). When there are none, every open issue without a milestone
is a candidate instead. The exception is `area: models`: no new model until every
configured model reaches its coverage target. Candidates rank by priority, then by the workstream
that the release has served least, then oldest first.

    python3 scripts/next_issue.py            # print the pick and the runners-up
    python3 scripts/next_issue.py --pull     # also add a pulled issue to the milestone
    python3 scripts/next_issue.py --json     # machine-readable
    python3 scripts/next_issue.py --unprioritised         # open issues with no priority
    python3 scripts/next_issue.py --prioritise 3525 high  # set the label and Linear's

Needs `gh` and LINEAR_API_KEY. The `work-an-issue` skill takes over from the pick.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import issue_state  # noqa: E402

PRIORITIES = {"urgent": 1, "high": 2, "medium": 3, "low": 4}  # Linear's numbering
NO_PRIORITY = 5
STARTED = ("In Progress", "In Review")
HELD_AREAS = ("area: models",)  # not pulled into a release (see the docstring)
RUNNERS_UP = 4


def gh_json(*args: str):
    """Run ``gh`` with ``args`` and parse its JSON output."""
    out = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def next_milestone() -> str | None:
    """The title of the oldest open milestone, which is the next release."""
    milestones = gh_json("api", f"repos/{issue_state.REPO}/milestones?state=open")
    return min((m["title"] for m in milestones), default=None)


def open_issues_all() -> list[dict]:
    """Every open issue, in a milestone or not."""
    fields = "number,title,labels,createdAt,milestone"
    return gh_json(
        "issue",
        "list",
        "--repo",
        issue_state.REPO,
        "--state",
        "open",
        "--limit",
        "1000",
        "--json",
        fields,
    )


def open_issues(milestone: str | None) -> list[dict]:
    """Open issues in ``milestone``, or with no milestone when it is None."""
    fields = "number,title,labels,createdAt,milestone"
    args = [
        "issue",
        "list",
        "--repo",
        issue_state.REPO,
        "--state",
        "open",
        "--limit",
        "1000",
        "--json",
        fields,
    ]
    issues = gh_json(*args, *(["--milestone", milestone] if milestone else []))
    return issues if milestone else [i for i in issues if not i.get("milestone")]


def linear_states() -> dict[str, dict]:
    """``{GitHub issue number: {"state": name, "priority": 1-4 or 0}}`` for open Linear issues."""
    query = (
        "query($k: String!, $a: String) { issues(first: 250, after: $a, filter: "
        '{team: {key: {eq: $k}}, state: {type: {nin: ["completed", "canceled"]}}}) '
        "{ pageInfo { hasNextPage endCursor } nodes { priority state { name } "
        "attachments { nodes { url } } } } }"
    )
    out, after = {}, None
    while True:
        page = issue_state.linear(query, {"k": issue_state.TEAM_KEY, "a": after})["issues"]
        for node in page["nodes"]:
            for attachment in node["attachments"]["nodes"]:
                if f"github.com/{issue_state.REPO}/issues/" in attachment["url"]:
                    number = attachment["url"].rsplit("/", 1)[1]
                    out[number] = {"state": node["state"]["name"], "priority": node["priority"]}
        if not page["pageInfo"]["hasNextPage"]:
            return out
        after = page["pageInfo"]["endCursor"]


def labels(issue: dict) -> list[str]:
    """The issue's label names."""
    return [label["name"] for label in issue.get("labels") or []]


def priority(issue: dict, linear: dict) -> int:
    """Linear's priority (1 urgent … 4 low), or the `priority:` label when Linear has none."""
    if linear.get("priority"):
        return linear["priority"]
    named = [
        PRIORITIES[n.split(": ", 1)[1]]
        for n in labels(issue)
        if n.startswith("priority: ") and n.split(": ", 1)[1] in PRIORITIES
    ]
    return min(named, default=NO_PRIORITY)


def area(issue: dict) -> str:
    """The issue's first `area:` label, or "none"."""
    return next((n for n in labels(issue) if n.startswith("area: ")), "none")


def rank(candidates: list[dict], states: dict, served: Counter) -> list[dict]:
    """Candidates by priority, then by how little their area has been served, then age.

    The order is greedy. Each issue placed counts against its area for the ones after it.
    So a run of picks spreads across workstreams, and no one area is drained first.
    """
    served, left, ordered = Counter(served), list(candidates), []

    def key(issue: dict):
        linear = states.get(str(issue["number"]), {})
        return (priority(issue, linear), served[area(issue)], issue["createdAt"])

    while left:
        best = min(left, key=key)
        ordered.append(best)
        left.remove(best)
        served[area(best)] += 1
    return ordered


def unstarted(issues: list[dict], states: dict) -> list[dict]:
    """Issues whose Linear state isn't In Progress or In Review, and not held back."""
    return [
        i
        for i in issues
        if states.get(str(i["number"]), {}).get("state") not in STARTED
        and issue_state.PROPOSED not in labels(i)
    ]


def pick(ranked: list[dict]) -> tuple[dict | None, list[dict]]:
    """The first eligible issue in ``ranked``, and up to RUNNERS_UP eligible ones after it."""
    chosen: list[dict] = []
    for issue in ranked:
        if issue_state.eligibility(str(issue["number"]))[0]:
            chosen.append(issue)
            if len(chosen) > RUNNERS_UP:
                break
    return (chosen[0], chosen[1:]) if chosen else (None, [])


def choose() -> dict:
    """Decide the next issue. Returns the milestone, whether it was pulled, the pick and others."""
    milestone = next_milestone()
    states = linear_states()
    in_release = open_issues(milestone) if milestone else []
    released = gh_json(
        "issue",
        "list",
        "--repo",
        issue_state.REPO,
        "--state",
        "all",
        "--limit",
        "1000",
        "--json",
        "labels",
        *(["--milestone", milestone] if milestone else []),
    )
    served = Counter(area(i) for i in released)
    best, others = pick(rank(unstarted(in_release, states), states, served))
    pulled = False
    if best is None:
        backlog = [i for i in open_issues(None) if area(i) not in HELD_AREAS]
        best, others = pick(rank(unstarted(backlog, states), states, served))
        pulled = best is not None
    return {"milestone": milestone, "pulled": pulled, "pick": best, "runners_up": others}


def unprioritised(issues: list[dict], states: dict) -> list[dict]:
    """Open issues that neither Linear nor a `priority:` label gives a priority."""
    return [i for i in issues if priority(i, states.get(str(i["number"]), {})) == NO_PRIORITY]


def set_priority(number: str, level: str) -> str:
    """Give issue ``number`` the `priority: <level>` label and the same priority in Linear."""
    subprocess.run(
        [
            "gh",
            "issue",
            "edit",
            number,
            "--repo",
            issue_state.REPO,
            "--add-label",
            f"priority: {level}",
        ],
        check=True,
        capture_output=True,
    )
    url = f"https://github.com/{issue_state.REPO}/issues/{number}"
    query = "query($u: String!) { attachmentsForURL(url: $u) { nodes { issue { id } } } }"
    nodes = issue_state.linear(query, {"u": url})["attachmentsForURL"]["nodes"]
    if not nodes:
        return f"#{number}: priority: {level} (not in Linear yet; the hourly sync copies it)"
    mutation = (
        "mutation($id: String!, $p: Int!) { issueUpdate(id: $id, input: {priority: $p}) "
        "{ success } }"
    )
    issue_state.linear(mutation, {"id": nodes[0]["issue"]["id"], "p": PRIORITIES[level]})
    return f"#{number}: priority: {level}"


def describe(issue: dict) -> str:
    """One line: number, title, area and priority label."""
    prio = next((n for n in labels(issue) if n.startswith("priority: ")), "no priority")
    return f"#{issue['number']} {issue['title']} ({area(issue)}, {prio})"


def pull_into(milestone: str, issue: dict) -> None:
    """Add ``issue`` to ``milestone`` and give it the release label."""
    version = milestone.lstrip("v")
    subprocess.run(
        [
            "gh",
            "issue",
            "edit",
            str(issue["number"]),
            "--repo",
            issue_state.REPO,
            "--milestone",
            milestone,
            "--add-label",
            f"release: {version}",
        ],
        check=True,
        capture_output=True,
    )


def main() -> None:
    """Print the next issue. With --pull, add a pulled issue to the milestone."""
    args = sys.argv[1:]
    if args[:1] == ["--prioritise"] and len(args) == 3 and args[2] in PRIORITIES:
        print(set_priority(args[1].lstrip("#"), args[2]))
        return
    if args[:1] == ["--unprioritised"]:
        for issue in unprioritised(open_issues_all(), linear_states()):
            print(describe(issue))
        return
    result = choose()
    if "--json" in sys.argv:
        print(json.dumps(result, indent=1))
        return
    if result["pick"] is None:
        sys.exit("No eligible issue, in the release or out of it.")
    where = f"pulled into {result['milestone']}" if result["pulled"] else result["milestone"]
    print(f"Next ({where}): {describe(result['pick'])}")
    for issue in result["runners_up"]:
        print(f"  then: {describe(issue)}")
    if result["pulled"] and "--pull" in sys.argv and result["milestone"]:
        pull_into(result["milestone"], result["pick"])
        print(f"Added #{result['pick']['number']} to {result['milestone']}.")
    elif result["pulled"]:
        print("Run again with --pull to add it to the release.")


if __name__ == "__main__":
    main()

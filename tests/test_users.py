"""Users pages: who made each experiment, from its contributor or its PR's author."""

import json
import subprocess
from datetime import timedelta

import pytest

from openinterp.builder import users
from openinterp.builder.agents import contributors_view
from openinterp.builder.compiler import compile_site
from openinterp.builder.tree import build_tree
from openinterp.builder.wiki import write_experiment_page
from openinterp.core.errata import record_ref


def later(record, hours, **submission):
    verdict = record.verdict.model_copy(
        update={"timestamp": record.verdict.timestamp + timedelta(hours=hours)}
    )
    sub = record.submission.model_copy(update=submission)
    return record.model_copy(update={"verdict": verdict, "submission": sub})


def test_the_recorded_contributor_wins_over_the_pr_author(record):
    named = later(record, 1, contributor="alice")
    (model,) = build_tree(
        [record, named], users={record_ref(record): "bob", record_ref(named): "bob"}
    )
    makers = {e.id: e.user for e in model.layers[0].units[0].experiments}
    assert makers == {"20261001T120000Z": "bob", "20261001T130000Z": "alice"}


def test_contributors_view_lists_each_persons_experiments_newest_first(record):
    reviewed = later(record, 2)
    reviewed = reviewed.model_copy(
        update={"verdict": reviewed.verdict.model_copy(update={"contributor": "carol"})}
    )
    (model,) = build_tree([record, later(record, 1), reviewed], users={record_ref(record): "bob"})
    experiments = model.layers[0].units[0].experiments
    people = {p.handle: p for p in contributors_view(experiments)}
    assert sorted(people) == ["bob", "carol"]
    assert people["bob"].experiments == 1 and people["bob"].href == "users/bob.html"
    assert [r["e"].id for r in people["carol"].reviewed] == ["20261001T140000Z"]
    assert people["carol"].experiments == 0 and people["carol"].reviews == 1


def test_the_site_has_a_page_per_user(tmp_path, record):
    compile_site([record], tmp_path, users={record_ref(record): "bob"})
    index = (tmp_path / "users" / "index.html").read_text()
    assert 'href="../users/bob.html">bob</a>' in index
    page = (tmp_path / "users" / "bob.html").read_text()
    assert "github.com/bob" in page and "1 experiment," in page
    assert "Fires on the closing quote" in page and "N117" in page
    agents = (tmp_path / "agents" / "index.html").read_text()
    assert 'href="../users/bob.html">bob</a>' in agents
    home = (tmp_path / "index.html").read_text()
    assert 'href="users/index.html" class=' in home and ">Users</a>" in home


def test_a_site_without_users_says_so(tmp_path, record):
    compile_site([record], tmp_path)
    assert "No users yet." in (tmp_path / "users" / "index.html").read_text()


def git(cwd, *args):
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd, check=True, capture_output=True,
    )  # fmt: skip


def test_merged_prs_reads_the_pr_number_from_merge_and_squash_commits(
    tmp_path, monkeypatch, record
):
    monkeypatch.chdir(tmp_path)
    git(tmp_path, "init", "-q", "-b", "main")
    write_experiment_page(record, tmp_path / "wiki")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "experiment: N117 (explore) (#12)")
    second = later(record, 1)
    git(tmp_path, "switch", "-qc", "branch")
    write_experiment_page(second, tmp_path / "wiki")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "an experiment")
    git(tmp_path, "switch", "-q", "main")
    git(tmp_path, "merge", "-q", "--no-ff", "branch", "-m", "Merge pull request #34 from me/branch")
    third = later(record, 2)
    write_experiment_page(third, tmp_path / "wiki")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "pushed straight to main")
    assert users.merged_prs("wiki") == {record_ref(record): 12, record_ref(second): 34}


def test_pr_authors_come_from_the_cache_then_github(monkeypatch):
    pages = [
        [
            {"number": 3, "user": {"login": "bob", "type": "User"}},
            {"number": 2, "user": {"login": "dependabot[bot]", "type": "Bot"}},
        ],
        [{"number": 1, "user": {"login": "alice", "type": "User"}}],
        [],
    ]
    asked = []

    def get(url):
        asked.append(url)
        return pages[len(asked) - 1]

    monkeypatch.setattr(users, "_get", get)
    assert users.pr_authors({1, 2, 3}, "o/r") == {1: "alice", 3: "bob"}
    assert len(asked) == 2 and "repos/o/r/pulls?" in asked[0]  # stops once it has all three
    cached = json.loads(users._cache_path().read_text())
    assert cached == {"1": "alice", "2": "", "3": "bob"}
    assert users.pr_authors({1, 2, 3}, "o/r") == {1: "alice", 3: "bob"}
    assert len(asked) == 2  # all from the cache


def test_pr_authors_without_a_network_names_nobody():
    assert users.pr_authors({1}, "o/r") == {}


@pytest.mark.parametrize("login", ["a@b.c", "-x"])
def test_logins_that_are_not_handles_are_dropped(monkeypatch, login):
    monkeypatch.setattr(
        users, "_get", lambda url: [{"number": 1, "user": {"login": login, "type": "User"}}]
    )
    assert users.pr_authors({1}, "o/r") == {}

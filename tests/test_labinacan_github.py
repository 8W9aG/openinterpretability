"""labinacan's GitHub reads: open PRs with gh, and PR authors with a cache (#6628)."""

import io
import subprocess

from labinacan.site import github


def test_open_prs_asks_gh_for_the_fields_and_reads_its_json(monkeypatch):
    seen = {}

    def run(args, **kw):
        seen["args"], seen["kw"] = args, kw
        return subprocess.CompletedProcess(args, 0, stdout='[{"number": 1}]')

    monkeypatch.setattr(github.shutil, "which", lambda name: "/bin/gh" if name == "gh" else None)
    monkeypatch.setattr(github.subprocess, "run", run)
    assert github.open_prs("number,url", limit=7) == [{"number": 1}]
    assert seen["args"] == ["gh", "pr", "list", "--state", "open", "--limit", "7",
                            "--json", "number,url"]  # fmt: skip
    assert seen["kw"] == {"check": True, "capture_output": True, "text": True, "timeout": 60}
    github.open_prs("number")
    assert seen["args"][6] == "300"  # the default limit


def test_open_prs_gives_none_without_gh_or_when_it_fails(monkeypatch):
    monkeypatch.setattr(github.shutil, "which", lambda name: None)
    assert github.open_prs("number") == []

    def fail(args, **kw):
        raise subprocess.TimeoutExpired(args, 60)

    monkeypatch.setattr(github.shutil, "which", lambda name: "/bin/gh")
    monkeypatch.setattr(github.subprocess, "run", fail)
    assert github.open_prs("number") == []


def test_get_json_sends_the_agent_and_the_token(monkeypatch):
    seen = []

    def urlopen(request, timeout):
        seen.append((request.full_url, dict(request.header_items()), timeout))
        return io.BytesIO(b'[{"n": 1}]')

    monkeypatch.setattr(github.urllib.request, "urlopen", urlopen)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "t")
    assert github.get_json("https://api.github.com/x", "lab") == [{"n": 1}]
    url, headers, timeout = seen[0]
    assert (url, timeout) == ("https://api.github.com/x", 30)
    assert headers == {"Accept": "application/vnd.github+json", "User-agent": "lab",
                       "Authorization": "Bearer t"}  # fmt: skip
    monkeypatch.setenv("GITHUB_TOKEN", "first")  # it comes before GH_TOKEN
    github.get_json("https://api.github.com/x", "lab")
    assert seen[1][1]["Authorization"] == "Bearer first"
    monkeypatch.delenv("GITHUB_TOKEN")
    monkeypatch.delenv("GH_TOKEN")
    github.get_json("https://api.github.com/x", "lab")
    assert "Authorization" not in seen[2][1]


def _pull(number, login, kind="User"):
    return {"number": number, "user": {"login": login, "type": kind}}


def test_authors_come_from_github_then_from_the_cache(tmp_path):
    cache, asked = tmp_path / "c" / "d" / "authors.json", []  # neither folder exists yet
    pages = {1: [_pull(3, "bob"), _pull(2, "bot", "Bot"), _pull(1, "bad!")], 2: []}

    def get(url):
        asked.append(url)
        return pages[int(url.rsplit("=", 1)[1])]

    found = github.pr_authors({1, 2, 3}, "o/r", cache=cache, get=get, valid=lambda s: "!" not in s)
    assert found == {3: "bob"}
    assert asked == [f"{github.API}/repos/o/r/pulls?state=closed&sort=created&direction=desc"
                     "&per_page=100&page=1"]  # fmt: skip
    assert cache.read_text() == '{"1": "", "2": "", "3": "bob"}'  # sorted, so it diffs well
    again = github.pr_authors({3}, "o/r", cache=cache, get=None, valid=None)  # asks nothing
    assert again == {3: "bob"}


def test_authors_page_until_found_or_the_last_page(tmp_path):
    asked = []
    pages = {1: [_pull(9, "a")], 2: [_pull(5, "b")], 3: []}

    def get(url):
        asked.append(int(url.rsplit("=", 1)[1]))
        assert len(asked) <= 3, "it asked for the same page again"  # fail fast, never loop
        return pages[asked[-1]]

    assert github.pr_authors({5}, "o/r", cache=None, get=get, valid=bool) == {5: "b"}
    assert asked == [1, 2]
    asked.clear()
    assert github.pr_authors({4}, "o/r", cache=None, get=get, valid=bool) == {}
    assert asked == [1, 2, 3]  # stops at the empty page


def test_without_a_network_the_cache_answers_and_a_warning_says_so(tmp_path, capsys):
    cache = tmp_path / "authors.json"
    cache.write_text('{"1": "alice"}')

    def offline(url):
        raise OSError("down")

    assert github.pr_authors({1, 2}, "o/r", cache=cache, get=offline, valid=bool) == {1: "alice"}
    assert capsys.readouterr().out == (
        "warning: can't read PR authors from GitHub (down); some users are unknown\n"
    )


def test_a_broken_cache_reads_as_empty(tmp_path):
    cache = tmp_path / "authors.json"
    cache.write_text("not json")
    get = lambda url: [_pull(1, "a")]  # noqa: E731
    assert github.pr_authors({1}, "o/r", cache=cache, get=get, valid=bool) == {1: "a"}


def test_a_pull_with_no_login_names_nobody(tmp_path):
    cache = tmp_path / "authors.json"
    pulls = [{"number": 1, "user": {"type": "User"}}]
    found = github.pr_authors({1}, "o/r", cache=cache, get=lambda url: pulls, valid=lambda s: True)
    assert found == {} and cache.read_text() == '{"1": ""}'


def test_the_cache_is_read_as_utf8_bytes(tmp_path):
    cache = tmp_path / "authors.json"
    cache.write_bytes('{"1": "zoë"}'.encode())
    assert github.pr_authors({1}, "o/r", cache=cache, get=None, valid=None) == {1: "zoë"}

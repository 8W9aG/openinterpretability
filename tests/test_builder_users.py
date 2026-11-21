"""PR authors from git history and GitHub's API, with the network replaced (builder/users.py)."""

import io
import json

from openinterpretability.builder import users
from openinterpretability.builder.users import _get as real_get


def test_a_folder_outside_git_names_no_prs(tmp_path, monkeypatch):
    # rev-parse prints nothing outside a repository, as it prints "true" in a shallow clone.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    assert users.merged_prs("wiki") == {}


def test_an_empty_page_ends_the_search(monkeypatch):
    asked = []
    monkeypatch.setattr(users, "_get", lambda url: asked.append(url) or [])
    assert users.pr_authors({7}, "o/r") == {}
    assert len(asked) == 1 and "page=1" in asked[0]
    assert json.loads(users._cache_path().read_text()) == {}


def test_with_the_cache_off_nothing_is_written(monkeypatch, tmp_path):
    monkeypatch.setenv(users.CACHE_ENV, "off")
    monkeypatch.chdir(tmp_path)
    pulls = [{"number": 7, "user": {"login": "bob", "type": "User"}}]
    monkeypatch.setattr(users, "_get", lambda url: pulls)
    assert users._cache_path() is None
    assert users.pr_authors({7}, "o/r") == {7: "bob"}
    assert list(tmp_path.iterdir()) == []


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _open(seen):
    def urlopen(request, timeout):
        seen.append((request, timeout))
        return _Response(b'[{"number": 1}]')

    return urlopen


def test_get_reads_json_with_the_token_when_one_is_set(monkeypatch):
    seen = []
    monkeypatch.setattr("urllib.request.urlopen", _open(seen))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_TOKEN", "t0ken")
    assert real_get("https://api.github.com/x") == [{"number": 1}]
    request, timeout = seen[0]
    assert request.full_url == "https://api.github.com/x" and timeout == 30
    assert request.get_header("Authorization") == "Bearer t0ken"
    assert request.get_header("Accept") == "application/vnd.github+json"


def test_get_sends_no_token_when_none_is_set(monkeypatch):
    seen = []
    monkeypatch.setattr("urllib.request.urlopen", _open(seen))
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert real_get("https://api.github.com/x") == [{"number": 1}]
    assert seen[0][0].get_header("Authorization") is None

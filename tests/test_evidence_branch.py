"""Evidence blobs ride on their own branch to the dataset (evidence_branch.py, #3254)."""

import json
import os
import subprocess

import pytest

from openinterpretability import evidence_branch as eb
from openinterpretability.core import evidence

GIT = ["git", "-c", "user.email=t@t", "-c", "user.name=t"]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    work = tmp_path / "repo"
    work.mkdir()
    subprocess.run([*GIT, "init", "-q"], cwd=work, check=True)
    monkeypatch.chdir(work)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "t")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@t")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "t")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@t")
    monkeypatch.setattr(evidence, "_fetch", lambda sha: None)  # never the network
    return work


def _slim_record(record, tmp_path, monkeypatch):
    store = tmp_path / "store"
    monkeypatch.setenv(evidence.DIR_ENV, str(store))
    slim = evidence.split(json.loads(record.model_dump_json()))
    path = tmp_path / "record.json"
    path.write_text(json.dumps(slim))
    return path, store


def test_blobs_go_to_a_parentless_commit_and_load_back(record, repo, tmp_path, monkeypatch):
    path, _ = _slim_record(record, tmp_path, monkeypatch)
    shas = eb.record_refs([path])
    assert shas  # the record refers to its top examples at least
    commit = eb.blob_commit(shas)
    parents = subprocess.run(["git", "rev-list", "--parents", "-n1", commit],
                             capture_output=True, text=True, check=True).stdout.split()  # fmt: skip
    assert parents == [commit]  # no parent: nothing of main's history comes with it
    names = subprocess.run(["git", "ls-tree", "-r", "--name-only", commit],
                           capture_output=True, text=True, check=True).stdout.split()  # fmt: skip
    assert sorted(names) == sorted(evidence.blob_name(s) for s in shas)
    fresh = tmp_path / "fresh-store"
    assert sorted(eb.load_branch(commit, fresh)) == sorted(shas)
    monkeypatch.setenv(evidence.DIR_ENV, str(fresh))
    assert eb.unresolved([path]) == []


def test_verify_names_a_record_whose_blob_is_missing(record, repo, tmp_path, monkeypatch):
    path, _ = _slim_record(record, tmp_path, monkeypatch)
    monkeypatch.setenv(evidence.DIR_ENV, str(tmp_path / "empty"))
    (problem,) = eb.unresolved([path])
    assert str(path) in problem and evidence.EVIDENCE_REPO in problem


def test_upload_creates_the_public_dataset_and_sends_each_blob(record, repo, tmp_path, monkeypatch):
    path, store = _slim_record(record, tmp_path, monkeypatch)
    shas = sorted(eb.record_refs([path]))
    calls = {}

    class FakeApi:
        exists = False

        def repo_exists(self, repo_id, repo_type):
            return self.exists

        def create_repo(self, repo_id, **kwargs):
            calls["create"] = (repo_id, kwargs)

        def upload_folder(self, repo_id, folder_path, **kwargs):
            from pathlib import Path

            calls["files"] = sorted(
                p.relative_to(folder_path).as_posix() for p in Path(folder_path).rglob("*.json")
            )

    assert eb.upload(shas, store, api=FakeApi()) == len(shas)
    assert calls["create"] == (evidence.EVIDENCE_REPO, {"repo_type": "dataset", "private": False})
    assert calls["files"] == sorted(evidence.blob_name(s) for s in shas)
    assert eb.upload([], store, api=FakeApi()) == 0
    calls.clear()
    existing = FakeApi()
    existing.exists = True  # a token with write access to the dataset only can't create it
    assert eb.upload(shas, store, api=existing) == len(shas)
    assert "create" not in calls


def test_a_draw_pushes_its_evidence_before_its_branch():
    from openinterpretability.draw_steps import _post

    commands = _post("k", "explore").commands
    assert commands.index("uv run open-interpretability evidence push") < commands.index("git push")


def test_a_draw_marks_its_pr_ready_before_its_last_push():
    # #6179: a push while the PR is a draft starts a run that skips every job. Marked ready
    # first, the last push starts a full run on the head commit.
    from openinterpretability.draw_steps import _post

    commands = _post("k", "explore").commands
    assert commands == [
        "uv run open-interpretability wiki",
        "make check lint test",
        'git add wiki && git commit -m "Explore k (<VERDICT>, <balanced accuracy>)"',
        "uv run open-interpretability evidence push",
        "gh pr ready",
        "git push",
    ]
    assert _post("k", "explore").title == "Post the result, whatever it is"
    assert _post("k", "explore").note == (
        "Commit the record's .sig if verify printed 'Signed as …'. `evidence push` sends the "
        "record's evidence blobs to their own branch, for CI to upload (issue #3254). Never "
        "abandon a claim silently: if you stop, say why in the PR and land it with only its "
        "ledger entry (wiki/ledger/), so the draw still counts. Don't close it."
    )
    assert not any("&&" in c and "gh pr ready" in c for c in commands)


def test_publish_uploads_only_what_records_need_and_the_dataset_lacks(
    record, repo, tmp_path, monkeypatch
):
    path, _ = _slim_record(record, tmp_path, monkeypatch)
    shas = sorted(eb.record_refs([path]))
    stray = evidence.put(b'{"not": "referenced"}')  # on the branch, named by no record
    commit = eb.blob_commit([*shas, stray])
    subprocess.run(["git", "update-ref", f"{eb.LOCAL_REFS}some-branch", commit], check=True)
    sent = []

    class FakeApi:
        def repo_exists(self, repo_id, repo_type):
            return True

        def list_repo_files(self, repo_id, repo_type):
            return [evidence.blob_name(shas[0])]  # one blob is already there

        def upload_folder(self, repo_id, folder_path, **kwargs):
            from pathlib import Path

            sent.extend(Path(p).stem for p in Path(folder_path).rglob("*.json"))

    fresh = tmp_path / "fresh-store"
    uploaded, missing = eb.publish_missing([path], fresh, api=FakeApi())
    assert sorted(sent) == shas[1:] and uploaded == len(shas) - 1 and missing == []
    assert stray not in sent


def test_load_all_loads_every_evidence_branch(record, repo, tmp_path, monkeypatch):
    from click.testing import CliRunner

    from openinterpretability.cli import cli

    path, _ = _slim_record(record, tmp_path, monkeypatch)
    shas = eb.record_refs([path])
    subprocess.run(["git", "update-ref", f"{eb.LOCAL_REFS}a", eb.blob_commit(shas)], check=True)
    fresh = tmp_path / "fresh-store"
    monkeypatch.setenv(evidence.DIR_ENV, str(fresh))
    monkeypatch.setattr(eb, "fetch_branches", lambda remote: None)
    result = CliRunner().invoke(cli, ["evidence", "load-all"])
    assert result.exit_code == 0, result.output
    assert f"Loaded {len(shas)} evidence blobs from 1 evidence branches." in result.output
    assert eb.unresolved([path]) == []


def test_a_blob_missing_from_the_dataset_is_read_from_its_evidence_branch(
    record, repo, tmp_path, monkeypatch
):
    # A record merged before the deploy published its blobs (#4530).
    path, store = _slim_record(record, tmp_path, monkeypatch)
    shas = eb.record_refs([path])
    commit = eb.blob_commit(shas)
    subprocess.run(["git", "update-ref", f"{eb.LOCAL_REFS}experiment/x", commit], check=True)
    for sha in shas:
        (store / evidence.blob_name(sha)).unlink()  # gone from the store, and not on the dataset
    monkeypatch.setenv(evidence.BRANCHES_ENV, "1")
    monkeypatch.setattr(evidence, "_BRANCHES", {"fetched": True})  # the branches are already here
    sha = next(iter(shas))
    assert evidence.sha256(evidence.get(sha)) == sha and (store / evidence.blob_name(sha)).exists()


def test_the_fallback_fetches_once_and_a_git_error_is_not_found(
    record, repo, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setenv(evidence.BRANCHES_ENV, "1")
    monkeypatch.setattr(evidence, "_BRANCHES", {})
    monkeypatch.setattr(eb, "fetch_branches", lambda *a: calls.append(1))
    monkeypatch.setenv(evidence.DIR_ENV, str(tmp_path / "empty"))
    for _ in range(2):
        with pytest.raises(evidence.EvidenceMissing):
            evidence.get("0" * 64)
    assert calls == [1]  # fetched once per process
    monkeypatch.setattr(evidence, "_BRANCHES", {})
    monkeypatch.setattr(eb, "fetch_branches", lambda *a: (_ for _ in ()).throw(OSError("no git")))
    with pytest.raises(evidence.EvidenceMissing):
        evidence.get("1" * 64)  # a git error is "not found", not a crash


def test_only_experiment_records_are_checked_for_blobs(monkeypatch):
    # An explanation record has no evidence blobs, and isn't an AuditRecord (#4793).
    from pathlib import Path

    from openinterpretability import cli_evidence_store

    added = [Path("wiki/experiments/m/L1/N2-positive/1.json"),
             Path("wiki/explanations/m/abc123/20261119T013311Z.json")]  # fmt: skip
    monkeypatch.setattr("openinterpretability.checks.changed_records", lambda *a: added)
    assert cli_evidence_store._added(Path("wiki"), "main") == added[:1]


def test_push_sends_the_blobs_to_their_branch_and_fetch_brings_them_back(
    record, repo, tmp_path, monkeypatch
):
    path, _ = _slim_record(record, tmp_path, monkeypatch)
    shas = eb.record_refs([path])
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    subprocess.run(["git", "remote", "add", "origin", str(remote)], check=True)
    assert eb.push("experiment/x", shas) == "evidence/experiment/x"
    listing = ["git", "ls-tree", "-r", "--name-only", "evidence/experiment/x"]
    names = subprocess.run(listing, cwd=remote, capture_output=True, text=True, check=True).stdout
    assert sorted(names.split()) == sorted(evidence.blob_name(s) for s in shas)
    eb.fetch_branches()
    refs = subprocess.run(["git", "for-each-ref", "--format=%(refname)", eb.LOCAL_REFS],
                          capture_output=True, text=True, check=True).stdout.split()  # fmt: skip
    assert refs == [f"{eb.LOCAL_REFS}experiment/x"]


def test_load_branch_with_only_copies_just_those_blobs(record, repo, tmp_path, monkeypatch):
    path, _ = _slim_record(record, tmp_path, monkeypatch)
    (sha,) = eb.record_refs([path])
    stray = evidence.put(b'{"not": "referenced"}')
    fresh = tmp_path / "fresh-store"
    assert eb.load_branch(eb.blob_commit([sha, stray]), fresh, only={sha}) == [sha]
    assert [p.stem for p in fresh.rglob("*.json")] == [sha]


def test_a_branch_blob_with_the_wrong_hash_is_refused(repo, tmp_path):
    def git(*args, env=None, data=None):
        done = subprocess.run(["git", *args], input=data, capture_output=True, check=True,
                              env={**os.environ, **(env or {})})  # fmt: skip
        return done.stdout.decode().strip()

    name = evidence.blob_name("0" * 64)
    oid = git("hash-object", "-w", "--stdin", data=b'{"tampered": true}')
    env = {"GIT_INDEX_FILE": str(tmp_path / "index")}
    git("update-index", "--add", "--cacheinfo", f"100644,{oid},{name}", env=env)
    commit = git("commit-tree", git("write-tree", env=env), "-m", "bad")
    with pytest.raises(evidence.EvidenceMissing, match=f"{commit}:{name} has the wrong hash"):
        eb.load_branch(commit, tmp_path / "store")
    assert not (tmp_path / "store").exists()


class _HubApi:
    """The Hub's client, without the network: no dataset yet, and it records uploads."""

    uploads = []

    def repo_exists(self, repo_id, repo_type):
        return False

    def create_repo(self, repo_id, **kwargs):
        pass

    def upload_folder(self, repo_id, folder_path, **kwargs):
        from pathlib import Path

        self.uploads.append(sorted(Path(p).stem for p in Path(folder_path).rglob("*.json")))


def test_upload_uses_the_hub_client_when_none_is_given(record, repo, tmp_path, monkeypatch):
    path, store = _slim_record(record, tmp_path, monkeypatch)
    shas = sorted(eb.record_refs([path]))
    monkeypatch.setattr("huggingface_hub.HfApi", _HubApi)
    monkeypatch.setattr(_HubApi, "uploads", [])
    assert eb.upload(shas, store) == len(shas)
    assert _HubApi.uploads == [shas]


def test_publish_with_no_dataset_reads_every_blob_from_the_first_branch_that_has_it(
    record, repo, tmp_path, monkeypatch
):
    path, _ = _slim_record(record, tmp_path, monkeypatch)
    shas = sorted(eb.record_refs([path]))
    monkeypatch.setattr("huggingface_hub.HfApi", _HubApi)
    monkeypatch.setattr(_HubApi, "uploads", [])
    fresh = tmp_path / "fresh-store"
    assert eb.publish_missing([path], fresh) == (0, shas)  # no branch holds them yet
    commit = eb.blob_commit(shas)
    for branch in ("a", "b"):  # the second branch is not read: nothing is left to find
        subprocess.run(["git", "update-ref", f"{eb.LOCAL_REFS}{branch}", commit], check=True)
    assert eb.publish_missing([path], fresh) == (len(shas), [])
    assert _HubApi.uploads == [shas]


def _bare_remote(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    subprocess.run(["git", "remote", "add", "origin", str(remote)], check=True)
    return remote


def _names(remote, branch="evidence/experiment/x"):
    listing = ["git", "ls-tree", "-r", "--name-only", branch]
    out = subprocess.run(listing, cwd=remote, capture_output=True, text=True, check=True).stdout
    return sorted(out.split())


def test_a_second_push_to_a_reused_branch_keeps_the_first_pushs_blobs(repo, tmp_path, monkeypatch):
    # #6166: a second revision of one neuron reused the branch. A push that replaced the
    # branch then lost a merged record's blob, before the deploy uploaded it.
    monkeypatch.setenv(evidence.DIR_ENV, str(tmp_path / "store"))
    first = evidence.put(b'{"first": 1}')
    second = evidence.put(b'{"second": 2}')
    remote = _bare_remote(tmp_path)
    eb.push("experiment/x", [first])
    (tmp_path / "store" / evidence.blob_name(first)).unlink()  # another machine pushes next
    eb.push("experiment/x", [second])
    assert _names(remote) == sorted(evidence.blob_name(s) for s in (first, second))


def test_a_push_reads_the_branch_again_when_another_push_lands_first(repo, tmp_path, monkeypatch):
    monkeypatch.setenv(evidence.DIR_ENV, str(tmp_path / "store"))
    first = evidence.put(b'{"first": 1}')
    second = evidence.put(b'{"second": 2}')
    remote = _bare_remote(tmp_path)
    reads = []
    real = eb.remote_head

    def stale_once(target, remote="origin"):
        reads.append(target)
        if len(reads) == 1:
            subprocess.run(
                ["git", "push", "-q", "origin", f"{eb.blob_commit([first])}:refs/heads/{target}"],
                check=True,
            )
            return None  # what the push read before the other push landed
        return real(target, remote)

    monkeypatch.setattr(eb, "remote_head", stale_once)
    eb.push("experiment/x", [second])
    assert len(reads) == 2  # the lease failed once, then the push read the branch again
    assert _names(remote) == sorted(evidence.blob_name(s) for s in (first, second))


def test_a_push_gives_up_when_other_pushes_keep_landing_first(repo, tmp_path, monkeypatch):
    monkeypatch.setenv(evidence.DIR_ENV, str(tmp_path / "store"))
    sha = evidence.put(b'{"x": 1}')
    other = evidence.put(b'{"y": 2}')
    _bare_remote(tmp_path)
    eb.push("experiment/x", [other])  # the branch's real head
    stale = eb.blob_commit([sha])  # a real commit, but never the branch's head
    monkeypatch.setattr(eb, "remote_head", lambda target, remote="origin": stale)
    with pytest.raises(RuntimeError, match="other pushes kept landing first"):
        eb.push("experiment/x", [sha])

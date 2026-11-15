"""Carry a pull request's evidence blobs to the dataset without them entering ``main`` (#3254).

A slim record (:mod:`openinterp.core.evidence`) refers to blobs by hash. The pull request
that adds it must not carry the blobs itself, or the squash merge would put them into
``main``'s history. So :func:`push` commits the blobs as one commit with no parent, and
pushes it to its own branch, ``evidence/<branch>``. The pull request's CI reads that branch
(:func:`load_branch`) and checks every record (:func:`unresolved`). It holds no write token.
After the merge, the deploy on ``main`` uploads the blobs that ``main``'s records refer to
and the public dataset lacks (:func:`publish_missing`). So the content of a pull request
that never merges is never published. A workflow deletes the evidence branch once its PR is
closed and its blobs are on the dataset. This uses git plumbing and a separate index, so it
never touches the work tree.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

from openinterp.core import evidence

BRANCH_PREFIX = "evidence/"


def _git(*args: str, env: dict | None = None, data: bytes | None = None) -> bytes:
    done = subprocess.run(
        ["git", *args], input=data, capture_output=True, check=True,
        env={**os.environ, **(env or {})},
    )  # fmt: skip
    return done.stdout


def record_refs(paths: Iterable[Path]) -> set[str]:
    """The blob hashes that the slim records at ``paths`` refer to."""
    found = set()
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for parent, field in evidence.FIELDS:
            value = (data.get(parent) or {}).get(field)
            if evidence.is_ref(value):
                found.add(value[evidence.REF_KEY])
    return found


def blob_commit(shas: Iterable[str], root: Path | None = None) -> str:
    """A commit with no parent whose tree holds each blob at its store path."""
    store = root or evidence.store_dir()
    with tempfile.TemporaryDirectory() as tmp:
        env = {"GIT_INDEX_FILE": str(Path(tmp) / "index")}
        for sha in sorted(shas):
            data = evidence.get(sha, store)  # checks the hash
            oid = _git("hash-object", "-w", "--stdin", data=data).decode().strip()
            info = f"100644,{oid},{evidence.blob_name(sha)}"
            _git("update-index", "--add", "--cacheinfo", info, env=env)
        tree = _git("write-tree", env=env).decode().strip()
    return _git("commit-tree", tree, "-m", "Evidence blobs (issue #3254)").decode().strip()


def push(branch: str, shas: Iterable[str], remote: str = "origin") -> str:
    """Push the blobs to ``evidence/<branch>`` on ``remote``; return the branch's name."""
    target = BRANCH_PREFIX + branch
    commit = blob_commit(shas)
    _git("push", "--force", remote, f"{commit}:refs/heads/{target}")
    return target


def load_branch(ref: str, root: Path | None = None, only: set[str] | None = None) -> list[str]:
    """Copy the blobs of ``ref`` into the local store, checking each hash; return them.

    With ``only``, it copies only the blobs whose hashes are in it.
    """
    loaded = []
    for line in _git("ls-tree", "-r", ref).decode().splitlines():
        meta, name = line.split("\t", 1)
        sha = Path(name).stem
        if only is not None and sha not in only:
            continue
        data = _git("cat-file", "blob", meta.split()[2])
        if evidence.sha256(data) != sha:
            raise evidence.EvidenceMissing(f"{ref}:{name} has the wrong hash")
        evidence.put(data, root)
        loaded.append(sha)
    return loaded


def upload(shas: Sequence[str], root: Path | None = None, api=None) -> int:
    """Upload the blobs to :data:`~openinterp.core.evidence.EVIDENCE_REPO`; return how many.

    The dataset is created, public, only if it doesn't exist yet. So a token with write
    access to that one dataset is enough once it exists.
    """
    if not shas:
        return 0
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    store = root or evidence.store_dir()
    if not api.repo_exists(evidence.EVIDENCE_REPO, repo_type="dataset"):
        api.create_repo(evidence.EVIDENCE_REPO, repo_type="dataset", private=False)
    with tempfile.TemporaryDirectory() as tmp:
        for sha in shas:
            target = Path(tmp) / evidence.blob_name(sha)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(evidence.get(sha, store))
        api.upload_folder(
            repo_id=evidence.EVIDENCE_REPO, repo_type="dataset", folder_path=tmp,
            commit_message=f"{len(shas)} evidence blobs (issue #3254)",
        )  # fmt: skip
    return len(shas)


LOCAL_REFS = "refs/evidence/"
"""Where :func:`fetch_branches` puts the remote's evidence branches."""


def fetch_branches(remote: str = "origin") -> None:
    """Fetch every ``evidence/*`` branch of ``remote`` to :data:`LOCAL_REFS`."""
    _git("fetch", "--quiet", "--force", remote, f"refs/heads/{BRANCH_PREFIX}*:{LOCAL_REFS}*")


def publish_missing(paths: Iterable[Path], root: Path | None = None, api=None):
    """Upload the blobs that the records at ``paths`` refer to and the dataset lacks.

    It reads them from the evidence branches under :data:`LOCAL_REFS`, so only blobs that
    a merged record names are published. Return ``(uploaded, still missing)``: the number
    uploaded, and the sorted hashes that no branch holds.
    """
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    wanted = record_refs(paths)
    have = set()
    if api.repo_exists(evidence.EVIDENCE_REPO, repo_type="dataset"):
        files = api.list_repo_files(evidence.EVIDENCE_REPO, repo_type="dataset")
        have = {Path(f).stem for f in files}
    need = wanted - have
    found: set[str] = set()
    refs = _git("for-each-ref", "--format=%(refname)", LOCAL_REFS).decode().split()
    for ref in refs:
        if need - found:
            found.update(load_branch(ref, root, only=need - found))
    return upload(sorted(found), root, api), sorted(need - found)


def unresolved(paths: Iterable[Path]) -> list[str]:
    """``path: reason`` for each record whose evidence can't be had or has a wrong hash."""
    from openinterp.core.schemas import AuditRecord

    problems = []
    for path in paths:
        try:
            AuditRecord.model_validate_json(Path(path).read_text(encoding="utf-8"))
        except evidence.EvidenceMissing as exc:
            problems.append(f"{path}: {exc}")
    return problems

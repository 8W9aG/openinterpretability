"""The patch store's refusals: wrong base, missing parent, bad branch names (#5229)."""

from __future__ import annotations

import pytest

from openinterpretability.patches.schema import Patch, PatchError
from openinterpretability.patches.store import PatchStore

BASE = "EleutherAI/pythia-70m@abc123"


def _patch(parents=(), base=BASE, **evidence) -> Patch:
    return Patch(base=base, parents=list(parents), evidence=evidence).sealed()


@pytest.fixture
def store(tmp_path):
    return PatchStore(tmp_path / "wiki", BASE)


def test_a_patch_on_another_base_model_is_refused(store):
    other = _patch(base="openai-community/gpt2@abc")
    with pytest.raises(PatchError, match="doesn't belong in"):
        store.write(other, sign=False)
    assert not store.path(other.id).exists()


def test_a_patch_whose_parent_is_missing_is_refused(store):
    orphan = _patch(parents=["f" * 64])
    with pytest.raises(PatchError, match="parent ffffffffffff is not in the store"):
        store.write(orphan, sign=False)


def test_writing_the_same_patch_twice_keeps_the_first_file(store):
    patch = _patch()
    path = store.write(patch, sign=False)
    path.write_text(path.read_text() + "\n")  # mark the file, to see that it is kept
    marked = path.read_text()
    assert store.write(patch, sign=False) == path
    assert path.read_text() == marked


def test_a_store_without_branches_lists_none(store):
    assert store.branches() == {}


@pytest.mark.parametrize("name", ["../escape", "-flag", "fix/../main", ""])
def test_a_bad_branch_name_is_refused(store, name):
    with pytest.raises(PatchError, match="not a branch name"):
        store.head(name)


def test_head_reads_a_branch_and_refuses_a_missing_one(store):
    patch = _patch(verdict="fixed")
    store.write(patch, sign=False)
    with pytest.raises(PatchError, match="no branch 'main'"):
        store.head("main")
    store.move("main", patch.id, create=True)
    assert store.head("main") == patch.id


def test_move_refuses_to_create_an_existing_branch_or_move_a_missing_one(store):
    patch = _patch(verdict="fixed")
    store.write(patch, sign=False)
    with pytest.raises(PatchError, match="branch 'fix' does not exist"):
        store.move("fix", patch.id)
    store.move("fix", patch.id, create=True)
    with pytest.raises(PatchError, match="branch 'fix' exists"):
        store.move("fix", patch.id, create=True)


def test_deleting_a_missing_branch_is_refused(store):
    with pytest.raises(PatchError, match="no branch 'gone'"):
        store.delete("gone")

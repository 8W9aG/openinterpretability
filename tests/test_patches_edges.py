"""Edges of the patch store, sessions and tensors that mutation testing found untested (#6491)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
import torch

from openinterpretability import signing
from openinterpretability.patches import schema, session, tensors
from openinterpretability.patches.schema import Op, Patch, PatchError
from openinterpretability.patches.store import PatchStore, model_slug, patches_dir

BASE = "org/model@rev"


def test_the_content_id_is_the_hash_of_one_canonical_json_text():
    # The id is a public name: any change to how the body is written changes every id.
    patch = Patch(base="m@r", message={"b": 1, "a": "é"})
    text = (
        '{"base":"m@r","evidence":{},"kind":"patch","message":{"a":"é","b":1},"ops":[],'
        f'"parents":[],"provenance":{{}},"rebased_from":null,"registry_version":{schema.REGISTRY_VERSION}}}'
    )
    assert patch.content_id() == hashlib.sha256(text.encode()).hexdigest()


def test_the_content_id_reads_parameters_in_their_json_form():
    # A parameter that only the JSON form can write, such as a path, still hashes.
    op = Op(op_id="m", type="add_module", params={"module": {"kind": "mlp"}, "where": Path("a/b")})
    assert len(Patch(base="m@r", ops=[op]).content_id()) == 64


def test_a_wrong_id_names_its_first_twelve_characters():
    patch = Patch(base="m@r", id="0123456789abcdefXYZ")
    with pytest.raises(PatchError, match=r"^patch 0123456789ab does not match"):
        patch.check_id()


def test_the_store_lays_out_patches_and_refs_by_these_exact_names(tmp_path):
    # Exact strings: a case-insensitive file system would hide "PATCHES" or "REFS".
    assert patches_dir(tmp_path) == tmp_path / "patches"
    assert model_slug("org/sub/model@rev@extra") == "org--sub--model"
    store = PatchStore(tmp_path, BASE)
    assert store.folder == tmp_path / "patches" / "org--model"
    assert store._ref("main") == store.folder / "refs" / "main"
    (store.folder / "refs").mkdir(parents=True)
    (store.folder / "refs" / "main").write_text("abc\n")
    assert store.branches() == {"main": "abc"}


def test_writing_signs_by_default_and_reads_back_in_utf8(tmp_path, monkeypatch):
    signed = []
    monkeypatch.setattr(signing, "auto_sign", signed.append)
    store = PatchStore(tmp_path, BASE)
    patch = Patch(base=BASE, message={"note": "café ✓"}).sealed()
    path = store.write(patch)
    assert signed == [path]
    assert path.read_text(encoding="utf-8") == patch.model_dump_json(indent=2) + "\n"
    assert store.get(patch.id).message == {"note": "café ✓"}


def test_a_full_id_is_read_directly_without_a_search(tmp_path, monkeypatch):
    store = PatchStore(tmp_path, BASE)
    patch = Patch(base=BASE).sealed()
    store.write(patch, sign=False)
    monkeypatch.setattr(Path, "glob", lambda *a: pytest.fail("a full id needs no search"))
    assert store.get(patch.id).id == patch.id


def test_main_names_a_refused_patch_by_its_first_twelve_characters(tmp_path):
    store = PatchStore(tmp_path, BASE)
    patch = Patch(base=BASE, evidence={"verdict": "not fixed"}).sealed()
    store.write(patch, sign=False)
    with pytest.raises(PatchError, match=re.escape(f"; {patch.id[:12]} is not fixed")):
        store.move("main", patch.id, create=True)


def test_a_draft_is_sealed_from_its_content_and_read_in_utf8(tmp_path, monkeypatch):
    draft = tmp_path / "draft.json"
    draft.write_text(Patch(base=BASE, id="stale", message={"n": "é"}).model_dump_json(), "utf-8")
    patch = session.read_draft(draft)
    assert patch.id == patch.content_id() and patch.message == {"n": "é"}


def test_a_session_reads_ancestors_from_the_store(tmp_path):
    store = PatchStore(tmp_path, BASE)
    parent = Patch(base=BASE).sealed()
    store.write(parent, sign=False)
    child = Patch(base=BASE, parents=[parent.id], message={"m": 1}).sealed()
    draft = tmp_path / "child.json"
    draft.write_text(child.model_dump_json(), encoding="utf-8")
    target = session.resolve(str(draft), tmp_path, "org/model")
    assert target.store.folder == store.folder
    assert target.lookup(parent.id).id == parent.id  # from the store
    assert target.lookup(target.patch.id) is target.patch
    with pytest.raises(PatchError, match="for org/model@rev, not org/other"):
        session.resolve(str(draft), tmp_path, "org/other")


def test_a_revision_may_hold_an_at_sign(tmp_path):
    # The model is what comes before the first "@": a revision such as "a@b" stays whole.
    draft = tmp_path / "d.json"
    draft.write_text(Patch(base="org/model@a@b").model_dump_json(), encoding="utf-8")
    assert session.resolve(str(draft), tmp_path, "org/model").patch.base == "org/model@a@b"


def test_the_tensor_store_lays_blobs_out_by_their_hash(tmp_path, monkeypatch):
    monkeypatch.setattr("openinterpretability.core.evidence.store_dir", lambda: tmp_path / "ev")
    assert tensors.store_root() == tmp_path / "ev" / "tensors"
    assert tensors.store_root(tmp_path / "x") == tmp_path / "x"
    ref = tensors.put_tensor(torch.ones(2), tmp_path / "x")
    sha = ref[tensors.KEY]
    assert (tmp_path / "x" / sha[:2] / f"{sha}.safetensors").exists()
    assert tensors._path(sha, None) == tmp_path / "ev" / "tensors" / sha[:2] / f"{sha}.safetensors"


def test_a_tensor_written_as_numbers_loads_as_float32():
    loaded = tensors.load([[1, 2], [3, 4]])
    assert loaded.dtype == torch.float32 and loaded.tolist() == [[1.0, 2.0], [3.0, 4.0]]

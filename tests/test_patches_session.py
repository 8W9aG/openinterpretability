import pytest
import torch

from openinterpretability.patches.schema import Op, Patch, PatchError
from openinterpretability.patches.session import resolve
from openinterpretability.patches.store import PatchStore
from openinterpretability.patches.tensors import put_tensor

BASE = "EleutherAI/pythia-70m@abc123"


@pytest.fixture(autouse=True)
def tensor_store(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENINTERPRETABILITY_EVIDENCE_DIR", str(tmp_path / "evidence"))


def stored_patch(wiki):
    op = Op(op_id="steer", type="steer", targets=["blocks.1.hook_resid_post"],
            params={"vector": put_tensor(torch.randn(4))})  # fmt: skip
    patch = Patch(base=BASE, parents=[], ops=[op]).sealed()
    PatchStore(wiki, BASE).write(patch, sign=False)
    return patch


def test_resolve_finds_a_stored_patch_by_its_id_prefix(tmp_path):
    wiki = tmp_path / "wiki"
    patch = stored_patch(wiki)
    target = resolve(patch.id[:10], wiki, "EleutherAI/pythia-70m")
    assert target.patch == patch
    assert target.lookup(patch.id) == patch


def test_resolve_refuses_a_stored_patch_for_another_model(tmp_path):
    wiki = tmp_path / "wiki"
    patch = stored_patch(wiki)
    # A store keyed by another model has its own folder, so the prefix is not found there.
    with pytest.raises(PatchError, match="no single patch"):
        resolve(patch.id[:10], wiki, "gpt2")

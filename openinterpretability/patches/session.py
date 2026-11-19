"""Run a command on a patched model: what ``--patch`` does (issue #5229).

``--patch`` names a stored patch (its id or a unique prefix) or a draft file. A draft is a
patch record with no ``id``, whose parents are already in the store. While the command runs,
the patch chain is applied to the loaded model (:func:`~.apply.applied`). The command's
result goes into the patch's evidence through ``--evidence-out``, never into the registry:
the patched model is not a registry model (#5220 §9.2). ``patch seal`` then writes the draft
with that evidence.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from openinterpretability.patches.history import replay
from openinterpretability.patches.schema import Patch, PatchError
from openinterpretability.patches.store import PatchStore


@dataclass(frozen=True)
class Target:
    """A patch to run with: its sealed form, and how to read its ancestors."""

    patch: Patch
    store: PatchStore

    def lookup(self, patch_id: str) -> Patch:
        """The patch ``patch_id``: this one, or an ancestor from the store."""
        return self.patch if patch_id == self.patch.id else self.store.get(patch_id)

    @property
    def key(self) -> str:
        """What caches keyed by the model must add, so a patched model has its own entries."""
        return f"+patch:{self.patch.id}"


def read_draft(path: Path) -> Patch:
    """A draft from ``path``, sealed so that it can be replayed. Its id isn't final yet."""
    patch = Patch.model_validate_json(path.read_text(encoding="utf-8"))
    return patch.model_copy(update={"id": ""}).sealed()


def resolve(name: str, wiki_dir: Path, model_name: str) -> Target:
    """The patch that ``name`` gives: a draft file, else a stored id or prefix."""
    path = Path(name)
    if path.is_file():
        patch = read_draft(path)
        store = PatchStore(wiki_dir, patch.base)
    else:
        store = PatchStore(wiki_dir, f"{model_name}@any")
        patch = store.get(name)
    if patch.base.split("@", 1)[0] != model_name:
        raise PatchError(f"the patch is for {patch.base}, not {model_name}")
    return Target(patch, store)


@contextmanager
def patched(model, target: Target | None):
    """Apply ``target`` to ``model`` for the ``with`` block; with None, do nothing."""
    if target is None:
        yield model
        return
    from openinterpretability.patches.apply import applied

    with applied(model, replay(target.patch.id, target.lookup)):
        yield model

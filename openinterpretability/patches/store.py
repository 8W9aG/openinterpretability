"""Patch records and branch refs in the wiki (issue #5229).

``wiki/patches/<model slug>/<id>.json`` holds each patch, beside the explanations of its
base model. It is signed as experiment records are (``signing.auto_sign``). A record is
write-once, because its name is its content's hash. ``refs/<name>`` holds a branch: the id
of the patch it points at. Each base model's ``main`` branch only moves to a patch whose
verdict is ``fixed`` (#5220 §9.1). So ``main`` is the accepted set of fixes for that base.
"""

from __future__ import annotations

import re
from pathlib import Path

from openinterpretability.patches.schema import Patch, PatchError, read_patch

MAIN = "main"
_BRANCH = re.compile(r"^[\w][\w./-]*$")


def patches_dir(wiki_dir: Path | str) -> Path:
    """Every base model's patches: ``patches/<model slug>/``."""
    return Path(wiki_dir) / "patches"


def model_slug(base: str) -> str:
    """The folder of a base model, from ``<model>@<revision>``."""
    return base.split("@", 1)[0].replace("/", "--")


class PatchStore:
    """The patches and branches of one base model."""

    def __init__(self, wiki_dir: Path | str, base: str):
        self.base = base
        self.folder = patches_dir(wiki_dir) / model_slug(base)

    def path(self, patch_id: str) -> Path:
        """Where the record of ``patch_id`` is."""
        return self.folder / f"{patch_id}.json"

    def write(self, patch: Patch, sign: bool = True) -> Path:
        """Write a sealed patch once, and sign it when this machine holds a trusted key."""
        from openinterpretability.signing import auto_sign

        patch.check_id()
        if model_slug(patch.base) != model_slug(self.base):
            raise PatchError(f"patch on {patch.base} doesn't belong in {self.folder}")
        for parent in patch.parents:
            if not self.path(parent).exists():
                raise PatchError(f"parent {parent[:12]} is not in the store")
        path = self.path(patch.id)
        if path.exists():
            return path  # the same content, already written
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(patch.model_dump_json(indent=2) + "\n", encoding="utf-8")
        if sign:
            auto_sign(path)
        return path

    def get(self, patch_id: str) -> Patch:
        """Read one patch, checking its hash. A unique prefix of the id is enough."""
        path = self.path(patch_id)
        if not path.exists():
            found = sorted(self.folder.glob(f"{patch_id}*.json"))
            if len(found) != 1:
                raise PatchError(f"no single patch {patch_id!r} in {self.folder}")
            path = found[0]
        return read_patch(path.read_text(encoding="utf-8"))

    def _ref(self, name: str) -> Path:
        if not _BRANCH.match(name) or ".." in name:
            raise PatchError(f"not a branch name: {name!r}")
        return self.folder / "refs" / name

    def branches(self) -> dict[str, str]:
        """Each branch and the patch id it points at."""
        root = self.folder / "refs"
        if not root.is_dir():
            return {}
        return {p.relative_to(root).as_posix(): p.read_text().strip()
                for p in sorted(root.rglob("*")) if p.is_file()}  # fmt: skip

    def head(self, name: str) -> str:
        """The patch id that branch ``name`` points at."""
        path = self._ref(name)
        if not path.exists():
            raise PatchError(f"no branch {name!r}")
        return path.read_text().strip()

    def move(self, name: str, patch_id: str, create: bool = False) -> None:
        """Point branch ``name`` at ``patch_id``. ``main`` takes only a fixed patch."""
        path = self._ref(name)
        if path.exists() == create:
            raise PatchError(f"branch {name!r} {'exists' if create else 'does not exist'}")
        patch = self.get(patch_id)
        if name == MAIN and patch.verdict != "fixed":
            short = patch.id[:12]
            raise PatchError(f"main only moves to a fixed patch; {short} is {patch.verdict}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(patch.id + "\n")

    def delete(self, name: str) -> None:
        """Remove branch ``name``. Its patches stay, as commits do."""
        path = self._ref(name)
        if not path.exists():
            raise PatchError(f"no branch {name!r}")
        path.unlink()

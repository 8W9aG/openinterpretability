"""The patch record, its operations and the versioned operation registry (issue #5229).

A patch never holds code. Each operation is declarative: a ``type`` from :data:`OP_TYPES`,
its ``targets`` (hook points or parameters of the canonical graph, :mod:`.graph`) and its
``params``. A new type, module or gate function is added here, in code, through a reviewed
PR, and :data:`REGISTRY_VERSION` goes up. A reader refuses an unknown type, and a patch that
needs a newer registry than this one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

REGISTRY_VERSION = 1
VERDICTS = ("fixed", "not fixed")  # evidence["verdict"], #5220 §9.1


@dataclass(frozen=True)
class OpType:
    """One operation type: its family, its required parameters, and how it combines."""

    family: str
    required: tuple[str, ...] = ()
    additive: bool = False  # residual-additive: two such ops at one hook point commute
    built: bool = True  # False: listed in the registry, but apply refuses it for now


OP_TYPES: dict[str, OpType] = {
    # Weight edits, on a parameter of the canonical graph.
    "lowrank_add": OpType("weight", ("A", "B"), additive=True),
    "dense_add": OpType("weight", ("delta",), additive=True),
    "sparse_add": OpType("weight", ("axis", "index", "values"), additive=True),
    "replace": OpType("weight", ("value",)),
    "scale": OpType("weight", ("factor",)),
    # Structural changes.
    "add_neurons": OpType("structural", ("W_in", "W_out"), additive=True),
    "add_head": OpType("structural", (), additive=True, built=False),
    "add_layer": OpType("structural", ("W1", "W2"), additive=True),
    "add_module": OpType("structural", ("module",), additive=True),
    "prune": OpType("structural", ("unit", "index")),
    "reroute": OpType("structural", (), built=False),
    # Gates, which other operations name by op_id.
    "gate": OpType("gate", ("function",)),
    # Memories.
    "table": OpType("memory", (), additive=True, built=False),
    "datastore": OpType("memory", (), additive=True, built=False),
    # Compiled modules and their adapters.
    "compiled": OpType("compiled", (), additive=True, built=False),
    "read_adapter": OpType("compiled", (), built=False),
    "write_adapter": OpType("compiled", (), built=False),
    # Activation interventions.
    "steer": OpType("activation", ("vector",), additive=True),
    "clamp": OpType("activation", ("index", "value")),
    "reft": OpType("activation", (), additive=True, built=False),
    # History: they change earlier operations, addressed as "<patch-id>:<op-id>".
    "set": OpType("history", ("params",)),
    "drop": OpType("history"),
    "order": OpType("history"),
}
GATE_FUNCTIONS = ("linear_probe", "threshold", "token_match", "attention_score", "mlp",
                  "and", "or", "not", "min", "product")  # fmt: skip
MODULES = ("linear", "mlp")  # the small blocks that add_module may insert


class PatchError(ValueError):
    """A patch that this reader can't accept, or a history operation that can't proceed."""


class Op(BaseModel):
    """One declarative alteration. ``gate`` names a gate op, in this patch or by full ref."""

    model_config = ConfigDict(extra="forbid")

    op_id: str
    type: str
    targets: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    gate: str | None = None
    identity_at_init: bool = False

    @model_validator(mode="after")
    def _known(self) -> Op:
        kind = OP_TYPES.get(self.type)
        if kind is None:
            raise ValueError(f"unknown operation type {self.type!r} (registry {REGISTRY_VERSION})")
        missing = [k for k in kind.required if k not in self.params]
        if missing:
            raise ValueError(f"{self.type} op {self.op_id!r} lacks {', '.join(missing)}")
        function = self.params.get("function") if self.type == "gate" else None
        if function is not None and function.get("kind") not in GATE_FUNCTIONS:
            raise ValueError(f"unknown gate function {function.get('kind')!r}")
        module = self.params.get("module") if self.type == "add_module" else None
        if module is not None and module.get("kind") not in MODULES:
            raise ValueError(f"unknown module kind {module.get('kind')!r}")
        if ":" in self.op_id:
            raise ValueError(f"op_id {self.op_id!r} may not hold ':', which separates a ref")
        return self


class Patch(BaseModel):
    """One patch record: the base, its parents, its ops and the evidence for it."""

    model_config = ConfigDict(extra="forbid")

    id: str = ""
    base: str  # "<model>@<revision>"
    parents: list[str] = Field(default_factory=list)
    registry_version: int = REGISTRY_VERSION
    ops: list[Op] = Field(default_factory=list)
    message: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    rebased_from: str | None = None
    kind: Literal["patch"] = "patch"

    @model_validator(mode="after")
    def _readable(self) -> Patch:
        if self.registry_version > REGISTRY_VERSION:
            raise ValueError(
                f"patch needs operation registry {self.registry_version}, "
                f"and this reader has {REGISTRY_VERSION}"
            )
        ids = [op.op_id for op in self.ops]
        if len(ids) != len(set(ids)):
            raise ValueError("op_id values must be unique in a patch")
        if "@" not in self.base:
            raise ValueError(f"base {self.base!r} must be '<model>@<revision>'")
        verdict = self.evidence.get("verdict")
        if verdict is not None and verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, not {verdict!r}")
        return self

    @property
    def verdict(self) -> str | None:
        """'fixed', 'not fixed', or None before the evidence is in."""
        return self.evidence.get("verdict")

    def content_id(self) -> str:
        """The SHA-256 of the record without ``id``: the patch's id, like a commit hash."""
        body = self.model_dump(mode="json", exclude={"id"})
        text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(text.encode()).hexdigest()

    def sealed(self) -> Patch:
        """This patch with its ``id`` set from its content."""
        return self.model_copy(update={"id": self.content_id()})

    def check_id(self) -> None:
        """Raise :class:`PatchError` if ``id`` is not the hash of the content."""
        if self.id != self.content_id():
            raise PatchError(f"patch {self.id[:12]} does not match its content hash")


def ref(patch_id: str, op_id: str) -> str:
    """The full address of an op: ``<patch-id>:<op-id>``."""
    return f"{patch_id}:{op_id}"


def read_patch(text: str) -> Patch:
    """Read and check one record. Refuse an unknown type, a newer registry or a bad id."""
    try:
        patch = Patch.model_validate_json(text)
    except ValueError as exc:
        raise PatchError(str(exc)) from exc
    patch.check_id()
    return patch

"""Patch history: replay, log, diff, merges, reverts and rebases (issue #5229).

Patches form a directed acyclic graph through ``parents``, as git commits do. Applying a
patch replays every ancestor from the base, in topological order (:func:`replay`). So a
revert only drops ops from the replayed history, and it is exact for every op type.

A merge is never trusted from its parents' evidence. :func:`conflicts` finds what blocks it,
and :func:`merge` refuses a parameter or order conflict that the merge doesn't resolve.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from openinterpretability.patches.schema import OP_TYPES, Op, Patch, PatchError, ref

Lookup = Callable[[str], Patch]
RESOLUTIONS = ("additive",)  # merge strategies built so far; TIES and task vectors are not


def ancestors(patch_id: str, lookup: Lookup) -> list[Patch]:
    """The patch and all its ancestors, parents before children (a topological order)."""
    order: list[Patch] = []
    seen: set[str] = set()

    def visit(pid: str, path: tuple[str, ...]) -> None:
        if pid in path:
            raise PatchError(f"patch {pid[:12]} is its own ancestor")
        if pid in seen:
            return
        patch = lookup(pid)
        for parent in patch.parents:
            visit(parent, (*path, pid))
        seen.add(pid)
        order.append(patch)

    visit(patch_id, ())
    bases = {p.base for p in order}
    if len(bases) != 1:
        raise PatchError(f"a patch chain mixes bases: {sorted(bases)}")
    return order


def _merged(old: dict, new: dict) -> dict:
    """``old`` with ``new``'s values, merging nested dicts."""
    out = dict(old)
    for key, value in new.items():
        both = isinstance(value, dict) and isinstance(out.get(key), dict)
        out[key] = _merged(out[key], value) if both else value
    return out


def _local(patch_id: str, op: Op) -> Op:
    """``op`` with its gate and its composite gates written as full refs."""

    def full(name: str) -> str:
        return name if ":" in name else ref(patch_id, name)

    params = op.params
    function = params.get("function")
    if op.type == "gate" and isinstance(function, dict) and "gates" in function:
        params = {**params, "function": {**function, "gates": [full(g) for g in function["gates"]]}}
    return op.model_copy(update={"gate": full(op.gate) if op.gate else None, "params": params})


def replay(patch_id: str, lookup: Lookup) -> list[tuple[str, Op]]:
    """The ops that applying ``patch_id`` installs, in order, each with its full ref."""
    ops: list[tuple[str, Op]] = []
    for patch in ancestors(patch_id, lookup):
        for raw in patch.ops:
            op = _local(patch.id, raw)
            _check_targets(op, ops)
            if op.type == "set":
                ops = [(r, _retuned(o, op) if r in op.targets else o) for r, o in ops]
            elif op.type == "drop":
                ops = [(r, o) for r, o in ops if r not in op.targets]
            elif op.type == "order":
                ops = _reordered(ops, op.targets)
            else:
                ops.append((ref(patch.id, op.op_id), op))
    return ops


def _retuned(old: Op, setter: Op) -> Op:
    return old.model_copy(update={"params": _merged(old.params, setter.params["params"])})


def _reordered(ops: list[tuple[str, Op]], order: Sequence[str]) -> list[tuple[str, Op]]:
    """``ops`` with the named refs moved, among their own places, into ``order``."""
    places = [i for i, (r, _) in enumerate(ops) if r in order]
    if len(places) != len(order):
        raise PatchError("an order op names refs that the history doesn't hold")
    by_ref = dict(ops)
    out = list(ops)
    for place, name in zip(places, order, strict=True):
        out[place] = (name, by_ref[name])
    return out


def _check_targets(op: Op, ops: list[tuple[str, Op]]) -> None:
    """A ``set`` or ``drop`` op must name ops that the history holds when it applies."""
    if op.type in ("set", "drop"):
        live = {r for r, _ in ops}
        missing = [t for t in op.targets if t not in live]
        if missing:
            raise PatchError(f"{op.type} op {op.op_id!r} names unknown ops {missing}")


def log(patch_id: str, lookup: Lookup) -> list[Patch]:
    """The patch's history, newest first."""
    return list(reversed(ancestors(patch_id, lookup)))


def diff(a: str, b: str, lookup: Lookup) -> dict[str, list[str]]:
    """The ops that only ``a`` applies, only ``b`` applies, and both with other params."""
    left, right = dict(replay(a, lookup)), dict(replay(b, lookup))
    changed = [r for r in left.keys() & right.keys() if left[r].params != right[r].params]
    return {"only_a": sorted(left.keys() - right.keys()),
            "only_b": sorted(right.keys() - left.keys()), "changed": sorted(changed)}  # fmt: skip


@dataclass(frozen=True)
class Conflict:
    """One reason two branches can't be merged as they are."""

    kind: str  # "parameter", "order" or "footprint"
    where: str
    refs: tuple[str, ...]


def _since(tip: str, common: set[str], lookup: Lookup) -> list[tuple[str, Op]]:
    """The ops that a branch applies beyond the patches both sides share."""
    shared = {r for pid in common for r, _ in replay(pid, lookup)}
    return [(r, op) for r, op in replay(tip, lookup) if r not in shared]


def _rows(op: Op) -> set[int] | None:
    """The rows or columns a sparse edit touches, or None for the whole parameter."""
    if op.type != "sparse_add":
        return None
    return {int(i) for i in op.params["index"]}


def _overlap(a: Op, b: Op) -> bool:
    ra, rb = _rows(a), _rows(b)
    same_axis = a.params.get("axis") == b.params.get("axis")
    return ra is None or rb is None or not same_axis or bool(ra & rb)


def conflicts(left: str, right: str, lookup: Lookup) -> list[Conflict]:
    """Parameter, order and footprint conflicts between two branch tips."""
    common = {p.id for p in ancestors(left, lookup)} & {p.id for p in ancestors(right, lookup)}
    a, b = _since(left, common, lookup), _since(right, common, lookup)
    found = []
    for ra, oa in a:
        for rb, ob in b:
            shared = set(oa.targets) & set(ob.targets)
            if not shared or oa.type == "gate" or ob.type == "gate":
                continue
            where = sorted(shared)[0]
            weight = OP_TYPES[oa.type].family == "weight" and OP_TYPES[ob.type].family == "weight"
            if weight and _overlap(oa, ob):
                found.append(Conflict("parameter", where, (ra, rb)))
            elif not (OP_TYPES[oa.type].additive and OP_TYPES[ob.type].additive):
                found.append(Conflict("order", where, (ra, rb)))
    found += _footprints(left, right, common, lookup)
    return found


def _footprints(left: str, right: str, common: set[str], lookup: Lookup) -> list[Conflict]:
    """Units in both sides' footprints: they need the merged interference check."""

    def units(tip: str) -> dict[str, str]:
        return {u: p.id for p in ancestors(tip, lookup) if p.id not in common
                for u in p.evidence.get("footprint", [])}  # fmt: skip

    ua, ub = units(left), units(right)
    return [Conflict("footprint", u, (ua[u], ub[u])) for u in sorted(ua.keys() & ub.keys())]


def merge(parents: Sequence[str], ops: Sequence[Op], message: dict[str, Any], lookup: Lookup,
          provenance: dict[str, Any] | None = None) -> Patch:  # fmt: skip
    """A merge patch of ``parents``. Refuse it while a conflict is unresolved.

    ``message["resolution"]`` resolves parameter conflicts (:data:`RESOLUTIONS`). An
    ``order`` op in ``ops`` resolves the order conflicts whose refs it names. A footprint
    conflict doesn't block the merge, but the message records it for the merge's own checks.
    """
    if len(parents) < 2:
        raise PatchError("a merge has two or more parents")
    base = lookup(parents[0]).base
    found = [c for i, a in enumerate(parents) for b in parents[i + 1 :]
             for c in conflicts(a, b, lookup)]  # fmt: skip
    ordered = {t for op in ops if op.type == "order" for t in op.targets}
    blocking = [c for c in found if c.kind == "order" and not set(c.refs) <= ordered]
    if message.get("resolution") not in RESOLUTIONS:
        blocking += [c for c in found if c.kind == "parameter"]
    if blocking:
        raise PatchError(f"unresolved merge conflicts: {blocking}")
    notes = [{"where": c.where, "refs": list(c.refs)} for c in found if c.kind == "footprint"]
    message = {**message, "footprint_conflicts": notes} if notes else dict(message)
    patch = Patch(base=base, parents=list(parents), ops=list(ops), message=message,
                  provenance=provenance or {})  # fmt: skip
    return patch.sealed()


def revert(patch_id: str, refs: Sequence[str], message: dict[str, Any], lookup: Lookup,
           provenance: dict[str, Any] | None = None) -> Patch:  # fmt: skip
    """A patch on ``patch_id`` that drops the ops ``refs`` from the replayed history."""
    live = {r for r, _ in replay(patch_id, lookup)}
    missing = [r for r in refs if r not in live]
    if missing:
        raise PatchError(f"nothing to revert at {missing}")
    op = Op(op_id="revert", type="drop", targets=list(refs))
    patch = Patch(base=lookup(patch_id).base, parents=[patch_id], ops=[op], message=message,
                  provenance=provenance or {})  # fmt: skip
    return patch.sealed()


def rebase(chain: Sequence[str], onto: str | None, base: str, lookup: Lookup) -> list[Patch]:
    """Copies of ``chain`` (oldest first, each the child of the one before) on a new parent.

    ``onto`` is the new parent, or None for the base itself; ``base`` is the new base
    revision. Each copy links its original (``rebased_from``), drops its evidence, which
    must be run again, and has its refs to earlier copies rewritten.
    """
    renamed: dict[str, str] = {}
    out: list[Patch] = []
    parent = onto
    for pid in chain:
        old = lookup(pid)
        ops = [_renamed(op, renamed) for op in old.ops]
        new = Patch(base=base, parents=[parent] if parent else [], ops=ops, message=old.message,
                    provenance=old.provenance, rebased_from=old.id).sealed()  # fmt: skip
        renamed[old.id] = new.id
        out.append(new)
        parent = new.id
    return out


def _renamed(op: Op, renamed: dict[str, str]) -> Op:
    def fix(name: str) -> str:
        pid, _, rest = name.partition(":")
        return f"{renamed[pid]}:{rest}" if rest and pid in renamed else name

    gate = fix(op.gate) if op.gate else None
    targets = [fix(t) for t in op.targets]
    return op.model_copy(update={"gate": gate, "targets": targets})

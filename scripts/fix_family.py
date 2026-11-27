"""The copy family and the measures that every fix arm shares (#7006, #5232).

``scripts/fix_arms.py`` fits the arms. This module reads the family's splits, encodes
prompts, and measures a model on them. ``Fit`` is one fitted change, which ``draft``
writes as a patch and ``checked`` reads back.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch

FAMILY = Path("docs/analyses/copy-family.jsonl")
MODEL = "EleutherAI/pythia-70m"
SHOTS = 4  # train prompts in arm E's prefix
SEED = 7006


@dataclass(frozen=True)
class Item:
    """One prompt of the family, with its right unit and the unit that the model copies."""

    prompt: str
    right: str
    copied: str


@dataclass
class Family:
    """The splits that the arms use: train and held-out wrong copies, and retain prompts."""

    train: list[Item]
    heldout: list[Item]
    retain: list[Item]
    right_heldout: list[Item] = field(default_factory=list)


@dataclass(frozen=True)
class Layout:
    """Where the arms act in one model: its copy heads, and its fact layers."""

    copy_heads: tuple[tuple[int, int], ...] = ((3, 3), (4, 0), (4, 7))
    fact_layers: tuple[int, ...] = (3, 4, 5)


def read_family(path: Path) -> Family:
    """The family's splits, from the JSONL file of #5232.

    Args:
        path: The family's file.

    Examples:
        >>> import tempfile
        >>> rows = [{"table": "metadata"},
        ...         {"table": "prompt", "prompt": "p", "right": " a", "copied": " b",
        ...          "kind": "wrong_copy", "split": "train"},
        ...         {"table": "prompt", "prompt": "q", "right": " c", "copied": " c",
        ...          "kind": "right_copy", "split": "retain"}]
        >>> f = Path(tempfile.mkdtemp()) / "family.jsonl"
        >>> _ = f.write_text("".join(json.dumps(r) + "\\n" for r in rows))
        >>> family = read_family(f)
        >>> [i.prompt for i in family.train], [i.prompt for i in family.retain]
        (['p'], ['q'])
    """
    splits: dict[tuple[str, str], list[Item]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("table") == "prompt":
            item = Item(row["prompt"], row["right"], row["copied"])
            splits.setdefault((row["kind"], row["split"]), []).append(item)
    return Family(splits.get(("wrong_copy", "train"), []),
                  splits.get(("wrong_copy", "heldout"), []),
                  splits.get(("right_copy", "retain"), []),
                  splits.get(("right_copy", "heldout"), []))  # fmt: skip


def token(tokenizer, unit: str) -> int:
    """The first token of ``unit``, which is the unit's only token in the family's model.

    Args:
        tokenizer: The model's tokenizer.
        unit: The unit's text, with its leading space.
    """
    return int(tokenizer.encode(unit, add_special_tokens=False)[0])


def encode(tokenizer, prompts: list[str], device) -> tuple[torch.Tensor, ...]:
    """The prompts as right-padded ids, their mask, and each prompt's last position.

    Args:
        tokenizer: The model's tokenizer.
        prompts: The prompts.
        device: Where the tensors go.
    """
    rows = [tokenizer.encode(p, add_special_tokens=False) for p in prompts]
    width = max(map(len, rows))
    ids = torch.tensor([r + [0] * (width - len(r)) for r in rows], device=device)
    mask = torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows], device=device)
    return ids, mask, mask.sum(1) - 1


def last_logits(model, tokenizer, prompts: list[str]) -> torch.Tensor:
    """The logits at each prompt's last position, ``[n, vocab]``, as float32.

    Args:
        model: The model, with any changes hooked in.
        tokenizer: Its tokenizer.
        prompts: The prompts.
    """
    device = next(model.parameters()).device
    ids, mask, last = encode(tokenizer, prompts, device)
    logits = model(input_ids=ids, attention_mask=mask).logits.float()
    return logits[torch.arange(len(prompts), device=device), last]


@torch.no_grad()
def measures(model, tokenizer, items: list[Item], prefix: str = "") -> dict:
    """The share of ``items`` whose top token is the right unit, and the mean p(right).

    Args:
        model: The model, with any change hooked in.
        tokenizer: Its tokenizer.
        items: The prompts.
        prefix: Text put before each prompt, for arm E.
    """
    if not items:
        return {"n": 0, "top1": None, "p_right": None}
    probs = torch.softmax(last_logits(model, tokenizer, [prefix + i.prompt for i in items]), -1)
    right = torch.tensor([token(tokenizer, i.right) for i in items], device=probs.device)
    rows = torch.arange(len(items), device=probs.device)
    return {"n": len(items), "top1": round(float((probs.argmax(-1) == right).float().mean()), 4),
            "p_right": round(float(probs[rows, right].mean()), 4)}  # fmt: skip


def evaluate(model, tokenizer, family: Family, prefix: str = "") -> dict:
    """:func:`measures` on each split of the family.

    Args:
        model: The model, with any change hooked in.
        tokenizer: Its tokenizer.
        family: The family.
        prefix: Text put before each prompt, for arm E.
    """
    return {name: measures(model, tokenizer, getattr(family, name), prefix)
            for name in ("train", "heldout", "retain", "right_heldout")}  # fmt: skip


def few_shot(family: Family, shots: int = SHOTS) -> str:
    """Arm E's prefix: the first ``shots`` train prompts, each completed with its right unit.

    Args:
        family: The family.
        shots: The number of examples.

    Examples:
        >>> items = [Item("A has two", " legs", " arms"), Item("B has one", " head", " eye")]
        >>> f = Family(items, [], [])
        >>> few_shot(f, 2)
        'A has two legs. B has one head. '
    """
    return "".join(f"{i.prompt}{i.right}. " for i in family.train[:shots])


@dataclass
class Fit:
    """One fitted change: its arm, its settings and its low-rank factors per site."""

    arm: str
    settings: dict
    ops: list[tuple[str, torch.Tensor, torch.Tensor]]  # (target, A, B), with ΔW = B @ A
    losses: list[float] = field(default_factory=list)

    @property
    def name(self) -> str:
        """The fit's file stem, such as ``B-l0.1-r4``."""
        parts = [self.arm] + [f"{k[0]}{v}" for k, v in self.settings.items()]
        return "-".join(parts)


def draft(fit: Fit, base: str, family: Family) -> dict:
    """The fit as a draft patch: one ``lowrank_add`` op per site, with its tensors stored.

    Args:
        fit: The fit.
        base: ``<model>@<revision>``.
        family: The family, whose train prompts the message lists.
    """
    from openinterpretability.patches.schema import Op, Patch
    from openinterpretability.patches.tensors import put_tensor

    ops = [Op(op_id=f"{fit.arm.lower()}{i}", type="lowrank_add", targets=[target],
              params={"A": put_tensor(a), "B": put_tensor(b)})
           for i, (target, a, b) in enumerate(fit.ops)]  # fmt: skip
    message = {"failure": "copy family (#5232)", "issue": 7006, "arm": fit.arm,
               "settings": fit.settings, "train": [i.prompt for i in family.train]}  # fmt: skip
    patch = Patch(base=base, ops=ops, message=message,
                  provenance={"by": "scripts/fix_arms.py", "issue": 7006})  # fmt: skip
    return patch.model_dump(mode="json", exclude={"id"})


def checked(path: Path, model) -> tuple[object, list]:
    """Read the draft at ``path`` back as #5229 does, and check it at zero strength.

    Args:
        path: The draft file.
        model: The base model.

    Returns:
        The sealed patch and its replayed ops.

    Raises:
        ZeroStartError: The patch changes the model at zero strength.
    """
    from openinterpretability.patches.apply import zero_start_check
    from openinterpretability.patches.history import replay
    from openinterpretability.patches.session import read_draft

    patch = read_draft(path)
    ops = replay(patch.id, {patch.id: patch}.__getitem__)
    device = next(model.parameters()).device
    zero_start_check(model, ops, torch.tensor([[1, 2, 3, 4]], device=device))
    return patch, ops

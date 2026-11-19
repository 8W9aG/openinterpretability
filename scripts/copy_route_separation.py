"""Do the #4781 route's units tell a wrong copy from a right one? (issue #5074).

pythia-70m answers "A week has seven days and a year has twelve" with " days". It copies the
first clause's unit, and here the copy is wrong. On "A dog has four legs and a cat has four"
the same copy is right. This script builds many prompts of one template from
:data:`FACTS`, and keeps those whose top token is the first clause's unit. So every kept
prompt is a copy, and the two groups differ only in whether the copy is right.

At the answer position it measures:

1. ``units``: each unit of the VERIFIED route (:data:`ROUTE_RECORD`). A neuron pole's
   activation and a head's strength |W_O(z − z̄)|, and their AUROC between the groups. Its
   null is every other unit of the same layer and kind.
2. ``directions``: after each layer, the residual stream's difference in group means, as a
   direction. It scores the direction by leave-one-out balanced accuracy, against
   :data:`RANDOM` seeded random directions.

Both measures are repeated on the object pairs alone (``_objects``), with no unit of time.

The rows go to ``--out``::

    uv run python scripts/copy_route_separation.py --out docs/analyses/copy-route-separation.jsonl
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import click
import torch

MODEL = "EleutherAI/pythia-70m"
ROUTE_RECORD = Path("wiki/answers/EleutherAI--pythia-70m/970a91e11d8c/20261118T234629Z.json")
RANDOM = 200  # random directions in the null
SEED = 0
ACTIVE = 0.1  # a null unit must be active on this share of the prompts
MEAN_TOKENS = 20_000  # corpus tokens for the heads' mean outputs, as explain uses
# (subject, count, unit): facts whose units share a category can be copied rightly.
FACTS = {
    "legs": [("spider", "eight"), ("ant", "six"), ("dog", "four"), ("cat", "four"),
             ("bird", "two"), ("horse", "four"), ("cow", "four"), ("chicken", "two"),
             ("beetle", "six"), ("duck", "two"), ("lion", "four"), ("table", "four")],
    "wheels": [("car", "four"), ("bike", "two"), ("tricycle", "three"), ("truck", "six"),
               ("wagon", "four"), ("bus", "six"), ("cart", "two"), ("scooter", "two")],
    "sides": [("triangle", "three"), ("square", "four"), ("pentagon", "five"),
              ("hexagon", "six"), ("octagon", "eight")],
    "strings": [("guitar", "six"), ("violin", "four"), ("banjo", "five"), ("harp", "forty")],
}  # fmt: skip
# Units of time, and other pairs whose second unit differs: copying them is wrong.
TIME = [("minute", "sixty", "seconds"), ("hour", "sixty", "minutes"), ("week", "seven", "days"),
        ("year", "twelve", "months"), ("decade", "ten", "years"), ("fortnight", "two", "weeks"),
        ("day", "twenty", "hours")]  # fmt: skip
OTHER = [("A hand has five fingers and a foot has five", " fingers", " toes"),
         ("A dozen has twelve eggs and a year has twelve", " eggs", " months")]  # fmt: skip


def article(word: str) -> str:
    """The article for ``word``: "An" before a vowel, else "A"."""
    return "An" if word[0] in "aeiou" else "A"


def candidates() -> list[dict]:
    """Every two-clause prompt: its copied unit, its right answer, and whether they match."""
    facts = [(s, n, u) for u, items in FACTS.items() for s, n in items] + TIME
    out = []
    for (s1, n1, u1), (s2, n2, u2) in itertools.permutations(facts, 2):
        prompt = f"{article(s1)} {s1} has {n1} {u1} and {article(s2).lower()} {s2} has {n2}"
        out.append({"prompt": prompt, "copied": f" {u1}", "right": f" {u2}"})
    out += [{"prompt": p, "copied": c, "right": r} for p, c, r in OTHER]
    return out


def single(tokenizer, text: str) -> int | None:
    """The id of ``text`` when it is one token, else None."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


@torch.no_grad()
def state(runner, means, prompt: str) -> tuple[int, dict, dict, torch.Tensor]:
    """The top token, every neuron and head value, and each layer's residual stream, last."""
    from openinterpretability.explain import _captured

    enc = runner.encode(prompt)
    out = runner.model(**enc, output_hidden_states=True)
    top = int(out.logits[0, -1].argmax())
    _, neurons, heads = _captured(runner, enc, True)
    acts = {la: a[0, -1].float().cpu() for la, a in neurons.items()}
    strengths = {}
    for la, z in heads.items():
        d = runner.head_dim(la)
        for h in range(runner.n_heads):
            cols = slice(h * d, (h + 1) * d)
            off = z[0, -1, cols].float().cpu() - means.heads[la][cols]
            strengths[(la, h)] = float((runner.head_output_weights(la, h).float() @ off).norm())
    resid = torch.stack([s[0, -1].float().cpu() for s in out.hidden_states[1:]])
    return top, acts, strengths, resid


def auroc(right: list[float], wrong: list[float]) -> float:
    """P(a right-copy value > a wrong-copy value), with ties as half."""
    wins = sum((a > b) + 0.5 * (a == b) for a in right for b in wrong)
    return wins / (len(right) * len(wrong))


def parse(label: str) -> tuple[int, int, str]:
    """``(layer, index, pole or "head")`` from a unit label such as ``L3/N7+``."""
    from openinterpretability.core.answer_records import parse_unit

    return parse_unit(label)


def unit_rows(route: list[str], kept: list[dict]) -> list[dict]:
    """Measure 1: each route unit's AUROC, and its percentile among units of its kind."""
    right = [k for k in kept if k["is_right"]]
    wrong = [k for k in kept if not k["is_right"]]

    def value(k, layer, index, pole):
        if pole == "head":
            return k["heads"][(layer, index)]
        a = float(k["acts"][layer][index])
        return max(a, 0.0) if pole == "positive" else max(-a, 0.0)

    def score(layer, index, pole):
        return auroc([value(k, layer, index, pole) for k in right],
                     [value(k, layer, index, pole) for k in wrong])  # fmt: skip

    null_cache: dict[tuple, list[float]] = {}
    rows = []
    for label in route:
        layer, index, pole = parse(label)
        key = (layer, pole == "head", pole)
        if key not in null_cache:
            width = len(kept[0]["acts"][layer]) if pole != "head" else None
            pool = range(8) if pole == "head" else range(width)
            # Only units active on ACTIVE of the prompts: a silent unit's AUROC is 0.5
            need = ACTIVE * len(kept)
            pool = [i for i in pool if sum(value(k, layer, i, pole) > 0 for k in kept) >= need]
            null_cache[key] = [abs(score(layer, i, pole) - 0.5) for i in pool]
        a = score(layer, index, pole)
        null = null_cache[key]
        pct = sum(x < abs(a - 0.5) for x in null) / len(null)
        rows.append({"unit": label, "auroc": round(a, 4), "null_percentile": round(pct, 4),
                     "beyond_null_95": pct >= 0.95})  # fmt: skip
    return rows


def loo_accuracy(x: torch.Tensor, y: torch.Tensor, direction=None) -> float:
    """Leave-one-out balanced accuracy of the difference-in-means direction, or a fixed one."""
    hits = torch.zeros(2)
    for i in range(len(y)):
        keep = torch.arange(len(y)) != i
        xr, yr = x[keep], y[keep]
        mu1, mu0 = xr[yr].mean(0), xr[~yr].mean(0)
        d = (mu1 - mu0) if direction is None else direction
        mid = ((mu1 + mu0) / 2) @ d
        sign = 1.0 if (mu1 @ d) >= (mu0 @ d) else -1.0
        hits[int(y[i])] += int(((x[i] @ d - mid) * sign > 0) == bool(y[i]))
    counts = torch.tensor([float((~y).sum()), float(y.sum())])
    return float((hits / counts).mean())  # balanced: the groups differ in size


def direction_rows(kept: list[dict], n_layers: int) -> list[dict]:
    """Measure 2: per layer, the difference-in-means direction against random directions."""
    y = torch.tensor([k["is_right"] for k in kept])
    gen = torch.Generator().manual_seed(SEED)
    rows = []
    for layer in range(n_layers):
        x = torch.stack([k["resid"][layer] for k in kept])
        acc = loo_accuracy(x, y)
        null = [loo_accuracy(x, y, torch.randn(x.shape[1], generator=gen)) for _ in range(RANDOM)]
        rows.append({"layer": layer, "loo_accuracy": round(acc, 4),
                     "null_mean": round(sum(null) / len(null), 4),
                     "null_95": round(sorted(null)[int(0.95 * len(null)) - 1], 4),
                     "beats_null": acc > sorted(null)[int(0.95 * len(null)) - 1]})  # fmt: skip
    return rows


@click.command()
@click.option("--corpus", type=click.Path(dir_okay=False, path_type=Path), default="corpus.jsonl")
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(corpus: Path, out: Path) -> None:
    """Build the prompts, keep the copies, and measure both groups."""
    from openinterpretability.cli import _load_runner
    from openinterpretability.evidence import read_corpus
    from openinterpretability.explain import unit_means

    runner = _load_runner(MODEL, "cpu", "float32")
    means = unit_means(runner, MODEL, read_corpus(corpus)[0], MEAN_TOKENS)
    route = json.loads(ROUTE_RECORD.read_text(encoding="utf-8"))["claim"]["route"]
    kept, seen = [], 0
    for c in candidates():
        copied, right = single(runner.tokenizer, c["copied"]), single(runner.tokenizer, c["right"])
        if copied is None or right is None:
            continue
        seen += 1
        top, acts, heads, resid = state(runner, means, c["prompt"])
        if top != copied:
            continue
        kept.append({**c, "is_right": copied == right, "acts": acts, "heads": heads,
                     "resid": resid})  # fmt: skip
    n_right = sum(k["is_right"] for k in kept)
    click.echo(f"{seen} prompts, {len(kept)} copies: {n_right} right, {len(kept) - n_right} wrong")
    rows = [{"table": "groups", "measured": seen, "kept": len(kept), "right_copy": n_right,
             "wrong_copy": len(kept) - n_right,
             "prompts": [{"prompt": k["prompt"], "copied": k["copied"], "right": k["right"],
                          "is_right": k["is_right"]} for k in kept]}]  # fmt: skip
    units = unit_rows(route, kept)
    rows += [{"table": "unit", **r} for r in units]
    rows += [{"table": "direction", **r} for r in direction_rows(kept, runner.n_layers)]
    objects = [k for k in kept if k["copied"].strip() in FACTS and k["right"].strip() in FACTS]
    rows += [{"table": "unit_objects", **r} for r in unit_rows(route, objects)]
    rows += [{"table": "direction_objects", **r}
             for r in direction_rows(objects, runner.n_layers)]  # fmt: skip
    click.echo(f"objects only: {len(objects)} copies, {sum(k['is_right'] for k in objects)} right")
    beyond = sum(r["beyond_null_95"] for r in units)
    click.echo(f"route units beyond the null's 95th percentile: {beyond} of {len(units)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    click.echo(f"wrote {len(rows)} rows → {out}")


if __name__ == "__main__":
    main()

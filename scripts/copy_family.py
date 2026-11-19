"""A family of copy-failure prompts for fixes, with splits and free-decoded text (issue #5232).

pythia-70m answers "A week has seven days and a year has twelve" with " days". It copies the
first clause's unit (#4781). A fix tested on the six prompts of the hallucination shortlist
can't show that it generalises. So this script builds a family of the same form:

    A <X> has <n> <unit of X> and a <Y> has <m>

from :data:`FACTS`, over every ordered pair. It measures pythia-70m on each prompt, in
float32, and keeps the prompts whose top token is the first clause's unit. Such a prompt is a
**wrong copy** when Y's unit differs, and a **right copy** when it is the same. Each row
holds the top 5 tokens, the right answer's probability and rank, and the copied unit's.

Splits come from :data:`SEED` (:func:`split`):

- wrong copies are grouped by their unordered pair. Some pairs go to ``train`` in one
  direction, and the reverse always goes to ``heldout``. The rest go to ``heldout``.
- a prompt with the same number in both clauses always goes to ``heldout``. These separate a
  learned relation from a memorised answer.
- right copies go half to ``retain`` (what a fix must not break), half to ``heldout``.

Each kept prompt has a greedy continuation and :data:`SAMPLES` sampled ones, of
:data:`DECODE_TOKENS` tokens. The first row is the family's metadata. It records #5074's
confound: a right copy here always shares its unit, so it shares a category.

``review-prompt`` writes the task for a frozen-fix hold-out. It asks a reviewer, who sees no
fix and no measure, for new failing prompts and new hard negatives.

    uv run python scripts/copy_family.py build --out docs/analyses/copy-family.jsonl
    uv run python scripts/copy_family.py review-prompt --family docs/analyses/copy-family.jsonl \\
        --out exchange/<id>/review.prompt.md
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import click

MODEL = "EleutherAI/pythia-70m"
SEED = 5232
TRAIN_SHARE = 0.6  # of the wrong-copy pairs whose two directions are split
DECODE_TOKENS = 20
SAMPLES = 2  # sampled continuations, with seeds 0, 1, …
CONFOUND = ("A right copy shares its unit with the first clause, so it also shares a category "
            "(#5074). A direction that finds 'same category' would separate right from wrong "
            "copies without tracking rightness.")  # fmt: skip
# (subject, count, unit): pairs within a unit can be copied rightly; across units, wrongly.
FACTS = {
    "legs": [("spider", "eight"), ("ant", "six"), ("dog", "four"), ("cat", "four"),
             ("bird", "two"), ("horse", "four"), ("cow", "four"), ("chicken", "two"),
             ("beetle", "six"), ("duck", "two"), ("lion", "four")],
    "wheels": [("car", "four"), ("bike", "two"), ("tricycle", "three"), ("truck", "six"),
               ("wagon", "four"), ("bus", "six"), ("cart", "two"), ("scooter", "two")],
    "sides": [("triangle", "three"), ("square", "four"), ("pentagon", "five"),
              ("hexagon", "six"), ("octagon", "eight")],
    "strings": [("guitar", "six"), ("violin", "four"), ("banjo", "five")],
}  # fmt: skip
# Facts whose unit is their own: copying the first clause's unit into them is wrong.
MEASURES = [
    ("minute", "sixty", "seconds"), ("hour", "sixty", "minutes"), ("week", "seven", "days"),
    ("year", "twelve", "months"), ("decade", "ten", "years"), ("fortnight", "two", "weeks"),
    ("day", "twenty-four", "hours"), ("century", "a hundred", "years"),
    ("kilometre", "a thousand", "metres"), ("metre", "a hundred", "centimetres"),
    ("kilogram", "a thousand", "grams"), ("dozen", "twelve", "eggs"),
    ("hand", "five", "fingers"), ("foot", "five", "toes"),
]  # fmt: skip


SILENT_H = {"hour"}  # words that start with a vowel sound though spelt with "h"


def article(word: str) -> str:
    """The article for ``word``: "An" before a vowel sound, else "A"."""
    return "An" if word[0] in "aeiou" or word in SILENT_H else "A"


def facts() -> list[tuple[str, str, str]]:
    """Every ``(subject, count, unit)``."""
    return [(s, n, u) for u, items in FACTS.items() for s, n in items] + MEASURES


def candidates() -> list[dict]:
    """Every ordered pair of facts as a prompt, with its copied unit and its right answer."""
    out = []
    for (s1, n1, u1), (s2, n2, u2) in itertools.permutations(facts(), 2):
        if s1 == s2:
            continue
        prompt = f"{article(s1)} {s1} has {n1} {u1} and {article(s2).lower()} {s2} has {n2}"
        out.append({"prompt": prompt, "first": s1, "second": s2, "copied": f" {u1}",
                    "right": f" {u2}", "shared_number": n1 == n2})  # fmt: skip
    return out


def split(rows: list[dict], seed: int = SEED) -> list[dict]:
    """``rows`` with ``split`` and ``reversed_of`` set (module docstring)."""
    rng = random.Random(seed)
    wrong = [r for r in rows if r["kind"] == "wrong_copy"]
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for r in wrong:
        by_pair.setdefault(tuple(sorted((r["first"], r["second"]))), []).append(r)
    for key in sorted(by_pair):
        group = sorted(by_pair[key], key=lambda r: r["prompt"])
        for r in group:
            r["split"], r["reversed_of"] = "heldout", None
        if len(group) == 2 and rng.random() < TRAIN_SHARE:
            one, other = rng.sample(group, 2)
            one["split"], other["reversed_of"] = "train", one["prompt"]
    for r in wrong:
        if r["shared_number"]:
            r["split"] = "heldout"
    right = sorted((r for r in rows if r["kind"] == "right_copy"), key=lambda r: r["prompt"])
    rng.shuffle(right)
    for i, r in enumerate(right):
        r["split"], r["reversed_of"] = ("retain" if i % 2 == 0 else "heldout"), None
    return rows


def metadata(rows: list[dict]) -> dict:
    """The family's first row: how it was made, and its counts by kind and split."""
    counts: dict[str, int] = {}
    for r in rows:
        counts[f"{r['kind']}/{r['split']}"] = counts.get(f"{r['kind']}/{r['split']}", 0) + 1
    return {"table": "metadata", "issue": 5232, "model": MODEL, "dtype": "float32",
            "seed": SEED, "train_share": TRAIN_SHARE, "decode_tokens": DECODE_TOKENS,
            "samples": SAMPLES, "counts": dict(sorted(counts.items())),
            "template": "A <X> has <n> <unit of X> and a <Y> has <m>",
            "confound": CONFOUND}  # fmt: skip


def single(tokenizer, text: str) -> int | None:
    """The id of ``text`` when it is one token, else None."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def measure_all(cands: list[dict], model: str) -> list[dict]:
    """The candidates that the model answers with the copied unit, measured and decoded."""
    import torch

    from openinterpretability.cli import _load_runner

    runner = _load_runner(model, "cpu", "float32")  # bfloat16 rounds pythia's logits
    out = []
    with torch.no_grad():
        for c in cands:
            copied, right = (single(runner.tokenizer, c[k]) for k in ("copied", "right"))
            if copied is None or right is None:
                continue
            probs = torch.softmax(runner.forward(runner.encode(c["prompt"]))[0, -1].float(), -1)
            if int(probs.argmax()) != copied:
                continue  # not a copy: the family is the copies
            top = probs.topk(5)
            out.append({**c, "kind": "right_copy" if copied == right else "wrong_copy",
                        "top": [(runner.tokenizer.decode([int(i)]), round(float(p), 4))
                                for p, i in zip(top.values, top.indices, strict=True)],
                        "right_p": round(float(probs[right]), 6),
                        "right_rank": int((probs > probs[right]).sum()) + 1,
                        "copied_p": round(float(probs[copied]), 6),
                        "decoded": decode(runner, c["prompt"])})  # fmt: skip
    return out


def decode(runner, prompt: str) -> dict:
    """A greedy continuation and SAMPLES sampled ones (temperature 1), of DECODE_TOKENS each."""
    import torch

    ids = runner.encode(prompt)["input_ids"]
    out = {"greedy": _generate(runner, ids, None)}
    for s in range(SAMPLES):
        out[f"sample_{s}"] = _generate(runner, ids, torch.Generator().manual_seed(s))
    return out


def _generate(runner, ids, generator) -> str:
    """Decode DECODE_TOKENS tokens after ``ids``: greedy, or sampled with ``generator``."""
    import torch

    out, past, step = [], None, ids
    for _ in range(DECODE_TOKENS):
        result = runner.model(input_ids=step, past_key_values=past, use_cache=True)
        past, logits = result.past_key_values, result.logits[0, -1].float()
        if generator is None:
            nxt = int(logits.argmax())
        else:
            nxt = int(torch.multinomial(torch.softmax(logits, -1), 1, generator=generator))
        out.append(nxt)
        step = torch.tensor([[nxt]], device=ids.device)
    return runner.tokenizer.decode(out)


def review_prompt(rows: list[dict], n: int, answer_path: str) -> str:
    """The frozen-fix hold-out task. It shows the template and train prompts, never a measure."""
    meta = next(r for r in rows if r.get("table") == "metadata")
    train = [r["prompt"] for r in rows if r.get("split") == "train"]
    shown = "\n".join(f"- {p}" for p in train[:40])
    template = {"failing": [{"prompt": "<a new prompt>", "right": "<its right answer token>",
                             "why": "<why the base model should copy wrongly>"}],
                "hard_negatives": [{"prompt": "<a new prompt>", "right": "<the copied unit>",
                                    "why": "<why copying is right here>"}],
                "reviewer_agent_model": "<your model id>"}  # fmt: skip
    return "\n".join([
        "# Frozen-fix hold-out: new prompts for the copy family (issue #5232)",
        "",
        f"A fix for {meta['model']}'s copy failure has been frozen. You don't see it, and you",
        "never see a measure. Write new prompts that the fix's makers could not have tuned on.",
        "",
        f"The family's template is: {meta['template']}. The model often answers with the first",
        "clause's unit. That is a **wrong copy** when the second subject's unit differs, and a",
        "**right copy** when it is the same.",
        "",
        f"Write {n} new **failing** prompts, where the base model should copy the wrong unit,",
        f"and {n} new **hard negatives**, where copying is right and a fix must not break it.",
        "Vary the form: other subjects and units, other word orders, extra words between the",
        "clauses, three-clause lists, and units the family doesn't use. Don't repeat these",
        "train prompts, or reverse them:",
        "",
        shown,
        "",
        f"Known confound: {meta['confound']} Include hard negatives whose units match across",
        "categories, where you can find them.",
        "",
        f"Write them as JSON to `{answer_path}`:",
        "",
        "```json",
        json.dumps(template, indent=1),
        "```",
    ])  # fmt: skip


@click.group()
def cli() -> None:
    """Build the copy family, or write its frozen-fix review prompt (issue #5232)."""


@cli.command()
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def build(out: Path) -> None:
    """Measure every candidate, keep the copies, split them and decode each."""
    rows = split(measure_all(candidates(), MODEL))
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [metadata(rows)] + [{"table": "prompt", **r} for r in rows]
    out.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
    click.echo(f"{json.dumps(metadata(rows)['counts'])} → {out}")


@cli.command(name="review-prompt")
@click.option("--family", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True)  # fmt: skip
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--n", default=20, show_default=True, help="New prompts of each kind to ask for.")
def review_prompt_command(family: Path, out: Path, n: int) -> None:
    """Write the frozen-fix hold-out task: new failing prompts and hard negatives."""
    rows = [json.loads(line) for line in family.read_text(encoding="utf-8").splitlines()]
    out.parent.mkdir(parents=True, exist_ok=True)
    answer = out.with_name(out.name.replace(".prompt.md", ".prompts.json"))
    out.write_text(review_prompt(rows, n, str(answer)), encoding="utf-8")
    click.echo(f"Wrote {out} (answer in {answer})")


if __name__ == "__main__":
    cli()

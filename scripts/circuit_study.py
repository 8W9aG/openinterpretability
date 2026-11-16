"""One circuit study (#4283): does GPT-2's wiring circuit C1 carry word completion?

The circuit comes from ``scripts/circuit_identity.py``: GPT-2's largest Leiden cluster of
the wiring graph. Its neuron members' explanations mostly describe completing a unit
that the previous tokens began (a split word, a collocation). So the task hypothesis is
**completing a two-token word**. Its measures follow ``docs/circuits-and-hubs.md``, with
mean ablation as in ``scripts/sva_circuit.py``:

* *Faithfulness*: the share of the metric that the circuit keeps alone.
* *Necessity*: the share of the metric lost when the circuit is ablated.
* *Specificity*: the same loss on two other tasks, agreement and name copying.
* *Minimality*: each member's own loss.
* *Null*: necessity of random neuron sets with the same layers.

Each metric is relative to ablating every neuron: (m − m(none)) / (m(all) − m(none)).
The circuit's two attention heads are not ablated, because the tools ablate neurons only::

    uv run python scripts/circuit_study.py --out docs/analyses/circuit-pilot.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import click
import torch
from sva_circuit import ablated_metric, by_layer, mean_activations

MODEL = "openai-community/gpt2"
CIRCUIT = [(4, 2999), (5, 1208), (5, 2567), (5, 616), (6, 37)]  # C1's neurons
CIRCUIT_HEADS = ["L7/H10", "L7/H2"]  # members that the neuron tools can't ablate
NULL_SETS = 64
SPECIFIC = 0.2  # a negative task passes when the circuit's loss there is below this
WORDS_FILE = Path("/usr/share/dict/words")  # the word list; the words drawn are recorded
WORD_COUNT = 400  # words drawn from the list, before keeping those GPT-2 splits in two
CONTEXT = "Yesterday I read a long article about"
NAMES = [" Mary", " John", " Alice", " Bob", " Sarah", " James", " Emma", " David"]


class PairTask:
    """Clean and corrupted prompts that differ in one token, with targets that differ too."""

    def __init__(self, clean, corrupted, good, bad):
        self.clean, self.corrupted = torch.tensor(clean), torch.tensor(corrupted)
        self.good, self.bad = torch.tensor(good), torch.tensor(bad)

    def metric(self, logits: torch.Tensor, part: slice = slice(None)) -> torch.Tensor:
        """logit(the clean prompt's target) − logit(the corrupted prompt's), per prompt."""
        last = logits[:, -1].float()
        rows = torch.arange(last.shape[0])
        good, bad = self.good[part].to(last.device), self.bad[part].to(last.device)
        return last[rows, good] - last[rows, bad]


def two_token_words(tok, seed: int) -> list[str]:
    """Lower-case words of 7+ letters that GPT-2 splits into exactly two pieces after a space."""
    words = sorted({w.strip() for w in WORDS_FILE.read_text().split() if w.strip().isalpha()
                    and w.strip().islower() and len(w.strip()) >= 7})  # fmt: skip
    drawn = random.Random(seed).sample(words, WORD_COUNT)
    return [w for w in drawn if len(tok.encode(" " + w)) == 2]


TOP = 10  # a word counts when GPT-2 ranks its second piece in its top 10 after the first


@torch.no_grad()
def predictable(runner, words: list[str]) -> list[str]:
    """The words whose second piece GPT-2 ranks in its top :data:`TOP` after the first piece."""
    tok, context, kept = runner.tokenizer, runner.tokenizer.encode(CONTEXT), []
    for word in words:
        first, second = tok.encode(" " + word)
        ids = runner.encode({"input_ids": torch.tensor([context + [first]])})
        top = runner.forward(ids)[0, -1].topk(TOP).indices.tolist()
        if second in top:
            kept.append(word)
    return kept


def word_task(tok, n: int, seed: int, words: list[str]) -> PairTask:
    """Completing a two-token word: clean ends on its first piece, the pair on another's."""
    pieces = [tok.encode(" " + w) for w in words]
    rng, clean, corrupted, good, bad = random.Random(seed), [], [], [], []
    context = tok.encode(CONTEXT)
    for _ in range(n):
        a, b = rng.sample(pieces, 2)
        if a[0] == b[0] or a[1] == b[1]:
            continue
        clean.append(context + [a[0]])
        corrupted.append(context + [b[0]])
        good.append(a[1])
        bad.append(b[1])
    return PairTask(clean, corrupted, good, bad)


def names_task(tok, n: int, seed: int) -> PairTask:
    """Name copying (indirect object): the clean prompt's answer is the other name."""
    rng, clean, corrupted, good, bad = random.Random(seed), [], [], [], []
    for _ in range(n):
        a, b, c = rng.sample(NAMES, 3)
        text = f"When{a} and{b} went to the store,{b} gave a drink to"
        swap = f"When{c} and{b} went to the store,{b} gave a drink to"
        ids, other = tok.encode(text), tok.encode(swap)
        if len(ids) != len(other):
            continue
        clean.append(ids)
        corrupted.append(other)
        good.append(tok.encode(a)[0])
        bad.append(tok.encode(c)[0])
    return PairTask(clean, corrupted, good, bad)


def agreement_task(tok, n: int, seed: int) -> PairTask:
    """Subject–verb agreement on one template: ' is' after a singular noun, ' are' after plural."""
    nouns = [(" dog", " dogs"), (" cat", " cats"), (" king", " kings"), (" girl", " girls"),
             (" doctor", " doctors"), (" teacher", " teachers"), (" bird", " birds")]  # fmt: skip
    is_, are = tok.encode(" is")[0], tok.encode(" are")[0]
    rng, clean, corrupted, good, bad = random.Random(seed), [], [], [], []
    for _ in range(n):
        one, many = rng.choice(nouns)
        singular = rng.random() < 0.5
        noun, other = (one, many) if singular else (many, one)
        ids, swapped = (
            tok.encode(f"The{noun} near the house"),
            tok.encode(f"The{other} near the house"),
        )
        if len(ids) != len(swapped):
            continue
        clean.append(ids)
        corrupted.append(swapped)
        good.append(is_ if singular else are)
        bad.append(are if singular else is_)
    return PairTask(clean, corrupted, good, bad)


def relative(runner, task, mean, keep=None, drop=None) -> float:
    """(m − m(none)) / (m(all) − m(none)) under the given ablation."""
    full = ablated_metric(runner, task, mean, None, None)
    empty = ablated_metric(runner, task, mean, {}, None)
    return (ablated_metric(runner, task, mean, keep, drop) - empty) / (full - empty)


def measure(runner, task, mean, neurons) -> dict:
    """Faithfulness and necessity of ``neurons`` on ``task``."""
    kept = relative(runner, task, mean, keep=by_layer(neurons))
    without = relative(runner, task, mean, drop=by_layer(neurons))
    return {"faithfulness": round(kept, 4), "necessity": round(1 - without, 4)}


def null_sets(runner, seed: int) -> list[list[tuple[int, int]]]:
    """Random neuron sets with the circuit's layers, one neuron per member."""
    rng = random.Random(seed)
    return [[(layer, rng.randrange(runner.mlp_width(layer))) for layer, _ in CIRCUIT]
            for _ in range(NULL_SETS)]  # fmt: skip


@click.command()
@click.option("--device", default="auto", show_default=True)
@click.option("--pairs", default=120, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(device: str, pairs: int, seed: int, out: Path) -> None:
    """Measure C1 on word completion, two negative tasks, its members and a null."""
    from openinterp.cli import _load_runner

    runner = _load_runner(MODEL, device, "float32", None)
    tok = runner.tokenizer
    words = predictable(runner, two_token_words(tok, seed))
    click.echo(f"{len(words)} two-token words that GPT-2 completes", err=True)
    tasks = {"word_completion": word_task(tok, pairs, seed, words),
             "agreement": agreement_task(tok, pairs, seed),
             "names": names_task(tok, pairs, seed)}  # fmt: skip
    means = dict(zip(tasks, mean_activations(runner, list(tasks.values())), strict=True))
    rows = []
    for name, task in tasks.items():
        base = ablated_metric(runner, task, means[name], None, None)
        rows.append({"table": "task", "task": name, "pairs": int(task.clean.shape[0]),
                     "clean_metric": round(base, 4),
                     **measure(runner, task, means[name], CIRCUIT)})  # fmt: skip
        click.echo(rows[-1], err=True)
    word, mean = tasks["word_completion"], means["word_completion"]
    for member in CIRCUIT:
        rows.append({"table": "member", "layer": member[0], "neuron": member[1],
                     **measure(runner, word, mean, [member])})  # fmt: skip
    for i, neurons in enumerate(null_sets(runner, seed)):
        rows.append({"table": "null", "set": i, **measure(runner, word, mean, neurons)})
    rows.append({"table": "circuit", "model": MODEL, "neurons": CIRCUIT,
                 "heads_not_ablated": CIRCUIT_HEADS, "words": words})  # fmt: skip
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    click.echo(f"wrote {len(rows)} rows → {out}", err=True)


if __name__ == "__main__":
    main()

"""A subject–verb agreement circuit of MLP neurons, traced on one model (issue #3225).

This replicates the method of Arora et al. (arXiv 2601.22594,
``docs/literature/neuron-basis-circuits.md``) on a registry model:

1. **Task.** Four templates end where the verb comes next. A clean prompt and its
   corrupted pair differ only in the number of the noun that the verb agrees with. The
   metric is logit(" is") − logit(" are"), signed by that noun's number.
2. **Attribution.** Integrated gradients, from each neuron's corrupted activation to its
   clean one, one layer at a time, with the corrupted prompt as input. It is summed over
   positions and averaged over the training pairs.
3. **Scores.** On held-out pairs, the top N neurons by attribution are the circuit.
   *Faithfulness*: the share of the metric left when every other neuron is mean-ablated.
   *Completeness*: the share left when the circuit is mean-ablated. Both are relative to
   ablating every neuron. A random set of N neurons is the reference.
4. **Registry.** How many of the circuit's neurons the registry has studied, how many
   are VERIFIED, and what their explanations say.

It reads the site's parquet (``open-interpretability build``) for step 4 and writes nothing in
``wiki/``::

    uv run python scripts/sva_circuit.py --site-dir build/site \\
        --out docs/analyses/sva-circuit.jsonl
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import click
import torch

# Nouns whose singular and plural are one token each in Qwen2.5's tokenizer.
NOUNS = [
    *("author", "pilot", "surgeon", "farmer", "manager", "customer", "officer", "teacher"),
    *("senator", "guard", "doctor", "student", "dancer", "lawyer", "writer", "painter"),
    *("driver", "player", "worker", "singer", "judge", "nurse", "minister", "boy", "girl"),
    *("king", "friend", "parent", "dog", "cat"),
]
IRREGULAR = {"child": "children", "woman": "women", "man": "men"}
PREPOSITIONS = ("near", "behind", "beside", "with")
VERBS = ("liked", "saw", "knew", "met")
TEMPLATES = ("simple", "across_pp", "across_rc", "within_rc")
SIZES = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000)
# Words that would mark an explanation as being about grammatical number or agreement.
NUMBER_WORDS = re.compile(
    r"\b(plural|singular|grammatical number|agreement|subject.verb|is/are)\b"
    r"|[\"'](is|are)[\"']",
    re.IGNORECASE,
)


def plural(noun: str) -> str:
    """The plural of ``noun``."""
    return IRREGULAR.get(noun, noun + "s")


def prompt(template: str, a: str, b: str, prep: str, verb: str) -> str:
    """The prompt; ``a`` is the head noun and ``b`` the second noun (both inflected)."""
    return {
        "simple": f"The {a}",
        "across_pp": f"The {a} {prep} the {b}",
        "across_rc": f"The {a} that the {b} {verb}",
        "within_rc": f"The {a} that the {b}",
    }[template]


def pairs(template: str, n: int, rng: random.Random) -> list[dict]:
    """``n`` (clean, corrupted, sign) pairs; half of them have a singular clean noun.

    The verb agrees with the head noun, except within a relative clause, where it agrees
    with the clause's subject. Only that noun's number differs between the two prompts.
    The other noun's number is drawn at random, and is the same in both.
    """
    nouns = [*NOUNS, *IRREGULAR]
    out = []
    for i in range(n):
        a, b = rng.sample(nouns, 2)
        prep, verb = rng.choice(PREPOSITIONS), rng.choice(VERBS)
        singular = i % 2 == 0
        agree, other = (b, a) if template == "within_rc" else (a, b)
        other = other if rng.random() < 0.5 else plural(other)
        clean, corrupted = (agree, plural(agree)) if singular else (plural(agree), agree)

        def text(form, other=other, prep=prep, verb=verb):
            head, second = (other, form) if template == "within_rc" else (form, other)
            return prompt(template, head, second, prep, verb)

        out.append({"clean": text(clean), "corrupted": text(corrupted),
                    "sign": 1.0 if singular else -1.0})  # fmt: skip
    return out


class Task:
    """One template's pairs as token tensors, with the metric."""

    def __init__(self, runner, rows: list[dict]):
        tok = runner.tokenizer
        self.clean = torch.tensor([tok.encode(r["clean"]) for r in rows])
        self.corrupted = torch.tensor([tok.encode(r["corrupted"]) for r in rows])
        self.sign = torch.tensor([r["sign"] for r in rows])
        self.is_, self.are = tok.encode(" is")[0], tok.encode(" are")[0]
        if self.clean.shape != self.corrupted.shape:
            raise ValueError("a clean prompt and its pair have different token counts")

    def metric(self, logits: torch.Tensor, part: slice = slice(None)) -> torch.Tensor:
        """Per prompt of ``part``: the signed logit difference at the last position."""
        last = logits[:, -1].float()
        return (last[:, self.is_] - last[:, self.are]) * self.sign[part].to(last.device)


def activations(runner, ids: torch.Tensor) -> dict[int, torch.Tensor]:
    """Every layer's MLP activations on ``ids``: ``{layer: [batch, seq, d_mlp]}``."""
    _, cache = runner.run_with_cache({"input_ids": ids}, list(range(runner.n_layers)))
    return cache


def integrated_gradients(runner, task: Task, steps: int, batch: int) -> dict[int, torch.Tensor]:
    """``{layer: [d_mlp]}``: each neuron's attribution, summed over positions and prompts."""
    total = {layer: torch.zeros(runner.mlp_width(layer)) for layer in range(runner.n_layers)}
    for start in range(0, task.clean.shape[0], batch):
        part = slice(start, start + batch)
        clean = activations(runner, task.clean[part])
        corrupted_ids = runner.encode({"input_ids": task.corrupted[part]})
        corrupted = activations(runner, task.corrupted[part])
        for layer in range(runner.n_layers):
            delta = clean[layer] - corrupted[layer]
            grads = torch.zeros_like(delta)
            for k in range(steps):
                alpha = (k + 0.5) / steps
                held: list[torch.Tensor] = []

                def patch(act, base=corrupted[layer], delta=delta, alpha=alpha, held=held):
                    x = (base + alpha * delta).detach().requires_grad_(True)
                    held.append(x)
                    return x

                with torch.enable_grad(), runner.hooks({layer: patch}):
                    value = task.metric(runner.forward(corrupted_ids), part).sum()
                    (grad,) = torch.autograd.grad(value, held[0])
                grads += grad / steps
            total[layer] += (delta * grads).sum(dim=(0, 1)).float().cpu()
    return total


def ablated_metric(runner, task: Task, means: dict, keep: dict | None, drop: dict | None):
    """The mean metric on the clean prompts, with neurons replaced by their means.

    With ``keep``, every neuron not in it is replaced. With ``drop``, only the neurons in
    it are. ``keep={}`` replaces every neuron. Both ``None`` replaces none.
    """
    hooks = {}
    for layer in range(runner.n_layers):
        width = runner.mlp_width(layer)
        mask = torch.zeros(width, dtype=torch.bool)  # True: replace with the mean
        if keep is not None:
            mask[:] = True
            mask[keep.get(layer, [])] = False
        if drop is not None:
            mask[drop.get(layer, [])] = True
        if not mask.any():
            continue
        m = mask.to(runner.device)

        def replace(act, layer=layer, m=m):
            return torch.where(m, means[layer].to(act.dtype), act)

        hooks[layer] = replace
    with torch.no_grad(), runner.hooks(hooks):
        logits = runner.forward(runner.encode({"input_ids": task.clean}))
    return float(task.metric(logits).mean())


def by_layer(neurons: list[tuple[int, int]]) -> dict[int, list[int]]:
    """``neurons`` as ``{layer: [neuron, ...]}``."""
    out: dict[int, list[int]] = {}
    for layer, idx in neurons:
        out.setdefault(layer, []).append(idx)
    return out


def mean_activations(runner, tasks: list[Task]) -> list[dict[int, torch.Tensor]]:
    """Per task, ``{layer: [seq, d_mlp]}``: the mean over its clean and corrupted prompts."""
    out = []
    for task in tasks:
        both = torch.cat([task.clean, task.corrupted])
        cache = activations(runner, both)
        out.append({layer: acts.mean(0) for layer, acts in cache.items()})
    return out


def scores(runner, tasks, means, neurons: list[tuple[int, int]]) -> dict:
    """Faithfulness and completeness of ``neurons``, averaged over the templates."""
    faithful, complete = [], []
    for task, mean in zip(tasks, means, strict=True):
        full = ablated_metric(runner, task, mean, None, None)
        empty = ablated_metric(runner, task, mean, {}, None)
        only = ablated_metric(runner, task, mean, by_layer(neurons), None)
        without = ablated_metric(runner, task, mean, None, by_layer(neurons))
        span = full - empty
        faithful.append((only - empty) / span)
        complete.append((without - empty) / span)
    return {
        "faithfulness": sum(faithful) / len(faithful),
        "completeness": sum(complete) / len(complete),
        "per_template_faithfulness": [round(f, 4) for f in faithful],
    }


def registry_rows(site_dir: Path, model: str) -> dict[tuple[int, int], list[dict]]:
    """The model's neuron-pole experiments, by (layer, neuron)."""
    import duckdb

    con = duckdb.connect()
    data = site_dir / "data" / "registry.parquet"
    con.execute(f"CREATE VIEW r AS SELECT * FROM read_parquet('{data}')")  # noqa: S608
    rows = con.execute(
        """SELECT slug, layer, neuron_idx, pole, status, unit_status, explanation
           FROM r WHERE kind = 'mlp_neuron_pole' AND model_name = ? AND sister_of IS NULL""",
        [model],
    ).fetchall()
    out: dict[tuple[int, int], list[dict]] = {}
    for slug, layer, idx, pole, status, unit_status, explanation in rows:
        out.setdefault((layer, idx), []).append(
            {
                "slug": slug,
                "pole": pole,
                "status": status,
                "unit_status": unit_status,
                "explanation": explanation,
            }  # fmt: skip
        )
    return out


@click.command()
@click.option("--model", default="Qwen/Qwen2.5-0.5B", show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--site-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--train", default=300, show_default=True, help="Training pairs per template.")
@click.option("--held-out", default=40, show_default=True, help="Held-out pairs per template.")
@click.option("--steps", default=10, show_default=True, help="Integrated-gradient steps.")
@click.option("--seed", default=0, show_default=True)
def main(model, device, site_dir, out, train, held_out, steps, seed):
    """Trace the circuit, score it against random neurons and compare it with the registry."""
    from openinterp.cli import _load_runner

    runner = _load_runner(model, device, "float32", None)
    rng = random.Random(seed)
    train_tasks, test_tasks = [], []
    for template in TEMPLATES:
        rows = pairs(template, train + held_out, rng)
        train_tasks.append(Task(runner, rows[:train]))
        test_tasks.append(Task(runner, rows[train:]))
    click.echo(f"{model}: {len(TEMPLATES)} templates, {train}+{held_out} pairs each")
    ranked = rank_neurons(runner, train_tasks, steps, train)
    curve = curve_rows(runner, train_tasks, test_tasks, ranked)
    neurons = neuron_rows(registry_rows(site_dir, model), ranked[: SIZES[5]])
    out.write_text("".join(json.dumps(r) + "\n" for r in [*curve, *neurons]))
    click.echo(f"wrote {len(curve)} curve rows and {len(neurons)} neurons → {out}")


def rank_neurons(runner, tasks: list[Task], steps: int, train: int):
    """Every neuron as (attribution, layer, neuron), largest attribution first."""
    attribution = {layer: torch.zeros(runner.mlp_width(layer)) for layer in range(runner.n_layers)}
    for template, task in zip(TEMPLATES, tasks, strict=True):
        part = integrated_gradients(runner, task, steps, batch=100)
        for layer, values in part.items():
            attribution[layer] += values / (len(TEMPLATES) * train)
        click.echo(f"attributed {template}")
    flat = [(float(v), layer, i) for layer, vs in attribution.items() for i, v in enumerate(vs)]
    return sorted(flat, key=lambda t: -abs(t[0]))


def curve_rows(runner, train_tasks, test_tasks, ranked) -> list[dict]:
    """Faithfulness and completeness at each of :data:`SIZES`, and for random neurons."""
    means = mean_activations(runner, train_tasks)
    every = sorted((layer, i) for _, layer, i in ranked)
    curve = []
    for n in SIZES:
        top = [(layer, i) for _, layer, i in ranked[:n]]
        row = {"table": "curve", "basis": "attribution", "size": n}
        row |= scores(runner, test_tasks, means, top)
        rand = [
            scores(runner, test_tasks, means, random.Random(s).sample(every, n)) for s in range(3)
        ]
        curve += [row, {"table": "curve", "basis": "random", "size": n,
                        "faithfulness": sum(r["faithfulness"] for r in rand) / 3,
                        "completeness": sum(r["completeness"] for r in rand) / 3}]  # fmt: skip
        click.echo(
            f"N={n}: faithfulness {row['faithfulness']:.3f}, completeness {row['completeness']:.3f}"
        )
    return curve


def neuron_rows(registry: dict, ranked) -> list[dict]:
    """One row per neuron of ``ranked``, with what the registry has on it."""
    rows = []
    for rank, (value, layer, i) in enumerate(ranked, 1):
        found = registry.get((layer, i), [])
        rows.append({
            "table": "neuron", "rank": rank, "layer": layer, "neuron_idx": i,
            "attribution": round(value, 6), "experiments": len(found),
            "verified_poles": sorted({r["pole"] for r in found if r["unit_status"] == "VERIFIED"}),
            "mentions_number": any(NUMBER_WORDS.search(r["explanation"] or "") for r in found),
            "slugs": sorted({r["slug"] for r in found}),
        })  # fmt: skip
    return rows


if __name__ == "__main__":
    main()

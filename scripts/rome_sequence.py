"""Sequential ROME edits at one layer of pythia-70m, and the VERIFIED poles above it (#5362).

Gupta et al. find that ROME and MEMIT forget slowly and then collapse as edits pile up.
This script makes :data:`EDITS` ROME edits in turn at layer :data:`rome_keep.LAYER`. Each
edit sets one city's country to a wrong one, and it is fitted on the model as the earlier
edits left it. The second moment C stays the base model's, as in ROME. A null sequence adds
random rank-one updates, the n-th with the norm of the n-th ROME edit.

At the base and after each count of edits in :data:`CHECKPOINTS`, a ``checkpoint`` row gives:

* the keep test of every VERIFIED pole, with ``random_delta_keep``'s batched readings.
* the edited layer's normalised weight distance ‖W_n − W_0‖ / ‖W_0‖.
* the share of the edits so far whose new object is the top prediction (retention).
* the language-model loss on the corpus sample.

A ``pole`` row for each pole and checkpoint gives its readings::

    uv run python scripts/rome_sequence.py --device cuda --out papers/rome-sequence/results.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nf4_keep import run_condition  # noqa: E402
from random_delta_keep import (  # noqa: E402
    base_columns,
    base_rows,
    cached_readings,
    changed,
    first_position_activations,
    probe_activations,
)
from rome_keep import (  # noqa: E402
    COV_TOKENS,
    LAYER,
    MODEL,
    contexts,
    down,
    fit_value,
    param_name,
    random_rank_one,
    rome_update,
    second_moment,
)

from openinterpretability.interference.keep import (  # noqa: E402
    SAMPLE_TOKENS,
    sample_windows,
    verified_records,
)

EDITS = 100  # edits in each sequence
# The edit counts at which both sequences are tested: dense early, where pythia-70m changes most.
CHECKPOINTS = (0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 60, 80, 100)
SEED = 5362
RELATION = " is located in the country of"
# Each city with its true country, so that the new object is never the true one.
CITIES = (
    ("Paris", "France"), ("Lyon", "France"), ("Marseille", "France"), ("Berlin", "Germany"),
    ("Munich", "Germany"), ("Hamburg", "Germany"), ("Rome", "Italy"), ("Milan", "Italy"),
    ("Naples", "Italy"), ("Madrid", "Spain"), ("Barcelona", "Spain"), ("Seville", "Spain"),
    ("Tokyo", "Japan"), ("Osaka", "Japan"), ("Kyoto", "Japan"), ("Beijing", "China"),
    ("Shanghai", "China"), ("Guangzhou", "China"), ("Delhi", "India"), ("Mumbai", "India"),
    ("Bangalore", "India"), ("Rio de Janeiro", "Brazil"), ("Sao Paulo", "Brazil"),
    ("Brasilia", "Brazil"), ("Toronto", "Canada"), ("Montreal", "Canada"),
    ("Vancouver", "Canada"), ("Guadalajara", "Mexico"), ("Monterrey", "Mexico"),
    ("Cancun", "Mexico"), ("Moscow", "Russia"), ("Saint Petersburg", "Russia"),
    ("Novosibirsk", "Russia"), ("Cairo", "Egypt"), ("Alexandria", "Egypt"), ("Luxor", "Egypt"),
    ("Nairobi", "Kenya"), ("Mombasa", "Kenya"), ("Lagos", "Nigeria"), ("Abuja", "Nigeria"),
    ("Lima", "Peru"), ("Cusco", "Peru"), ("Santiago", "Chile"), ("Valparaiso", "Chile"),
    ("Athens", "Greece"), ("Thessaloniki", "Greece"), ("Istanbul", "Turkey"),
    ("Ankara", "Turkey"), ("Izmir", "Turkey"), ("Tehran", "Iran"), ("Isfahan", "Iran"),
    ("Baghdad", "Iraq"), ("Basra", "Iraq"), ("Jerusalem", "Israel"), ("Tel Aviv", "Israel"),
    ("Haifa", "Israel"), ("Warsaw", "Poland"), ("Krakow", "Poland"), ("Gdansk", "Poland"),
    ("Stockholm", "Sweden"), ("Gothenburg", "Sweden"), ("Oslo", "Norway"),
    ("Bergen", "Norway"), ("Helsinki", "Finland"), ("Tampere", "Finland"),
    ("Copenhagen", "Denmark"), ("Aarhus", "Denmark"), ("Dublin", "Ireland"),
    ("Cork", "Ireland"), ("Edinburgh", "Scotland"), ("Glasgow", "Scotland"),
    ("Aberdeen", "Scotland"), ("Manchester", "England"), ("Liverpool", "England"),
    ("Birmingham", "England"), ("Sydney", "Australia"), ("Melbourne", "Australia"),
    ("Brisbane", "Australia"), ("Perth", "Australia"), ("Buenos Aires", "Argentina"),
    ("Cordoba", "Argentina"), ("Mendoza", "Argentina"), ("Havana", "Cuba"),
    ("Hanoi", "Vietnam"), ("Saigon", "Vietnam"), ("Bangkok", "Thailand"),
    ("Chiang Mai", "Thailand"), ("Seoul", "Korea"), ("Busan", "Korea"),
    ("Karachi", "Pakistan"), ("Lahore", "Pakistan"), ("Islamabad", "Pakistan"),
    ("Lisbon", "Portugal"), ("Porto", "Portugal"), ("Vienna", "Austria"),
    ("Salzburg", "Austria"), ("Brussels", "Belgium"), ("Antwerp", "Belgium"),
    ("Zurich", "Switzerland"), ("Geneva", "Switzerland"), ("Bern", "Switzerland"),
    ("Amsterdam", "Netherlands"), ("Rotterdam", "Netherlands"), ("Kiev", "Ukraine"),
    ("Odessa", "Ukraine"), ("Bucharest", "Romania"), ("Budapest", "Hungary"),
    ("Casablanca", "Morocco"), ("Marrakech", "Morocco"), ("Rabat", "Morocco"),
    ("Addis Ababa", "Ethiopia"), ("Bogota", "Colombia"), ("Medellin", "Colombia"),
    ("Caracas", "Venezuela"), ("Jakarta", "Indonesia"), ("Bali", "Indonesia"),
    ("Florence", "Italy"), ("Venice", "Italy"), ("Turin", "Italy"), ("Valencia", "Spain"),
    ("Bilbao", "Spain"), ("Cologne", "Germany"), ("Frankfurt", "Germany"),
    ("Dresden", "Germany"), ("Nice", "France"), ("Bordeaux", "France"),
    ("Toulouse", "France"), ("Hiroshima", "Japan"), ("Nagoya", "Japan"),
    ("Sapporo", "Japan"), ("Shenzhen", "China"), ("Wuhan", "China"), ("Chengdu", "China"),
    ("Kolkata", "India"), ("Chennai", "India"), ("Hyderabad", "India"),
    ("Calgary", "Canada"), ("Ottawa", "Canada"), ("Quebec", "Canada"),
    ("Adelaide", "Australia"), ("Canberra", "Australia"), ("Leeds", "England"),
    ("Bristol", "England"), ("Oxford", "England"), ("Cambridge", "England"),
    ("Kazan", "Russia"), ("Sochi", "Russia"), ("Giza", "Egypt"), ("Kano", "Nigeria"),
    ("Malmo", "Sweden"), ("Trondheim", "Norway"), ("Galway", "Ireland"),
    ("Lviv", "Ukraine"), ("Kharkiv", "Ukraine"), ("Debrecen", "Hungary"),
    ("Fez", "Morocco"), ("Cali", "Colombia"), ("Maracaibo", "Venezuela"),
    ("Surabaya", "Indonesia"), ("Bandung", "Indonesia"),
)  # fmt: skip


def make_facts(tokenizer, n: int = EDITS, seed: int = SEED) -> list[tuple[str, str, str]]:
    """``n`` counterfactual facts: each city's country set to another, one-token country.

    Args:
        tokenizer: The model's tokenizer, which checks that each new object is one token.
        n: How many facts.
        seed: The seed of the cities' order and their new countries.

    Returns:
        ``(subject, relation, new object)`` triples, each city once.

    Raises:
        ValueError: There are fewer than ``n`` cities.
    """
    countries = sorted({c for _, c in CITIES if len(tokenizer(" " + c)["input_ids"]) == 1})
    cities = [pair for pair in CITIES if pair[1] in countries]
    if len(cities) < n:
        raise ValueError(f"{len(cities)} cities, fewer than {n}")
    rng = random.Random(seed)
    rng.shuffle(cities)
    return [(city, RELATION, " " + rng.choice([c for c in countries if c != true]))
            for city, true in cities[:n]]  # fmt: skip


def top_share(model, tokenizer, facts) -> float:
    """The share of ``facts`` whose new object is the model's top prediction, with no prefix.

    Args:
        model: The model, as it is now.
        tokenizer: Its tokenizer.
        facts: The facts.
    """
    if not facts:
        return 0.0
    device = next(model.parameters()).device
    hits = 0
    with torch.no_grad():
        for subject, relation, obj in facts:
            ids = torch.tensor([tokenizer(subject + relation)["input_ids"]], device=device)
            hits += int(model(ids).logits[0, -1].argmax()) == tokenizer(obj)["input_ids"][0]
    return hits / len(facts)


def sample_loss(model, batches: list[torch.Tensor]) -> float:
    """The model's mean next-token loss on the corpus sample.

    Args:
        model: The model, as it is now.
        batches: The corpus sample.
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        losses = [float(model(b.to(device), labels=b.to(device)).loss) for b in batches]
    return sum(losses) / len(losses)


def distance(delta: torch.Tensor, weight: torch.Tensor) -> float:
    """‖ΔW‖ / ‖W_0‖, the edited layer's normalised weight distance.

    Args:
        delta: The summed change of the layer's output matrix.
        weight: The layer's base output matrix.
    """
    return float(delta.double().norm() / weight.double().norm())


def rome_edits(model, tokenizer, facts, cov, ctx_windows) -> list[tuple[torch.Tensor, ...]]:
    """Fit each fact's ROME edit in turn, each on the model as the earlier edits left it.

    Args:
        model: The base model. It is put back as it was after each fit.
        tokenizer: Its tokenizer.
        facts: The facts, in the order of the sequence.
        cov: The base model's second moment C at the edited layer.
        ctx_windows: Corpus batches that give the fit's prefixes.

    Returns:
        Each edit's Λ and u, in order, on the CPU.
    """
    layer, name = down(model, LAYER), param_name(LAYER)
    delta, edits = torch.zeros_like(layer.weight, device="cpu"), []
    for fact in facts:
        with changed(model, {name: delta}):
            ctxs = contexts(tokenizer, ctx_windows, fact[0])
            key, value = fit_value(model, tokenizer, ctxs, fact, LAYER)
            lam, u = rome_update(layer.weight.detach(), layer.bias.detach(), cov, key, value)
        edits.append((lam.cpu(), u.cpu()))
        delta += torch.outer(lam, u).cpu()
    return edits


def deltas(edits, kind: str) -> list[torch.Tensor]:
    """The summed change after each edit: ROME's own, or random edits of the same norms.

    Args:
        edits: Each ROME edit's Λ and u, from :func:`rome_edits`.
        kind: ``rome``, or ``random`` for the matched-norm null.

    Returns:
        The summed change of the layer's output matrix after 1, 2, … edits.
    """
    out, total = [], None
    for i, (lam, u) in enumerate(edits):
        a, b = (lam, u) if kind == "rome" else random_rank_one(lam, u, SEED + i)
        step = torch.outer(a, b)
        total = step if total is None else total + step
        out.append(total.clone())
    return out


def checkpoint(env: dict, kind: str, n: int, delta, facts) -> list[dict]:
    """The keep test and the model's measures after the first ``n`` edits of one sequence.

    Args:
        env: The run's shared state: ``runner``, ``batches``, ``records``, ``base`` and
            ``state`` for ``run_condition``, and ``weight``, the base output matrix.
        kind: ``rome`` or ``random``.
        n: The number of edits made.
        delta: The summed change after ``n`` edits.
        facts: The facts of the sequence.

    Returns:
        The checkpoint row, then one row for each pole.
    """
    runner, change = env["runner"], {param_name(LAYER): delta}
    fields = {"model": MODEL, "condition": kind, "edits": n, "edit_layer": LAYER,
              "sample_tokens": SAMPLE_TOKENS}  # fmt: skip
    summary, *poles = run_condition(runner, env["batches"], env["records"], env["base"],
                                    env["state"], change, fields)  # fmt: skip
    with changed(runner.model, change):
        measures = {
            "retention": round(top_share(runner.model, runner.tokenizer, facts[:n]), 4),
            "latest": round(top_share(runner.model, runner.tokenizer, facts[n - 1 : n]), 4)
            if n
            else None,
            "loss": round(sample_loss(runner.model, env["batches"]), 5),
        }
    above = [p for p in poles if p["layer"] > LAYER and p["base_passes"]]
    summary |= {"table": "checkpoint", **measures,
                "distance": round(distance(delta, env["weight"]), 6),
                "keeps_above": sum(p["keeps"] for p in above), "poles_above": len(above),
                "keeps_below": sum(p["keeps"] for p in poles if p["layer"] <= LAYER)}  # fmt: skip
    return [summary, *poles]


@click.command()
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--edits", "n_edits", default=EDITS, show_default=True)
def main(wiki_dir, out, device, n_edits) -> None:  # pragma: no cover - loads the model
    """Fit the ROME sequence, then test both sequences at each checkpoint.

    Args:
        wiki_dir: The registry that holds the VERIFIED records.
        out: The JSONL file to write. It is written again after each checkpoint.
        device: The device of the model.
        n_edits: The number of edits in each sequence.
    """
    from openinterpretability.cli import _load_runner

    records = verified_records(wiki_dir, MODEL)
    runner = _load_runner(MODEL, device, "float32", None)
    model, tokenizer = runner.model, runner.tokenizer
    for p in model.parameters():
        p.requires_grad_(False)
    batches = sample_windows(tokenizer)
    cache, firsts = probe_activations(runner, records), first_position_activations(runner, records)
    on_base = {r.submission.identifier.slug: cached_readings(runner, r, cache, firsts)
               for r in records}  # fmt: skip
    env = {"runner": runner, "batches": batches, "records": records,
           "base": base_columns(runner, batches, records),
           "state": {"on_base": on_base, "cache": cache, "firsts": firsts},
           "weight": down(model, LAYER).weight.detach().cpu().clone()}  # fmt: skip
    facts = make_facts(tokenizer, n_edits)
    cov = second_moment(model, sample_windows(tokenizer, COV_TOKENS), LAYER)
    edits = rome_edits(model, tokenizer, facts, cov, sample_windows(tokenizer, COV_TOKENS)[-3:])
    lines = base_rows(records, on_base, MODEL)
    lines += [{"table": "fact", "index": i, "subject": f[0], "object": f[2].strip(),
               "edit_norm": round(float(lam.norm() * u.norm()), 6)}
              for i, (f, (lam, u)) in enumerate(zip(facts, edits, strict=True))]  # fmt: skip
    zero = torch.zeros_like(env["weight"])
    for kind in ("rome", "random"):
        steps = deltas(edits, kind)
        for n in (c for c in CHECKPOINTS if c <= n_edits):
            rows = checkpoint(env, kind, n, steps[n - 1] if n else zero, facts)
            lines += rows
            click.echo(json.dumps({k: rows[0][k] for k in ("condition", "edits", "keeps_above",
                       "keeps_below", "distance", "retention", "loss")}), err=True)  # fmt: skip
            out.write_text("".join(json.dumps(r) + "\n" for r in lines))


if __name__ == "__main__":
    main()

"""How many of GPT-2's transcoder latents are neurons in disguise? (issue #3608).

Shabalin and Piotrowski (arXiv 2610.24521) call a transcoder feature a *neuron feature*
when its encoder and its decoder both align with one MLP neuron. This script measures that
for Dunefsky et al.'s GPT-2 small transcoders. The vectors are those of
:mod:`openinterpretability.harness.crossref`:

* a latent reads with its encoder (``latent_reads``, ``ln_2`` folded in, centred) and
  writes with its decoder (``latent_writes``).
* a neuron reads with ``c_fc`` (``mlp_reads``, ``ln_2`` folded in) and writes ``w_out``.

A latent and a neuron pole score the smaller of their two cosines. Each cosine is
oriented as crossref orients it, so + means the latent reads what drives the pole and
writes what the pole writes. A latent's score is its best pole's. The unsigned score
uses |cos| on each side with no pole.

    uv run python scripts/neuron_features.py --out docs/analyses/neuron-features.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import torch

from openinterpretability.harness.crossref import _unit, latent_reads, latent_writes
from openinterpretability.harness.dictionaries import DICTIONARIES, load_spec
from openinterpretability.harness.wiring import _matrix, mlp_reads

AUDITS = Path("docs/analyses/transcoder-baseline.jsonl")
DICTIONARY = "gpt2-small-transcoders"
THRESHOLDS = (0.6, 0.7, 0.9)
CHUNK = 2048
CHANCE_DRAWS = 4096  # random read and write pairs, for each layer's chance score


def scores(reads: torch.Tensor, writes: torch.Tensor, n_in: torch.Tensor, n_out: torch.Tensor):
    """Each latent's best oriented score, its neuron and pole, and its best unsigned score."""
    best, neuron, pole, unsigned = [], [], [], []
    for start in range(0, reads.shape[0], CHUNK):
        r = reads[start : start + CHUNK] @ n_in.T  # [c, width]
        w = writes[start : start + CHUNK] @ n_out.T
        positive, negative = torch.minimum(r, w), torch.minimum(-r, -w)
        both = torch.stack([positive, negative])  # [2, c, width]
        flat = both.permute(1, 0, 2).reshape(r.shape[0], -1)
        top = flat.max(dim=-1)
        best.append(top.values)
        neuron.append(top.indices % r.shape[1])
        pole.append(top.indices // r.shape[1])
        unsigned.append(torch.minimum(r.abs(), w.abs()).amax(dim=-1))
    return torch.cat(best), torch.cat(neuron), torch.cat(pole), torch.cat(unsigned)


def layer_scores(runner, layer: int):
    """The four tensors of :func:`scores` for every latent of ``layer``'s transcoder.

    Also the 99th percentile of the score of random read and write pairs. A read is
    centred, as LayerNorm centres a latent's read.
    """
    spec = DICTIONARIES[DICTIONARY]
    d = load_spec(spec, layer, runner.device)
    reads = _unit(latent_reads(runner, spec, layer, d))
    writes = _unit(latent_writes(runner, spec, layer, d))
    n_in = _unit(mlp_reads(runner, layer)["in"])
    n_out = _unit(_matrix(runner.down_proj(layer)).T)
    g = torch.Generator().manual_seed(3608 + layer)
    r, w = (torch.randn(CHANCE_DRAWS, reads.shape[1], generator=g) for _ in range(2))
    chance = scores(_unit(r - r.mean(-1, keepdim=True)), _unit(w), n_in, n_out)[0]
    return (*scores(reads, writes, n_in, n_out), float(chance.quantile(0.99)))


def shares(values: torch.Tensor) -> dict[str, float]:
    """The share of ``values`` at or above each threshold."""
    return {f"share_{t}": round(float((values >= t).float().mean()), 4) for t in THRESHOLDS}


@click.command()
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(out: Path) -> None:
    """Score the audited latents and every latent of each layer, and write the rows."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from openinterpretability.harness.hooks import NeuronHookRunner

    name = DICTIONARIES[DICTIONARY].model_name
    model = AutoModelForCausalLM.from_pretrained(name).eval()
    runner = NeuronHookRunner(model, AutoTokenizer.from_pretrained(name))
    audits = [json.loads(line) for line in AUDITS.open()]
    audited = [a for a in audits if a["arm"] == "latent" and a["outcome"] == "live"]
    rows = []
    for layer in range(runner.n_layers):
        best, neuron, pole, unsigned, chance = layer_scores(runner, layer)
        rows.append({"kind": "layer", "layer": layer, "latents": len(best),
                     "chance_99": round(chance, 4),
                     "median_score": round(float(best.median()), 4), **shares(best),
                     **{f"unsigned_{k}": v for k, v in shares(unsigned).items()}})  # fmt: skip
        for a in audited:
            at, feature = a["unit"].split("/")
            if int(at[1:]) != layer:
                continue
            f = int(feature[1:])
            rows.append({
                "kind": "latent", "unit": a["unit"], "verdict": a["verdict"],
                "score": round(float(best[f]), 4), "unsigned": round(float(unsigned[f]), 4),
                "neuron": int(neuron[f]), "pole": ("positive", "negative")[int(pole[f])],
            })  # fmt: skip
        click.echo(json.dumps(rows[-1] if rows[-1]["kind"] == "layer" else rows[-1]), err=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


if __name__ == "__main__":
    main()

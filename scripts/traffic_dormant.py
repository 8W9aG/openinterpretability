"""Each pythia-70m neuron's active shares on the traffic map's corpus sample (#5231).

The issue's second known case is that dormant neurons should carry almost no traffic. A
pole is dormant when it is active on under 1% of tokens (``scripts/dormant_rank.py``,
#5134). This script writes one ``activity`` row for each neuron, with the active shares of
its two poles on the sample of ``scripts/traffic_map.py``. ``scripts/traffic_stats.py``
joins them to the sites.

    uv run python scripts/traffic_dormant.py --device cuda \
        --out papers/traffic-map/results.activity.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dormant_rank import active_shares  # noqa: E402

from openinterpretability.interference.keep import layer_columns, sample_windows  # noqa: E402

MODEL = "EleutherAI/pythia-70m"


@click.command()
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cpu", show_default=True)
def main(out: Path, device: str) -> None:  # pragma: no cover - loads the model
    """Write each neuron's positive and negative active shares on the sample.

    Args:
        out: The JSONL file to write.
        device: The device of the model.
    """
    from openinterpretability.cli import _load_runner

    runner = _load_runner(MODEL, device, "float32", None)
    batches = sample_windows(runner.tokenizer)
    lines = []
    for layer in range(runner.n_layers):
        width = runner.model.gpt_neox.layers[layer].mlp.dense_4h_to_h.in_features
        shares = active_shares(layer_columns(runner, batches, layer, list(range(width))))
        lines += [{"table": "activity", "layer": layer, "index": j,
                   "positive": round(float(shares[j]), 6),
                   "negative": round(float(shares[width + j]), 6)}
                  for j in range(width)]  # fmt: skip
    out.write_text("".join(json.dumps(r) + "\n" for r in lines))


if __name__ == "__main__":
    main()

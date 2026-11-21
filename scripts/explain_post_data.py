"""Measure the data of the explain post's figures that the records don't keep.

The post (posts/explain/) follows one explanation of EleutherAI/pythia-70m: "The sun rises
in the" → " sky". The explanation record and its answer record hold most of what the
figures draw. This script measures the rest, with the record's pinned inputs:

- **baseline:** the mean of three neurons in each corpus window, and their values on the
  prompt, for the figure of how a baseline is found;
- **coverage:** the answer's log-probability clean, with every unit ablated, and with only
  the explained units kept;
- **minimal set:** the coverage after each step of the greedy pruning.

The explained set is the record's, not the registry's now, so the numbers match the
record. The script fails if its coverage differs from the record's.

Run it with ``uv run python scripts/explain_post_data.py``. It writes
posts/explain/media/data.json. It takes about a minute on a laptop CPU.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
RECORD = ROOT / "wiki/explanations/EleutherAI--pythia-70m/6c78ac114096/20261119T013311Z.json"
OUT = ROOT / "posts/explain/media/data.json"
NEURONS = ((5, 1296), (0, 354), (5, 1677))  # (layer, neuron): two top poles and the route's last
TOLERANCE = 1e-3  # the largest coverage difference from the record that the script accepts


def _windows(runner, texts, tokens: int, window: int) -> list[list[int]]:
    """The windows of corpus tokens that ``unit_means`` reads, in its order."""
    ids = [i for t in texts for i in runner.tokenizer.encode(t)][:tokens]
    return [ids[i : i + window] for i in range(0, len(ids) - window + 1, window)]


def baseline(runner, texts, inputs, prompt: str) -> dict:
    """Each neuron's mean in each window, past the window's first token, and on the prompt."""
    import torch

    windows = _windows(runner, texts, inputs.means_tokens, inputs.means_window)
    layers = sorted({la for la, _ in NEURONS})
    found: dict[str, list[float]] = {f"L{la}/N{n}": [] for la, n in NEURONS}
    store: dict[int, torch.Tensor] = {}

    def keep(layer):
        def hook(act):
            store[layer] = act

        return hook

    with torch.no_grad(), runner.hooks({la: keep(la) for la in layers}, None, None):
        for window in windows:
            runner.forward(runner.encode(torch.tensor([window])))
            for la, n in NEURONS:
                found[f"L{la}/N{n}"].append(round(float(store[la][0, 1:, n].float().mean()), 5))
        runner.forward(runner.encode(prompt))
        on_prompt = {
            f"L{la}/N{n}": [round(float(v), 5) for v in store[la][0, :, n].float()]
            for la, n in NEURONS
        }
    return {"window": inputs.means_window, "windows": found, "prompt": on_prompt}


def pruning(share, keep: list, full: float, floor_share: float) -> list[dict]:
    """The greedy pruning of ``explain_coverage.minimal_set``, with the coverage at each try.

    It makes the same choices as ``minimal_set``. Each item is one unit that it tried to
    remove: the unit, the coverage without it, and whether it stayed out.
    """
    floor = floor_share * full
    drops = []
    for i, unit in enumerate(keep):
        rest = share(keep[:i] + keep[i + 1 :])
        drops.append(((full - rest) if rest is not None else float("inf"), i, unit))
    current, steps = list(keep), []
    for _, _, unit in sorted(drops, key=lambda d: (d[0], d[1])):
        trial = [u for u in current if u != unit]
        value = share(trial) if trial else None
        removed = value is not None and value >= floor
        if removed:
            current = trial
        steps.append({"unit": unit, "coverage": None if value is None else round(value, 5),
                      "removed": removed, "left": len(current)})  # fmt: skip
    return steps


def main() -> int:
    """Measure the post's data, check it against the record and write it."""
    from openinterpretability import explain_groups
    from openinterpretability.cli import _load_runner
    from openinterpretability.core.explanation_records import ExplanationRecord
    from openinterpretability.explain import PoleAttribution, explain, unit_means
    from openinterpretability.explain_coverage import (
        MINIMAL,
        _log_prob,
        _share,
        keep_hooks,
        prompt_coverage,
    )
    from openinterpretability.reproduce_explanation import means_corpus

    record = ExplanationRecord.model_validate_json(RECORD.read_text(encoding="utf-8"))
    inputs, model = record.inputs, record.model_name
    env = inputs.environment
    runner = _load_runner(model, "cpu", env.dtype, env.model_revision)
    texts = means_corpus(record, {})
    means = unit_means(runner, model, texts, inputs.means_tokens)
    groups = explain_groups.superpositions(model) + explain_groups.circuits(model)
    result = explain(runner, model, record.prompt, means, {}, token=inputs.token_id,
                     position=record.position, top=inputs.top, edges=0, groups=groups)  # fmt: skip
    pinned = set(inputs.explained)
    order = [
        (u, a, s, PoleAttribution(*u, 0.0, 0.0).label in pinned) for u, a, s, _ in result.order
    ]
    args = (runner, record.prompt, means, inputs.token_id, record.position)
    coverage = prompt_coverage(*args, order, draws=inputs.random_draws, seed=inputs.random_seed)
    if abs(coverage["coverage"] - record.prompt_coverage) > TOLERANCE:
        print(f"coverage {coverage['coverage']:.4f} differs from the record's "
              f"{record.prompt_coverage:.4f}", file=sys.stderr)  # fmt: skip
        return 1
    clean, ablated = coverage["clean"], coverage["ablated"]

    def share(units):
        hooks = keep_hooks(runner, means, units)
        return _share(clean, ablated, _log_prob(runner, record.prompt, inputs.token_id,
                                                record.position, hooks))  # fmt: skip

    keep = [u for u, _, _, done in order if done]
    label = {u: PoleAttribution(*u, 0.0, 0.0).label for u in keep}
    steps = pruning(share, keep, coverage["coverage"], MINIMAL)
    for step in steps:
        step["unit"] = label[step["unit"]]
    data = {
        "record": str(RECORD.relative_to(ROOT)),
        "baseline": baseline(runner, texts, inputs, record.prompt),
        "coverage": {k: round(coverage[k], 5) for k in ("clean", "ablated", "kept", "coverage")}
        | {
            "units": coverage["units"],
            "random_draws": [round(d, 5) for d in coverage["random_draws"]],
        },
        "pruning": {"floor": MINIMAL, "full": round(coverage["coverage"], 5), "steps": steps},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: coverage {coverage['coverage']:.4f}, "
          f"{sum(1 for s in steps if not s['removed'])} units kept by the pruning")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())

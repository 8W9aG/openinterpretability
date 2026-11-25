"""How many VERIFIED explanations are token-set rules: the ceiling of weight-only checks (#5144).

Each standing VERIFIED neuron pole's explanation has a label, from a fixed rubric:

* ``token-set``: whether the pole fires can be decided from the current token alone;
* ``context``: deciding needs other tokens, such as the previous token or the topic;
* ``mixed``: a token set that the explanation gates by context.

A weight-only check (ROTATE #3446, #5062, #3457) can test only a token-set rule, so the
token-set share caps what it can confirm. Subagents gave the labels, and two more relabelled a
seeded fifth of the poles without seeing the first labels. This script joins the labels with
each record's ``token_r2``, then writes the shares, the agreement and the behavioural check.

    uv run python scripts/token_set_ceiling.py --labels labels/ \\
        --out papers/token-set-ceiling/results.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from base_instruct_delta import auroc  # noqa: E402

from openinterpretability.interference.keep import wilson  # noqa: E402

LABELS = ("token-set", "mixed", "context")
LAYERS = {  # each configured model's blocks, for a pole's relative depth
    "EleutherAI/pythia-70m": 6,
    "openai-community/gpt2": 12,
    "Qwen/Qwen2.5-0.5B": 24,
    "Qwen/Qwen2.5-1.5B": 28,
}
DEPTHS = ((0.0, 1 / 3, "early"), (1 / 3, 2 / 3, "middle"), (2 / 3, 1.01, "late"))
BOOT, SEED = 1000, 5144


def kappa(first: list[str], second: list[str]) -> float:
    """Cohen's kappa between two raters' labels of the same items.

    Args:
        first: The first rater's labels.
        second: The second rater's labels, in the same order.

    Returns:
        The kappa: 1 for full agreement, 0 for agreement at chance.

    Examples:
        >>> kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"])
        1.0
        >>> kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"])
        0.0
    """
    n = len(first)
    seen = sorted(set(first) | set(second))
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / n
    chance = sum((first.count(k) / n) * (second.count(k) / n) for k in seen)
    return 1.0 if chance == 1 else round((observed - chance) / (1 - chance), 4)


def depth_band(depth: float) -> str:
    """A relative depth's third of the model: early, middle or late.

    Args:
        depth: The layer over the last layer, from 0 to 1.

    Returns:
        The band's name.

    Examples:
        >>> depth_band(0.0), depth_band(0.5), depth_band(1.0)
        ('early', 'middle', 'late')
    """
    return next(name for low, high, name in DEPTHS if low <= depth < high)


def share_row(poles: list[dict], **keys) -> dict:
    """The token-set share of ``poles``, with each label's count and a 95% Wilson interval.

    Args:
        poles: The pole rows.
        **keys: What the row describes, such as ``model`` and ``depth``.

    Returns:
        The row.
    """
    counts = {label: sum(p["label"] == label for p in poles) for label in LABELS}
    n, k = len(poles), counts["token-set"]
    return {"table": "share", **keys, "poles": n, **counts,
            "token_set_share": round(k / n, 4) if n else None,
            "ci95": wilson(k, n) if n else None}  # fmt: skip


def token_r2_test(poles: list[dict], boot: int = BOOT, seed: int = SEED) -> dict:
    """``token_r2`` as an AUROC for a token-set label, with a 95% bootstrap interval over poles.

    Args:
        poles: The pole rows.
        boot: The bootstrap draws.
        seed: Their seed.

    Returns:
        The row.
    """
    scores = [p["token_r2"] for p in poles]
    hits = [p["label"] == "token-set" for p in poles]
    rng, draws = random.Random(seed), []
    for _ in range(boot):
        idx = [rng.randrange(len(poles)) for _ in poles]
        if 0 < sum(hits[i] for i in idx) < len(idx):
            draws.append(auroc([scores[i] for i in idx], [hits[i] for i in idx]))
    draws.sort()
    pairs = list(zip(scores, hits, strict=True))
    median = lambda xs: round(sorted(xs)[len(xs) // 2], 4)  # noqa: E731
    return {"table": "token_r2", "poles": len(poles), "auroc": round(auroc(scores, hits), 4),
            "ci95": [round(draws[int(0.025 * len(draws))], 4),
                     round(draws[int(0.975 * len(draws)) - 1], 4)],
            "median_token_set": median([s for s, h in pairs if h]),
            "median_other": median([s for s, h in pairs if not h])}  # fmt: skip


def read_labels(folder: Path, pattern: str) -> dict[str, dict]:
    """Every label file in ``folder`` that matches ``pattern``, by slug.

    Args:
        folder: The folder of label files.
        pattern: The files' glob, such as ``labels-*.json``.

    Returns:
        ``{slug: {"label", "reason"}}``.
    """
    out = {}
    for path in sorted(folder.glob(pattern)):
        for row in json.loads(path.read_text()):
            out[row["slug"]] = {"label": row["label"], "reason": row["reason"]}
    return out


def result_rows(poles: list[dict], second: dict[str, dict]) -> list[dict]:
    """The paper's rows: each pole, the shares, the agreement and the ``token_r2`` check.

    Args:
        poles: The pole rows, each with its first label.
        second: The second rater's labels, by slug.

    Returns:
        The rows.
    """
    rows = [{"table": "pole", **p} for p in poles]
    rows.append(share_row(poles, model="all", depth="all"))
    for model in LAYERS:
        mine = [p for p in poles if p["model"] == model]
        rows.append(share_row(mine, model=model, depth="all"))
        for *_, band in DEPTHS:
            rows.append(share_row([p for p in mine if p["band"] == band], model=model, depth=band))
    both = [p for p in poles if p["slug"] in second]
    first, other = [p["label"] for p in both], [second[p["slug"]]["label"] for p in both]
    ceiling = [lab == "token-set" for lab in first], [lab == "token-set" for lab in other]
    same = sum(a == b for a, b in zip(first, other, strict=True))
    rows.append({"table": "agreement", "poles": len(both), "kappa": kappa(first, other),
                 "kappa_token_set": kappa(*([str(x) for x in c] for c in ceiling)),
                 "agree": round(same / len(both), 4)})  # fmt: skip
    rows.append(token_r2_test(poles))
    return rows


@click.command()
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option("--labels", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(wiki_dir: Path, labels: Path, out: Path) -> None:
    """Join the labels with the records, and write the paper's rows.

    \f
    Args:
        wiki_dir: The registry.
        labels: The folder of ``labels-*.json`` and ``second-labels-*.json``.
        out: Where to write the rows.
    """
    from openinterpretability.interference.keep import verified_records

    first, second = (
        read_labels(labels, "labels-*.json"),
        read_labels(labels, "second-labels-*.json"),
    )
    poles = []
    for model, blocks in LAYERS.items():
        for record in verified_records(wiki_dir, model):
            ident = record.submission.identifier
            depth = round(ident.layer / (blocks - 1), 4)
            poles.append({"slug": ident.slug, "model": model, "layer": ident.layer,
                          "depth": depth, "band": depth_band(depth),
                          "token_r2": round(record.submission.confounds.token_r2, 6),
                          **first[ident.slug]})  # fmt: skip
    out.write_text("".join(json.dumps(r) + "\n" for r in result_rows(poles, second)))


if __name__ == "__main__":  # pragma: no cover
    main()

"""The base-to-instruct output KL at each VERIFIED pole's own texts, against its keep (#5141).

For each Qwen2.5 base and instruct pair, each VERIFIED pole of the base model has two sets
of texts. Its probes are the hard negatives and positive holdouts of its verdict. Its top
contexts are its scan's top examples. For each text, this reads the next-token
``KL(instruct ‖ base)`` at each position, and averages it over the text. That is the same
direction as ``scripts/base_instruct_patch.py``. It reads each text twice:

* *plain*: the text alone;
* *chat*: the text as one user turn, rendered by the instruct model's chat template for both
  models (as ``interference.keep.chat_strength`` does). Only the user turn's positions count.

A pole's four measures are then each tested as an AUROC for predicting that its explanation
fails on the instruct model. The labels are the keep outcomes in
``docs/analyses/base-instruct*.jsonl``, plain measures against plain labels and chat
measures against chat labels. The null shuffles the labels within each layer, as for the
weight change (#5133) and the Fisher (#5118). It trains nothing.

    uv run python scripts/base_instruct_kl.py --device mps \\
        --out docs/analyses/base-instruct-kl.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from base_instruct_delta import auroc, shuffled_p  # noqa: E402

from openinterpretability.interference.keep import verified_records  # noqa: E402

# Each pair: the base model, and its keep rows on plain text and in the chat template.
PAIRS = {
    "Qwen/Qwen2.5-0.5B": ("base-instruct-0.5b.jsonl", "base-instruct-0.5b-chat.jsonl"),
    "Qwen/Qwen2.5-1.5B": ("base-instruct.jsonl", "base-instruct-chat.jsonl"),
}
MEASURES = ("probes", "top")
MAX_TOKENS = 256  # a text's first tokens: probes are short, and top contexts are windows
BATCH_POSITIONS = 2048  # padded positions in one pass: the logits are positions × 151,936


def mean_kl(target: torch.Tensor, other: torch.Tensor, keep: list[bool] | None = None) -> float:
    """The mean of ``KL(target ‖ other)`` over a text's positions.

    Args:
        target: The instruct model's logits, ``[seq, vocab]``.
        other: The base model's logits, ``[seq, vocab]``.
        keep: Which positions count, or None for all of them.

    Returns:
        The mean KL, in nats.

    Examples:
        >>> a = torch.tensor([[2.0, 0.0], [0.0, 0.0]])
        >>> mean_kl(a, a)
        0.0
        >>> round(mean_kl(a, torch.zeros(2, 2), [True, False]), 4)
        0.3278
    """
    log_p = torch.log_softmax(target.float(), -1)  # float32: MPS has no float64
    log_q = torch.log_softmax(other.float(), -1)
    per = (log_p.exp() * (log_p - log_q)).sum(-1)
    if keep is not None:
        per = per[torch.tensor(keep, dtype=torch.bool)]
    return float(per.mean()) if len(per) else float("nan")


def turn_positions(offsets: list[tuple[int, int]], start: int, end: int) -> list[bool]:
    """Which tokens of a rendered chat lie in the user turn's text, from their offsets.

    Args:
        offsets: Each token's ``(start, end)`` characters in the rendered chat.
        start: Where the user's text starts in the rendered chat.
        end: Where it ends.

    Returns:
        One flag per token.

    Examples:
        >>> turn_positions([(0, 5), (5, 9), (9, 12), (12, 20)], 6, 12)
        [False, True, True, False]
    """
    return [a < end and b > start for a, b in offsets]


def encoded(tokenizer, texts: list[str], template=None) -> tuple[torch.Tensor, list[list[bool]]]:
    """The texts' token ids, right-padded into one batch, and which positions count in each.

    Args:
        tokenizer: The base model's tokenizer, which both models share.
        texts: The texts.
        template: The tokenizer whose chat template renders each text as a user turn, or
            None for plain text.

    Returns:
        The ids, ``[texts, longest]``, and each text's flags. A padded position never counts.
    """
    rows, flags = [], []
    for text in texts:
        text = clipped(tokenizer, text)
        if template is None:
            ids = tokenizer(text)["input_ids"]
            keep = [True] * len(ids)
        else:
            rendered = template.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=False
            )
            start = rendered.rindex(text)  # the user turn, after any system turn
            enc = tokenizer(rendered, return_offsets_mapping=True)
            ids, keep = (
                enc["input_ids"],
                turn_positions(enc["offset_mapping"], start, start + len(text)),
            )
        rows.append(ids)
        flags.append(keep)
    width = max(len(r) for r in rows)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ids = torch.tensor([r + [pad] * (width - len(r)) for r in rows])
    return ids, [f + [False] * (width - len(f)) for f in flags]


def clipped(tokenizer, text: str) -> str:
    """``text`` cut to its first :data:`MAX_TOKENS` tokens, so a long top context stays cheap.

    Args:
        tokenizer: The tokenizer.
        text: The text.

    Returns:
        The text, or its first tokens decoded.
    """
    ids = tokenizer(text)["input_ids"]
    return text if len(ids) <= MAX_TOKENS else tokenizer.decode(ids[:MAX_TOKENS])


def batches(lengths: list[int], budget: int = BATCH_POSITIONS) -> list[list[int]]:
    """Indices of texts grouped so each group's padded size stays within ``budget``.

    Args:
        lengths: Each text's length in tokens.
        budget: The most padded positions in one group.

    Returns:
        The groups, in order. A text longer than the budget is a group alone.

    Examples:
        >>> batches([3, 3, 3, 5], budget=9)
        [[0, 1, 2], [3]]
        >>> batches([10, 1], budget=8)
        [[0], [1]]
    """
    groups: list[list[int]] = []
    for i, n in enumerate(lengths):
        if groups:
            width = max(n, *(lengths[j] for j in groups[-1]))
            if width * (len(groups[-1]) + 1) <= budget:
                groups[-1].append(i)
                continue
        groups.append([i])
    return groups


def texts_kl(base, instruct, texts: list[str], template=None) -> list[float]:
    """Each text's mean ``KL(instruct ‖ base)``, plain or as a chat user turn, in one batch.

    Right padding leaves each text's own positions as they would be alone. The model is
    causal, so a position never sees the padding after it.

    Args:
        base: The base model's runner.
        instruct: The instruct model's runner.
        texts: The texts.
        template: The tokenizer whose chat template renders each turn, or None for plain text.

    Returns:
        One mean KL per text.
    """
    out: list[float] = [float("nan")] * len(texts)
    lengths = [len(base.tokenizer(clipped(base.tokenizer, t))["input_ids"]) + 32 for t in texts]
    for group in batches(lengths, BATCH_POSITIONS):  # + 32 for a chat template's tokens
        ids, flags = encoded(base.tokenizer, [texts[i] for i in group], template)
        with torch.no_grad():
            a = instruct.run_with_interventions(ids, {})
            b = base.run_with_interventions(ids, {})
        for row, i in enumerate(group):
            out[i] = mean_kl(a[row], b[row], flags[row])
    return out


def pole_texts(record) -> dict[str, list[str]]:
    """A pole's probes and its top contexts.

    Args:
        record: The pole's standing VERIFIED record.

    Returns:
        ``{"probes": [...], "top": [...]}``.
    """
    probes = [r.prompt for r in record.verdict.adversarial_test_results
              if r.kind in ("hard_negative", "positive_holdout")]  # fmt: skip
    return {"probes": probes, "top": [e.text for e in record.submission.top_examples]}


def pole_row(base, instruct, record) -> dict:
    """One pole's four KL measures: probes or top contexts, plain or in the chat template.

    Args:
        base: The base model's runner.
        instruct: The instruct model's runner.
        record: The pole's standing VERIFIED record.

    Returns:
        The pole's row.
    """
    ident, texts = record.submission.identifier, pole_texts(record)
    row = {"table": "pole", "slug": ident.slug, "layer": ident.layer}
    for measure in MEASURES:
        for mode, template in (("plain", None), ("chat", instruct.tokenizer)):
            values = texts_kl(base, instruct, texts[measure], template)
            values = [v for v in values if v == v]  # a text with no position gives nan
            row[f"kl_{measure}_{mode}"] = round(sum(values) / len(values), 6) if values else None
    return row


def keep_labels(path: Path) -> dict[str, bool]:
    """Each pole's keep outcome from a ``base-instruct`` rows file, for the poles that reproduced.

    Args:
        path: The rows file.

    Returns:
        ``{slug: keeps}``.
    """
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["slug"]: r["keeps"] for r in rows if r.get("table") == "pole" and r["base"]["passes"]}


def auroc_rows(base_name: str, poles: list[dict], labels: dict[str, dict[str, bool]]) -> list[dict]:
    """Each KL measure as an AUROC for a failure, with its within-layer shuffle p.

    Args:
        base_name: The base model.
        poles: The pole rows.
        labels: The keep labels by mode, ``{"plain": {slug: keeps}, "chat": {...}}``.

    Returns:
        One row per measure and mode.
    """
    out = []
    for measure in MEASURES:
        for mode, keeps in labels.items():
            key = f"kl_{measure}_{mode}"
            used = [p for p in poles if p["slug"] in keeps and p[key] is not None]
            scores = [p[key] for p in used]
            fails = [not keeps[p["slug"]] for p in used]
            layers = [p["layer"] for p in used]
            out.append({"table": "test", "base": base_name, "measure": measure, "mode": mode,
                        "poles": len(used), "fails": sum(fails),
                        "auroc": round(auroc(scores, fails), 4),
                        "p_within_layer_shuffle": shuffled_p(scores, fails, layers)})  # fmt: skip
    return out


@click.command()
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--analyses", default="docs/analyses", type=click.Path(file_okay=False, path_type=Path)
)
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cpu", show_default=True)
def main(wiki_dir: Path, analyses: Path, out: Path, device: str) -> None:
    """Read each pair's KL at each VERIFIED pole's texts, test it, and write the rows.

    \f
    Args:
        wiki_dir: The registry.
        analyses: The folder of the ``base-instruct`` rows, which hold the keep labels.
        out: Where to write the rows.
        device: The device to run the models on.
    """
    from openinterpretability.cli import _load_runner

    rows = []
    for base_name, (plain, chat) in PAIRS.items():
        base = _load_runner(base_name, device, "float32", None)
        instruct = _load_runner(f"{base_name}-Instruct", device, "float32", None)
        records = verified_records(wiki_dir, base_name)
        poles = []
        for i, record in enumerate(records, 1):
            poles.append(pole_row(base, instruct, record))
            print(f"{base_name}: {i} of {len(records)} poles", flush=True)
        labels = {"plain": keep_labels(analyses / plain), "chat": keep_labels(analyses / chat)}
        rows += [{**p, "base": base_name} for p in poles] + auroc_rows(base_name, poles, labels)
        del base, instruct
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":  # pragma: no cover
    main()

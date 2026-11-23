"""Does instruction tuning move VERIFIED poles more on chat text than on plain text (#5126)?

For each neuron pole of ``--base`` that stands VERIFIED, the script reads the record's own
probes in four conditions: the base model and its ``-Instruct`` model, each on plain text and
inside the chat template. Both models read the same rendered template, the instruct
model's (``interference/template.py``). The scale rules are those of ``base_instruct.py``.
From the four readings come three changes for each pole: the template's own effect, the
tuning's effect on plain text and the tuning's effect on chat text. The script also gives
each pole's base-to-instruct correlation on a plain sample and on a chat sample, and its
mean strength on the template's own tokens. It writes no record.

    uv run python scripts/template_tuning.py --device cpu \\
        --out papers/template-or-tuning/results.jsonl
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openinterpretability.interference.keep import (  # noqa: E402
    GATES,
    pearson,
    readings,
    sample_stats,
    sample_windows,
    verified_records,
    wilson,
)
from openinterpretability.interference.template import (  # noqa: E402
    chat_columns,
    chat_sample,
    pole_mean,
)

BASE = "Qwen/Qwen2.5-0.5B"
CONDITIONS = ("base_plain", "base_chat", "instruct_plain", "instruct_chat")
# Each comparison: its name, then the conditions before and after the change.
CHANGES = (("template", "base_plain", "base_chat"),
           ("tuning_plain", "base_plain", "instruct_plain"),
           ("tuning_chat", "base_chat", "instruct_chat"))  # fmt: skip


def binomial_p(k: int, n: int) -> float:
    """The two-sided exact binomial p-value of ``k`` of ``n`` at one half.

    Args:
        k: The count on one side.
        n: The count on both sides.

    Returns:
        The p-value, 1 when ``n`` is 0.

    Examples:
        >>> binomial_p(0, 0), round(binomial_p(1, 10), 4)
        (1.0, 0.0215)
    """
    if not n:
        return 1.0
    low = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(low + 1)) / 2**n)


def chat_stats(base, instruct, sample: list[dict], records: list) -> dict[str, dict]:
    """Each pole's correlation on the chat sample's text, and its strength on the template.

    Args:
        base: The base model's runner.
        instruct: The instruct model's runner.
        sample: The documents of :func:`chat_sample`.
        records: The VERIFIED records.

    Returns:
        ``{slug: {"pearson_chat", "template_base", "template_instruct"}}``.
    """
    wanted: dict[int, list[int]] = defaultdict(list)
    for r in records:
        ident = r.submission.identifier
        if ident.neuron_idx not in wanted[ident.layer]:
            wanted[ident.layer].append(ident.neuron_idx)
    a = chat_columns(base, sample, wanted)
    b = chat_columns(instruct, sample, wanted)
    out = {}
    for r in records:
        ident = r.submission.identifier
        col = wanted[ident.layer].index(ident.neuron_idx)
        on_a, on_b = a[ident.layer], b[ident.layer]
        out[ident.slug] = {
            "pearson_chat": pearson(on_a["text"][:, col], on_b["text"][:, col]),
            "template_base": round(pole_mean(on_a["template"][:, col], ident.pole), 6),
            "template_instruct": round(pole_mean(on_b["template"][:, col], ident.pole), 6),
        }
    return out


def compare(rows: list[dict]) -> list[dict]:
    """For each change, how many poles it moves on keep and on each rate, with paired tests.

    Args:
        rows: The ``pole`` rows. Only poles that keep on the base model as plain text count.

    Returns:
        One ``change`` row for each of :data:`CHANGES`. ``lost`` and ``gained`` count the
        poles whose keep flips, with an exact McNemar p. For each rate, ``down`` and ``up``
        count the poles whose rate falls or rises, with a sign-test p and the mean change.
    """
    poles = [r for r in rows if r["base_plain"]["passes"]]
    out = []
    for name, before, after in CHANGES:
        lost = sum(r[before]["passes"] and not r[after]["passes"] for r in poles)
        gained = sum(r[after]["passes"] and not r[before]["passes"] for r in poles)
        row = {"table": "change", "change": name, "before": before, "after": after,
               "poles": len(poles), "keep_before": sum(r[before]["passes"] for r in poles),
               "keep_after": sum(r[after]["passes"] for r in poles), "lost": lost,
               "gained": gained, "p_keep": round(binomial_p(lost, lost + gained), 6)}  # fmt: skip
        for gate in GATES:
            deltas = [r[after][gate] - r[before][gate] for r in poles]
            down, up = sum(d < 0 for d in deltas), sum(d > 0 for d in deltas)
            row[gate] = {"mean_change": round(sum(deltas) / len(deltas), 4), "down": down,
                         "up": up, "p_sign": round(binomial_p(down, down + up), 6)}  # fmt: skip
        out.append(row)
    return out


def summary(rows: list[dict], base_name: str) -> dict:
    """The ``summary`` row: the keep share in each condition, with a Wilson interval.

    Args:
        rows: The ``pole`` rows.
        base_name: The base model.

    Returns:
        The row.
    """
    poles = [r for r in rows if r["base_plain"]["passes"]]
    keeps = {c: sum(r[c]["passes"] for r in poles) for c in CONDITIONS}
    return {"table": "summary", "base": base_name, "instruct": f"{base_name}-Instruct",
            "poles": len(rows), "reproduced": len(poles), "keeps": keeps,
            "keep_ci95": {c: wilson(k, len(poles)) for c, k in keeps.items()}}  # fmt: skip


@click.command()
@click.option("--wiki-dir", default="wiki", type=click.Path(file_okay=False, path_type=Path))
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--base", "base_name", default=BASE, show_default=True)
@click.option("--corpus", default="corpus.jsonl", show_default=True)
def main(wiki_dir: Path, out: Path, device: str, base_name: str, corpus: str) -> None:
    """Read each VERIFIED pole in the four conditions, and write the rows.

    Args:
        wiki_dir: The registry's wiki.
        out: The JSON Lines file to write.
        device: The device of the models.
        base_name: The base model. Its instruct model is ``<base>-Instruct``.
        corpus: The corpus of the two samples.
    """
    from openinterpretability.cli import _load_runner

    records = verified_records(wiki_dir, base_name)
    base = _load_runner(base_name, device, "float32", None)
    instruct = _load_runner(f"{base_name}-Instruct", device, "float32", None)
    render = instruct.tokenizer  # one template for both models
    plain = sample_stats(base, instruct, sample_windows(base.tokenizer, corpus=corpus), records)
    chat = chat_stats(base, instruct, chat_sample(render, base.tokenizer, corpus), records)
    rows = []
    for record in records:
        ident = record.submission.identifier
        ratio = plain[ident.slug]["ratio"] or 1.0
        rows.append({
            "table": "pole", "slug": ident.slug, "layer": ident.layer, "pole": ident.pole,
            "base_plain": readings(base, record),
            "base_chat": readings(base, record, 1.0, chat=True, template=render),
            "instruct_plain": readings(instruct, record, ratio),
            "instruct_chat": readings(instruct, record, ratio, chat=True, template=render),
            "ratio": plain[ident.slug]["ratio"], "pearson_plain": plain[ident.slug]["pearson"],
            **chat[ident.slug],
        })  # fmt: skip
    head = summary(rows, base_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in [head, *compare(rows), *rows]))
    click.echo(json.dumps(head))


if __name__ == "__main__":
    main()

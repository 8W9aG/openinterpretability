"""Neuronpedia's explanations as a blind proposer baseline.

The command is `open-interpretability neuronpedia-baseline`.

Does Neuronpedia's explanation of a dictionary latent explain it as well as the agent's
hypothesis? The rule was fixed before scoring. Each experiment on a latent that
Neuronpedia hosts takes the latent's first machine-written explanation, word for word
(:func:`openinterp.crossref.machine_explanation`). Human-written ones are
left out, so the baseline is one automatic explainer.

The explanation is then scored on the record's own held-out windows, so the two scores
are paired. The Patchscope baseline is scored the same way (:mod:`openinterp.baseline`).
Nothing is sent to Neuronpedia. The explanations are read once through its public API
and kept in the cache of :mod:`openinterp.crossref`. Each row credits Neuronpedia and the model that
wrote the explanation, and links to the latent's page. The author's account is not kept.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from openinterp.baseline import METRICS, BaselineRow, paired_report

CREDIT = "Neuronpedia"


def reading(explainer: str | None) -> str:
    """The ``reading`` of a row: who wrote the baseline, or that there was none."""
    return f"{CREDIT}, {explainer}" if explainer else f"no {CREDIT} explanation"


def report(rows: Sequence[BaselineRow], links: dict[str, str | None]) -> str:
    """Return the comparison as a Markdown page.

    ``links`` maps each row's slug to its latent's Neuronpedia page. The page gives the
    rule, the paired table, and every explanation with its credit and link.
    """
    have = [r for r in rows if r.baseline_explanation is not None]
    judged = sum(r.baseline_judge_model is not None for r in rows)
    units = {r.slug for r in rows}
    covered = {r.slug for r in have}
    explainers = Counter(r.reading.removeprefix(f"{CREDIT}, ") for r in have)
    credits = ", ".join(f"{k} on {n} experiments" for k, n in explainers.items()) or "none"
    intro = (
        f"Explanations from [Neuronpedia](https://www.neuronpedia.org), credited to the "
        f"model that wrote them ({credits}). "
        f"The rule, fixed before scoring: each latent's first machine-written explanation, "
        f"word for word. Human-written ones are left out. Neuronpedia has one for "
        f"{len(covered)} of the {len(units)} audited latents it hosts. Each is scored on the "
        f"same held-out windows as the agent's explanation. Detection and fuzzing pair only "
        f"experiments whose agent explanation was judged by the same model as the "
        f"baseline's ({judged} of {len(rows)}). Embedding uses the record's own encoder. "
        f"The sign test counts experiments, and a unit with several experiments counts "
        f"more than once."
    )
    after = [
        "Per experiment, agent → Neuronpedia on each metric (— where unpaired). The "
        "agent's explanation is on the unit's page.",
        "",
        "| Unit | Status | Neuronpedia's explanation | Detection | Fuzzing | Embedding |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in sorted(rows, key=lambda r: (r.slug, r.experiment_id)):
        link = links.get(r.slug)
        unit = f"[{r.slug}]({link})" if link else r.slug
        theirs = _cell(r.baseline_explanation) if r.baseline_explanation else "—"
        pairs = " | ".join(_pair(r, name) for name in METRICS)
        after.append(f"| {unit} | {r.status} | {theirs} | {pairs} |")
    return paired_report(
        rows,
        title="Neuronpedia explanations as a blind proposer baseline",
        generated_by="`open-interpretability neuronpedia-baseline`",
        method="`openinterp/neuronpedia_baseline.py` and docs/PROTOCOL.md §4",
        intro=intro,
        after=after,
    )


def _pair(row: BaselineRow, name: str) -> str:
    """``agent → baseline`` on metric ``name``, or ``—`` if either is missing."""
    a, b = row.agent[name], row.baseline[name]
    return "—" if a is None or b is None else f"{a:.2f} → {b:.2f}"


def _cell(text: str) -> str:
    """``text`` made safe for one Markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|").replace("<", "&lt;")

# Verdict stability across scan corpora

Issue #3187 asks how often a rerun, another reviewer or another corpus changes a verdict. `verdict-stability.md` measured reruns, and `reviewer-crossover.md` measured other reviewers. This analysis measures another corpus. Run on 2026-11-14 with `scripts/corpus_stability.py`. The rows are in `corpus-stability.jsonl`. Reported, not gated.

**Answer.** With the hypothesis and the probes fixed, scanning on another corpus changes VERIFIED-or-not for 7 of 40 records, and the status for 10 of 40. Weighted to the registry's mix, that is about 17% of first readings. That is the same size as the rerun noise (18%, `verdict-stability.md`).

## Why the corpus can change a verdict

The registry can't measure this from its records alone. A rescan writes a new hypothesis, so no hypothesis has been read on two corpora: there were 0 such pairs on 2026-11-14. But a verdict reads the corpus in one place, the scan of the unit:

- **The scale.** The scan's maximum activation is the scale. A probe counts as active at 0.25 × the scale (`docs/PROTOCOL.md` §4), and counterfactual Δs are divided by it.
- **The confound baselines** use the same threshold.
- **The default-on gate** reads the scan's active share.

Each probe's raw activation is stored in the record. So the same hypothesis and probes can be judged again with another corpus's scan, with no LLM and no new probe measurement. The proposer's view of the unit can also change with the corpus, and so the hypothesis. That is not measured here.

## Method

- **Records.** 40 first readings of random native neuron poles under protocol 0.8. All of them were scanned on the default corpus (pile-10k, `ded48cfc…`). There are 10 for each pair of model and outcome: GPT-2 small and Qwen2.5-0.5B, VERIFIED and not. They were drawn with seed 0 from 265 GPT-2 and 158 Qwen records.
- **The other corpus.** fineweb-edu (`3e64a923…`), rebuilt from its pinned source with `open-interpretability corpus --same-as`. It has the same 500 documents of 3,000 characters at most.
- **Rescan.** Each pole was scanned on fineweb-edu with its record's token budget and windows (`open-interpretability audit --evidence-out`). One Qwen record had scanned every document with no budget, so its rescan did too.
- **Judge again.** Each record's stored probe and counterfactual results went through `review_metrics` and `decide` with the new scan:
  - its scale, the threshold that follows from it, and its active share
  - the confound baselines from its own confound profile
- **Control.** The same code with each record's own scan gave the stored status and balanced accuracy for all 40 records.

## Results

| Records | VERIFIED-or-not changed | Status changed |
| --- | --- | --- |
| All 40 | 7 (17.5%, 95% CI 8.7–32%) | 10 (25%, 14–40%) |
| GPT-2 small, VERIFIED (10) | 2 | 2 |
| GPT-2 small, not VERIFIED (10) | 2 | 5 |
| Qwen2.5-0.5B, VERIFIED (10) | 3 | 3 |
| Qwen2.5-0.5B, not VERIFIED (10) | 0 | 0 |

The sample has as many VERIFIED records as not, and the registry has fewer VERIFIED ones. So the rates are weighted by the registry's share of VERIFIED first readings in each model. That gives 20% for GPT-2 small and 12% for Qwen2.5-0.5B, so 17% over both.

| Stored → with fineweb-edu | Records |
| --- | --- |
| VERIFIED → VERIFIED | 15 |
| VERIFIED → NEEDS_REVISION | 5 |
| NEEDS_REVISION → NEEDS_REVISION | 10 |
| NEEDS_REVISION → VERIFIED | 2 |
| REJECTED → REJECTED | 5 |
| REJECTED → NEEDS_REVISION | 3 |

- **The scale moves both ways.** fineweb-edu's maximum is a median 1.06 times pile-10k's, with quartiles 0.88 and 1.21 and a range of 0.60–2.29. A smaller maximum lowers the threshold, so more hard negatives become active. A larger one raises it, so more holdouts drop out.
- **Every change is at a gate's edge.** Balanced accuracy changes by 0.036 on average, and by a median of 0. The 10 changes are:
  - VERIFIED to NEEDS_REVISION: specificity failed 2 times, sensitivity 2 times, and the confound baseline 2 times. One record failed two gates.
  - NEEDS_REVISION to VERIFIED: 2 records, by specificity or the confound baseline.
  - REJECTED to NEEDS_REVISION: 3 records that rose above the 0.6 line.
- **One fifth of VERIFIED records lose it** (5 of 20). So a VERIFIED verdict holds the scale that its own corpus gives, and part of it rests on that choice.

## What it means for the question

A rerun and another corpus each change about one verdict in six. A later reading by another reviewer changes about one in three:

| Change | VERIFIED-or-not flips | Source |
| --- | --- | --- |
| A rerun by the same reviewer | 18% | `verdict-stability.md` |
| Another corpus, same probes | 17% (weighted) | this analysis |
| A later reading by another reviewer model, with new probes | 33% (61 of 187 pairs) | `registry-stats.md`, not a random sample |

The corpus effect works only through the scale and the scan's statistics, so it could be reduced without new reviews. For example, a scale taken over a larger or pooled corpus would vary less between corpora. The protocol doesn't say which corpus is right: both corpora here are pinned, and either is a valid scan.

## Limits

- **40 records, one other corpus.** The intervals are wide. A corpus less like pile-10k than fineweb-edu, such as code, could change more.
- **The hypothesis is fixed.** A proposer that sees the new corpus's top examples could write a different hypothesis. That is a larger effect, and it needs agents.
- **The probes were written against the original scan's examples.** The reviewer never saw activations, but it did see the hypothesis, which came from those examples.
- **One seed per scan sample.** The scan reads a seeded sample of documents, so the scale also varies within a corpus. This analysis doesn't separate that from the change of corpus.

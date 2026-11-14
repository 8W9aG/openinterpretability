# Registry-wide statistics over random neuron poles

This analysis does the first item of `TODO.md` (*Protocol and science*): the registry's verdict rates over units chosen at random. The random draws that the scan stopped on count as outcomes. The other items of that section build on these numbers. The registry was read on 2026-11-02, at 1,594 experiments.

The rows are in `registry-stats.jsonl`, written by `scripts/registry_stats.py` from the parquet that `openinterp build` exports. Reported, not gated.

## Method

- **Units.** MLP neuron poles whose first reviewed experiment has `selection = "random"`. A follow-up on a unit records `manual`, so the first experiment decides. Sister studies test a borrowed explanation and are left out. That gives 1,088 units with 1,431 experiments: 323 GPT-2 small, 736 Qwen2.5-0.5B and 29 Qwen2.5-1.5B. Heads (5, all manual) and dictionary latents (the learned-basis baseline, `transcoder-baseline.md`) are not pooled with them.
- **Three shares VERIFIED.**
  - *First reading*: the verdict of the unit's first experiment. Follow-ups can't raise it, so it is the share used below unless a table says otherwise.
  - *Standing*: the unit's status under `docs/PROTOCOL.md` §3 step 9 (`unit_status`).
  - *Ever*: any of the unit's experiments was VERIFIED.
- **Draws.** `wiki/draws/` holds the random draws the scan stopped on, dead or default-on (protocol 0.8). The first neuron-pole draw was recorded at 2026-10-29T00:16Z. Before then a stopped scan left no file. So draw rates use only units drawn since then: the draws plus the random units first reviewed since then.
- **Intervals and tests.** 95% Wilson intervals. Differences across layers are tested with Pearson's chi-square on VERIFIED against not, with its p from 10,000 permutations of the verdicts over the units.
- **Replications.** A *reading* of a hypothesis is an experiment on the unit with the same explanation and a probe set of its own. A re-measurement (`reuses_probes_of`) is not a reading. Each later reading is compared with the hypothesis's first.

## Verdict rates

| Set | Units | VERIFIED (first) | 95% CI | NEEDS_REVISION | REJECTED | VERIFIED (standing) | VERIFIED (ever) | CONTESTED |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All | 1,088 | 297 (27.3%) | 24.7–30.0% | 444 | 347 | 333 (30.6%) | 353 (32.4%) | 9 |
| GPT-2 small | 323 | 68 (21.1%) | 17.0–25.8% | 151 | 104 | 75 (23.2%) | 76 | 0 |
| … positive poles | 252 | 62 (24.6%) | 19.7–30.3% | 128 | 62 | 70 (27.8%) | 70 | 0 |
| … negative poles | 71 | 6 (8.5%) | 3.9–17.2% | 23 | 42 | 5 (7.0%) | 6 | 0 |
| Qwen2.5-0.5B | 736 | 220 (29.9%) | 26.7–33.3% | 286 | 230 | 249 (33.8%) | 268 | 9 |
| … positive poles | 382 | 103 (27.0%) | 22.8–31.6% | 152 | 127 | 121 (31.7%) | 130 | 4 |
| … negative poles | 354 | 117 (33.1%) | 28.4–38.1% | 134 | 103 | 128 (36.2%) | 138 | 5 |
| Qwen2.5-1.5B | 29 | 9 (31.0%) | 17.3–49.2% | 7 | 13 | 9 (31.0%) | 9 | 0 |

About a quarter of random audited poles are VERIFIED on their first reading. Follow-ups raise that to 30.6% standing and 32.4% ever. The 56 units VERIFIED only after their first reading are what the *Account for retries* item asks about. 89% of first readings (973 of 1,088) were self-reviews: the proposer's model reviewed its own hypothesis.

These shares are not a property of the models alone. They depend strongly on the agents and the scan corpus (below), so they compare like with like only within one agent pair and one corpus.

## Scan stops

Random draws since 2026-10-29T00:16Z, by model and pole:

| Model | Pole | Drawn | Audited | Default-on | Dead | Stopped |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-2 small | negative | 158 | 27 | 131 | 0 | 82.9% |
| GPT-2 small | positive | 164 | 164 | 0 | 0 | 0% |
| Qwen2.5-0.5B | both | 175 | 175 | 0 | 0 | 0% |
| Qwen2.5-1.5B | both | 29 | 29 | 0 | 0 | 0% |

- **41% of GPT-2 draws stop at the scan** (131 of 322), all of them negative poles that are default-on. That agrees with the 87% of GPT-2 small's negative poles that `position-zero.md` measured as default-on. No Qwen draw stopped.
- **With the stopped draws counted, 11.5% of GPT-2 small's random draws are VERIFIED** on first reading (37 of 322), against 19.4% of the 191 audited. A registry-wide rate without the draws overstates GPT-2's share of poles with a verified explanation by about two thirds.
- **No neuron pole was dead** in 526 draws. The share dead before the draws were recorded can't be known. Among dictionary latents, 21 of 61 random draws were dead.
- The pole of a random draw is random unless `--pole positive` is passed, as the learned-basis baseline did for GPT-2. Records before issue #3205 don't say whether it was passed, so the split between poles is not exactly uniform. Since then, `provenance.pole_forced` says it.

## By layer

GPT-2 small, first reading (p = 0.42 for no difference between layers):

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Units | 21 | 37 | 22 | 26 | 25 | 32 | 25 | 28 | 19 | 32 | 21 | 35 |
| VERIFIED | 2 | 6 | 2 | 8 | 8 | 6 | 7 | 8 | 6 | 5 | 4 | 6 |
| Share | 10% | 16% | 9% | 31% | 32% | 19% | 28% | 29% | 32% | 16% | 19% | 17% |

Qwen2.5-0.5B, first reading (p = 0.21):

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Units | 30 | 31 | 33 | 27 | 35 | 30 | 29 | 16 | 25 | 34 | 29 | 25 |
| VERIFIED | 13 | 8 | 9 | 10 | 16 | 9 | 9 | 6 | 6 | 11 | 9 | 4 |
| Share | 43% | 26% | 27% | 37% | 46% | 30% | 31% | 38% | 24% | 32% | 31% | 16% |

| Layer | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Units | 29 | 43 | 35 | 33 | 33 | 39 | 30 | 29 | 29 | 41 | 27 | 24 |
| VERIFIED | 11 | 11 | 8 | 6 | 11 | 12 | 8 | 5 | 6 | 11 | 15 | 6 |
| Share | 38% | 26% | 23% | 18% | 33% | 31% | 27% | 17% | 21% | 27% | 56% | 25% |

By quarter of depth, all three models pooled: 28.7%, 27.9%, 27.8% and 24.8% (p = 0.76).

No layer effect is detectable at about 30 units a layer. GPT-2's layers 0–2 have the lowest shares (10 of 80, 12.5%). Many of their positive poles have a first-token floor (`position-zero.md`), which may make them harder to explain, but that wasn't tested here. Qwen2.5-1.5B has 29 units over 19 layers, too few to split. The per-layer rows with standing and intervals are in the JSONL.

## Replications

146 later readings of a hypothesis on a random unit, each compared with the hypothesis's first reading:

| First reading → later reading | VERIFIED | NEEDS_REVISION | REJECTED |
| --- | --- | --- | --- |
| VERIFIED (57) | 39 | 16 | 2 |
| NEEDS_REVISION (47) | 19 | 18 | 10 |
| REJECTED (42) | 2 | 15 | 25 |

- **The status agrees in 56% of pairs** (82 of 146). VERIFIED or not agrees in 73% (107), Cohen's κ = 0.44.
- **A VERIFIED hypothesis is VERIFIED again 68% of the time** (39 of 57). One that was not VERIFIED becomes VERIFIED 24% of the time (21 of 89), nearly always from NEEDS_REVISION.
- **Most later readings are by another reviewer model** (129). Those agree in 54% of pairs, against 71% for the 17 by the same model (12 of 17). The same-model pairs are too few to measure run-to-run noise, which the *Measure verdict stability* item does.
- The later readings weren't drawn at random. The planner asks for `replicate` after VERIFIED and for `independent_review` after a self-review. So these are agreement rates given a verdict, not one registry-wide agreement rate.

The 56% agrees with `prompt-trim-ab.md`, where one prompt rerun on the same units agreed about half the time. A single reading is a noisy measurement of a hypothesis, and a registry share VERIFIED inherits that noise.

## What the rates depend on

**The proposer.** First-reading VERIFIED by proposer model (each mostly reviewed by itself, Opus by Sonnet):

| Proposer | GPT-2 small | Qwen2.5-0.5B | Qwen2.5-1.5B |
| --- | --- | --- | --- |
| Claude Opus 5.5 | 24 / 54 (44%) | 49 / 64 (77%) | — |
| deepseek-flash | 30 / 125 (24%) | 138 / 388 (36%) | 7 / 14 |
| muse-spark-1.3 | 14 / 144 (10%) | 23 / 254 (9%) | 2 / 15 |
| gemini-3.8-flash | — | 7 / 17 | — |
| glm-5.3-flash | — | 3 / 9 | — |
| Claude Haiku 5.5 | — | 0 / 4 | — |

The proposer–reviewer pair moves the rate by a factor of 4–8. The reviewer crossover (`reviewer-crossover.md`) found that most of this is the submission, and that Sonnet is the most lenient reviewer. The pairs ran on different models in different proportions: muse-spark proposed 45% of GPT-2's units and 35% of Qwen2.5-0.5B's. So comparisons between models must be made within one pair. Within Opus → Sonnet and within deepseek-flash, Qwen2.5-0.5B's rate is higher than GPT-2's. Within muse-spark they are the same.

**The scan corpus.** 444 units were scanned on the 552-token built-in sample, not a real corpus:

| Model | Built-in sample | Real corpus |
| --- | --- | --- |
| GPT-2 small | 22 / 155 (14%) | 46 / 168 (27%) |
| Qwen2.5-0.5B | 28 / 289 (10%) | 192 / 447 (43%) |

Part of this is the proposer mix, because muse-spark ran most built-in scans. Within one proposer the effect holds for Qwen2.5-0.5B: deepseek-flash 5 / 41 (12%) against 133 / 347 (38%), muse-spark 18 / 230 (8%) against 5 / 24 (21%). It doesn't hold for GPT-2 small: deepseek-flash 10 / 43 (23%) against 20 / 82 (24%), muse-spark 12 / 112 (11%) against 2 / 32 (6%). The built-in scans also ran earlier, under older protocols. The planner already draws `rescan` for these units.

**The protocol.** 396 of the 1,088 first readings were judged under the current protocol (0.8). The rest were judged under 0.2–0.7. A re-evaluation is a later experiment, so it doesn't change a first reading.

## Consequences for the other items

- *Measure verdict stability*: same-reviewer replications in the registry are too few (17) to estimate run-to-run noise. The planned reruns are needed.
- *Account for retries*: 56 of 353 ever-VERIFIED units (16%) needed more than one experiment.
- *Check whether VERIFIED predicts held-out detection*: compare within one proposer–reviewer pair and one corpus, or adjust for them. Both move the share VERIFIED more than the model does.
- Any headline "share of neurons with a verified explanation" must name its denominator. For GPT-2 small, with stopped draws counted, it is about 60% of the audited share.

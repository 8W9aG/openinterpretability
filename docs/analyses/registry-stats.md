# Registry-wide statistics over random neuron poles

This analysis gives the registry's verdict rates over units chosen at random. The random draws that the scan stopped on count as outcomes. It is the evidence for research question 1, *Are native MLP neurons clean?* (#3186, `docs/research-questions.md`). The registry was read on 2026-11-14, at 2,617 experiments. The first reading was on 2026-11-02, at 1,594 experiments. Where a number changed in a way that matters, the text says so.

The rows are in `registry-stats.jsonl`, written by `scripts/registry_stats.py` from the parquet that `open-interpretability build` exports. Reported, not gated.

## Method

- **Units.** MLP neuron poles whose first reviewed experiment has `selection = "random"`. A follow-up on a unit records `manual`, so the first experiment decides. Sister studies test a borrowed explanation and are left out. That gives 1,711 units with 2,290 experiments: 758 GPT-2 small, 794 Qwen2.5-0.5B and 159 Qwen2.5-1.5B. Heads and dictionary latents (the learned-basis baseline, `transcoder-baseline.md`) are not pooled with them.
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
| All | 1,711 | 430 (25.1%) | 23.1–27.2% | 780 | 501 | 477 (27.9%) | 520 (30.4%) | 16 |
| GPT-2 small | 758 | 160 (21.1%) | 18.4–24.2% | 385 | 213 | 168 (22.2%) | 183 | 5 |
| … positive poles | 645 | 153 (23.7%) | 20.6–27.2% | 354 | 138 | 166 (25.7%) | 176 | 5 |
| … negative poles | 113 | 7 (6.2%) | 3.0–12.2% | 31 | 75 | 2 (1.8%) | 7 | 0 |
| Qwen2.5-0.5B | 794 | 240 (30.2%) | 27.1–33.5% | 317 | 237 | 277 (34.9%) | 303 | 11 |
| … positive poles | 405 | 112 (27.7%) | 23.5–32.2% | 164 | 129 | 136 (33.6%) | 147 | 4 |
| … negative poles | 389 | 128 (32.9%) | 28.4–37.7% | 153 | 108 | 141 (36.2%) | 156 | 7 |
| Qwen2.5-1.5B | 159 | 30 (18.9%) | 13.5–25.7% | 78 | 51 | 32 (20.1%) | 34 | 0 |
| … positive poles | 85 | 15 (17.6%) | 11.0–27.1% | 42 | 28 | 16 (18.8%) | 18 | 0 |
| … negative poles | 74 | 15 (20.3%) | 12.7–30.8% | 36 | 23 | 16 (21.6%) | 16 | 0 |

About a quarter of random audited poles are VERIFIED on their first reading. Follow-ups raise that to 27.9% standing and 30.4% ever. 90 units were VERIFIED only after their first reading. 93% of first readings (1,596 of 1,711) were self-reviews: the proposer's model reviewed its own hypothesis.

- **The shares fell since 2026-11-02**, from 27.3% to 25.1% first reading and from 30.6% to 27.9% standing. The new units are mostly GPT-2 small and Qwen2.5-1.5B, which have lower shares. Half of them came from muse-spark, the proposer with the lowest share (below). Qwen2.5-0.5B's own share barely moved (29.9% to 30.2%).
- **Qwen2.5-1.5B has a lower share than Qwen2.5-0.5B**: 18.9% against 30.2% on first reading. On 2026-11-02 it had 29 units, too few to tell them apart. Its proposers were half muse-spark and half deepseek-flash, so part of this is the proposer mix.
- **GPT-2 small's negative poles hardly ever stand**: 2 of 113 are VERIFIED standing, fewer than the 7 VERIFIED on first reading. After follow-ups, 5 of the 7 no longer stand.

These shares are not a property of the models alone. They depend strongly on the agents and the scan corpus (below), so they compare like with like only within one agent pair and one corpus.

## Scan stops

Random draws since 2026-10-29T00:16Z, by model and pole:

| Model | Pole | Drawn | Audited | Default-on | Dead | Stopped |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-2 small | negative | 504 | 69 | 435 | 0 | 86.3% |
| GPT-2 small | positive | 559 | 557 | 0 | 2 | 0.4% |
| Qwen2.5-0.5B | both | 233 | 233 | 0 | 0 | 0% |
| Qwen2.5-1.5B | both | 159 | 159 | 0 | 0 | 0% |

- **41% of GPT-2 draws stop at the scan** (437 of 1,063). All but 2 of them are negative poles that are default-on. That agrees with the 87% of GPT-2 small's negative poles that `position-zero.md` measured as default-on. No Qwen draw stopped.
- **With the stopped draws counted, 12.1% of GPT-2 small's random draws are VERIFIED** on first reading (129 of 1,063), against 20.6% of the 626 audited. A registry-wide rate without the draws overstates GPT-2's share of poles with a verified explanation by about two thirds.
- **2 neuron poles were dead** in 1,455 draws, both GPT-2 positive poles. The share dead before the draws were recorded can't be known. 21 dictionary-latent draws were dead.
- The pole of a random draw is random unless `--pole positive` is passed, as the learned-basis baseline did for GPT-2. Records before issue #3205 don't say whether it was passed, so the split between poles is not exactly uniform. Since then, `provenance.pole_forced` says it.

## By layer

GPT-2 small, first reading (p = 0.075 for no difference between layers):

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Units | 44 | 80 | 65 | 61 | 55 | 69 | 62 | 70 | 46 | 84 | 54 | 68 |
| VERIFIED | 8 | 13 | 9 | 13 | 17 | 16 | 21 | 19 | 9 | 18 | 7 | 10 |
| Share | 18% | 16% | 14% | 21% | 31% | 23% | 34% | 27% | 20% | 21% | 13% | 15% |

Qwen2.5-0.5B, first reading (p = 0.23):

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Units | 31 | 33 | 35 | 30 | 38 | 32 | 31 | 20 | 26 | 35 | 30 | 28 |
| VERIFIED | 14 | 10 | 10 | 10 | 19 | 9 | 10 | 7 | 7 | 11 | 9 | 4 |
| Share | 45% | 30% | 29% | 33% | 50% | 28% | 32% | 35% | 27% | 31% | 30% | 14% |

| Layer | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Units | 34 | 47 | 37 | 36 | 36 | 39 | 31 | 32 | 31 | 47 | 29 | 26 |
| VERIFIED | 12 | 11 | 9 | 8 | 12 | 12 | 9 | 6 | 7 | 12 | 15 | 7 |
| Share | 35% | 23% | 24% | 22% | 33% | 31% | 29% | 19% | 23% | 26% | 52% | 27% |

Qwen2.5-1.5B has 159 units over 28 layers, 2–11 a layer, too few to show (p = 0.45). By quarter of depth, all three models pooled: 26.1%, 25.5%, 26.6% and 22.3% (p = 0.46).

No layer effect is detectable at 30–80 units a layer. GPT-2 small comes closest (p = 0.075). Its middle layers 4–7 have the highest share (73 of 256, 29%). Its first three and last two layers have the lowest (47 of 311, 15%). Many of its early positive poles have a first-token floor (`position-zero.md`), which may make them harder to explain, but that wasn't tested here. The per-layer rows with standing and intervals are in the JSONL.

## Replications

231 later readings of a hypothesis on a random unit, each compared with the hypothesis's first reading:

| First reading → later reading | VERIFIED | NEEDS_REVISION | REJECTED |
| --- | --- | --- | --- |
| VERIFIED (86) | 51 | 31 | 4 |
| NEEDS_REVISION (87) | 30 | 46 | 11 |
| REJECTED (58) | 3 | 19 | 36 |

- **The status agrees in 58% of pairs** (133 of 231). VERIFIED or not agrees in 71% (163), Cohen's κ = 0.37.
- **A VERIFIED hypothesis is VERIFIED again 59% of the time** (51 of 86), down from 68% on 2026-11-02. One that was not VERIFIED becomes VERIFIED 23% of the time (33 of 145), nearly always from NEEDS_REVISION.
- **Most later readings are by another reviewer model** (187). Those agree in 54% of pairs, against 73% for the 44 by the same model (32 of 44). Same-model pairs mix run-to-run noise with the planner's choice of which units to re-read, so the *Measure verdict stability* item measures that noise instead (`verdict-stability.md`).
- The later readings weren't drawn at random. The planner asks for `replicate` after VERIFIED and for `independent_review` after a self-review. So these are agreement rates given a verdict, not one registry-wide agreement rate.

The 58% agrees with `prompt-trim-ab.md`, where one prompt rerun on the same units agreed about half the time. A single reading is a noisy measurement of a hypothesis, and a registry share VERIFIED inherits that noise.

## What the rates depend on

**The proposer.** First-reading VERIFIED by proposer model (each mostly reviewed by itself, Opus by Sonnet):

| Proposer | GPT-2 small | Qwen2.5-0.5B | Qwen2.5-1.5B |
| --- | --- | --- | --- |
| Claude Opus 5.5 | 24 / 54 (44%) | 49 / 64 (77%) | — |
| deepseek-flash | 90 / 312 (29%) | 149 / 417 (36%) | 23 / 76 (30%) |
| muse-spark-1.3 | 46 / 392 (12%) | 32 / 283 (11%) | 7 / 83 (8%) |
| gemini-3.8-flash | — | 7 / 17 | — |
| glm-5.3-flash | — | 3 / 9 | — |
| Claude Haiku 5.5 | — | 0 / 4 | — |

The proposer–reviewer pair moves the rate by a factor of 3–7. The reviewer crossover (`reviewer-crossover.md`) found that most of this is the submission, and that Sonnet is the most lenient reviewer. The pairs ran on different models in different proportions: muse-spark proposed 52% of GPT-2's units, 36% of Qwen2.5-0.5B's and 52% of Qwen2.5-1.5B's. So comparisons between models must be made within one pair. Within deepseek-flash, Qwen2.5-0.5B's rate (36%) is higher than GPT-2's (29%) and Qwen2.5-1.5B's (30%). Within Opus → Sonnet, Qwen2.5-0.5B's is higher than GPT-2's. Within muse-spark the three are about the same.

**The scan corpus.** 444 units were scanned on the 552-token built-in sample, not a real corpus. None of the units added since 2026-11-02 were:

| Model | Built-in sample | Real corpus |
| --- | --- | --- |
| GPT-2 small | 22 / 155 (14%) | 138 / 603 (23%) |
| Qwen2.5-0.5B | 28 / 289 (10%) | 212 / 505 (42%) |

Part of this is the proposer mix, because muse-spark ran most built-in scans. Within one proposer the effect holds for Qwen2.5-0.5B: deepseek-flash 5 / 41 (12%) against 144 / 376 (38%), muse-spark 18 / 230 (8%) against 14 / 53 (26%). It is small or absent for GPT-2 small: deepseek-flash 10 / 43 (23%) against 80 / 269 (30%), muse-spark 12 / 112 (11%) against 34 / 280 (12%). The built-in scans also ran earlier, under older protocols. The planner already draws `rescan` for these units.

**The protocol.** 1,019 of the 1,711 first readings were judged under the current protocol (0.8). The rest were judged under 0.2–0.7, 616 of them under 0.7. A re-evaluation is a later experiment, so it doesn't change a first reading.

## Consequences for the other items

- *Measure verdict stability*: the registry's same-reviewer replications (44) are not a random sample, so they can't replace the planned reruns.
- *Account for retries*: 90 of 520 ever-VERIFIED units (17%) needed more than one experiment.
- *Check whether VERIFIED predicts held-out detection*: compare within one proposer–reviewer pair and one corpus, or adjust for them. Both move the share VERIFIED more than the model does.
- Any headline "share of neurons with a verified explanation" must name its denominator. For GPT-2 small, with stopped draws counted, it is about 60% of the audited share.

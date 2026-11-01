# Reviewer-effect crossover: how much of a verdict is the reviewer

This analysis does the *Reviewer-effect crossover* item of `TODO.md` (*Protocol and science*). Verdict rates in the registry differ a lot by agent pair. Opus proposing with Sonnet reviewing verified 135 of 203 records, while deepseek-flash reviewing itself verified 221 of 597 and muse-spark-1.3 reviewing itself 46 of 438. In the registry each proposer mostly has one reviewer, so these rates can't say whether the hypothesis or the reviewer decides the verdict. Here a fixed set of submissions is reviewed again from scratch by three reviewer models on the same prompt, so the two can be separated.

The rows are in `reviewer-crossover.jsonl` (one per submission, with every arm's result), written by `scripts/reviewer_crossover.py`. Reported, not gated.

## Method

- **Submissions.** 30 registry records judged under protocol 0.8, all randomly selected native MLP neuron poles (`selection = "random"`). 10 come from each of the registry's three main proposers: deepseek-flash, muse-spark-1.3 and Claude Opus 5.5. They are drawn with `random.Random(0)`, with one record per unit. 16 are GPT-2 small and 14 Qwen2.5-0.5B. Each submission is reused exactly as recorded: its hypothesis, counterfactuals, scan and confound profile.
- **Reviewer arms.** Each submission is reviewed three times, by `neuron-reviewer` subagents running Claude Sonnet 5.5, Opus 5.5 and Haiku 5.5. Each gets the reviewer prompt `openinterp review --emit-prompt` writes for the submission, with 20 probes. The three prompts are identical apart from the answer path, and they don't name the proposer. Every agent read only its own prompt and the `interpretability-illusions-review` skill, and every agent reported the model ID it was meant to run. Two agents wrote 21 probes, one Sonnet and one Haiku. Each was asked to drop one, before any measurement.
- **The original arm.** The record's own probe set, written by its original reviewer (deepseek-flash, muse-spark-1.3 or Sonnet), re-measured by the same code. That reviewer saw an older prompt and is not blind to the proposer, so this arm is a reference, not a fourth arm of the crossover.
- **Verdict.** Every probe set is measured on its unit under the current policy (0.8) by `review_probes`, the code the null calibration uses: probes, counterfactuals, the lookahead check and the confound baseline, without causal metrics or wiring, which aren't gated. No token check was run in any arm, so a mislabelled probe counts against the hypothesis, as it would in a review without one.
- **Analysis.** VERIFIED counts per proposer and arm; pairwise agreement between arms; and a two-way ANOVA (submission × reviewer, one review per cell) over the three new arms, for balanced accuracy and for VERIFIED as 0/1. P-values are from 10,000 permutations of the arms within each submission.

## Verdicts

VERIFIED, by proposer (rows) and reviewer (columns), out of 10 submissions per proposer:

| Proposer | Original reviewer | Sonnet | Opus | Haiku |
| --- | --- | --- | --- | --- |
| deepseek-flash | 4 | 6 | 4 | 2 |
| muse-spark-1.3 | 2 | 0 | 0 | 0 |
| Claude Opus 5.5 | 6 | 7 | 3 | 3 |
| **All 30** | 12 | 13 | 7 | 5 |

Mean balanced accuracy:

| Proposer | Original reviewer | Sonnet | Opus | Haiku |
| --- | --- | --- | --- | --- |
| deepseek-flash | 0.68 | 0.84 | 0.79 | 0.70 |
| muse-spark-1.3 | 0.60 | 0.66 | 0.63 | 0.63 |
| Claude Opus 5.5 | 0.87 | 0.89 | 0.78 | 0.73 |

### The submission decides most of the verdict

| Outcome | Submission | Reviewer | Residual | Reviewer F (p) |
| --- | --- | --- | --- | --- |
| Balanced accuracy | 71% | 8% | 20% | 12.3 (< 0.001) |
| VERIFIED (0/1) | 59% | 6% | 34% | 5.4 (0.005) |

The table splits the variance across the 90 new reviews. The submission accounts for most of it: 71% of the variance in balanced accuracy and 59% of the variance in VERIFIED. The reviewer accounts for 6–8%, a small effect but a real one (permutation p ≤ 0.005; Cochran's Q on VERIFIED, 9.45, p = 0.005). The residual mixes run-to-run noise with submission × reviewer interaction, and with one review per cell the two can't be separated. It is about three times the reviewer's share for balanced accuracy and five times for VERIFIED.

About a quarter of the submission share (19% of the total) is which model proposed (permutation p = 0.016). Of the 30 reviews of muse-spark-1.3's hypotheses, none was VERIFIED. The two muse-spark records the registry has as VERIFIED, both reviewed by muse-spark itself (Qwen L17/N1022− and L20/N4066−), were VERIFIED by no new reviewer: their balanced accuracies range from 0.38 to 0.64.

### Sonnet is the lenient reviewer, and it inflates the Opus → Sonnet rate

Sonnet verified 13 of 30, Opus 7 and Haiku 5. Sonnet's median balanced accuracy is 0.85, against 0.71 for Opus and 0.66 for Haiku. On average Sonnet scores a submission 0.06 higher than Opus and 0.11 higher than Haiku.

| Pair | Only the first VERIFIED | Only the second VERIFIED | Sign test p |
| --- | --- | --- | --- |
| Sonnet vs Opus | 7 | 1 | 0.07 |
| Sonnet vs Haiku | 8 | 0 | 0.008 |
| Opus vs Haiku | 4 | 2 | 0.69 |
| Original vs Sonnet | 3 | 4 | 1.0 |
| Original vs Opus | 6 | 1 | 0.13 |
| Original vs Haiku | 7 | 0 | 0.016 |

Every disagreement between Sonnet and Haiku goes the same way. For the registry's largest pair this matters: Opus's ten hypotheses were VERIFIED 7 times by Sonnet but 3 times each by Opus and Haiku. So with a stricter Claude reviewer, the registry's 66% for Opus → Sonnet would be about 30%. Haiku and Opus fail more hypotheses on sensitivity (15 and 13 failures, against 7 for Sonnet): their positive holdouts fire less often than Sonnet's.

Haiku is the strictest reviewer here, which doesn't make it the best one. Its holdouts may simply be worse examples of the claimed feature. In a verdict, a strict reviewer and a careless one look alike; only reading the probes against the hypotheses would tell them apart.

### Agreement between reviewers is moderate

| Arms | Same verdict | Same VERIFIED call | κ (VERIFIED) | Mean \|Δ balanced accuracy\| |
| --- | --- | --- | --- | --- |
| Sonnet vs Opus | 17/30 | 22/30 | 0.43 | 0.099 |
| Sonnet vs Haiku | 13/30 | 22/30 | 0.41 | 0.146 |
| Opus vs Haiku | 17/30 | 24/30 | 0.38 | 0.105 |
| Original vs Sonnet | 17/30 | 23/30 | 0.52 | 0.143 |
| Original vs Opus | 17/30 | 23/30 | 0.48 | 0.114 |
| Original vs Haiku | 19/30 | 23/30 | 0.46 | 0.145 |

Two reviewers make the same VERIFIED call on about three submissions in four (κ 0.38–0.52), and give the same three-way verdict on about half. Of the 30 submissions, 16 were VERIFIED by none of the three new reviewers, 3 by all three, and 11 by one or two. This is close to the run-to-run agreement of `prompt-trim-ab.md`, where rerunning the whole pipeline gave the same verdict on 3 of 8 units. So one review's verdict is a noisy reading of the submission, whichever model writes it. The new reviewers agree with each other no better than with the original reviewers, though those saw an older prompt.

### Which gates fail

| Gate failed | Original reviewer | Sonnet | Opus | Haiku |
| --- | --- | --- | --- | --- |
| `specificity` | 11 | 12 | 13 | 14 |
| `sensitivity` | 9 | 7 | 13 | 15 |
| `beats_confound_baseline` | 12 | 9 | 14 | 14 |
| `counterfactual_reproduction` | 2 | 2 | 2 | 2 |

Specificity fails about equally often under every reviewer: all of them can write hard negatives that fire. The reviewers differ mainly in sensitivity, that is, in whether their holdouts fire, and so in whether the hypothesis beats the confound baseline. `counterfactual_reproduction` is the proposer's own gate and doesn't depend on the reviewer. No other gate failed.

## What it means for the registry

- **Registry rates per agent pair mostly reflect the proposer.** Over a fixed set of hypotheses the submission explains about eight times as much variance as the reviewer, and muse-spark's hypotheses fail under every reviewer. The low rates for muse-spark records are mostly the hypotheses, not muse-spark's reviewing.
- **The reviewer still shifts the rate by about a factor of two.** Sonnet verifies roughly twice as often as Opus and 2.6 times as often as Haiku. Rates should not be compared across reviewer models without saying which reviewer each used, and registry-wide statistics should report the reviewer mix. The rotated-basis null's native arm (18 of 30 VERIFIED, `null-calibration.md`) was reviewed by Sonnet and is likely at the lenient end. Its comparison with the rotated arm is unaffected, since Sonnet reviewed both arms.
- **A single review is noisy.** A third or more of the variance is residual, so the *verdict stability* item (TODO) should split that residual into same-reviewer noise and interaction. Independent reviews by different models would reduce it, as would replications.

## Caveats

- **30 submissions, one review per cell.** The reviewer effect is clear, but the per-proposer rates have wide intervals (10 per cell), and noise can't be separated from interaction.
- **Three Claude reviewers only.** deepseek-flash and muse-spark were not run as new reviewers, so the original arm is the only view of them, and it confounds model with prompt version and with blinding.
- **No token check and no label audit.** The pipeline normally lets the reviewer check its probes against the tokenizer. Here no arm did, so all are compared on their first draft. Some differences between reviewers may be labelling errors, not strictness.
- **Protocol 0.8 records only, drawn from three proposers.** Opus has 20 such records; deepseek-flash and muse-spark have over a hundred each.

## Reproduce

```
uv run python scripts/reviewer_crossover.py prepare      # exchange/crossover/cNN/<arm>/review.prompt.md
# one neuron-reviewer agent per prompt, with model sonnet, opus or haiku, writes review.probes.json
uv run python scripts/reviewer_crossover.py measure --arm original
uv run python scripts/reviewer_crossover.py measure --arm sonnet --agent-model external/claude-sonnet-5-5
uv run python scripts/reviewer_crossover.py measure --arm opus --agent-model external/claude-opus-5-5
uv run python scripts/reviewer_crossover.py measure --arm haiku --agent-model external/claude-haiku-5-5
uv run python scripts/reviewer_crossover.py report --out docs/analyses/reviewer-crossover.jsonl
```

`prepare` takes a few seconds; each `measure` takes about 3–4 minutes on Apple's GPU. Each reviewer agent took 15–70 seconds.

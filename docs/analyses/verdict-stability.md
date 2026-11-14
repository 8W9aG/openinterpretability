# Verdict stability

Run on 2026-11-13 for issue #3196, with `scripts/verdict_stability.py`. The numbers are in `verdict-stability.json`. Every registry-wide claim rests on single verdicts. The question: if the same reviewer reviews the same submission again, with the same prompt, how often does the verdict change? And which part of a verdict's variance is that noise?

## Method

- **Submissions.** The reviewer crossover's 30 (`reviewer-crossover.md`): random protocol-0.8 native neuron poles, 10 each from deepseek-flash, muse-spark-1.3 and Claude Opus 5.5 as proposer.
- **Reviews.** The crossover already had one review of each by Sonnet, Opus and Haiku. Two more `neuron-reviewer` subagents on Claude Sonnet 5.5 reviewed each submission, on the identical prompt that the crossover's Sonnet reviewer got, with 20 probes each. So each submission has three Sonnet reviews that differ only by chance.
- **Measurement.** Every probe set is measured on its unit under protocol 0.8 by the crossover's `review_probes`, so the numbers compare with the crossover's.
- **Variance.** Balanced accuracy is split four ways. *Same-reviewer noise* is the pooled variance among each submission's three Sonnet reviews. The crossover's two-way ANOVA (submission × reviewer, one review per cell) gives the submission and reviewer parts and a residual. The *interaction* is that residual less the same-reviewer noise.

## Results

### Agreement between reviews by one reviewer

| | Pairs that agree | Fleiss' κ | All three agree |
| --- | --- | --- | --- |
| Status (VERIFIED, NEEDS_REVISION, REJECTED) | 67% | 0.47 | 50% of submissions |
| VERIFIED or not | 82% | 0.61 | — |

- **Of 30 submissions, 14 got at least one VERIFIED review. Only 6 got three.** A majority, two of three, verifies 12.
- **Two gates cause the flips.** `specificity` changes between two reviews in 20% of pairs, and `beats_confound_baseline` in 16%. `sensitivity` changes in 4%. The other six gates never change: they depend on the submission, not on the probes.

### Where balanced accuracy's variance comes from

| Part | Share |
| --- | --- |
| Submission | 60% |
| Same-reviewer noise | 21% |
| Reviewer model | 11% |
| Reviewer × submission | 9% |

- **The submission is most of it.** That is the part a verdict should measure.
- **Most of the crossover's unexplained residual is noise.** About 70% of it is a reviewer disagreeing with itself, and 30% is a reviewer model reacting to a particular submission.
- **The noise is twice the reviewer effect.** Which probes one review happens to write moves a verdict more than which model writes them.

## Limits

- **One reviewer model, 30 submissions.** Opus and Haiku may be more or less consistent than Sonnet.
- **Three reviews per submission.** The noise estimate has wide uncertainty, about ±35% of its value.
- **Specificity's flips come with the hard negatives.** A review writes about 13 of 20 probes as hard negatives. One failed negative moves specificity by about 0.08, and the gate is at 0.80, so a single probe can flip it.

## What this means

- **A single verdict is a noisy measurement.** About one review in five would flip VERIFIED-or-not if repeated. Registry rates from single verdicts carry that noise, as `registry-stats.md` found when it compared later readings with the first.
- **Fix the specificity gate's noise first.** Options, in order of cost:
  - more hard negatives per review (for example 20 negatives and 10 holdouts), so one probe moves specificity less.
  - a margin or interval rule on specificity, instead of a hard 0.80 line.
  - a majority over two or three independent reviews before VERIFIED, which costs two or three times as much review.
- **The second and third change the decision rule,** so they would need a `PROTOCOL_VERSION` bump (`docs/PROTOCOL.md` §6). The first changes only how many probes a review asks for. Issue #3509 chose it, measured below.

## 30 probes a review (issue #3509)

Run on 2026-11-13. The same 30 submissions, each reviewed twice more by Sonnet (`claude-sonnet-5-5`, `neuron-reviewer`). Each prompt asked for 30 probes: 20 hard negatives and 10 positive holdouts. Both reviews were measured under today's policy. The rows are in `verdict-stability-30.json`.

    uv run python scripts/verdict_stability.py prepare --probes 30 --arms sonnet30,sonnet30b
    uv run python scripts/verdict_stability.py measure --agent-model external/claude-sonnet-5-5 --arms sonnet30,sonnet30b
    uv run python scripts/verdict_stability.py report --arms sonnet30,sonnet30b --out docs/analyses/verdict-stability-30.json

| Measure | 20 probes (3 reviews) | 30 probes (2 reviews) |
| --- | --- | --- |
| VERIFIED-or-not, agreement over pairs | 0.82 | 0.90 |
| Fleiss' κ on VERIFIED | 0.61 | 0.79 |
| `specificity` flip rate | 20% | 7% |
| `beats_confound_baseline` flip rate | 16% | 3% |
| `sensitivity` flip rate | 4% | 7% |
| Same-reviewer variance of balanced accuracy | 0.0055 | 0.0032 |
| Same-reviewer variance of specificity | 0.0067 | 0.0049 |
| VERIFIED per review, mean | 10.7 | 11.5 |

- **Repeated reviews agree more.** One review in ten flips VERIFIED-or-not, against one in five with 20 probes. The noise in balanced accuracy falls by 42%.
- **The flips left the specificity gate.** It flips in 7% of pairs, against 20%. One failed negative now moves specificity by 0.05, not about 0.08.
- **The rule is no stricter.** About as many reviews end VERIFIED. Mean sensitivity is a little lower (0.73 against 0.79), with 10 holdouts in place of about 7.

**Limits.** 30 submissions and two reviews each, so the gain in agreement has a wide interval. One pair of 20-probe reviews also agreed at 0.90. 14 of the 60 reviews first wrote 31 or 32 probes, all extra hard negatives. Each reviewer was asked to drop its weakest extra negatives, with no activations shown. Two reviews wrote 19 negatives and 11 holdouts, and were kept.

**Adopted.** `openinterp review` asks for 30 probes, split 20 and 10. `ReviewPolicy` is unchanged, so `PROTOCOL_VERSION` stays 0.8. Older records keep their 20-probe reviews and are not re-reviewed.

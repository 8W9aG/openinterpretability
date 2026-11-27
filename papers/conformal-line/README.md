# What error rate does the review's firing line imply? A conformal line for 784 explained neuron poles

The review counts a test prompt as firing a pole when its peak reaches a quarter of the pole's maximum, the same share in every model. We asked what error rate that line implies, and what a line with a stated error rate would change. On held-out corpus text, the fixed line fires a median 23.1% of the time in Pythia-70m and 0.7% in Qwen2.5-1.5B. A split-conformal line set for a 5% error rate fires about 5% of the time in all four models, as its guarantee says. With that line in place, 451 of the 784 explanations would still pass their pass thresholds. The guarantee assumes that new text is like the calibration text, and the second model's test prompts are not. In Qwen2.5-1.5B, 26.2% of the test prompts predicted quiet fire at that line, five times the bound. In Pythia-70m it is 3.1%. A line that is fair to corpus text is therefore not fair to adversarial test prompts, and the gap differs by model.

**Status:** Preprint. Work left: #6551 (choosing the pass thresholds to bound false passes), #6552 (a line from only the windows the explanation doesn't cover)

## Introduction

Following Cammarata (2026), a *pole* is one sign of a neuron's activation. Our adversarial review decides whether an explanation of a pole holds by testing it on test prompts. A second model writes hard negatives, which the explanation predicts leave the pole quiet, and positive test prompts, which it predicts fire the pole. A test prompt fires the pole when its peak strength reaches a quarter of the pole's recorded maximum, which we call the *firing line*. The explanation passes when at least 80% of hard negatives stay quiet and at least 60% of positive test prompts fire. We call such an explanation, and its pole, *passing*.

That line is one share of the maximum for every pole and every model. It states no error rate. A pole whose maximum is far above its usual activity gets a line that ordinary text almost never reaches. A pole whose maximum is close to its bulk gets a line that ordinary text reaches often. So the same 0.80 specificity threshold may be easy for one pole and hard for another.

Conformal prediction sets a threshold from held-out data, with a guarantee on its error rate that needs no model of the data (Angelopoulos & Bates, 2021). This paper gives each passing pole a conformal line. Its contributions are the answers to three questions:

1. **What error rate does the fixed line imply?** On held-out corpus text, the median pole fires on 23.1% of windows in Pythia-70m, 11.1% in GPT-2, 1.6% in Qwen2.5-0.5B and 0.7% in Qwen2.5-1.5B (Table 1). *Limit:* one corpus, and windows rather than test prompts.
2. **Does the conformal line keep its promise, and what verdicts would it change?** At $\alpha = 0.05$ the median held-out coverage is 0.950, and the lowest of the 784 poles is at 0.913 (Figure 1). With that line, 451 of 784 explanations still pass both pass thresholds. 174 fail on specificity and 159 on sensitivity, and none fail on both (Table 2). *Limit:* only the two pass thresholds are read again, because the other tests don't depend on the line.
3. **Does the guarantee hold on the second model's test prompts?** At $\alpha = 0.05$, the test prompts predicted quiet fire at 3.1% in Pythia-70m and 6.2% in GPT-2, near the bound. In Qwen2.5-0.5B the rate is 16.7%, and in Qwen2.5-1.5B 26.2% (Figure 2, Table 3). *Limit:* this compares test prompts with 20-token windows, and test prompts vary in length.

These results do not show that any explanation is wrong. They show that the firing line is a choice, and that its error rate depends on the model and on the text.

The paper bears on research question 2, *How reliable is one verdict?* ([research questions](../../docs/research-questions.md)).

## Related work

Split conformal prediction holds out a calibration set, scores each of its points, and takes the $\lceil (n+1)(1-\alpha) \rceil$-th smallest score as a threshold. If the calibration points and a new point are independent and identically distributed, the new point's score is at most that threshold with probability at least $1 - \alpha$ (Angelopoulos & Bates, 2021, Section 1.1, Theorem 1). The guarantee holds for any score and any distribution, but only for new points from the calibration distribution. The same text names distribution shift as the case where a test point comes from another distribution, and points to weighted conformal methods for it (Angelopoulos & Bates, 2021, Section 7).

In our earlier work, the reliability of a verdict has been studied by repeating reviews, swapping hypotheses and reproducing records (`docs/research-questions.md`, question 2). None of those studies sets the firing line itself. This paper keeps every record's test prompts and changes only the line.

## Method

**Poles.** We took each MLP neuron pole whose newest record has a passing explanation, on 2026-11-24. There are 784: 154 in Pythia-70m, 206 in GPT-2, 350 in Qwen2.5-0.5B and 74 in Qwen2.5-1.5B.

**The record's line.** The review measures each test prompt's peak strength over its tokens. A test prompt fires when its peak is at least `activation_threshold` times `max_activation` (`agents/review_measurements.py`). The record stores every test prompt's peak.

**Windows.** For each model we cut `corpus.jsonl`, the pinned corpus, into windows of 20 tokens. That is about a test prompt's length: the test prompts' median is 14 words. A seeded shuffle (seed 0) splits them into 1,000 calibration windows and 1,000 held-out windows. One forward pass per window reads each pole's peak strength, skipping the first token where the record has a first-token floor. On 25 Pythia-70m poles, 14 of them with a floor, these peaks equal the review's own `harness.units.peak_strength` exactly.

**The conformal line.** The score is a window's peak, and $n = 1000$. The quantile is the $\lceil (n+1)(1-\alpha) \rceil$-th smallest calibration peak, for $\alpha = 0.05$ and $\alpha = 0.01$. The theorem bounds the chance that a new peak is *above* the quantile, but a test prompt fires when its peak *reaches* the line. So we put the line one floating-point step above the quantile, which keeps the bound for the review's rule when peaks tie. Layer-0 poles often tie, because their activation depends on the token alone. The source notes that ties need a tie-breaking step for the upper bound on coverage (Angelopoulos & Bates, 2021, Section 1). This step is our own, for the lower bound under the review's rule.

**Re-reading the verdicts.** With a new line, each stored test prompt passes or fails again, as the review decides: it passes when "reaches the line" matches its prediction. The rates weight near-duplicate test prompts as the review does (`core.stats.effective_counts`), and group test prompts by kind. With the record's own line, this reproduces the stored specificity and sensitivity of all 784 records exactly. A pole *passes its pass thresholds* when both rates reach the record's own thresholds, 0.80 and 0.60.

**Test prompts predicted quiet.** For the guarantee, the relevant test prompts are those predicted quiet: hard negatives, and the positive test prompts of an inverse claim. We count how many of them reach the line, pooled over each model's poles.

## The fixed line on corpus text

![The fixed line fires on 0.7% to 23% of corpus windows, and the conformal line on about 5%](media/fire.svg)

**Figure 1.** The fixed line fires on 0.7% to 23% of corpus windows, and the conformal line on about 5%. The share of 1,000 held-out corpus windows on which each pole fires. Left: the record's own line. Right: the conformal line at $\alpha = 0.05$. Dots above 50% sit on the top edge and are counted above it.

| Model | Poles | Fires at the record's line | Fires at the 5% line | Coverage at the 5% line: median, 5th percentile | Coverage at the 1% line: median, 5th percentile |
| --- | ---: | ---: | ---: | --- | --- |
| Pythia-70m | 154 | 23.1% | 4.7% | 0.953, 0.936 | 0.991, 0.982 |
| GPT-2 | 206 | 11.1% | 5.0% | 0.950, 0.933 | 0.991, 0.982 |
| Qwen2.5-0.5B | 350 | 1.6% | 5.0% | 0.950, 0.934 | 0.991, 0.984 |
| Qwen2.5-1.5B | 74 | 0.7% | 4.9% | 0.951, 0.933 | 0.991, 0.983 |

**Table 1.** The median share of held-out windows on which a pole fires, and the conformal line's coverage. The 5% line is at $\alpha = 0.05$ and the 1% line at $\alpha = 0.01$. Coverage is the share of held-out windows under the line.

The record's line is a share of the pole's maximum, and that maximum comes from the pole's own scan. In Qwen2.5 the maxima sit far above the bulk of activity, so the line is high and fires rarely. In Pythia-70m and GPT-2 the maxima are closer to the bulk, so the same share fires often. Under the record's line, 33 Pythia-70m poles fire on more than half of all windows. The conformal line instead fixes the rate, and its coverage stays within sampling noise of $1 - \alpha$ in every model. With 1,000 held-out windows, the standard error of a coverage near 0.95 is about 0.007.

## Verdicts under a conformal line

| Model | Poles | Median ratio of the 5% line to the record's | Pass at the 5% line | Fail specificity | Fail sensitivity | Pass at the 1% line |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pythia-70m | 154 | 1.90 | 92 | 7 | 55 | 37 |
| GPT-2 | 206 | 1.51 | 131 | 10 | 65 | 67 |
| Qwen2.5-0.5B | 350 | 0.64 | 203 | 112 | 35 | 208 |
| Qwen2.5-1.5B | 74 | 0.32 | 25 | 45 | 4 | 42 |
| All | 784 | 1.02 | 451 | 174 | 159 | 354 |

**Table 2.** The pass thresholds under the conformal line. Every one of the 784 passes both thresholds with the record's own line. No pole fails both thresholds at the 5% line.

The two kinds of failure follow the line's direction. Where the conformal line is higher than the record's, as in Pythia-70m and GPT-2, weak positive test prompts stop firing and sensitivity fails. Where it is lower, as in Qwen2.5, hard negatives start to fire and specificity fails. At $\alpha = 0.01$ every line rises. So more Pythia-70m and GPT-2 explanations fail on sensitivity, and fewer Qwen2.5 ones fail on specificity.

## Test prompts predicted quiet

![Quiet test prompts fire far above the bound in Qwen2.5, and near it in Pythia-70m and GPT-2](media/hard.svg)

**Figure 2.** Quiet test prompts fire far above the bound in Qwen2.5, and near it in Pythia-70m and GPT-2. The share of each model's test prompts predicted quiet that reach the conformal line. The dashed line is $\alpha$, the rate that the guarantee bounds on corpus text.

| Model | Test prompts predicted quiet | Fire at the record's line | Fire at the 5% line | Fire at the 1% line |
| --- | ---: | ---: | ---: | ---: |
| Pythia-70m | 3,030 | 9.4% | 3.1% | 0.9% |
| GPT-2 | 2,887 | 9.4% | 6.2% | 1.6% |
| Qwen2.5-0.5B | 4,469 | 7.4% | 16.7% | 6.7% |
| Qwen2.5-1.5B | 1,244 | 8.9% | 26.2% | 11.7% |
| All | 11,630 | 8.6% | 11.5% | 4.4% |

**Table 3.** Test prompts predicted quiet that fire, pooled over each model's poles.

The guarantee needs new text from the calibration distribution. The second model writes hard negatives to break an explanation, so they are not drawn from the corpus. In Pythia-70m the test prompts predicted quiet fire less often than corpus windows. In Qwen2.5 they fire three to five times as often as $\alpha$.

## Discussion

The fixed line was chosen as a share of the maximum so that one rule fits every pole. These results show that it does not fit every model in the same way. The same rule lets 23.1% of Pythia-70m's corpus windows fire the median pole, and 0.7% of Qwen2.5-1.5B's. A specificity threshold of 0.80 therefore asks less of a Qwen2.5 explanation's hard negatives than the corpus would suggest, and more of a Pythia-70m one.

A conformal line fixes the rate on corpus text, and keeps that promise in every model. It does not fix the rate on the second model's test prompts, which are adversarial by design. We did not test why Qwen2.5's test prompts cross the line so often. One possibility is that the conformal line in Qwen2.5 sits low, in the bulk of the activations, where near-miss text reaches it easily. Either way, no single $\alpha$ gives the same false-fire rate on test prompts in every model.

So a conformal line alone would not make the verdicts comparable across models. It would change which explanations pass, in opposite directions by model, without a guarantee on the test prompts that decide the verdict. A rule with a stated error rate on false passes would need labelled examples of wrong explanations (#6551).

## Limitations

- **One corpus for every pole.** Each record pins its own scan corpus, but all lines here come from `corpus.jsonl`. A pole's line would differ on its own corpus.
- **Every window calibrates, not only the uncovered ones.** The issue asked for the windows that the explanation does not cover. Labelling those needs a judge per window, so this paper uses all windows. Covered windows raise the line, so these lines are, if anything, too high. #6552 tests how much this matters.
- **Windows of one length.** A test prompt's peak is over its own tokens, and test prompts vary in length. A longer text has more chances to cross the line. So the comparison in Table 3 is approximate.
- **Windows are not independent.** Neighbouring windows come from the same document, and the split is by window, not by document. The guarantee assumes independent, identically distributed scores.
- **Only the pass thresholds.** A passing verdict also needs counterfactuals, a lookahead check, an independent review and a confound control. The line doesn't enter those, so a pole that passes its pass thresholds here still passes the rest.

## Conclusion

The review's fixed firing line implies a false-fire rate on corpus text that differs about thirtyfold between models. A split-conformal line removes that difference on corpus text. But it changes 333 of 784 test prompt verdicts at $\alpha = 0.05$, and it gives no guarantee on the second model's adversarial test prompts. Three steps would take this further:

- Choose the pass thresholds to bound the rate of false passes, with Learn then Test or conformal risk control (#6551).
- Calibrate the line only on windows that the explanation does not cover, and see whether the verdicts move (#6552).
- Decide whether the review should adopt a conformal line. That is a change to the decision rule, so it needs its own issue and a `PROTOCOL_VERSION` bump (`docs/PROTOCOL.md` §6).

## References

- Angelopoulos, A. N., & Bates, S. (2021). *A gentle introduction to conformal prediction and distribution-free uncertainty quantification*. Preprint, version 6 of 7 December 2022. [arXiv:2107.07511](https://arxiv.org/abs/2107.07511).
- Cammarata, N. (2026). *Neural networks are already clean*. Goodfire research essay. [already-clean.goodfire.com](https://already-clean.goodfire.com/).

## Appendices

### Reproduce

```bash
uv run python scripts/conformal_line.py --out papers/conformal-line/results.jsonl
uv run python scripts/paper_figures.py
```

The script reads the newest record of each pole from `wiki/` on the day it runs. It ran on an M1 Max in about 15 minutes, with the models already downloaded. The window shuffle uses seed 0.

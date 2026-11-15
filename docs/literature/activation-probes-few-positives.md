# Activation Probes Are Reliable With a Handful of Positive Examples

**Citation.** Riya Tyagi (MIT) and Stefan Heimersheim (FAR.AI). "Activation Probes Are Reliable With a Handful of Positive Examples". arXiv 2610.39766v1, 29 October 2026. [Paper](https://arxiv.org/abs/2610.39766). [Code](https://github.com/Centrattic/scarcity-probes). The arXiv comment says "Presented at the Mechanistic Interpretability workshop at NeurIPS 2025". That year does not agree with the arXiv date, and we did not resolve it.

**Status.** Not applicable. `open-interpretability probe-compare` already trains in the regime that the paper recommends, so nothing changes. The paper also does not address our open problem, which is too few positives in the held-out set.

**Read.** The abstract page and the v1 HTML full text, on 2026-11-13: Sections 1–4, Appendices A.1–A.8, Tables 1–4 and the figure captions. A model read the text and reported it to us, with quotes. The HTML has no tables of results, and we could not read values from the plots. So every number below is one that the text states.

## Summary

**Claim.** A monitor for rare misbehaviour has many negative examples and only a few positive ones. In that case, train the probe on all the negatives, not on a balanced set. Generated positives can help, but only a little.

**Method.** The paper probes one layer per model. It uses these models:

- Gemma-2-9B, after layer 20, for spam (Enron-Spam, with SMS Spam as the out-of-distribution test).
- Llama-3.3-70B-Instruct, after layer 50, for honesty (the MASK benchmark).
- Qwen3 from 0.6B to 14B, for the scaling trend.

It uses four probe types. The first is logistic regression (L2, C = 1, class-weighted loss). The second is a logistic probe on the top 1,024 SAE or transcoder features. The others are a cosine-to-class-centroid probe and an attention probe. Each probe type uses four ways to pool tokens: mean, max, last token and softmax. The training sets have 1–10 or 50–150 positives, against 1,750 negatives (Enron-Spam) or 340 (MASK). The comparison is against balanced training. Each test set has equal classes. The metrics are AUC and recall at a 1% false-positive rate. The authors keep a spam example only when the model's own few-shot answer is right. Gemma and Llama use 10 seeds, and Qwen uses 5. There is no validation set, because the authors say that positives are too rare to hold out.

**Findings, as the text reports them.** The figures are the mean over the 5 probes with the best AUC. The table gives imbalanced training minus balanced training.

| Setting | AUC | Recall at 1% FPR |
| --- | --- | --- |
| Enron-Spam, 1–10 positives | +0.045 | +0.093 |
| MASK, 1–10 positives | +0.055 | +0.034 |
| Enron-Spam, 50–150 positives | +0.000 | +0.015 |
| MASK, 50–150 positives | +0.007 | +0.041 |
| SMS Spam (out of distribution), 1–10 positives | +0.012 | −0.031 |
| SMS Spam (out of distribution), 50–150 positives | −0.011 | −0.059 |

Generated positives (gpt-4o-mini) at 2× add 0.026 AUC and 0.104 recall. The authors say that this is about the same as doubling the real positives. At 10× and 20×, the probes get worse. Larger Qwen3 models need fewer positives, "a promising, though not perfectly monotonic, trend". The text gives no numbers for that trend.

## Where the claims go beyond the evidence

- **The title is wider than the result.** The paper shows that imbalanced training beats balanced training at 1–10 positives. It does not show that such probes are reliable in an absolute sense. The absolute AUCs are only in the plots.
- **The gain does not transfer.** On SMS Spam, the out-of-distribution test, recall at 1% FPR falls with imbalanced training. The authors call that result "inconclusive". A monitor in deployment meets that case.
- **Two concepts, and one of them is easy.** The authors say that spam is "a relatively simple concept". Honesty on MASK is the only harder concept, and it uses one model.
- **The selection of the top 5 probes is not described.** The headline means are over the 5 probes with the best AUC. The text does not say which data gave that AUC. There is no validation set, so the selection may use the test set. That would inflate the means.
- **No intervals or tests.** The means come from 5 or 10 seeds, with no confidence interval and no significance test.
- **The paper does not agree with itself.** Section 3.2 says that 2× upsampling is best, and the Figure 3 caption says 3×. The caption names "13 probe architectures", but Table 1 lists 4 types. Section 2.2 lists Qwen3-32B, but Table 3 does not have it. Appendix A.4 gives upsampling factors 2–5, but the results use 10× and 20×.
- **Workshop paper, version 1.** The venue year in the comment does not agree with the arXiv date.

## Use in this repo

**Our probes already train this way.** `open-interpretability probe-compare` (`openinterp/harness/probing.py`) fits a logistic probe and a mass-mean probe on the MLP hidden layer and on the residual stream. It keeps at most 2,000 positive and 8,000 negative tokens (`--max-positive`, `--max-negative`) and weights the loss by class (`fit_logistic`). That is the paper's recommendation: keep all the negatives and balance them in the loss, not by sampling. So the paper adds no reason to change the defaults.

**Our thin row is thin at test, not at training.** The GPT-2 L0/N1923+ row in `docs/analyses/probe-comparison.jsonl` trained on 27 positives against 8,000 negatives. The paper suggests that a probe trained like this can work. But the row's held-out set has 15 positives, and `docs/analyses/probe-comparison.md` already calls it too thin to estimate any probe. The paper does not help with this. Its test sets have equal classes, and the text does not give their size. With 15 positives, one miss moves recall by 0.067. A score from that row needs an interval, and no number of training negatives gives one.

**The setting is different.** The paper probes sequence-level concepts in models from 0.6B to 70B, and it pools the tokens. Our probes are per token, in Qwen2.5-0.5B and GPT-2 small. They test whether one neuron pole carries a concept, and not whether a monitor catches a rare event. The scaling trend is to larger models, so it does not apply to ours.

**One idea carries over.** The paper reports recall at a 1% false-positive rate next to AUC. For a pole, a similar number is its recall at the protocol's line on held-out tokens. It counts only when that line fires on 1% or less of the negatives. `probe-compare` reports balanced accuracy at the line, which mixes the two error rates. Issue #3537 holds that follow-up, together with intervals for rows with few held-out positives.

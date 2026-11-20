# Can a gate tell a wrong copy from a right one?

Run on 2026-11-20 for issue #5233, with `scripts/copy_gate.py`. The rows are in `copy-gate.jsonl`, and its first row is the metadata. Arms G and I of the fix plan (#5220) add a module that stays silent unless a gate opens. Its interference is then the gate's false opens, and its reach is the gate's recall. So before any module, this page asks whether a usable gate exists, on the copy family of #5232.

**Answer.** Only late. After the last layer, a linear probe that also learned from corpus text catches 58% of held-out failures. It opens on 5 of 97 held-out right copies and on none of 18,000 corpus tokens. At the layers where a module would sit inside the network, 2 and 3, the same kind of gate catches 12% to 21%. A gate read from the copy mechanism does worse than the text probes from layer 2 on.

## Method

- **Data.** The copy family of #5232 on pythia-70m, in float32 ([`copy-family.md`](copy-family.md)). The gate must open on a wrong copy and stay shut on a right one.
  - **Fit:** train's 155 wrong copies against retain's 97 right copies.
  - **Test:** held-out's 700 wrong copies against its 97 right copies. Nothing is chosen on held-out.
  - **Corpus:** 20,000 tokens of `corpus.jsonl`. The first 2,000 may serve as right answers in fitting, and the other 18,000 measure false opens on ordinary text.
- **Text-keyed gates.** After each layer, at the answer position, three probes on the residual stream:
  - **diff_means:** the difference-in-means direction of #5074.
  - **logistic:** an L2-regularised logistic regression, with the two groups weighted equally.
  - **logistic_corpus:** the same, with the 2,000 fitting corpus tokens added as right answers.
- **Mechanism-keyed gate.** A logistic regression on four numbers:
  - the attention of the copy heads L3/H3, L4/H0 and L4/H7 from the answer position to the first clause's unit.
  - the cosine of the two subjects' last-token residual states after layer 2, a subject-mismatch signal.

  In corpus text there are no subjects or units to find. So its corpus false opens use a proxy: the attention-only gate, read as each copy head's strongest attention to an earlier token.
- **Thresholds.** Each threshold comes from out-of-fold scores on the fitted data (5 folds), at 5% false opens on the fitted right copies. For logistic_corpus, also at 5% on the fitting corpus tokens. Fitting and threshold both use only fitted data.
- **Measures.** Held-out AUROC with a 95% bootstrap interval (1,000 resamples). The null is the held-out AUROC of 200 seeded random directions, each signed on the fitted data, and the table gives its 95th percentile. Also the held-out ROC's recall at 5% false opens: a reading of the curve, not a threshold a gate could set. Then the operating point: held-out recall, held-out right-copy false opens and corpus false opens, at the fitted threshold.

## Results

**Table 1.** The text-keyed gates on held-out prompts. "Recall at 5%" reads the held-out ROC. The last three columns use the threshold set on fitted data.

| Layer | Probe | AUROC (95% interval) | Random 95th percentile | Recall at 5% | Recall | Right-copy false opens | Corpus false opens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | diff_means | 0.73 (0.67–0.78) | 0.68 | 0.16 | 0.14 | 0.031 | 0.598 |
| 2 | logistic | 0.92 (0.88–0.95) | 0.68 | 0.66 | 0.45 | 0.021 | 0.518 |
| 2 | logistic_corpus | 0.82 (0.77–0.86) | 0.68 | 0.19 | 0.12 | 0.010 | 0.000 |
| 3 | diff_means | 0.76 (0.70–0.81) | 0.69 | 0.16 | 0.06 | 0.031 | 0.300 |
| 3 | logistic | 0.92 (0.89–0.96) | 0.69 | 0.72 | 0.60 | 0.031 | 0.760 |
| 3 | logistic_corpus | 0.81 (0.76–0.86) | 0.69 | 0.26 | 0.21 | 0.031 | 0.000 |
| 4 | logistic | 0.95 (0.92–0.98) | 0.70 | 0.87 | 0.82 | 0.041 | 0.205 |
| 4 | logistic_corpus | 0.87 (0.82–0.91) | 0.70 | 0.53 | 0.46 | 0.041 | 0.000 |
| 5 | logistic | 0.98 (0.96–0.99) | 0.71 | 0.93 | 0.95 | 0.052 | 0.078 |
| 5 | logistic_corpus | 0.91 (0.87–0.94) | 0.71 | 0.54 | 0.58 | 0.052 | 0.000 |

Layers 0 and 1, and diff_means at layers 4 and 5, are in `copy-gate.jsonl`. None of them does better than the rows above.

**Table 2.** The mechanism-keyed gate on held-out prompts.

| Gate | AUROC (95% interval) | Recall at 5% | Recall | Right-copy false opens | Corpus false opens |
| --- | --- | --- | --- | --- | --- |
| Attention and mismatch | 0.77 (0.73–0.82) | 0.38 | 0.17 | 0.021 | no positions in corpus text |
| Attention only | 0.65 (0.60–0.70) | 0.16 | 0.09 | 0.021 | 0.448 (proxy) |

- **The residual stream holds the difference.** Every logistic probe from layer 1 on beats the random-direction null by a wide margin. The best, after layer 5, has an AUROC of 0.98. The plain difference-in-means direction does much worse, which fits #5074's 0.73.
- **A gate fitted on templates alone opens on ordinary text.** The template-only logistic probes open on 8% to 100% of corpus tokens. They learned what separates the two kinds of template prompt, and ordinary text is neither.
- **Corpus negatives close the gate on ordinary text, at a cost in recall.** With 2,000 corpus tokens in fitting, no probe opens on any of the 18,000 test tokens. Recall falls, most at the early layers.
- **The mechanism gate is weaker.** Its subject-mismatch term carries most of the weight (−1.00, against at most 0.29 for an attention term). The copy heads attend a little more to the unit when the copy is right, as #5074 found for their strength. Its attention-only part opens on 45% of corpus tokens, because copy heads attend strongly to earlier tokens in ordinary text too.

## The operating point for #5220

- **For a module after the last layer** (one that acts on the logits): the logistic_corpus probe after layer 5. It reaches a held-out recall of 0.58 (409 of 700, 95% interval 0.55 to 0.62). It opens on 5 of 97 right copies, 5.2% (interval 2.2% to 11.5%), and on 0 of 18,000 corpus tokens (upper bound 0.02%). That is near the line this issue's plan set, at least 50% recall at no more than 5% false opens. The right-copy rate is one copy above it, with a wide interval.
- **For a module inside the network** (arms G and I place one at a site that the route passes through): no gate here is usable. At layers 2 and 3 the corpus-safe probe catches 12% and 21%. The probe that catches more opens on half or more of ordinary text.
- **So this family supports a late gate, not a mid-network one.** If arms G and I need a gate at layer 2 to 4, #5220's fallback applies: gradient arms with a retain loss. This is evidence for the maintainer's plan, not a change to it.

## Limits

- **The confound of #5232.** A right copy shares its unit with the first clause, and so its category. A probe that finds "same category" would score well without tracking rightness. The mismatch term of the mechanism gate reads exactly that. The frozen-fix hold-out of #5232, with hard negatives whose units match across categories, is the test this page doesn't run.
- **97 right copies.** Each held-out false-open rate rests on 97 prompts, so one prompt moves it by a point.
- **One corpus, 20,000 tokens.** A probe that learned from 2,000 of them may still open on text unlike `corpus.jsonl`.
- **Not scored like a unit.** The issue asks to score the gate's stated meaning with `open-interpretability score` and the adversarial reviewer. That command scores registry units from their records, and a gate isn't one yet (#5462).
- **One model and one template**, pythia-70m.

## Reproduce

```bash
uv run python scripts/copy_gate.py --out docs/analyses/copy-gate.jsonl
```

It reads the family and `corpus.jsonl`, and runs in about five minutes on a CPU. The seed is 5233.

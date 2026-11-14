# Evaluating Neuron Explanations

**Citation.** Tuomas Oikarinen, Ge Yan and Tsui-Wei Weng (UC San Diego). "Evaluating Neuron Explanations: A Unified Framework with Sanity Checks". arXiv 2506.05774v1, 6 June 2025. [Paper](https://arxiv.org/abs/2506.05774). [Code and results](https://github.com/Trustworthy-ML-Lab/Neuron_Eval). ICML 2025. Funded in part by the US National Science Foundation, a Hellman Fellowship, an Intel award and the Nvidia Academic Grant Program.

**Status.** Used. `scripts/score_sanity.py` (#3237, `docs/analyses/score-sanity.md`) runs this paper's two sanity tests on every stored score of `open-interpretability score`.

**Read.** The abstract page and the v1 HTML, on 2026-11-14. A model read the text and reported it to us, with short quotes. All of it came back as the paper's own text, appendices included. We did not see the figures.

## Summary

**Claim.** Many metrics used to evaluate neuron explanations barely move when the explanation's concept labels change a lot. One framework covers them all, and two simple sanity tests show which ones to trust.

**Method.**

- **Framework.** An evaluation is a metric applied to two vectors over a probing set: the unit's activations, and the concept's presence as the explanation defines it. The concept vector can come from human raters, labelled data, a model or an LLM simulator. The framework covers 18 metrics from 19 earlier studies. Among them are recall, precision, F1, IoU, accuracy, balanced accuracy, AUC, AUPRC, Pearson and Spearman correlation, cosine, WPMI and MAD.
- **Missing-labels test.** Each positive concept label is set to 0 with probability 0.5.
- **Extra-labels test.** Zeros are set to 1 at random until the positive labels double.
- **Pass rule.** A metric passes when it falls, by more than ε = 0.001 on a 0–1 scale, for over 90% of units in every test. How much it falls does not matter.
- **Settings.** The tests use ground-truth concept labels in 8 settings. Six are vision: ViT-B-16, ResNet-50 and ResNet-18 on ImageNet, Places365 and CUB. Two are language: the final-layer neurons for the 500 commonest tokens of GPT-2 small and GPT-2 XL, on OpenWebText.
- **Theory.** An ideal unit whose activations equal the concept, at active shares γ from 0.499 down to 0.0001, with 1,000 runs of 500,000 points each.
- **Meta-evaluation.** How well each metric ranks a unit's correct explanation above wrong ones, measured as AUPRC.

**Findings, as the text reports them.**

| Metric | Missing labels | Extra labels | Verdict |
| --- | --- | --- | --- |
| Pearson correlation | passes | passes | passes |
| Cosine | passes | passes | passes, but sensitive to a constant shift in activations |
| AUPRC | passes | passes | passes |
| F1 and IoU | passes | passes | pass |
| Balanced accuracy | passes (98.7%) | fails (53.7%) | fails on rare concepts |
| Recall | passes | fails (0%) | fails |
| Precision | fails (45.7%) | passes | fails |
| AUC | passes | fails | fails on rare concepts |
| Correlation on top-and-random samples | 87.8% | 60.3% | fails |
| Spearman correlation | fails | fails | fails |

The percentages are the share of units whose score falls, averaged over the 8 settings. In theory, balanced accuracy falls by 0.25 under missing labels at every γ. Under extra labels it falls by γ / (2(1 − γ)): 0.498 at γ = 0.499, 0.0056 at 0.1, and nothing measurable at 0.0001. So it fails exactly when the concept is rare. In the meta-evaluation, correlation ranks best (mean rank 1.6), then cosine (2.3) and AUPRC (3.9). Balanced accuracy is mid-table (7.3).

On the GPT-2 settings alone, balanced accuracy falls under extra labels for only 9.7% (small) and 9.5% (XL) of units. Its meta-AUPRC is 0.984 and 0.997, against 0.992 and 0.999 for correlation.

## Where the claims go beyond the evidence

- **Ground-truth labels only.** The sanity tests use dataset labels as the concept vector. They were not run on LLM-simulated or human labels, which is how most explanation scores are made.
- **Final-layer units.** Most settings read final-layer neurons, whose concepts are the model's own classes or tokens. The authors warn that these units "are systematically different from other units".
- **Passing is necessary, not sufficient.** The authors say so: failing shows a metric should not be relied on, but passing does not make it good.
- **No intervals.** The averages across settings have no confidence intervals or tests.
- **Small inconsistencies.** Table 3 marks WPMI and inverse AUPRC as failing, and the text calls them borderline. Section 5 says 10 settings and Table 4 says 8. Appendix E.2's text and table give different numbers for cosine's failure.

## Use in this repo

**Our scores already run its tests.** `scripts/score_sanity.py` (#3237) applies both tests to every stored detection and fuzzing score, with the judge's answers as the concept labels. There, balanced accuracy passed both tests wherever the judge beat chance, and recall, precision and F1 each failed one (`docs/analyses/score-sanity.md`).

**The two results agree because our samples are balanced.** The paper finds that balanced accuracy fails the extra-labels test on rare concepts. Our detection windows are half active by construction (`sample_heldout`), so γ is about 0.5. That is the case where the paper's own theory has balanced accuracy pass. For an ideal unit, it falls by p/2 under either of our tests. So `open-interpretability score` may report balanced accuracy *because* its sample is balanced. A score taken on ordinary text, where a pole is active on a small share of tokens, should use correlation, AUPRC or F1. `docs/analyses/score-sanity.md` now says this.

**F1 behaves the other way on our samples.** The paper's F1 passes both tests, but on our balanced windows F1 rose under extra labels. With half the windows active, a random extra "active" answer hits an active window half the time. So recall gains about as much as precision loses. Both results fit the same arithmetic at different γ.

**Our windows are not top-and-random.** Correlation on top-and-random samples fails both tests here, because oversampling the top makes it "more similar to Recall". `open-interpretability score` takes its active windows from five bins across the activation range, plus as many random inactive ones. So it avoids the top-heavy sample that the paper warns about.

**The null-explanation control covers the case that the tests don't.** The paper's tests perturb labels at random. Our null-explanation control (`docs/analyses/null-explanation-control.md`) replaces the explanation itself with an empty one or another unit's, which is closer to a real wrong explanation.

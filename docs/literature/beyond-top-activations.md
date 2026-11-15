# Beyond Top Activations

**Citation.** Tuomas Oikarinen, Ge Yan, Akshay Kulkarni and Tsui-Wei Weng (UC San Diego). "Beyond Top Activations: Efficient and Reliable Crowdsourced Evaluation of Automated Interpretability". arXiv 2506.07985, v1 9 June 2025, v2 2 December 2025. [Paper](https://arxiv.org/abs/2506.07985). The paper says its code "will be available" at [Trustworthy-ML-Lab/Efficient-Interpretability-Eval](https://github.com/Trustworthy-ML-Lab/Efficient-Interpretability-Eval). We did not check that the repository exists. Preprint, no venue given. The authors' institution judged the crowd study exempt from IRB review.

**Status.** Planned. Issue #3595 brings the paper's importance sampling to the inactive windows of `open-interpretability score`.

**Read.** The abstract page and the v2 HTML, on 2026-11-14. A model read the text in four passes and reported it to us, with short quotes. Every pass came back as the paper's own text, from the introduction to the end of the appendices. We did not see the figures, so values that are only on plots are not here.

## Summary

**Claim.** An evaluation of neuron explanations that shows raters only the top-activating inputs measures recall and nothing else. So it favours explanations that are too broad. An evaluation over the whole input set solves that, but it costs far more. Two methods cut the cost by about 40 times.

**Method.**

- **Units.** 2,048 neurons of ResNet-50 layer 4 and 3,072 MLP neurons of ViT-B-16 layer 11, both trained on ImageNet. The probing set is the ImageNet validation set.
- **Metric.** The Pearson correlation between a neuron's activations and the presence of the explained concept, over the whole probing set. The authors say it "captures both sensitivity and specificity".
- **Model-guided importance sampling (MG-IS).** It draws the inputs that raters see from a mix. 80% are drawn in proportion to |activation × SigLIP concept score|, and 20% uniformly. Weights of p/q keep the estimate unbiased.
- **Bayesian rating aggregation (BRAgg).** It models each rating as wrong with a fixed rate, 23% as measured on Mechanical Turk. It combines the ratings with a prior, uniform or from SigLIP.
- **The study.** Mechanical Turk raters marked which of 15 inputs show the concept. Five explanation methods were rated on 100 neurons per model, with 180 inputs per neuron and 3 raters per input. The methods were Linear Explanations with SigLIP labels, CLIP-Dissect, INVERT, MAIA and DnD.

**Findings, as the text reports them.**

| Setting | Result |
| --- | --- |
| MG-IS, simulation | The same accuracy as uniform sampling with about 13 times fewer inputs (abstract) or about 15 times (body) |
| MG-IS + BRAgg, simulation with 23% noise | 27.5% relative correlation error at 550 ratings per neuron. Uniform sampling with a majority vote needs 22,560 |
| The same, on Mechanical Turk (10 neurons, 600 inputs, 9 raters each) | 19.8% error at 550 ratings per neuron. Uniform sampling doesn't reach it in the tested range |
| Cost | $2.16 per neuron and explanation, $2,160 for the whole study, against an estimated $90,000 without the two methods |
| Crowd study, mean correlation (BRAgg with a SigLIP prior) | LE(SigLIP) 0.21, CLIP-Dissect 0.15, INVERT 0.13, MAIA 0.11, DnD 0.10 |
| Rater agreement, Fleiss' κ | 0.19 (ViT) and 0.27 (ResNet-50) in the study. Three of the authors reached 0.77 |

The authors explain why the two LLM-based methods (MAIA, DnD) score low: they look at high activations, and their descriptions are inconsistent.

## Where the claims go beyond the evidence

- **Vision only.** Both models are ImageNet classifiers, and both layers are late. The authors say other methods might do better on early layers, other datasets or SAE latents. Nothing here is about language-model neurons.
- **The winner shares the sampler's model.** LE(SigLIP) is the best method, and SigLIP also drives MG-IS and the BRAgg prior. The paper varies the sampler and the prior, but not the explanation model. So it can't rule out that the evaluation favours LE(SigLIP).
- **Low rater agreement.** κ of 0.19–0.27 means that the raters often disagree. BRAgg's single error rate treats all disagreement as noise, not as a real ambiguity in the concept.
- **Small details that don't match.** The abstract says 13 times and the body 15 times. Three passages name different SigLIP checkpoints for the same role.
- **The code is promised, not released.**

## Use in this repo

**Our detection already scores beyond top activations.** `open-interpretability score` takes its windows from held-out text (`sample_heldout`, `openinterp/harness/explanation_scores.py`). It sorts the active windows by peak and splits them into five equal bins. Each bin gives 5 windows, so the weakest activations count as much as the top ones. That answers the paper's first point: an evaluation of top activations alone measures only recall.

**Our inactive windows are uniform, which is the gap that MG-IS fills.** Detection adds as many inactive windows as active ones, drawn at random from the held-out text. Most of them have nothing to do with the explanation. So the half of detection that tests false positives is easy. It rarely shows the judge a window that the explanation would wrongly call active. MG-IS draws such windows on purpose, and weights keep the estimate unbiased. Issue #3595 tries that, using the confound profile's trigger tokens or a small embedding model as the concept proxy. For verdicts, the reviewer's hard negatives do this job already.

**Our numbers are not population numbers, and theirs are.** The paper's correlation is an estimate over the whole probing set. Our detection's balanced accuracy is measured on a sample with equal numbers of active and inactive windows. It says how well the judge separates the two, not how often the explanation is right on ordinary text. That is acceptable for comparing explanations of one unit. It is not a rate to quote, and the score's documentation should say so.

**The rater model is about humans.** BRAgg's 23% error rate is for crowd workers. Our judges are single LLM calls, and the null-explanation control already checks them: with an empty explanation or another unit's, a Haiku judge scored 0.50 (`docs/analyses/null-explanation-control.md`). If we use several judges for one score, a Bayesian aggregation like BRAgg would be the way to combine them.

**Related work.** The same group wrote *Evaluating Neuron Explanations*, whose sanity checks issue #3237 applies to the metrics of `open-interpretability score`.

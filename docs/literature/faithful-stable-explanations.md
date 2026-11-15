# Faithful and Stable Neuron Explanations

**Citation.** Ge Yan, Tuomas Oikarinen and Tsui-Wei (Lily) Weng (UC San Diego). "Faithful and Stable Neuron Explanations for Trustworthy Mechanistic Interpretability". arXiv 2512.18092v1, 19 December 2025. [Paper](https://arxiv.org/abs/2512.18092). [Code](https://github.com/Trustworthy-ML-Lab/Trustworthy_Explanations). The HTML names the Mechanistic Interpretability Workshop at NeurIPS 2025. The authors say that an LLM checked their grammar and wording.

**Status.** Not applicable. It studies vision neurons, with explanations picked from a fixed concept set. The registry already measures both of its quantities: probe-count uncertainty through `builder/significance.py`, and stability through verdict stability and replicate reviews.

**Read.** The abstract page and the v1 HTML, on 2026-11-14. A model read the text in three passes and reported it to us, with short quotes. All of it came back as the paper's own text, appendices A–J included. We did not see the figures, so values that are only on plots are not here.

## Summary

**Claim.** Neuron identification is "the inverse of machine learning". So learning theory gives it guarantees on two things. *Faithfulness* asks whether the chosen concept is the one that best matches the neuron. *Stability* asks whether the choice holds across probing sets.

**Method.**

- **Setup.** There is a neuron, a finite set of candidate concepts, and a similarity metric such as accuracy, AUROC or IoU. The explanation is the concept with the highest similarity on a probing set.
- **Faithfulness.** Suppose every concept's measured similarity is within r of its true value. Then the chosen concept is within 2r of the best one (Theorem 3.1, Corollary 3.2). For accuracy, r = √(log(2/δ) / (2n)) by Hoeffding, with n probing samples. AUROC's rate gets worse as the concept gets rarer.
- **Stability.** A bootstrap over the probing set, with K = 100 resamples. *Bootstrap Explanation* (BE) returns the smallest set of concepts that covers a share t of the resamples' choices. Theorem 4.1 gives a lower bound on the chance that the set holds the best concept.
- **Experiments.**
  - simulations with known concept frequencies, 1,000 repetitions each
  - ResNet-50, ConvNeXt-base and ViT-B/16, explained by CLIP-Dissect and NetDissect
  - the Broden concepts (1,198) and a 20,000-word list

**Findings, as the text reports them.**

| Setting | Result |
| --- | --- |
| Simulation, a regular concept | Accuracy converges fastest. AUROC is faster than IoU, recall and precision |
| Simulation, a rare concept (frequency 0.001) | "AUROC and recall are much worse than precision and IoU" |
| Simulation, 1,000 concepts | Every metric's error falls about as 1/√n |
| Share of resamples that give the top concept (CLIP-Dissect against NetDissect) | ResNet-50 66.9% against 79.2%. ViT-B/16 51.2% against 91.3%. ConvNeXt-base 53.7% against 27.8% |
| Concepts needed to cover 90% of resamples | ResNet-50 3.11 against 2.08. ViT-B/16 5.93 against 1.34. ConvNeXt-base 5.61 against 6.99 |
| Cost of BE with K = 100 | About 30% more time and 15% more memory |

So a single explanation from CLIP-Dissect is often only one of several concepts that a slightly different probing set would give.

## Where the claims go beyond the evidence

- **Vision only.** The authors say "In this work we focus on vision models". Language-model neurons are only cited.
- **A fixed concept set.** The bounds need a finite list of candidate concepts. A free-text explanation, as an LLM writes it, has no such list.
- **The bounds give no sample size.** They are O(1/√n) with constants left open. So they say how the error shrinks, not how many probes a reliable explanation needs.
- **Bootstrap resamples are not independent draws.** Theorem 4.1 treats the K resamples as independent trials, but they all come from one probing set. The paper does not discuss that.
- **Missing details.** The layer and probing set of Section 4.3 are not named, and neither is the number of neurons averaged in Tables 3 and 4. There are no seeds or intervals. The statement and the proof of Theorem 4.1 write the binomial terms differently.

## Use in this repo

**The uncertainty it bounds is already in our verdicts.** A review measures specificity on hard negatives and sensitivity on holdouts. `builder/significance.py` puts a Beta posterior on each, after clustering near-duplicate probes. The site's P(useful) is the chance that their balanced accuracy is above the 0.6 rejection line. That is a direct form of the paper's "within r of the true value", and it widens when there are fewer probes.

**Its rate gives the size of that uncertainty.** With the paper's Hoeffding rate for accuracy and δ = 0.05, our 20 hard negatives give r ≈ 0.30, and our 10 holdouts r ≈ 0.43. So one review's specificity of 0.85 is consistent with a true specificity anywhere from about 0.55 to 1. That fits `docs/analyses/verdict-stability.md`: three reviews of the same submission disagree on the status in a third of pairs.

**Its stability test is the registry's replicate review.** BE re-runs identification on resampled probing sets. The registry's `replicate` follow-ups (`open-interpretability next`) and the verdict-stability study re-test a hypothesis with fresh probes. A hypothesis that holds up under that is stable in the paper's sense.

**Rare concepts need more than balanced metrics.** The paper finds AUROC and recall slow for rare concepts. That matches `evaluating-neuron-explanations.md`, from the same group: on ordinary text, where a pole is seldom active, use a metric that handles rare concepts. Our detection windows are balanced, so they avoid the problem.

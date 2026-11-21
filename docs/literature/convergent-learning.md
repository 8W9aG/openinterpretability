# Convergent Learning: Do Different Neural Networks Learn the Same Representations?

**Citation.** Yixuan Li, Jason Yosinski, Jeff Clune, Hod Lipson and John Hopcroft, of Cornell University, the University of Wyoming and Columbia University. The first two authors contributed equally. "Convergent Learning: Do different neural networks learn the same representations?" *ICLR 2016*. [arXiv:1511.07543](https://arxiv.org/abs/1511.07543), version 1 of 24 November 2015, last revised as version 3 on 28 February 2016. A preliminary version was at the NIPS 2015 Feature Extraction workshop. Code is at [github.com/yixuanli/convergent_learning](https://github.com/yixuanli/convergent_learning).

**Mismatch.** The abstract on the arXiv page lists three findings, and the abstract in the HTML of version 3 lists four. This page quotes the HTML.

**Status.** Used. The registry's sister search matches each pole to its best-correlated pole in another model, and keeps mutual best matches. That is a strict form of this paper's one-to-one matching. `papers/explanations-across-models` cites it three times.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 3 for this page's author, a model (Claude Opus 5.5). It read Sections 1 to 6 and Appendices S.1 to S.3 in the source's own words, in two fetches. Appendix S.4 came back only as a summary, and this page uses nothing from it. The figures are images, so their values aren't used.

## Summary

**Claim.** Networks trained from different random starts learn some of the same features, but not all. In the paper's words, "some features are learned reliably in multiple networks, yet other features are not consistently learned". Where features recur, they sometimes recur as single units and sometimes as small groups.

**Method.**

- **Networks.** Four AlexNet-style networks on ImageNet, trained in Caffe "using four different random initializations". Their top-1 accuracies range from 58.65% to 58.84%.
- **Semi-matching.** Each unit of Net1 takes its most correlated unit of Net2, so several units can share one partner.
- **One-to-one matching.** A "bipartite matching": an assignment without replacement that maximises the summed correlation. The Hopcroft-Karp algorithm finds it.
- **Groups.** A sparse linear map from one network's layer to the other's, with an L1 penalty. Hierarchical clustering then finds features shared by small groups of units.
- **Data.** Correlations and mutual information over the units' activations on images.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Units of Net1 with no Net2 match above 0.5 (Section 3) | conv1 17%, conv2 37%, conv3 63%, conv4 92%, conv5 75% |
| Best layers for one-to-one matches (Section 3) | conv1 and conv5. The middle layers match worse |
| Sparse map error, no penalty (Table 1) | conv1 0.170, conv2 0.372. conv3 to conv5 stay high, 0.434 to 0.484 |
| Code type in early layers (Section 4) | "a mix of a local and a slightly distributed code" |

## Where the claims go beyond the evidence

- **No random baseline.** The paper compares matches with a unit's correlation with itself and with units of the same network, but not with random directions or untrained networks. So a correlation of 0.5 has no chance level to compare with.
- **One architecture, one pair.** Most detailed results compare Net1 with Net2 only, with no spread across the other pairs and no intervals.
- **The upper layers are left open.** The authors say "it is difficult to draw any strong conclusions regarding those layers".
- **Greedy clustering.** The authors say the clustering "could in some cases be suboptimal".

## Use in this repo

- **Our citations check out.** `papers/explanations-across-models` cites the paper three times.
  - It "match[es] the units of two vision networks, trained from different random starts, by the correlation of their activations". The paper trains four and reports most detail on two, so this matches.
  - The sister search's mutual step "is a strict form of the one-to-one matching". A mutual best match is a one-to-one pairing that each side would also pick alone, so this is fair. The step is the paper's own, and it says so.
  - The quote "some features are learned reliably in multiple networks, yet other features are not consistently learned" is in the abstract.
  - The five authors, 2016 and ICLR 2016 match.
- **The rotated baseline is not from this paper.** The sister search compares each match with a random rotation of the layer, as Gurnee et al. (2024) do ([`universal-neurons.md`](universal-neurons.md)). This paper has no such baseline. `papers/explanations-across-models` credits the baseline to Gurnee et al., which is right.
- **Groups are a case the sister search can miss.** The paper finds features shared by small groups of units, not single units. A pole-to-pole search finds only one-to-one matches. Our reading: a feature spread over a few poles in one model would show as weak matches, not as a sister. The registry's superpositions are the closest unit to that case.
- **No question for the maintainer.**
- **Related pages.** [`universal-neurons.md`](universal-neurons.md) takes this method to GPT-2, with the rotated baseline.

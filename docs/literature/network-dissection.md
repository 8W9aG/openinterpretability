# Network Dissection: Quantifying Interpretability of Deep Visual Representations

**Citation.** David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva and Antonio Torralba, of CSAIL, MIT. The first two authors contributed equally. "Network Dissection: Quantifying Interpretability of Deep Visual Representations". *CVPR 2017*, an oral presentation. [arXiv:1704.05796](https://arxiv.org/abs/1704.05796), version 1 of 19 April 2017, the only version. Code and data are at [netdissect.csail.mit.edu](http://netdissect.csail.mit.edu).

**Mismatch.** Table 1's concept counts sum to 1,376, but Figure 3's caption says the probe runs "1197 segmentation tasks". The paper doesn't reconcile them.

**Status.** Used. `papers/neuron-development` cites it as the precedent for following unit interpretability through training. Its rotation test is the measured version of the random-direction comparison that the registry's nulls make.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 1 for this page's author, a model (Claude Opus 5.5). It read the whole text in the source's own words, in two fetches: Sections 1 to 4 and the references. The figures are images, so their values aren't used.

## Summary

**Claim.** The interpretability of a vision network's units can be measured: score each unit against a large set of labelled concepts. Interpretability is a property of the basis, not of the representation's power. Rotating the units keeps what a network can do but removes most of its interpretable units.

**Method.**

- **Concepts.** Broden, a merged dataset of pixel-labelled scenes, objects, parts, materials, textures and colours.
- **Score.** Threshold each unit's activation map at its top 0.5% of values. Then measure the overlap (IoU) with each concept's mask over the whole dataset. A unit is a detector when its IoU exceeds 0.04, and takes its best concept.
- **Networks.** AlexNet, GoogLeNet, VGG-16 and ResNet-152, supervised on ImageNet and Places, and AlexNets trained with ten self-supervised tasks.
- **Rotation test.** Rotate the 256 conv5 units of a Places205 AlexNet by a random orthogonal matrix, and count unique detectors again.
- **Training.** Follow the detectors of one AlexNet's conv5 over 2.4 million training iterations.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Random rotation of conv5 (Section 3.2) | 80% fewer unique detectors, with the same discriminative power |
| Over training (Section 3.5) | Object and part detectors appear at about 10,000 iterations. No unit changes category |
| Across networks (Section 3.4) | ResNet > VGG > GoogLeNet > AlexNet, and Places > ImageNet |
| Batch normalisation (Section 3.5) | "seems to decrease interpretability significantly" |
| Wider conv5, 256 to 768 units (Section 3.7) | Many more detectors, at 0.5% lower accuracy. Wider still adds little |
| Human check (Section 3.1, Table 3) | 37% (conv1) to 71% (conv5) of the labels were judged descriptive |

## Where the claims go beyond the evidence

- **Only Broden's concepts count.** The authors say so: a unit that matches a concept absent from Broden "will not score well for interpretability".
- **One layer for the rotation.** The 80% drop is for conv5 of one AlexNet. The paper doesn't say whether it holds elsewhere.
- **A fixed IoU cut-off.** The 0.04 threshold is a choice. The authors say their qualitative results don't depend on it.
- **One run per setting.** The training-dynamics and regularisation results come from single runs, apart from three repeats that converge to similar levels.

## Use in this repo

- **Our citations check out.** `papers/neuron-development` cites the paper twice.
  - That such research explains a network "one unit at a time" fits the method, which scores each unit alone.
  - That it "measured how well individual units of vision networks align with semantic concepts" matches Section 2.
  - That it "followed that alignment across training iterations" matches Section 3.5.
  - The five authors, 2017 and CVPR 2017 match.
- **The rotation test is the measured answer to Szegedy et al.** [Szegedy et al. (2014)](intriguing-properties.md) judged by eye that random directions read like units. This paper measures it on one layer and finds 80% fewer detectors after rotation. The registry's random-direction nulls ask the same question for each unit.
- **Concept labels against written explanations.** The paper's units are scored against fixed labels. The registry's are scored against an explanation written for each unit and tested by probes. `papers/neuron-development` makes that step and marks it as its own.
- **Batch normalisation as a caution.** The paper finds that whitening each layer lowers interpretability without lowering accuracy. Our reading: the registry's models use layer norm and RMSNorm, not batch norm, so this result doesn't carry over directly.
- **No question for the maintainer.**
- **Related pages.** [`intriguing-properties.md`](intriguing-properties.md) is the claim this paper tests. [`universal-neurons.md`](universal-neurons.md) compares units across networks.

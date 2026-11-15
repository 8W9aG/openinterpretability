# Is a Linear Probe Evidence of a Linear Representation?

**Citation.** Nadir Kutluozen (Cold Spring Harbor Laboratory), Christian Internò (Bielefeld University) and David Klindt (Cold Spring Harbor Laboratory). "Is a Linear Probe Evidence of a Linear Representation?". arXiv 2610.22655v1, 15 October 2026. [Paper](https://arxiv.org/abs/2610.22655). [Code and data](https://github.com/klindtlab/linear-probe-nullspace). The arXiv comment says "Accepted at NeurIPS 2026", and we did not look at the proceedings. The acknowledgements name a Goodfire research grant, and say that the authors did the work with Silico, Goodfire's research agent.

**Status.** Planned. Issue #3559 adds a random-init arm to `open-interpretability probe-compare`, which the paper's argument calls for. Our untrained-model null already applies the same control to verdicts.

**Read.** The abstract page and the v1 HTML, on 2026-11-13. A model read the text and reported it to us. Sections 1–6 (theory, simulation, SVG-World, discussion and limitations) came back as the paper's own text. The appendices came back only as the model's summary of them. So every number below is from Sections 4–5. We did not see the figures.

## Summary

**Claim.** The usual test, "a linear probe decodes the factor, so the model represents it linearly", is too weak. A rich enough feature map gives a linear readout of almost anything, and a random one does too. So high probe accuracy fits two different geometries. One is a *linear world model*, where the features are a linear map of the factors plus small noise. The other is a kernel-like geometry with no such structure.

**Method.** The paper defines a linear world model as `f(x) = W z + μ + ε`, with the noise small and uncorrelated with the factors `z`. That is strictly stronger than linear decodability. It proposes three tests that a random kernel fails:

- **D1, reverse predictivity.** Regress the features on the factors, and report the R². It also derives a ceiling for that R² from the feature spectrum.
- **D2, spectral concentration.** The same R², on only the top k whitened principal components, with k the number of factors.
- **D3, out-of-distribution transfer.** Train the probe on one visual style and test it on another.

The tests run in three settings:

- A simulation with known ground truth: random Fourier features into a ReLU MLP, either at random init or trained to read out the factors.
- *SVG-World*, rendered scenes with 12 position factors in two styles. It uses six vision transformers (DINOv3 S/B/L, DINOv2-B, DeiT-B and CLIP-B), each against a random-init copy, and ResNet-50 as a control.
- dSprites, 3D Shapes, MPI3D and CelebA.

The probes are ridge regressions on the CLS token or the mean of the patch tokens. There are no single units, no nonlinear probes and no interventions.

**Findings, as the body text reports them.**

| Setting | Result |
| --- | --- |
| Simulation, width 1,024, 5 seeds | Forward R² is 1.000 whether or not the readout has weight decay. Reverse R² is 0.9935 with weight decay and 0.7841 without. At random init, reverse R² levels off near 0.2 |
| DINOv3-B/16, SVG-World, CLS | Forward R² 0.88 against 0.33 at random init (fixed ridge). With a penalty chosen by cross-validation, the random init reaches 0.66 |
| The same | Reverse R² 0.18 against 0.02 at random init. The gap is 0.187 (95% bootstrap interval 0.182–0.191) in one style and 0.143 (0.139–0.148) in the other |
| The same, spectral concentration | Pretrained needs 17 components for 90% of the variance that the factors predict. Random inits need 55–59 |
| Six transformers | Reverse-R² gaps over random init of 0.145–0.235. ResNet-50: 0.105 |
| DINOv3 S/16 → L/16 | Forward R² rises from 0.855 to 0.914. Reverse R² falls from 0.257 to 0.169 |
| Cross-style transfer, patch mean | ρ 0.65 against 0.23 at random init |
| CelebA | 0.28–0.36 of the reverse-R² ceiling, against 0.07–0.11 at random init |

The conclusion: pretraining adds structure that a random kernel lacks, but the models are far from a linear world model of these factors. Better forward decoding can come with worse reverse predictivity, as from DINOv3 S to L.

## Where the claims go beyond the evidence

- **The tests are relative to a chosen set of factors.** A low score rules out a linear world model *of those factors*. It does not show that the model has no clean structure. The authors say this.
- **Variance-based tests miss sparse features.** D1 and D2 are linear and weighted by variance. So they underweight sparse, low-variance or superposed factors, which are what neuron and dictionary work looks for. The authors say this, and suggest a sparse-dictionary basis, which they did not test.
- **Vision only.** All the encoders are vision models, and the factors are positions and face attributes. The paper shows nothing for language models or for token-level concepts.
- **The ridge penalty matters.** At random init, forward R² goes from 0.33 to 0.66 with a penalty chosen by cross-validation. So the forward gap that the paper reports depends on a fixed penalty.
- **The sign is not stable.** On two of six encoder–benchmark pairs, the sign of the comparison depends on how the benchmark encodes the factor.
- **No causal test.** Interventions and steering are future work. So "linear world model" here is a statement about the geometry of the features, not about what the model uses.
- **Appendix numbers we did not read.** The summary's appendix tables differ a little from the body, for example 0.996 against 0.9935. We used only the body.

## Use in this repo

**Our verdicts already have the paper's control.** The untrained-model null (`docs/analyses/untrained-null.md`, #3202) audits 20 GPT-2 poles twice, on the trained weights and on a random-init copy. The trained poles verify 9 of 20 times and the random ones 2 of 20 times. That is the paper's argument applied to our decision rule. A random network still has structure, so its rate is not zero, and the gap is what counts.

**Our dense probes have no such control.** `docs/analyses/probe-comparison.md` reads dense probes at AUROC ≥ 0.99 on a 4,864-wide layer as "every concept is linearly decodable from the layer almost perfectly". By the paper's argument, a layer that wide may decode these labels at random init too. So that line shows less than it seems to. The comparison it serves still stands: the probe is a ceiling for the pole, not evidence of a representation. But "how far the pole falls short of the layer" needs the random arm's ceiling as well. Issue #3559 adds that arm.

**The argument does not reach a single pole.** One neuron's strength is a one-feature readout with a fitted threshold, not a rich feature map. A single feature with one threshold can't fit an arbitrary label. And the untrained null verified only 2 of 20 random units. The paper's kernel argument is about the width of the readout, and k = 1 has none.

**Reverse predictivity has a single-unit form.** For a pole, D1 becomes: how much of the pole's activity does the concept explain? A pole that fires on the concept but also on much else decodes the concept well and has low reverse R². The repo reads that today through probes and through the explanation scores, not as one number. Issue #3441, *Calibrated concept profiles*, is the nearest open work.

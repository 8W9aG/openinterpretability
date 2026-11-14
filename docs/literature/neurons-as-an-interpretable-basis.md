# Neurons as an Interpretable Basis

**Citation.** Curt Tigges, Matthew Kowal, Thomas McGrath, Owen Lewis and Nick Cammarata (Goodfire). "Neurons as an Interpretable Basis". Goodfire Research, 10 November 2026. [Post](https://www.goodfire.com/research/neurons-as-an-interpretable-basis), with a PDF and a Markdown version on the same page. It calls itself "a companion to *Neural Networks Are Already Clean*", the essay behind the `native-neuron-audit` skill. It comes with the [Neuron Atlas](https://www.goodfire.com/research/public_neuron_atlas), which gives full pages for 2,734 neurons and short labels for "all 10 million neuron-poles". A company research post, with no peer review.

**Status.** Planned. Its split of a pole into modes is a template for issue #3215, polysemantic experiments. Its quadrant and band results bear on the `native-neuron-audit` skill and on the protocol's line.

**Read.** The HTML page and its Markdown version, on 2026-11-14. A model read the text and reported it to us, with short quotes. The main text came back as the post's own text, to about 220,000 of the Markdown's 416,000 characters. We did not read the rest of the appendix, the PDF or the interactive figures. One appendix result, on circuits, came back only as a model's summary, so this page leaves it out.

## Summary

**Claim.** The MLP neurons of a large MoE language model are more interpretable than people assume. They are already sparse in what they do. Each pole has few modes, and usually one main mode. Descriptions written by an agent can find a neuron's firings, simulate its activations and write inputs that drive it.

**Method.**

- **Model.** Qwen3.6-35B-A3B, with about 5.26 million neurons. Each of its 40 layers has 256 experts of 512 neurons. A token goes to 8 routed experts and one shared expert.
- **Units.** The SwiGLU product times the expert's router weight. Positive and negative poles are read apart. The post splits firings into four sign quadrants of gate and up. Q1 (gate+, up+) and Q3 (gate−, up−) give positive values, and Q2 and Q4 give negative ones.
- **Sparsity.** It mean-ablates activations by percentile band, or keeps only the top k per expert or per layer. It measures the loss and KL on 8.4 million tokens of web text.
- **Modes.** For each firing, the post takes the top 400 gate and up inputs and clusters them into *subcircuits* (median 19 per pole). Claude Opus 5.5 then groups those into *modes*. Each mode gets its share of the pole's firings and of its KL damage under ablation.
- **Descriptions.** A "Neuron Scientist" agent (Claude Fable 5, Claude Opus 5.5 or GPT-6 Sol) writes them from 200 documents per neuron.
- **Tests.** Detection against distractors, mode classification, simulation of activation levels, and input synthesis.

**Findings, as the text reports them.**

| Setting | Result |
| --- | --- |
| Keep the top 320 of 4,096 routed neurons per layer (92.2% ablated) | Loss 1.886 → 1.940 (+2.9%), KL 0.140 |
| Ablate the middle band, P25–P75 (49.7% of activations) | Loss +0.010 |
| Ablate P5–P95 (88.8%) | Loss +0.091 |
| Random 64 per expert, the control (87.5%) | Loss 4.214 |
| Sign quadrants, all routed firings | Q1 and Q2 13–14% each, Q3 35–36%, Q4 36–37% |
| Sign quadrants, top 100 firings | Q1 39–50%, Q2 43–57%, Q3 1–8%, Q4 0–5% |
| Modes, 5,468 poles | 2,556 have one mode, 2,673 have two or more, and 239 have none that reads. 69% have one mode that causes at least 90% of the KL damage |
| Detection, 496 poles, Opus 5.5 judge (chance 0.50) | 0.95 against random text, 0.89 against firings from the same expert |
| Simulation, 486 poles | 73% of the activation restored overall, against a ceiling of 96% for a perfect reader |
| Input synthesis, 128 poles | 120 (94%) exceed the neuron's maximum on the 10B-token corpus |
| Q3 ablated, 24 GSM8K problems | Mean reasoning length falls from 1,550 to 176 tokens |

The post also finds that a pole's positive and negative values "usually (but not always)" have unrelated roles. It says that "neurons are never actually off": a routed neuron always has some value.

## Where the claims go beyond the evidence

- **One model.** Every result is on one MoE model. The authors say that they "must also establish how far these findings generalize across models and behaviors".
- **The judges and agents are the same model families as the describers.** Claude and GPT models write the descriptions, group the modes and judge the tests. The post gives no human check of the descriptions.
- **Mode counts depend on the method.** The authors say that their mode counts "are not definitive counts of the concepts represented by a neuron". A different clustering of the subcircuits would give different counts.
- **Few intervals.** Only the Q3 reasoning test (24 problems) has 95% Wilson intervals. The other ranges in the text are reported ranges, not intervals, and no seeds are given.
- **The essay and the post give different shares.** The essay said that about 90% of poles had one main job in a Qwen3-32B audit. Here 47% of poles have one mode, and 69% have one mode that causes almost all the damage. The two measures differ, but a reader of the essay alone would take the higher number.
- **The detection score is near the ceiling partly by design.** Mode classification from inputs scores 0.97. The authors call it "a near-ceiling reference rather than an independent test".

## Use in this repo

**It reads the protocol's unit.** The registry also reads each pole of a SwiGLU neuron apart (`docs/PROTOCOL.md` §1). The post's finding that the two poles usually have unrelated roles supports that choice. Its model is a MoE with a router weight on each neuron. Our Qwen2.5 models are dense, and GPT-2 uses GELU. So shares such as the quadrant split may not transfer.

**It puts numbers on the skill's sign-regime check.** `native-neuron-audit` tells a proposer to check whether a positive pole's top examples come from `g+u+` or from the double-negative `g-u-` regime. In the post, `g-u-` (Q3) holds 35–36% of all firings but only 1–8% of the top 100. So top examples nearly always show Q1, and the Q3 role stays hidden from a scan that reads top examples only. The post also finds that Q3 matters: ablating it cuts the model's reasoning length by nine tenths. The skill now cites these numbers.

**It bears on the protocol's line.** The protocol counts a pole as active at 0.25 × its maximum. In the post, ablating the middle half of activations costs almost nothing (+0.010 loss), and ablating all but the outer 10% costs +0.091. So most of a neuron's function is in its large values, which is the regime that the line measures. That supports the line, though the post measured it on a different model.

**Its modes are a template for polysemantic experiments.** Issue #3215 asks for an experiment class for polysemantic units. The post splits a pole into modes and gives each its share of firings and of damage. That is a concrete form: an explanation per mode, each with its own share, tested on its own firings. About half the post's poles have more than one mode. So the registry's one-explanation form fits about half of all poles well, and #3594 (probe every part of a multi-part explanation) matters for the rest.

**Its detection scores are far above ours.** The post's detection reaches 0.89–0.95 with an Opus judge, against about 0.6 for our stored scores (`docs/analyses/verified-vs-detection.md`). A quarter of our scores are at or below chance (#3600). The settings differ. Its judge is Claude Opus 5.5, and ours are mostly small, fast models: deepseek-flash and muse-spark-1.3 judged over 90% of our stored detection scores. Its distractors are chosen, while our inactive windows are random. And its describer read 200 documents per neuron. Each difference is a lead for #3600.

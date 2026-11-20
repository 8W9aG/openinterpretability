# Net2Net: Accelerating Learning via Knowledge Transfer

**Citation.** Tianqi Chen, Ian Goodfellow and Jonathon Shlens, of Google, Mountain View. Chen is also a PhD student at the University of Washington. "Net2Net: Accelerating Learning via Knowledge Transfer". [arXiv:1511.05641](https://arxiv.org/abs/1511.05641), version 4 of 23 April 2016. Versions 1 to 3 are of 18 November 2015, 19 November 2015 and 7 January 2016. The arXiv comment says "ICLR 2016 submission", and DBLP lists it under ICLR 2016. We didn't check the proceedings. The text gives no code link, no funding and no disclosure. The experiments used TensorFlow.

**Mismatch.** The abstract claims "a new state of the art" on ImageNet. The body gives 78.5% on the authors' own validation set, with no metric named and no table of earlier results. Algorithm 1 divides by a count that the text indexes by another unit, so the algorithm and the formula disagree on one index.

**Status.** Planned. #5220 names it, with Transformer-Patcher, as background for arm I, an added block whose output starts at zero. The fix campaign is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 4 for this page's author, a model (Claude Opus 5.5). It read the source's own text in one fetch: Sections 1 to 4, the references and Appendices A and B. The maths came as LaTeX text. We didn't see any figure as an image. Figures 4 to 6 hold the only accuracy curves, so the training speed appears here only as the captions describe it. We didn't read versions 1 to 3.

## Summary

**Claim.** A trained network can be grown into a wider or deeper one that computes the same function at the start. Training the larger network from there reaches the same accuracy as training it from random weights, but much sooner.

**Method.**

- **Net2WiderNet.** A random map sends each new unit to an old one, and each old unit is kept at least once. A copy takes its source's incoming weights, so it has the same activation. Each outgoing weight is divided by the number of copies of its source, so the next layer receives the same sum. Biases, convolution channels, batch norm and concatenation need matching remaps.
- **Net2DeeperNet.** A new layer goes after an old one, with its weight matrix set to the identity. That keeps the function only if $\phi(I\phi(v)) = \phi(v)$ for all $v$. This holds for ReLU and, with replicated columns, for maxout. It fails for the logistic sigmoid. Batch norm needs its scale and shift set from training statistics, so that the new layer undoes the normalisation.
- **Exactness.** Net2DeeperNet is exact and needs no noise. Net2WiderNet is exact only with a randomised training method such as dropout. Otherwise "one should add a small amount of noise to all but the first copy of each column." The start is then close to the teacher, not the same.
- **After growth.** Every parameter trains, with nothing frozen. The student's learning rate starts at about one tenth of the teacher's.
- **Setting.** Inception-BN on ImageNet. The teacher for widening has about 30% of the parameters. The baselines are random padding for the new width, random weights for the new depth, and the larger model trained from scratch.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Net2WiderNet against random weights (Figure 4 caption) | it "converges to roughly its final validation set accuracy after roughly 3×10^6 minibatches". Random weights need "roughly an additional 2×10^6 minibatches" |
| Final accuracy, wider model (Section 3.2) | the same for both starts: "the true size of the model governs the accuracy" |
| Net2DeeperNet against random weights (Section 3.3) | it reaches "good accuracy much faster". The text gives no number |
| Wider and deeper model (Section 3.4) | "78.5% on our ImageNet validation set". The larger models were not trained from scratch |

## Where the claims go beyond the evidence

- **The speed results are curves with no numbers in the text.** The text gives only the minibatch counts in one caption. It gives no wall-clock time and no compute.
- **One run of each, with no variance.** No seeds, runs or intervals appear.
- **The claim of a new best result has no comparison.** The authors say the large models were not trained from scratch "due to resource and time constraints". So 78.5% has no baseline of the same size.
- **Nothing about the copied units after training.** The paper doesn't measure whether copies split apart, or how far the old units move.
- **"Lifelong learning" is a motivation, not a test.** No experiment adds a task.

The authors state limits themselves. Net2DeeperNet needs an activation that can represent the identity. Batch norm needs statistics from the data. Noise makes the wider start approximate.

## Use in this repo

- **The issue's two questions, as the source answers them.**
  - *Which starts keep the function:* copying units with outgoing weights divided by the count, which is exact unless noise is added. And an identity layer, which is exact for ReLU but not for every activation.
  - *What the old units do in training:* the paper doesn't measure it. All weights train, and the paper calls that "safe," because training starts from the teacher's function.
- **The patch format already uses a simpler exact start.** In a residual network, a block whose output scale is zero adds nothing, whatever its activation. The patch format checks exactly that. Each patch runs with every output scale at zero, and the check fails if any logit moves beyond float32 noise ([`docs/patches.md`](../patches.md)). So arm I of #5220 doesn't need Net2DeeperNet's identity condition, which Qwen's SwiGLU and GPT-2's GELU would fail.
- **Net2Net trains everything, and the fix plan does not.** Net2Net lets old and new parts co-adapt. Arm I keeps the base model frozen, so interference can come only from where the module switches on. That is the difference the fix plan depends on, and Net2Net gives no evidence for it either way.
- **A widened copy is a duplicate unit for the registry.** Our reading, not the paper's: right after Net2WiderNet, two copies of a neuron have the same activation on every input. Each copy carries an equal share of the original's output. A registry explanation of each would be the same, and an ablation of one would show only its share of the effect. If a fix ever widens a layer, the registry would need to treat the copies as one unit until training splits them.
- **Questions for the maintainer, recorded on this page only.**
  - Should a fix that adds width keep the base frozen, as arm I does, or let the base train, as Net2Net does?
  - Should the interference report (#5230) measure how far the old units' activations move, which Net2Net never did?
- **Related pages.** [Transformer-Patcher](transformer-patcher.md) adds one trained neuron for each mistake, with its start keyed to the mistake. [`progressive-neural-networks.md`](progressive-neural-networks.md) grows a network with the old columns frozen. [`ffn-key-value-memories.md`](ffn-key-value-memories.md) is the view of a neuron as a key and a value.

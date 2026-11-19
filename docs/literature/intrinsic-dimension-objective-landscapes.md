# Measuring the Intrinsic Dimension of Objective Landscapes

**Citation.** Chunyuan Li (Duke University, work done as an intern at Uber AI Labs), Heerad Farkhoor, Rosanne Liu and Jason Yosinski (Uber AI Labs). "Measuring the Intrinsic Dimension of Objective Landscapes". [arXiv:1804.08838](https://arxiv.org/abs/1804.08838), version 1 of 24 April 2018, the only version. Venue: ICLR 2018, as the PDF header and the arXiv comment state. The paper text gives no link to code or data. It names no funding. It thanks the OpusStack team inside Uber for the computing platform. The experiments use TensorFlow and Keras.

**Status.** Planned. #4782 would use it in its strand on LoRA and low-rank fine-tuning. No code, skill or protocol in this repo depends on it. The maintainer designs any campaign to fix a model, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we downloaded the v1 PDF (23 pages) with `curl` and converted it with `pdftotext`. The reader was this page's author, a model (Claude Opus 5.5). We read in the source's own text the abstract, Sections 1 to 4, the acknowledgements and the references. We also read the supplementary Sections S5 to S11. We read Tables 1, S2, S3, S4 and S5 as extracted text. We did not see the figures as images, so this page gives only their captions and the numbers that the text states. We did not read the OpenReview discussion or any code.

## Summary

**Claim.** Train a network only inside a random d-dimensional subspace of its D weights, and raise d until a solution appears. That d measures the intrinsic dimension of the problem. It is often far below D, and it changes little as a network grows wider or deeper.

**Method.**

- **Random subspace training.** The weights are θ = θ₀ + Pθ_d. The start θ₀ and the D × d matrix P are random and frozen, and θ_d starts at zero. Training changes only θ_d. Each column of P has unit length and spreads over all D weights. The columns are not orthogonalised, and the authors rely on the near-orthogonality of random vectors in high dimension.
- **Definition.** The intrinsic dimension is the codimension of the solution set, D = d_int + s. Here s is the dimension of the set of solutions. In a toy problem with D = 1,000 and 10 sum constraints, d_int is 10.
- **Threshold.** d_int90 is the smallest d that reaches 90% of a baseline's performance. The baseline is usually the best directly trained model. The authors chose 90% "after looking at a number of dimension vs. performance plots". d_int100 varied too much to use.
- **Projections.** A dense Gaussian P for small networks. A very sparse random projection for CIFAR-10. The Fastfood transform, which forms P implicitly, for ImageNet and Pong.
- **Problems.** Fully connected (FC) networks and LeNet on MNIST, MNIST with shuffled pixels and MNIST with shuffled labels. FC, LeNet and a 20-layer ResNet on CIFAR-10. SqueezeNet on ImageNet. A DQN on CartPole and two easier variants. Evolution strategies on InvertedPendulum, Humanoid and Atari Pong.
- **Sweep.** 20 FC networks on MNIST, with 1 to 5 hidden layers and widths 50, 100, 200 and 400. Each d has 3 runs, and the standard deviation of d_int90 comes from 50 bootstrap samples. Other experiments use one run.
- **Baselines.** Directly trained networks. 1,000 small, directly trained FC networks of random depth and width, to compare parameter efficiency. Weight decay and dropout, to compare subspace training as a regulariser.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Toy problem (Section 2) | D = 1,000, d_int90 = d_int100 = 10 |
| MNIST, FC 784–200–200–10 (Section 3.1) | D = 199,210, d_int90 about 750, which is 0.4% of D. Stored as two seeds and 750 floats: 3.2 kB against 793 kB, a factor of 260 |
| MNIST, LeNet (Section 3.1) | D = 44,426, d_int90 = 290, a compression of about 150 times |
| MNIST, 20 FC networks (Figure 3 caption) | D varies by a factor of 24.1 and d_int90 by 1.33, with a global baseline of 100% accuracy. With a baseline for each model, 1.42 (Figure S7) |
| MNIST, shuffled pixels (Section 3.1) | FC stays at 750. LeNet rises from 290 to 1,400 |
| MNIST, shuffled labels, FC with 5 layers of 400 (Table S2) | D = 959,610. d_int90 is 190k for all 50,000 labels (3.8 floats per label), 130k for 50% (5.2) and 90k for 10% (18.0) |
| CIFAR-10 (Table 1) | FC 9,000 and LeNet 2,900 |
| CIFAR-10, to above 50% validation accuracy (Section 3.2) | FC 9k, LeNet 2.9k, ResNet 1k |
| CIFAR-10, ResNet with 280k weights (S8) | 1,000 to 2,000 to reach LeNet's baseline, 20,000 to 50,000 to reach its own |
| ImageNet, SqueezeNet (S9) | D = 1,248,424. Accuracy 34.34% at d = 500k, below the threshold, so d_int90 is over 500k. Each d took 6 to 7 days on 4 GPUs |
| Reinforcement learning (Table 1, S6) | InvertedPendulum 4, Humanoid 700, Atari Pong 6,000 (ES). CartPole 25, Pole 23, Cart 7 (DQN) |
| LeNet variants (S10) | MNIST: standard 290, untied 600, global tied filters 425, FC 2,000. CIFAR-10: 1,000, 2,750, 2,500 and 35,000 |
| Training stability (S5.3) | Direct training of deep FC networks fails above depth 4 with SGD. Subspace training stays stable up to 10 layers |
| Projection cost (Table S4) | With d at 1% of D = 1M: dense 1.0742 s (extrapolated), sparse 0.0019 s, Fastfood 0.0195 s per forward and backward pass |

**Optimisers and rotation (Section 2.1).** The authors note that a random rotation of the full basis (d = D) does not change the path of SGD. It does change the path of RMSProp and Adam, which scale each axis on its own.

## Where the claims go beyond the evidence

- **"Extra parameters" rests on one sweep.** The abstract says that extra parameters "serve directly to increase the dimensionality of the solution manifold". The evidence is 20 FC networks on MNIST, at a 90% threshold. The authors themselves warn against a claim that all FC networks on MNIST have d_int90 near 750.
- **d_int90 is an upper bound, measured with random directions.** A random subspace finds a solution only when it meets the solution set. So d_int90 bounds the true intrinsic dimension from above, and a learned subspace could need fewer. The authors state this in the MDL discussion.
- **The 90% threshold was chosen after looking at the data.** The authors expect most conclusions to hold at another threshold, but they test no other one. The CIFAR-10 FC results move by a factor of 5.0 or 1.52, depending on the baseline. So the baseline changes the number a lot.
- **Most estimates are one run.** Only the MNIST FC sweep has 3 runs and a bootstrap. The authors argue that the variance from P is small, but they do not test the variance from θ₀ separately.
- **The numbers disagree within the source.** MNIST LeNet is 290 in Section 3.1, 275 in Table S5 and 225 in S7. Shuffled-pixel LeNet is 1,400 in Table 1 and 650 in Table S5. CIFAR-10 LeNet is 2,900 in Table 1 and 2,500 in S6.2 and S7. S6.2 gives MNIST FC as 650. CIFAR-10 FC has D = 656,810 in Table 1 and 1,055,610 in Table S5. Table S5 gives the CartPole FC network D = 199,210, but S6.1 describes a network with two layers of 400.
- **The baseline descriptions disagree.** S5.1 says the baseline for each model gives a "slightly lower intrinsic dimension". The Figure S7 caption says that it gives a higher d_int90 for larger models.
- **No language models and no units.** Every network is an image classifier or a small control policy. The paper says nothing about single neurons, explanations or what the trained weights compute.

The authors state several limits themselves: the upper-bound nature of d_int, the loose MDL bound, the unfinished ImageNet estimate and the one-run estimates.

## Use in this repo

- **The repo has no subspace training and no fine-tuning code.** `git grep` finds no LoRA, adapter or subspace training in `openinterpretability/` or `scripts/`. [`fixing-models.md`](fixing-models.md) also notes that the repo has no fine-tuning or weight-edit code. So the paper's runs cannot be repeated here as they are.
- **A low-dimensional update is not a change to few neurons.** Each column of P spreads over all D weights. So a solution found in a 750-dimensional subspace moves every neuron's weights a little. This is our inference from Section 2.1, not a claim of the paper. It agrees with the point in [`lora-learns-less-forgets-less.md`](lora-learns-less-forgets-less.md): a LoRA change moves every neuron of a layer inside one shared subspace. So a small intrinsic dimension does not predict that only a few VERIFIED poles move after a fine-tune.
- **The repo's base-instruct numbers fit a small, spread update.** [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) tests each VERIFIED pole of Qwen2.5 on its instruct model at the same index. On Qwen2.5-1.5B, 38 of 43 hypotheses keep, and every activation correlation is 0.953 or more. On Qwen2.5-0.5B, 252 of 311 keep, at a median correlation of 0.974. A fine-tune that needs few directions, each spread thin over the units, would move each pole a little and break few. That reading is ours. The paper measures no language model, and the analysis does not record how Qwen tuned the instruct models.
- **Redundancy gives units room to drift.** If D = d_int + s and s is large, many directions leave the loss unchanged. Training can then move units along the solution set without a change in the score. In [`papers/neuron-development/`](../../papers/neuron-development/README.md), Pythia-70m's loss is within 1% of its final value at step 16,000. Yet its 86 VERIFIED poles correlate with their final form at a median of only 0.62 there. Adjacent checkpoints correlate at a median of 0.78 to 0.82. The paper's solution manifold is one possible account of this movement. The paper does not test it, and neither does the repo.
- **Sister neurons are not required by the loss.** The paper finds d_int90 nearly constant across network sizes, so extra weights add redundancy. A larger model thus has many solutions that do not reuse a smaller model's units. That is one reason a match across sizes is not a given. [`docs/analyses/sister-neurons.md`](../analyses/sister-neurons.md) finds sisters for about a fifth of Qwen2.5-0.5B's VERIFIED poles in Qwen2.5-1.5B. The paper's result is on MNIST FC networks, so this is a weak link only.
- **The paper's random rotation is in weight space, and the repo's null is in activation space.** The rotated-basis null in [`docs/analyses/null-calibration.md`](../analyses/null-calibration.md) rotates the MLP's hidden activations of Qwen2.5-0.5B. There, 18 of 30 native positive poles reach VERIFIED against 1 of 30 rotated units. The paper rotates the parameter vector instead, and it measures performance, not units. So its results are not a null for the repo's explanations.
- **The optimiser remark bears on the native-basis premise.** Section 2.1 says that Adam and RMSProp depend on the rotation of the weight basis, and SGD does not. An optimiser that scales each weight on its own thus favours the axes of the weights. That is one candidate reason, beside the elementwise nonlinearity, why a neuron basis could be privileged. The paper states the property but does not test its effect on units. The repo does not record each registry model's optimiser.
- **Shared machinery for memorisation is an interference question.** Table S2 shows fewer floats per label as the network memorises more random labels. The authors read this as shared machinery. This bears on #4782's interference strand, but the paper does not look inside the network.
- **The next paper in this strand reads language models.** #4852 asks for a review of the follow-up work on the intrinsic dimension of language model fine-tuning. That review, not this one, should test the claims on models like the registry's.
- **The open issues already cover the weight-change tests.** #5133 reads the rank and the per-pole norm of Δ = instruct minus base, against a null of shuffled poles. #5118 tests diagonal Fisher, and #5125 runs a sequence of narrow fine-tunes. This paper suggests one more null for #5133: a Haar-random rotation of Δ's neuron axis. It would show whether the change sits on neurons more than a spread update would be. That is a comment for #5133, not a new issue.
- **A question for the maintainer.** Should a weight fix be limited to a random or chosen low-dimensional subspace, so that it moves the VERIFIED poles less? That is a choice in the fix campaign. So this page records it as a question and opens no issue.

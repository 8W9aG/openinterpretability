# Gradient Episodic Memory for Continual Learning

**Citation.** David Lopez-Paz and Marc'Aurelio Ranzato (both Facebook Artificial Intelligence Research). "Gradient Episodic Memory for Continual Learning". *Advances in Neural Information Processing Systems 30* (NIPS 2017), Long Beach. [arXiv:1706.08840](https://arxiv.org/abs/1706.08840), version 6 of 13 September 2022 (version 1: 26 June 2017, version 5: 4 November 2017). The arXiv comment says "Published at NIPS 2017", and the venue agrees with the dates. Code: [facebookresearch/GradientEpisodicMemory](https://github.com/facebookresearch/GradientEpisodicMemory). The paper gives no funding statement. The acknowledgements thank Martin Arjovsky for the quadratic-program view of the method.

**Status.** Planned. #4782 (catastrophic interference, weight-space programming and LoRA) would use it. The maintainer designs any campaign to fix a model, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we downloaded the arXiv PDF (version 6) and read its text, extracted with `pdftotext`. We read in the source's own text the abstract, sections 1 to 6, Algorithm 1, Tables 1 to 3 and Appendix A (the hyper-parameter grids). From Appendix B we read the summary lines (final accuracy, backward and forward transfer) of all 15 evaluation matrices, and the full MNIST-permutation matrices. We skimmed the reference list. We did not see Figure 1 as an image: its bars and curves came back as axis labels only. No model read the source for us.

## Summary

**Claim.** A learner sees a stream of tasks, each example once. It can keep a small memory of examples from each past task. Before each update, it projects the gradient so that the loss on no past task's memory goes up. This limits forgetting and still lets learning on a new task improve old tasks.

**Method.**

- **The setting.** The learner sees triplets of input, task descriptor and target, one task after another, each example once. The descriptors are integers here. The authors leave rich descriptors and zero-shot learning to future work.
- **Three metrics.** R is a T × T matrix: R_i,j is the test accuracy on task j after the learner finishes task i. ACC is the mean of the last row. Backward transfer (BWT) is the mean of R_T,i − R_i,i over the first T − 1 tasks. Forward transfer (FWT) is the mean of R_i−1,i minus the accuracy at random initialisation. A large negative BWT is forgetting.
- **The episodic memory.** A budget of M examples splits into m = M/T for each task. The memory keeps the last m examples of each task.
- **The constraint.** The update minimises the loss on the current example, subject to no increase of the loss on any past task's memory (Equation 6). The method assumes that the loss is locally linear and that the memory is representative. So the constraint becomes a dot product: the update g must satisfy ⟨g, g_k⟩ ≥ 0 for each past task's memory gradient g_k (Equation 7).
- **The projection.** If a constraint fails, the update becomes the closest gradient, in squared ℓ2 distance, that meets all of them (Equation 8). The dual of this quadratic program has one variable for each past task, not one for each parameter (Equation 11). A small constant γ added to the dual solution biases the update towards positive backward transfer.
- **Data.** Twenty tasks in each benchmark. MNIST permutations: each task applies a fixed pixel permutation. MNIST rotations: each task rotates the digits by a fixed angle between 0 and 180 degrees. Incremental CIFAR-100: each task has 5 classes not in any other task. MNIST tasks have 1,000 training examples, CIFAR tasks 2,500.
- **Networks.** For MNIST, a fully connected network with two hidden layers of 100 ReLUs. For CIFAR-100, a ResNet18 with a third of the feature maps, and a separate linear classifier for each task. Plain SGD, batches of 10.
- **Baselines.** One network for all tasks ("single"). One smaller network for each task ("independent"). One input layer for each task ("multimodal", MNIST only). Also EWC, and iCARL (CIFAR only). GEM, iCARL and EWC have the architecture of "single" plus episodic memory.
- **Selection.** A grid search set every hyper-parameter (Appendix A), and the paper reports the best results for each model. GEM's grid has one memory size, 5,120, for all three benchmarks. γ was 0.5 for all three.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| MNIST permutations, final ACC / BWT / FWT (Appendix B.1) | single 0.6018 / −0.1980 / 0.0093. independent 0.4523 / 0 / 0. multimodal 0.7561 / −0.0275 / 0.0059. EWC 0.6185 / −0.1653 / 0.0054. GEM 0.8260 / 0.0247 / 0.0088 |
| MNIST rotations, final ACC / BWT / FWT (Appendix B.2) | single 0.5307 / −0.0896 / 0.4254. independent 0.6241 / 0 / 0. multimodal 0.7585 / −0.0243 / 0.0177. EWC 0.5461 / −0.2047 / 0.5524. GEM 0.8607 / 0.0048 / 0.6647 |
| CIFAR-100, final ACC / BWT / FWT (Appendix B.3) | single 0.4631 / −0.1226 / −0.0006. independent 0.4235 / 0 / 0. iCARL 0.5462 / −0.0830 / 0. EWC 0.4984 / −0.0799 / −0.0077. GEM 0.6783 / 0.0042 / −0.0078 |
| CPU training time, seconds (Table 1) | Permutations: single 11, independent 11, multimodal 14, EWC 179, GEM 77. Rotations: 11, 16, 13, 169 and 135 |
| CIFAR-100 ACC against total memory size 200 / 1,280 / 2,560 / 5,120 (Table 2) | GEM 0.487 / 0.579 / 0.633 / 0.654. iCARL 0.436 / 0.494 / 0.500 / 0.508 |
| MNIST rotations ACC/BWT after 1, 2 and 5 epochs per task (Table 3) | single on shuffled data 0.83/−0.00, 0.87/−0.00, 0.89/−0.00. single 0.53/−0.08, 0.49/−0.25, 0.43/−0.40. EWC 0.55/−0.19, 0.59/−0.17, 0.61/−0.11. GEM 0.86/+0.05, 0.88/+0.02, 0.89/−0.02 |
| The cost (section 4.4) | The dual has T = 20 variables, against p = 1,109,240 parameters for CIFAR-100. The bottleneck is one gradient for each past task at each step |

The authors read Table 3 as GEM matching the "oracle performance upper-bound" of training on shuffled data from all tasks.

## Where the claims go beyond the evidence

- **One run, no intervals and no test.** Each number comes from one run. The paper gives no seeds, no spread and no statistical test, but says that GEM "performs significantly better" than EWC. On MNIST permutations the BWT gap to the next method, multimodal, is 0.05.
- **Hyper-parameters chosen by the reported results.** The paper reports the best result of the grid search for each model. The paper names no validation set, so the choice may use the test data. γ came from the same search.
- **The memory is a quarter of the data.** On MNIST, 5,120 memories over 20 tasks is 256 for each task, out of 1,000 training examples. The paper says that EWC also has an episodic memory, but it gives neither its size nor its use. So we cannot tell whether the comparison is at equal access to the old data. Table 2 tests the memory size on CIFAR-100 only.
- **The figures in the text and the appendix disagree.** Table 3 gives GEM's one-epoch BWT on rotations as +0.05, but Appendix B.2.5 gives 0.0048. Table 2 gives GEM 0.654 and iCARL 0.508 at memory 5,120. Appendix B.3 gives 0.6783 and 0.5462, and the grid chose memory 5,120 for both. The paper does not say which runs differ.
- **The "upper bound" is beaten.** After one epoch, GEM's 0.86 is above the 0.83 of shuffled training. So shuffled training at that budget is not an upper bound, and "matches" is a claim about the 5-epoch column only.
- **Small networks, small images and task labels at test time.** The MNIST networks have 200 hidden units. CIFAR-100 uses a separate classifier for each task, and every test query names its task. The forward-transfer claim also needs care. Rotations give a large FWT to every method, because the tasks are near copies. On CIFAR-100, GEM's FWT is −0.0078, not "negligible or positive" as section 4.4 says.
- **The constraint is first-order.** The authors state that the dot-product test holds where the loss is locally linear and the memory is representative. The learning rate chosen for GEM is 0.1, and the paper does not measure whether a projected step still raised a memory's loss.

The authors name three limits themselves. GEM makes no use of structured task descriptors, and its memory policy is simple. Each step needs one backward pass for each task.

## Use in this repo

- **Nothing in the repo uses this method today.** A search of the code, docs, papers and skills finds no gradient projection, episodic memory or GEM. The repo has never trained a model. The sister reviews of EWC ([`elastic-weight-consolidation.md`](elastic-weight-consolidation.md)) and synaptic intelligence ([`synaptic-intelligence.md`](synaptic-intelligence.md)) report the same.
- **The R matrix already exists for explanations.** GEM's matrix scores every task after each stage of training. [`papers/neuron-development/`](../../papers/neuron-development/README.md) does the same for 86 VERIFIED Pythia-70m poles at 20 checkpoints, with each pole's recorded probes. 51 explanations pass at step 16,000, but 13 of those fail again later. In the paper's terms that is negative backward transfer on explanations, with no new task at all. Ordinary training is the change here, and the poles' median correlation with their final form is only 0.62 at that step.
- **A zero BWT on outputs need not keep an explanation.** GEM constrains only the loss on stored examples. A step can keep those losses and still move the units that carry them. The neuron-development poles move at a median adjacent-checkpoint correlation of 0.78 to 0.82 while the loss is nearly flat. So a GEM-style constraint on outputs is no evidence that a pole's explanation survives. A check would run the recorded probes again. This is our inference. The paper never looks at a hidden unit.
- **The constraint names the test for a fix.** GEM's rule is that an update must not raise the loss on the stored examples of earlier tasks. For pythia-70m's " days" error (#4781, #5015), the stored examples would be the prompts where copying is right. [`fixing-models.md`](fixing-models.md) names those hard negatives in `docs/analyses/hallucination-shortlist.jsonl` as the test that a fix must pass. The dot product ⟨g, g_k⟩ is a cheap first look at whether two goals conflict. Our attribution patching already uses gradients at the activations (`openinterpretability/core/wiring.py`). How or whether a campaign uses any of this is the maintainer's decision.
- **Shared units are where gradients conflict.** A negative ⟨g, g_k⟩ means two tasks pull shared parameters in opposite directions. That is the superposition case, where one neuron serves several features (`openinterpretability/explain_groups.py`, the superposition records). The paper measures conflict over all parameters at once, not per unit. A per-neuron share of the dot product would show which units carry the conflict. This is our suggestion, and the paper does not test it.
- **A question for the maintainer.** Should a gradient-conflict measure between the " months" target and the right-copy prompts be recorded as evidence for #4781-style fixes? It needs gradients of the model's loss, not training. We record it as a question and open no issue, because the fix campaign is the maintainer's.
- **No single-neuron or explanation content.** The paper's units are whole networks and their accuracy. Nothing in it bears on our review protocol, probes or verdicts.

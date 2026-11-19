# Overcoming Catastrophic Forgetting with Hard Attention to the Task

**Citation.** Joan Serrà, Dídac Surís, Marius Miron and Alexandros Karatzoglou. All four are at Telefónica Research, Barcelona. Surís is also at Universitat Politècnica de Catalunya, and Miron at Universitat Pompeu Fabra. "Overcoming Catastrophic Forgetting with Hard Attention to the Task". *Proceedings of the 35th International Conference on Machine Learning*, Stockholm, PMLR 80, 2018. [arXiv:1801.01423](https://arxiv.org/abs/1801.01423), version 3 of 29 May 2018 (version 1: 4 January 2018). The arXiv comment says "Includes appendix. Accepted for ICML 2018", and the venue agrees with the dates. Code: [github.com/joansj/hat](https://github.com/joansj/hat). The paper gives no funding statement.

**Status.** Planned. #4782 (catastrophic interference, weight-space programming and LoRA) would use it. The maintainer designs any campaign to fix a model, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we downloaded the arXiv PDF (version 3) and read its text, extracted with `pdftotext`. We read all of it in the source's own text. That is the abstract, Sections 1 to 5 and Appendices A to E, with Tables 1 to 8. We did not see the figures as images. Figures 3 to 10 are curves, bars and a mask, so this page gives only what the text says of them. The extraction lost some notation in the equations, and we took their meaning from the text around them. No model read the source for us.

## Summary

**Claim.** A network can learn, for each task, an almost binary mask over its units. The masks of past tasks then block the gradient on the weights that those tasks use. The authors say this cuts forgetting "by 45 to 80%" against the best earlier method in four setups.

**Method.**

- **The mask.** Each layer l has a task embedding e for task t. The mask is a = σ(s · e), a sigmoid gate with a scale s. It multiplies the layer's output unit by unit. A "unit" is a fully connected neuron or a convolutional filter. The last layer has a hard-coded binary mask, which is the same as one output head for each task.
- **The gradient block.** After each task, the cumulative mask is the element-wise maximum of the new mask and the old cumulative mask. On a later task, Equation 2 scales the gradient on weight (i, j). The factor is 1 − min(cumulative mask of output unit i, cumulative mask of input unit j). So a weight between two units that a past task uses does not change.
- **Annealing and compensation.** During each epoch, s rises linearly from 1/s_max to s_max (Equation 3). At test time s = s_max, so the gate is close to a step. The authors found that the embeddings learned slowly under this schedule. A compensation rescales their gradient to undo the effect of the annealed sigmoid.
- **Sparsity.** A loss term R adds an L1 penalty on the masks of units that past tasks do not use (Equations 4 and 5). Its weight c sets how much capacity each task takes. The two hyperparameters are s_max (stability) and c (compactness).
- **Main setup.** Eight image data sets, each one task, in random order. They are CIFAR10, CIFAR100, FaceScrub, FashionMNIST, NotMNIST, MNIST, SVHN and TrafficSigns. All inputs are 32 × 32 × 3. The network is AlexNet-like: 3 convolutional layers and 2 fully connected layers of 2,048 units. Each method has 7.1 M parameters. Training is plain SGD, learning rate 0.05, batch 64, at most 200 epochs.
- **The metric.** The forgetting ratio ρ (Equation 6) measures accuracy on a past task after task t. It puts a random classifier at ρ ≈ −1 and a multitask network at ρ ≈ 0. The headline is its average over the tasks seen so far.
- **Baselines.** SGD with dropout, SGD with all layers frozen except the last (SGD-F), EWC, IMM (mean and mode), LWF, LFL, PathNet and PNN. A grid search on one seed's task sequence chose each method's hyperparameters. Each method ran 10 seeds, which set the task order, data split, batch order and weights.
- **Other setups (Appendix C).** Incremental CIFAR: 10 tasks of 2 CIFAR10 or 20 CIFAR100 classes. Permuted MNIST: 10 tasks, with two-layer networks of 100, 500 and 2,000 hidden units. Split MNIST: digits 0–4 and 5–9, with the network of Lee et al. (2017).

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Eight tasks, after task 2 (Table 1) | HAT ρ≤2 = −0.02 (s.d. 0.03). Best baseline EWC −0.08 (0.06). The authors call this a 75% cut in forgetting |
| Eight tasks, after task 8 (Table 1) | HAT ρ≤8 = −0.06 (0.01). Best baseline PNN −0.11 (0.01), then PathNet −0.17 (0.23), EWC −0.25 (0.03), SGD −0.66 (0.03). The authors call this a 45% cut |
| Eight tasks, all t (Table 3) | HAT falls from −0.01 at t = 1 to −0.06 at t = 8. PNN starts at −0.10 at t = 1. The authors say its fixed share of capacity for each task causes this |
| Incremental CIFAR, 10 tasks (Table 6) | HAT ρ≤10 = −0.09 (0.01). Best baseline EWC −0.18 (0.06). The authors call this a 55% cut |
| Permuted MNIST, 10 tasks (Table 7) | HAT large (5.8 M parameters) 98.6% mean accuracy. SI 97.1% and EWC 96.9% at 5.8 M. HAT small (0.1 M) 91.6%, SI 86.0% at 0.1 M |
| Split MNIST, 2 tasks (Table 8) | HAT 99.0% (s.d. 0.0). CAB 94.9% (0.3), IMM-mode 94.1% (0.3), SGD 71.3% (1.5). All at 1.9 M parameters |
| Hyperparameters (Figure 4, seed 0) | "Any variation" within s_max ∈ [25, 800] and c ∈ [0.1, 2.5] "results in reasonable performance". The default is s_max = 400, c = 0.75 |
| Compression of one task (Table 4) | With c = 1.5, HAT keeps 1.2% of the weights on MNIST (99.5% to 99.4%) and 21.4% on CIFAR100 (52.7% to 49.1%) |
| Training time, eight tasks (Table 5) | HAT 2.2 h in total on one Titan X, against 1.4 h for SGD and 3.9 h for EWC |

## Where the claims go beyond the evidence

- **The network needs the task's identity at test time.** The mask comes from the task's embedding, and the output head is one for each task. So the network must know which task an input belongs to. This is the multi-head setting that the authors cite as routine. It says nothing about one shared output, as in a language model.
- **Image classification only.** The authors say so in the conclusion. All networks are small convolutional or fully connected ones. The paper does not test its hope that HAT applies "beyond the catastrophic forgetting problem".
- **Hyperparameters were chosen on one of the test sequences.** The grid search used "a task sequence determined by a single seed". The text does not say that this seed is outside the 10 reported ones. Figure 4, the sensitivity study, also uses seed 0 only.
- **Most literature numbers were not re-run.** In Tables 7 and 8, the authors copy several baseline scores from other papers, as the superscripts mark. Their settings may differ from HAT's. The main eight-task study did re-run all baselines with the same seeds.
- **No significance tests.** The tables give standard deviations over 10 runs. The "45 to 80%" cuts mix two measures: forgetting ratio in two setups and error rate in two others. The 75% cut after task 2 rests on −0.02 against −0.08, with standard deviations of 0.03 and 0.06.
- **A standard deviation of 0.0.** Split MNIST gives HAT 99.0% with a standard deviation of 0.0 over 10 runs. That may be rounding, but the paper does not say.
- **A small inconsistency.** Table 2 lists SVHN with 100 classes, but SVHN has 10 digit classes.

The authors give credit to PackNet and DEN, which appeared while they wrote HAT. Appendix D reports the variants that did worse, and says it is not a formal set of results.

## Use in this repo

- **Nothing in the repo uses this method today.** A search of the code, docs, papers and skills finds no hard attention mask, PackNet or per-unit gradient block. The sister reviews in this batch are EWC ([`elastic-weight-consolidation.md`](elastic-weight-consolidation.md)), synaptic intelligence ([`synaptic-intelligence.md`](synaptic-intelligence.md)) and PackNet (#4817).
- **HAT's unit is our unit.** HAT protects whole neurons and filters, not single weights. Our registry explains one MLP neuron pole at a time. So HAT is the closest of these methods to our grain. Its mask multiplies a unit's output, which is the same operation as our ablations of a neuron. EWC and SI score single weights instead.
- **The gradient block is a form of gradient routing.** Equation 2 stops the gradient on weights between units that past tasks use. [`gradient-routing.md`](gradient-routing.md) masks gradients by data point to place a capability in a chosen region. HAT masks them by task to keep a region still. Both choose where training may change the network before it trains.
- **A frozen pole keeps its weights, not its explanation.** Equation 2 freezes a weight only when a past task's mask holds both of its units. A neuron's input weights from unmasked units can still change. So its inputs can change, and its activation can change with them. [`papers/neuron-development/`](../../papers/neuron-development/README.md) shows how far poles move with no mask at all. At step 16,000 the 86 VERIFIED Pythia-70m poles correlate with their final form at a median of 0.62. Adjacent checkpoints correlate at a median of 0.78 to 0.82. This is our inference. The paper measures task accuracy, not what a unit does.
- **Instruction tuning moved most poles little, with no mask.** [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) found 38 of 43 hypotheses keep on Qwen2.5-1.5B-Instruct (88%), at a median activation correlation of 0.983. On Qwen2.5-0.5B, 252 of 311 keep (81%), at a median of 0.974. HAT's problem is that SGD forgets fast on new tasks. That fine-tune did not show it at the level of single poles.
- **Weight reuse between tasks is the shared-unit question.** Figure 6 reports the share of one task's weights that later tasks reuse. In the repo, superposition records (`openinterpretability/explain_groups.py`) mark units that several features share. Those are the units where a mask for one behaviour would also block another. [`complementary-learning-systems.md`](complementary-learning-systems.md) makes the same point for focused learning.
- **Evidence for a fix of #4781, not a design.** A fix of pythia-70m's " days" error must keep the prompts where copying is right (see [`fixing-models.md`](fixing-models.md)). HAT's answer is to freeze the units that the old behaviour uses. It needs a task label at test time, and a language model has none. So HAT does not carry over as it stands.
- **A question for the maintainer.** Could a VERIFIED answer route supply HAT's mask, with the route's units frozen or left free in a weight fix? That is a choice in the fix campaign, so this page records it as a question and opens no issue.
- **No single-neuron or explanation content.** The paper monitors which units each task uses, but it never says what a unit does. It has no probes, explanations or random baselines for units. Nothing in it bears on our review protocol or our verdicts.

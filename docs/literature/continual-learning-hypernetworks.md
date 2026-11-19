# Continual Learning with Hypernetworks

**Citation.** Johannes von Oswald, Christian Henning, Benjamin F. Grewe and João Sacramento, Institute of Neuroinformatics, University of Zürich and ETH Zürich. Von Oswald and Henning are equal contributors. "Continual learning with hypernetworks". International Conference on Learning Representations (ICLR) 2020. arXiv 1906.00695 [cs.LG]: v1 of 3 June 2019, v2 of 10 October 2019, v3 of 12 February 2020 and v4 of 11 April 2022. We read v4. [Paper](https://arxiv.org/abs/1906.00695). Code: [chrhenning/hypercl](https://github.com/chrhenning/hypercl). The data are the public MNIST, CIFAR-10 and CIFAR-100 sets. Funding: the Swiss National Science Foundation (CRSII5-173721), ETH project funding (ETH-20 19-01) and the Swiss Data Science Center (C17-18, P18-03). The paper gives no statement about tools or help from an LLM. The ICLR 2020 venue agrees with the v3 date and with the arXiv comment.

**Status.** Planned. #4782 would cite it in its strand on editing, merging and task vectors, and #5125 tests the question that it bears on. No code, skill or protocol in this repo depends on it. The maintainer designs any campaign to fix a model, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we downloaded the v4 PDF from arXiv with `curl`, and converted it with `pdftotext`. The reader was this page's author, a model (Claude Opus 5.5). We read all 28 pages in the source's own text. That is the abstract, Sections 1 to 5, the references and Appendices A to F. We read Tables 1 to 6 as extracted text. We did not see the figures as images. For Figures 2 to 5 and A2 to A8, we have the captions and the text about them. We also have the few labels that the extraction kept. We did not open the linked code.

## Summary

**Claim.** A network that learns tasks one after another forgets the old ones. The authors move the problem one level up. A task-conditioned hypernetwork makes all the weights of a target network from a learned task embedding. To keep an old task, the hypernetwork only has to keep making the same weights for that task's embedding. A simple regularizer does this, with no stored data. The authors report results at the state of the art on the standard benchmarks, little forgetting over 100 tasks, and transfer to later tasks.

**Method.**

- **The model (Section 2.1).** A hypernetwork `f_h(e, Θ_h)` maps a task embedding `e` to the target network's weights. Each task has its own embedding, which trains by backpropagation with `Θ_h`. After a task, its embedding is frozen and stored.
- **The regularizer (Equation 2).** While the model learns task T, a penalty keeps the hypernetwork's output for each old embedding close to its output before task T. The strength is `β_output`. A one-step look-ahead `ΔΘ_h` from Adam enters the penalty. The penalty needs a frozen copy of `Θ_h` and the old embeddings. It needs no old data.
- **Chunking (Section 2.2).** A smaller hypernetwork makes the weights in chunks. A learned chunk embedding, shared by all tasks, picks the chunk. This lets the hypernetwork have fewer trainable parameters than the target network, which the authors call the compressive regime. Appendix E proves that a chunked network can approximate any continuous function on a compact set.
- **Unknown task identity (Section 2.3).** HNET+ENT picks the embedding whose output has the lowest entropy. HNET+R protects a replay VAE's decoder with a hypernetwork and trains the classifier on replayed data. HNET+TIR adds a task-inference classifier trained on the replayed data.
- **Scenarios.** In CL1 the system receives the task identity. In CL2 a single output head must solve all tasks, and the identity is not given. In CL3 the system must infer the identity.
- **Data and target networks.** A 1D polynomial regression toy problem. Permuted MNIST with 10, 50, 100 and 250 tasks, and split MNIST with five binary tasks, on fully connected networks. Split CIFAR-10/100 on a ResNet-32 and on the smaller network of Zenke et al. (2017).
- **Baselines.** Online EWC, synaptic intelligence (SI) and deep generative replay with distillation (DGR+distill). The MNIST baseline numbers are from van de Ven and Tolias (2019). HAT is in Appendix D. Fine-tuning and training each task from scratch are the lower references, and a replay upper bound (RPL-UB) is in Table 2.
- **Metrics.** Test accuracy on each task after all tasks ("final"), and right after the model learns that task ("during"). Tables give the mean ± SEM over 20 seeds. Figures 3 and A2 to A5 use 5 seeds.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Table 1, permuted MNIST, 10 tasks, CL1, task-averaged accuracy (± SEM, n = 20) | EWC 95.96, SI 94.75, DGR 97.51, HNET 97.57, HNET+R 97.87 |
| Table 1, same, CL3 | EWC 33.88, SI 29.31, DGR 96.38, HNET+ENT 91.75, HNET+TIR 97.59, HNET+R 97.76 |
| Table 1, split MNIST, CL2 | EWC 64.32, SI 65.36, DGR 96.83, HNET+ENT 87.01, HNET+TIR 94.43, HNET+R 98.00 |
| Table 1, split MNIST, CL3 | EWC 19.96, SI 19.99, DGR 91.79, HNET+ENT 69.48, HNET+TIR 89.59, HNET+R 95.30 |
| Table 4, permuted MNIST, 100 tasks, HNET+TIR (± SEM, n = 5) | CL1 96.12 during and 96.18 after. CL2 95.97 after, CL3 96.00 after |
| Table 5, split CIFAR-10/100, CL1, during against after (n = 5) | ResNet-32 82.36 against 82.34. Zenke et al.'s network 74.75 against 71.29 |
| Appendix D, split CIFAR-10/100 on Zenke et al.'s network | SI 73.85% task-averaged, HNET 71.29% ± 0.32% |
| Table 6, permuted MNIST, HAT against HNET | Hidden layers 100, 100: 91.6 against 95.92. 100 tasks at 1000, 1000: 86.04 against 94.98 |
| Appendix D, permuted MNIST, 250 tasks, regularizer on at most 32 random old tasks per step | 94.19 ± 0.16 final, against 95.54 ± 0.05 during, 5 seeds |
| Appendix C, hypernetwork size against target size | Permuted MNIST VAE: 2,211,907 against 2,227,024. ResNet-32: 457,336 against 468,540 |
| Appendix D, fine-tuning without protection, permuted MNIST, 100 tasks | A fine-tuned target network declines gradually. A hypernetwork with `β_output` = 0 falls almost at once to chance level |

The text also reports these results without a table number. Figure A7 says that split MNIST stays in the high 90s down to a compression ratio near 1%. Figure 4 shows a 2D embedding space for split MNIST. A large region of it solves the first task, and that region holds the later tasks' embeddings.

## Where the claims go beyond the evidence

- **Small image benchmarks only.** Every result is on MNIST variants or CIFAR. The largest target network is a ResNet-32 of about 0.47 M weights. "Scalable and generic" in the conclusion rests on these settings.
- **The baseline numbers come from another paper.** The authors take the MNIST baselines from van de Ven and Tolias (2019). They say that those authors tuned each baseline for each scenario, while HNET used one setting for all. That favours the baselines. But the HNET search space was large and partly manual, and no text says which data split chose the settings.
- **The two MNIST tables disagree.** HNET+R appears in Table 1 and in Table 2 with different values and the same n. Split MNIST CL3 is 95.30 in Table 1 and 94.97 in Table 2. CL2 is 98.00 against 97.88. The text gives no reason.
- **The paper is inconsistent about architecture sensitivity.** Appendix C says that the hypernetwork architecture "seems to have a strong influence on the performance". Figure A7 says that performance stays stable over a large pool of architectures. The search grid in Appendix C also lists `β_output` = 0.005 twice.
- **No forgetting holds on ResNet-32 only.** The claim that hypernetworks "essentially eliminate" forgetting rests on 82.36 against 82.34. On the smaller CIFAR network, accuracy falls from 74.75 to 71.29, and SI does as well. The authors report both.
- **The fine-tuning reference on ResNet-32 differs in more than the penalty.** HNET stores batch-norm running statistics for each task. Fine-tuning keeps updating them. So part of fine-tuning's loss on CIFAR can come from those statistics, not from the weights.
- **Some comparisons use few seeds or different setups.** The baseline hyperparameter search of Figure A6 uses one seed. The HAT numbers from Serra et al. (2018) have no statistics, and the authors say that the two methods use different training setups.
- **The method needs task boundaries.** The authors say themselves that their methods need to know when a task ends. CL1 also needs the task identity at test time. HNET+ENT and HNET+TIR infer it, at a cost on split MNIST CL3: 69.48 and 89.59, against 95.30 for HNET+R.
- **Storage is not one embedding per task only.** Appendix A says that the regularizer also needs a frozen copy of the hypernetwork. The ResNet-32 runs store batch-norm statistics per task, which grow with the number of units.

The authors state several limits themselves. The regularizer allows only weak backward transfer. The cost of the regularizer grows with the number of tasks, and they test a random subset of old tasks to bound it. They give no constructive bound on the compression that a chunked hypernetwork can reach.

## Use in this repo

- **No configured model is task-conditioned.** GPT-2, Pythia and Qwen2.5 use one set of weights for all text, and no network writes them. The paper avoids interference by giving each task its own generated weights. A VERIFIED pole in a protected old task would keep by construction. So the method does not measure the question of #5125, whether an explanation keeps through shared weights.
- **Plain fine-tuning damaged a large target network gradually here as well.** In Appendix D, fine-tuning a target network on 100 permutations gives a gradual decline, not collapse. [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) finds the same kind of result for units without any penalty. 38 of 43 poles keep on Qwen2.5-1.5B (88%), and 252 of 311 on Qwen2.5-0.5B (81%). The settings differ, so this is agreement in direction only.
- **The paper's "during" and "after" columns are the design of #5125.** Tables 3 to 5 compare each task's accuracy right after it trained with its accuracy after all tasks. #5125 re-tests the VERIFIED poles after each fine-tune stage. The same split would separate a pole that fails at one stage from one that fails slowly over several. That is a suggestion for a comment on #5125, not a new issue.
- **Keep the values that a check reads.** The regularizer keeps the generated weights, not the target network's outputs on stored data. The authors argue that keeping a few points is easier than keeping many input-output pairs. For a pole, the points that matter are its record's probes, 12 to 20 hard negatives and 6 to 10 holdouts. An anchor on those activations is the same idea in activation space, near [`learning-without-forgetting.md`](learning-without-forgetting.md). Whether a fix for #4781 uses such an anchor is the maintainer's choice.
- **The regularizer pulls the same way on every output.** Appendix D says that an EWC-like weight on each output is possible, but the paper does not test it. So it gives no evidence on #5118 (Fisher importance) or #5133 (the weight change at a pole). The neighbouring pages [`elastic-weight-consolidation.md`](elastic-weight-consolidation.md) and [`synaptic-intelligence.md`](synaptic-intelligence.md) cover the weighted forms.
- **Shared weights still carry the interference, one level up.** Without the penalty the hypernetwork falls to chance almost at once, faster than the target network. So all tasks interfere inside `Θ_h`, and the penalty holds them apart. Figure 4 shows tasks that compete for regions of the embedding space. The paper never looks at single units of the hypernetwork or the target. So it adds nothing to the superposition records in `openinterpretability/explain_groups.py`.
- **A learned weight writer, with no unit content.** Like [`hypernetworks.md`](hypernetworks.md) and [`text-to-lora.md`](text-to-lora.md), the paper writes weights from a small learned input that no one reads. #3381 and #4471 ask for weights that a person or agent can read. A 24-number embedding that selects about 2 M weights is compact, but nothing in the paper shows that its parts line up with units. [`hard-attention-to-the-task.md`](hard-attention-to-the-task.md) reviews the per-neuron mask method that Appendix D compares with.
- **Late drift in Pythia is not what this method would stop.** [`papers/neuron-development/`](../../papers/neuron-development/README.md) finds that Pythia-70m's 86 VERIFIED poles keep changing after the loss settles. At step 16,000 their median correlation with the final form is 0.62. Adjacent checkpoints correlate at a median of 0.78 to 0.82. That is drift inside one task, pretraining, with no boundary. The authors say that their method needs task boundaries. So it has no handle on that drift.
- **No follow-up issue.** #5125, #5118 and #5133 already cover the tests that this paper bears on, and the paper adds no test of single units.
- **A question for the maintainer.** Should a fix for #4781 keep the activations of VERIFIED poles on their own probes, as this paper keeps generated weights? That is a choice in the fix campaign, so this page records it and opens no issue.
- **No single-neuron content.** The paper names no neuron, explanation, probe or random baseline for units. Its measures are task accuracies.

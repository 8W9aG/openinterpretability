# A Comprehensive Survey of Continual Learning: Theory, Method and Application

**Citation.** Liyuan Wang, Xingxing Zhang, Hang Su and Jun Zhu. "A Comprehensive Survey of Continual Learning: Theory, Method and Application". [arXiv:2302.00487](https://arxiv.org/abs/2302.00487) [cs.LG], version 3, 6 February 2024. Version 1 dates from 31 January 2023, and version 2 from 11 June 2023. All four authors are at the Department of Computer Science and Technology, Tsinghua University, Beijing. The paper also names its Institute for AI, BNRist Center, THBI Lab and Tsinghua-Bosch Joint Center for ML. Jun Zhu is the corresponding author. The arXiv comment says that "the concise version" is in IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). The issue gives the year as 2024. We did not see the TPAMI version, so we can't give its volume or pages, or say how it differs from version 3. The arXiv PDF still carries the template header "Journal of LaTeX Class Files, vol. 14, no. 8, August 2015". It is a survey, so it has no code or data. Funding: the National Key Research and Development Program of China, NSFC and BNRist. Tsinghua gave support through the Institute for Guo Qiang, the Tsinghua-OPPO Joint Research Center and its High Performance Computing Center. L.W. also had a Shuimu Tsinghua Scholar grant.

**Status.** Planned. #4782 (catastrophic interference, weight-space programming and LoRA) would use it. It is a map of the methods that the other #4782 pages review one at a time. It also bears on #5118 and #5125. The maintainer designs any campaign to fix a model, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we downloaded version 3 of the PDF (33 pages) from arXiv with `curl`, and extracted its text with `pdftotext`. A model (claude-opus-5-5) read it, not a person. The list below says what we read and how closely.

- **Read in full, in the source's own text:** the abstract and Sections 1 to 4. These are the introduction, the setup and metrics, the theory and the five method families. We also read Section 6.5 (natural language processing), Section 6.6 (beyond task performance), Section 7 (discussion and neuroscience), Section 8 (conclusion) and the acknowledgments.
- **Read through once, but kept out of this summary:** Section 5 and Sections 6.1 to 6.4. Section 5 is about visual classification: task-agnostic inference, few labels, and online and task-free learning. Sections 6.1 to 6.4 cover object detection, segmentation, conditional generation and reinforcement learning. These sections are mostly lists of method names, and none bears on language models or units.
- **Skimmed:** the 527 references. We looked up about 25 entries that this page names, and read no cited paper.
- **Not seen:** the nine figures, which we did not render. Text extraction garbled most equations, so we read each one through the text around it. Table 1's rows came out in the wrong order, and we matched them to their scenarios by their conditions.

The survey reports no experiments of its own, so the table below holds its framework and its theory, not results.

## Summary

**Claim.** The general aims of continual learning are a proper trade-off between stability and plasticity, and adequate generalisation within and between tasks, at low resource cost. The authors sort the methods into five families: regularisation, replay, optimisation, representation and architecture. They say that this taxonomy extends the usual ones.

**Method.** A survey of the literature to early 2024, with 527 references. It has these parts:

- **Setup (Section 2).** A task is a set of training samples from one distribution. The learner sees the tasks in sequence, with little or no access to old samples. Table 1 defines eight scenarios by what changes and what the learner knows: instance-incremental, domain-incremental, task-incremental, class-incremental, task-free, online, blurred-boundary and continual pre-training.
- **Metrics (Section 2.3).** Average accuracy and average incremental accuracy measure the whole sequence. Forgetting measure and backward transfer measure stability. Intransience and forward transfer measure plasticity. The survey also lists linear probes for "representation forgetting", the top eigenvalue of the Hessian for flatness, and storage and compute costs.
- **Theory (Section 3).** It casts continual learning as sequential Bayesian inference, where each posterior becomes the next task's prior. Two approximations follow: an online Laplace approximation, which gives the quadratic penalty of [EWC](elastic-weight-consolidation.md), and online variational inference. It then states a bound on how much an old task can lose. It also collects results on task similarity and on how hard the problem is.
- **Methods (Section 4).** Each family has sub-directions, listed below.
- **Applications (Sections 5 and 6).** Vision, reinforcement learning, language, federated learning, machine unlearning and robustness.
- **Discussion (Section 7).** Trends, links to diffusion models, foundation models, transformers and multi-modal models, and links to neuroscience.

**The five families (Section 4).** The other #4782 pages review many of these methods one at a time, so this list names them and links them.

| Family | Sub-directions | Methods with their own page here |
| --- | --- | --- |
| Regularisation | Weight regularisation: a quadratic penalty on each weight's change, scaled by its importance. Function regularisation: knowledge distillation from the old model | [EWC](elastic-weight-consolidation.md), [SI](synaptic-intelligence.md), [LwF](learning-without-forgetting.md) |
| Replay | Experience replay of stored samples, generative replay (pseudo-rehearsal) and feature replay | [GEM](gradient-episodic-memory.md), [rehearsal and pseudorehearsal](rehearsal-and-pseudorehearsal.md) |
| Optimisation | Gradient projection, meta-learning, and a loss-landscape view (flat minima, mode connectivity) | [OGD](orthogonal-gradient-descent.md), [GEM](gradient-episodic-memory.md) |
| Representation | Self-supervised learning, pre-training for later continual learning, and continual pre-training | none |
| Architecture | Parameter allocation (masks or pruning over a fixed or growing network), model decomposition, modular networks | [PackNet](packnet.md), [HAT](hard-attention-to-the-task.md), [progressive networks](progressive-neural-networks.md) |

The survey says that the families are closely linked. For example, "regularization and replay ultimately act to rectify the gradient directions in optimization".

**Findings, as the source reports them.** All are the survey's statements, or results that it cites. None is a new measurement.

| Setting | What the survey states |
| --- | --- |
| A second-order expansion around the optimum of task *i* (Eq. 19–20) | The loss on task *i* after a move Δθ to task *j*'s optimum is at most ½ λ_max ‖Δθ‖² above its minimum. λ_max is the top eigenvalue of task *i*'s Hessian. So flatter minima and nearer task optima forget less |
| Separate output heads, in the neural tangent kernel regime (Eq. 21) | "an increase in task similarity may lead to more forgetting", because two similar solutions become hard to tell apart |
| One shared solution for all tasks | Finding it is "NP-hard in general", because each new task makes the feasible region more "narrow and irregular". The survey cites Knoblauch et al. (2020) |
| SI, MAS and RWalk importance | All are "tantamount to an approximation of the FIM", the Fisher information matrix. The survey cites Benzing (2022) |
| Self-supervised learning and large-scale pre-training | Both "suffer from less catastrophic forgetting". Empirically both give representations that are more orthogonal, sparse and uniformly scattered, and wider loss basins |
| Pre-training for later continual learning | The benefit is larger with more pre-training data, a larger model and a contrastive loss |
| Network shape | Wider networks forget less, through more orthogonal and sparse gradients and lazier training. Batch normalisation biases its moments toward the current task. Dropout acts as a gate that makes pathways for each task |
| Feature replay | The "central challenge" is representation shift, which the survey calls "feature-level catastrophic forgetting" |
| Foundation models (Section 7.2) | Fine-tuning a foundation model "tends to forget pre-trained knowledge". The survey cites Lin et al. (2023) |
| Language (Section 6.5) | Language work shares the vision scenarios and methods. Its mark is heavy use of pre-trained transformers, so adapters, prompt tuning and instructions carry most methods |
| Neuroscience (Section 7.3) | The hippocampus compresses replay in time: "an experience of around 6 seconds can be compressed into 0.2 seconds". The survey maps complementary learning systems onto replay buffers and generative models |

The survey's trend (Section 7.1) is a move from stability alone toward plasticity and generalisation between tasks. Its main tools for that are flat minima, ensembles and pre-trained representations.

## Where the claims go beyond the evidence

- **"Comprehensive" covers mostly vision.** Section 5 uses visual classification as its only worked case. Language has one subsection (6.5), and it is a list of methods with no comparison. The authors say that current work "mainly focus[es] on visual classification".
- **No comparison of methods.** The survey reports no benchmark table. It says which methods exist and why, not which works better and by how much. A reader can't rank the five families from it.
- **The bounds rest on strong assumptions.** Eq. 20 assumes a second-order expansion and a zero gradient at the old optimum. Eq. 22 assumes that all tasks share one minimiser. The survey states each assumption, but its conclusions about flatness read wider than the bounds.
- **Cited results stand in for causes.** The survey cites work where pre-training and self-supervision "suffer from less catastrophic forgetting". It attributes this "empirically" to orthogonal, sparse representations. It does not say whether any cited work tests that cause by intervention.
- **The neuroscience mapping is by analogy.** Section 7.3 pairs each family with a brain mechanism. No section tests whether the pairing predicts anything.
- **Small errors in the text.** One sentence repeats "to accommodate" twice. Section 5.3 writes "TSCL", which seems to mean TFCL. The PDF keeps a 2015 template header.
- **Credit.** The authors define every scenario and metric formally, which makes the vocabulary easy to reuse. They state that the merits and limits of experience replay "remain largely open".

## Use in this repo

- **A second survey covers the experiments.** [De Lange et al. (2021)](continual-learning-survey.md) compares 11 of these methods under one protocol. This survey reports no experiments of its own.
- **The registry already measures one kind of forgetting, but per unit.** [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) tests each VERIFIED hypothesis again after instruction tuning. On Qwen2.5-1.5B, 38 of 43 keep (88%, 95% CI 76–95%), and activations correlate at a median r of 0.983. On Qwen2.5-0.5B, 252 of 311 keep (81%, 76–85%). In the survey's terms this is backward transfer, read for units through their own probes. The survey's nearest unit-level tool is a linear probe for representation forgetting.
- **The scale claim agrees in direction only.** The survey says that larger pre-trained models forget less. In `base-instruct.md` the larger model keeps more, 88% against 81%, but the two intervals overlap. The two models also have different poles and probe sets, so this is not a test of scale.
- **The Fisher result bears on #5118.** #5118 asks whether a pole's diagonal Fisher predicts whether it keeps after instruction tuning. The survey reports that SI, MAS and RWalk importance all approximate the Fisher. If that holds, one Fisher test speaks for that family too. Eq. 20 also puts the size of the weight move ‖Δθ‖ beside curvature. So the base-to-instruct weight change of each pole's rows may explain as much as its Fisher. That suggestion belongs on #5118 as a comment, and we open no new issue.
- **"Representation shift" is what the neuron-development paper measures.** In Pythia-70m's pretraining, [`papers/neuron-development/`](../../papers/neuron-development/README.md) follows 86 VERIFIED poles. At step 16,000 the loss is within 1% of its final value, but the median correlation of the poles with their final form is 0.62. Adjacent checkpoints correlate at a median of 0.78 to 0.82. Pretraining on one corpus is not a task sequence, so the survey's metrics don't apply directly. The nearest scenario in Table 1 is continual pre-training.
- **Sequential fine-tunes are #5125.** #5125 fine-tunes a base model and its instruct model on a sequence of narrow tasks, and tests the poles again after each stage. The survey's metrics give that analysis a vocabulary. Forgetting measure and backward transfer apply to a pole's pass or fail over stages as well as to accuracy.
- **Similar prompts may be the hardest to keep.** The survey's neural tangent kernel result says that more similar tasks can forget more. [`fixing-models.md`](fixing-models.md) names the prompts where copying is right as the test that a fix of the #4781 " days" error must pass. Those prompts are the most similar to the error. This is evidence for the maintainer's campaign, and it designs nothing.
- **Some methods act on neurons, but rate them by task loss.** NPC, UCL, AGS-CL and PackNet-style pruning pick "important neurons" by importance estimates, uncertainty or activation value. That importance is a share of task loss, not an explanation of what the neuron computes. The registry's poles could label such neurons, but the survey suggests no such use.
- **Interference between units has no unit-level account here.** The survey blames shared parameters for interference between tasks, and says that wider networks forget less through orthogonal, sparse gradients. It says nothing about superposition or about directions over several neurons, which [`openinterpretability/explain_groups.py`](../../openinterpretability/explain_groups.py) handles.
- **Nothing on explanations, probes of units or nulls.** The survey has no content on single-unit explanations, adversarial probes or random baselines. Its use here is as a map and a vocabulary for #4782, not as a method.
- **No follow-up issue.** The questions that the survey raises are already open as #5118 and #5125, or belong to the maintainer's fix campaign.

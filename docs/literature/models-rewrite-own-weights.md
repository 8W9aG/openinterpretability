# Language Models Can Rewrite Their Own Weights

**Citation.** Jiajun Fan, Neil He, Changyi Yang, Andy Mao, Xi Zhu, Chaoran Cheng, Kaiqu Liang, Shuaike Shen, Zetian Li, Ruihan Guo and Ge Liu. "Language Models Can Rewrite Their Own Weights". arXiv 2611.21885, v1 16 November 2026, the only version. [Paper](https://arxiv.org/abs/2611.21885). Preprint, no venue given. The HTML shows the logos of UIUC, CMU, Rutgers, Princeton and Stanford, but the text doesn't say which author is at which. The paper says that the task prompt "will be released with the code", and it links no code or data. It gives no funding statement and no statement about help from an LLM in the writing. The models that it studies are Qwen3.5 models. The link first given to us was on arxivb.org, another domain, so we used the arxiv.org page.

**Status.** Planned. #4782 (weight-space programming and interference) would use it. The maintainer designs the campaign to repair models. So this page reports evidence and records questions only.

**Read.** A web-fetch agent read the abstract page and the v1 HTML on 2026-11-19. It read the body text in full, with Appendices A to E and Algorithms 1 to 4. It also read the values of Tables 1 to 13 and the figure captions. The first fetch gave later sections as a summary only. The agent fetched them again by offset until it had the source's own text to the end. We did not see the figures as images, the PDF or the TeX source. Numbers that only a figure gives are not here.

## Summary

**Claim.** A 27B language model can find the cause of its own errors and edit its own weights to remove it. It uses no gradients and no training. The edit makes it more accurate on problems that it has not seen. The model can also repair a damaged copy of itself. Of four model sizes, only the largest gains.

**Method.**

- **Models.** One family, Qwen3.5 at 2B, 4B, 9B and 27B. Their layers mix Gated DeltaNet linear attention with full attention, and thinking is on.
- **What an edit is.** An edit replaces some named tensors of the weights with tensors of the same shape, written in bf16. The framework refuses an edit to the input embedding, the final normalisation or the output layer. It also refuses a whole tensor times a constant.
- **A session.** The unedited model answers 256 practice problems (64 in a repair session), and the model sees the graded answers. Then, each turn, it writes a rationale and one action:
  - Python code, on a CPU, with gradients off.
  - A generation pass, of the unedited model, the edited model or both on the same seeds.
  - A read of hidden states, MLP activations or log-probabilities on up to 256 texts.

  A session has 20 forward passes, 200 code runs and at most 160 turns. The code can't load a model, start a process or reach the network. The edit on disk at the end is the one that the model submits. The framework gives helper functions, such as removing a direction and taking a group-mean difference.
- **Second protocol.** Before the model answers SUBMIT or WITHDRAW, the framework tests the edit once on 256 reserved practice problems. There were eight such sessions of the 27B model, and each one starts from the base weights.
- **The 27B model's own edit, first protocol.** It weakens what a matrix writes along one unit direction u: W − αuuᵀW, with α = 0.05. It applies this to the attention and MLP output matrices of layer 8. u is the unit difference between the mean layer-8 hidden states of 20 answers that finish and 20 that run out of tokens. The model edited layers 56 and 48 first, and moved to layer 8 when those edits failed.
- **The 27B model's edits, second protocol.** Session 8 raises the time-step bias of all 48 linear-attention layers by 0.5. Session 2 subtracts 5 times the projection on one direction from layer 63's output matrices.
- **Repair of a damaged copy.** The framework adds Gaussian noise to one block of weights. The 27B model computes each output matrix's maximum against its median weight magnitude. It names layer 30's MLP output matrix, at 1,397 times its median. Then it clips each weight above 15 times its matrix's median, in that matrix and 19 output matrices of layers 0 to 10. That changes 0.013% of their weights.
- **Data and tests.** The practice and held-out problems are math problems from the DeepScaleR pool, with benchmark overlaps removed. A paired test compares the edited and unedited model on the same seeds, per problem. It uses a one-sided sign test, or a sign-flip permutation test over problems with 100,000 flips.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| 27B, second protocol, session 8, 490 held-out problems, 8 answers each, 16,384 tokens | 57.2% → 63.8%, +6.6 points, 95% interval [5.0, 8.1], p < 10⁻⁵ |
| 27B, second protocol, session 2, same test | 61.3%, +4.0 points, p < 10⁻⁵ |
| 27B, second protocol, session 1, same test | 57.8%, +0.5 points, p = 0.25 |
| 27B, second protocol, sessions 3 to 7 | Withdrawn, so no edit |
| 27B, first protocol, its submitted layer-8 edit, held-out test | 57.2% → 58.3%, +1.1 points, p = 0.038 |
| 27B, first protocol, the 39 practice problems used to find the edit | 22.4% → 29.6%, +7.3 points, p = 0.009 |
| 27B repair, 200 held-out problems, 1 answer each | Damaged 0.0%, repaired 33.0%, base 58.5%. 66 problems turn right and none turn wrong |
| 27B repair of the named matrix only, same problems | 55.0% |
| 27B repair, six image benchmarks, 100 problems each | Base 84.3%, damaged 7.3%, repaired 70.2%, pooled |
| 27B repair, rows of the repaired matrix against the original | Cosine similarity 0.002, as for random values. Norms: original 3.85, damaged 825, repaired 39.7 |
| 2B | Makes no edit |
| 4B, own weights | Withdraws its edit. Held-out accuracy would go from 41.0% to 0.0% |
| 9B, own weights | Submits an edit. Held-out accuracy goes from 47.5% to 0.0% |
| 2B, 4B and 9B repair | 0.0% |

## Where the claims go beyond the evidence

- **The 6.6-point edit passed a check that the framework ran.** The abstract calls the session 8 edit "its own edit". That session's edit passed the framework's test on 256 reserved problems before it was submitted. The edit that the model submitted with no check, under the first protocol, gains 1.1 points (p = 0.038).
- **One of eight sessions, with no correction.** Two of the eight second-protocol sessions are significant. The 6.6-point result is the largest of them. The paper reports all eight, but it adjusts none of the p-values for the eight comparisons.
- **"Accuracy only rises" has no test.** The abstract says that over the first eight sessions the accuracy only rises. Table 13 gives point estimates of each session's model against the base. It doesn't compare successive models.
- **No random edit.** No session tests a random direction at the same layer and strength as the layer-8 edit. No session clips at random either. So the tests don't show that the chosen direction matters more than any small edit would. The 4B and 9B models' noise edits are their own failed attempts, not a control.
- **Side effects are measured on math alone.** The edits of the model's own weights are tested on held-out math problems only. The paper gives no general text benchmark and no perplexity. The image benchmarks test the repair only.
- **One model family, and one session per size.** Every model is a Qwen3.5. Each size has one complete session on its own weights. The 27B model has two, and three more sessions stopped early and were not analysed. "Only the largest gains" rests on these four runs.
- **The damage is made to be found.** The authors say so themselves: a damaged block "is a failure with a single cause by construction". The repaired weights are not the original weights, since the rule only caps the random values.
- **An unexplained gap.** The full 20-matrix repair (33.0%) scores below the repair of the named matrix alone (55.0%), on the same 200 problems. The paper doesn't discuss it.
- **Small inconsistencies.** One figure caption says the image test takes the first 100 problems. Appendix E.6 says they are 100 of 300, drawn with one seed. Generation is not exactly reproducible, as the authors say.

The authors state some limits themselves. They report the repeats between practice and held-out problems, and they show that significance holds without them. They recommend that a person check each edit that a model makes to itself before it is used.

## Use in this repo

- **The edits are directions and tensors, not units.** The model weakens one direction of a layer's output, or clips the outliers of whole matrices. No edit names a neuron or a head. So the paper says nothing about whether a VERIFIED unit explains what an edit changes. A test of that question would need the edit expressed in the registry's units.
- **Its direction is a difference of group means.** The layer-8 direction is the mean hidden state of answers that finish, minus that of answers that run out of tokens. #5074 asks the same kind of question of the #4781 route: do its units separate wrong-copy prompts from right-copy prompts? The paper reports no probe accuracy for its direction. Our version should report how well the route's units separate the two groups.
- **Removing a direction is a weight edit of what `remove_direction` does to activations.** `explain_groups.remove_direction` removes a superposition's direction from the activations where it is positive. W − αuuᵀW removes a share of a direction from what a matrix writes, on every token. A route's units could give u, so an edit like this one could test whether removing a route's direction changes the answer. Whether the campaign should edit weights or activations is the maintainer's question.
- **The missing random control is #5069's control.** #5069 asks for random units of the same kinds and layers as a control for the steering samples. This paper shows why the control matters for a weight edit too. With no random direction at layer 8, a gain of 1.1 points can't be attributed to the direction.
- **The outlier search resembles our hub norm notes.** The repair found the damaged matrix by its maximum weight against its median, at 1,397 times. `hubs.py` reports a hub center's output-weight norm against its layer's median and its percentile. Both use a unit's size against its layer as evidence of an artefact. Here the artefact was made on purpose, so the method's success is easier than in a real model.
- **What it adds to the synthesis's table of tests.** `docs/literature/fixing-models.md` lists the tests that papers used to show that an edit has no collateral damage. This paper adds one design and shows one gap. The design is a paired test on the same seeds, with a check on reserved problems before the edit is submitted. The gap is that it tests side effects on the task's own domain only.
- **No single-neuron content.** The paper reads hidden states and finds a loop with a period of about 32 tokens. It gives no unit explanations and no probes, so nothing in it bears on a pole's verdict.

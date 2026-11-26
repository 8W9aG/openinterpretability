# Parasite Features: Causal Abstraction Through Spurious Pathways

**Citation.** Denis Sutter, Qing Yao, Sasha Boguraev, Julian Minder, Antonio Di Maio, Thomas Hofmann, Kyle Mahowald and Tiago Pimentel. Two of them made an equal contribution. The HTML's author markup is scrambled, so the mapping of authors to affiliations is uncertain. The affiliations include ETH Zurich, UT Austin and EPFL. "Parasite Features: Causal Abstraction Through Spurious Pathways". [arXiv:2611.26711](https://arxiv.org/abs/2611.26711) [cs.LG], version 1 of 19 November 2026, the only version. The arXiv comment gives the venue as NeurIPS 2026. The licence is CC BY 4.0. The code is at [github.com/dounick/parasite-features](https://github.com/dounick/parasite-features). The Swiss National Science Foundation, the US National Science Foundation, Coefficient Giving and a UT Austin fellowship fund the work. The paper makes no statement about the use of AI tools.

**Status.** Planned. #3229 would compare a pole's interchange intervention with a learned DAS direction for the same variable. This paper shows that such a learned direction can pass for the wrong reason, so #3229 needs its control. No verdict of the registry rests on a learned alignment map, so no protocol rule changes.

**Read.** On 2026-11-26 a `web-fetch` subagent read the abstract page, the whole HTML of version 1 in four parts and the code's README. It read them for this page's author, a model (Claude Opus 5.5). It had the source's own text for Sections 1 to 10 and Appendices B to E, with Table 2. We saw no figure as an image, so the values of Figures 2 to 5 are missing except where the text states them. We did not read the PDF or the code.

## Summary

**Claim.** A learned alignment map, such as Distributed Alignment Search (DAS), can join two unrelated directions of a network into one apparent causal variable. One direction tells the inputs apart, and the other steers the output. The joint feature passes the intervention tests of causal abstraction, although the model does not use it. The authors warn that claims "that rest on high intervention accuracy alone should thus be treated with care".

**Method.**

- **Causal abstraction.** An algorithm with an intermediate variable explains a network when some alignment map makes interventions on the network match interventions on the algorithm. The intervention swaps the variable's value from a source input into a base input.
- **Directions.** A *discriminative direction* separates two classes of hidden states by a margin. A *steering direction* moves the output true or false when the hidden state is pushed far along it. A parasite feature is a learned map that combines the two to pass the intervention test.
- **Theorems.** Theorem 1 says that an invertible linear map can make any such pair pass, under conditions on the margins and spreads. It holds "even if directions ... are orthogonal". Theorem 2 gives stricter conditions for a rotation, the map that DAS classically learns. Both are sufficient conditions, not necessary ones.
- **Maps.** The main map is a rotation of the hidden state followed by an invertible linear map on a 2-dimensional subspace. A rotation alone is learned in one dimension.
- **Tasks.** *Parasitic Agreement* in Pythia-1b: a constructed task in which a subject's gender decides the verb's number, which the model does not solve. *Carry-One* in Llama3.1-8b: the carry of two-digit addition, at layer 20. A sentiment task and a faithful subject-verb agreement task in the appendices and Section 7.2. GPT2-Medium and Qwen2-1.5B in Appendix E.
- **Diagnostics.** Intervention landscapes over the two directions, and the angle between the input and output directions. Also an ablation of the found subspace against random subspaces of the same size.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Carry-One, Llama3.1-8b, clean accuracy (Section 7.1) | 99.1% |
| Same, with the learned 2D subspace ablated | 98.8% |
| Same, with a random 2D subspace ablated, 10 trials | 99.12% ± 0.13 |
| Carry-One ± n steering, n = 1 to 10 (Appendix A) | Every n aligns with at least 96.6% accuracy |
| Sentiment, Llama3.1-8b layer 22 (Appendix A) | Odds of 3.6 on held-out interventions across classes |
| Faithful subject-verb agreement, layer 2 (Section 7.2) | Odds 4.3, with the landscape of a faithful feature |
| Appendix E, Table 2, best odds | Qwen2-1.5B: 2.83 for rotation and linear, 2.03 for rotation. GPT2-Medium: 2.53 and 2.46 |

## Where the claims go beyond the evidence

**One natural case.** Carry-One is the only task in which the model does the task and the feature is shown to be unused. The model cannot solve Parasitic Agreement, by construction. So that task shows that the map can pass, not that DAS finds parasites in practice.

**No seeds or intervals.** The paper reports no seeds and no repeat runs. Parasitic Agreement has 50 held-out pairs. The only spread is ± 0.13 on the random-subspace control, and the paper does not say whether it is a standard deviation or an error.

**Ablation is evidence, not proof.** The authors say so themselves: "a small accuracy drop alone does not fully establish the subspace is unused". A model could also route around a lost subspace.

**The diagnostics are not tested as tests.** Orthogonality of the two directions is offered as "one such signal", and the landscapes as a diagnosis. Neither has a false-positive rate measured over many faithful and parasitic features.

**Probes and SAEs are speculation.** Section 8 suggests that linear probes and SAEs could meet the same problem, but no experiment tests either.

The authors credit the limits of their results. They note that the conditions are sufficient and not necessary, and that "finding parasitic features does not necessarily mean the hypothesized feature does not exist".

## Use in this repo

**Our verdicts rest on no learned map.** A pole is a fixed native unit, and its explanation passes on activations (`docs/PROTOCOL.md` §4). The pole must fire on held-out positives and stay silent on hard negatives. No interchange intervention decides a pole's verdict, so the paper's construction does not apply to it. The causal metrics that a record carries, sufficiency and necessity on minimal pairs, are recorded and not gated.

**The theorems would apply to a fixed basis too.** The theorems are stated for directions, not for a learned basis. Suppose that one neuron separates the inputs and another steers the output. Then a set of the two could pass an interchange test in the same way. This matters for any claim that two units carry one variable, such as a superposition unit judged by intervention.

**The answer experiments already run the paper's control.** An answer route must pass `necessity`: mean-ablating the route cuts at least half as much as ablating every unit. It must also beat random sets of its layers and kinds (`beats_null`), and its patching must beat the same random sets (`patching_null`). That is the paper's subspace ablation against random subspaces, as a gate. A parasitic route, sufficient under patching but unused, would fail `necessity`.

**#3229 needs the control.** #3229 plans to compare a pole's interchange intervention with the ceiling of a learned one-dimensional DAS direction. This paper shows that the ceiling can be parasitic. The comparison should also ablate the DAS direction and check that the behaviour drops beyond random directions. It should also report the angle between the direction's input and output parts. We add this to #3229 as a comment.

**The illusions checklist.** The reviewer's checklist already asks for ablation when only patching shows an effect, after Makelov et al. (`interpretability-illusions-review`). Parasite features are a second reason for that rule, for learned maps. We add them to the skill's list of papers.

**Related pages.** ReFT's page notes that #3229 would use the distributed interchange intervention that LoReFT extends (`reft-representation-finetuning.md`). The page on *Pretty Linear Liars* warns that a linear probe can find a direction that is not the model's own (`pretty-linear-liars.md`). This paper makes the same point for interventions.

# Mechanistic Failure Modes of Weight Pruning in Transformer Language Models

**Citation.** Nabeelah Faumi and Deshan Sumanathilaka (Informatics Institute of Technology, Colombo, Sri Lanka). "Mechanistic Failure Modes of Weight Pruning in Transformer Language Models". [arXiv:2610.37126](https://arxiv.org/abs/2610.37126) [cs.LG], cross-listed in cs.CL. Version 1 of 27 October 2026 is the only version. The arXiv comment says "12 pages, 10 Tables". Preprint, no venue given. The licence is CC BY 4.0. The text gives no code or data link, and no funding, acknowledgement or AI-use statement.

**Mismatch.** The arXiv abstract page gives the sparsity levels as "3%, 50%, 70%", whereas the HTML abstract and the body give 30%. The title says weight pruning, but the method removes whole heads and whole neurons. The text also disagrees about which neurons magnitude pruning removes. Section 3.2 says "the lowest-scoring neurons are removed globally". Sections 4.3, 4.5 and 4.7 say that it removes the largest, highest-weight or highest-activating neurons. The paper's account of the sign flip rests on the second reading.

**Status.** Planned. #6400 would test whether a pruned pole's downstream effect survives in this repo, and this paper motivated it. #6164 extends `papers/pruning-keep/` to GPT-2. No code, skill or protocol here depends on the paper.

**Read.** On 2026-11-26 a `web-fetch` subagent read the abstract page and the HTML of version 1 in two parts. It read them for this page's author, a model (Claude Opus 5.5). It had the source's own text for Sections 1 to 6, the Limitations, the references, Appendices A.1 to A.6 and most tables. The tool allows only short verbatim quotes, so much of that text came back paraphrased. Parts of Tables 2 and 5 came back garbled. It did not see the PDF or the TeX source.

## Summary

**Claim.** Pruning a language model can break a circuit in distinct ways, which a loss or accuracy number alone does not show. The paper names three: a component's contribution is zeroed, dampened, or inverted in sign. In GPT-2 small, magnitude pruning of 30% of the MLP neurons leaves the IOI circuit's heads in place. But it reverses the sign of the Name Mover heads' direct effect on the answer.

**Method.**

- **Model.** GPT-2 small (117M parameters, 12 layers) only, in TransformerLens, with no retraining after pruning.
- **Design.** Two criteria (magnitude and random), two kinds of component (attention heads and MLP neurons) and three sparsity levels (30%, 50% and 70%). That gives 12 pruned models. A head's magnitude is the L2 norm of its query weights. A neuron's is the L2 norm of its input weights. Magnitude pruning ranks globally, and random pruning removes the same share in each layer. Sparsity counts components, not parameters.
- **Tasks.** Indirect object identification (IOI), 300 sentences after Wang et al. (2022), with the mean logit difference of the right name over the wrong one. Greater-Than (Hanna et al., 2023) is a second circuit. TinyStories perplexity measures general ability.
- **Circuit tools.** Activation patching over the 26 IOI heads, direct logit attribution (DLA) by class of head, and path patching over four routes.
- **Baselines.** The unpruned model, and one random mask for each condition, with seed 42.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Unpruned, IOI | Mean logit difference 2.942, accuracy 94.8%. TinyStories perplexity 11.54 |
| Magnitude, neurons, 30% | Logit difference −2.475, accuracy 17.7%, perplexity 74.61 |
| Random, heads, 30% | Logit difference −0.706, accuracy 36.4%, perplexity 140.10 |
| Name Mover DLA, magnitude neurons 30% | From +3.576 to −1.102 |
| Backup Name Mover DLA, same setting | From +0.427 to −0.511 |
| Activation patching, same setting | The Name Movers are degraded, not reversed (Table 9) |
| Magnitude, heads, 50% and 70% | All four classes' DLA exactly zero. Accuracy 45.7% and 47.9% |
| Greater-Than, magnitude neurons 30% | Accuracy 0.2% |

## Where the claims go beyond the evidence

- **One model, two circuits.** Every result is on GPT-2 small, with two synthetic tasks. The authors say so, and they call the taxonomy a hypothesis. The page credits that.
- **One random mask for each condition.** The random baseline has one draw, with seed 42. No spread over masks is reported, so a random condition's single number can't be told from chance. The authors do not list this among their limits.
- **The sign flip is in one measure only.** DLA shows the Name Movers reversed, but activation patching shows them degraded, not reversed. The headline names the stronger reading.
- **"The heads stay intact" holds by construction.** Neuron pruning does not touch any head's weights. So the check that all 26 IOI heads keep their weights after neuron pruning cannot fail.
- **The direction of magnitude pruning is unclear.** Section 3.2 says that the method removes the smallest neurons. If so, the account in Sections 4.3 to 4.7 does not fit. The reader can't tell which pruning produced the numbers.
- **Few intervals.** Some patching effects carry Bonferroni-corrected tests, but the main accuracies and logit differences have no intervals. The ± values are not defined.
- **Other internal disagreements.** Section 4.5 says neuron pruning harms perplexity more than head pruning at the same sparsity. Table 3 gives the opposite for random pruning at 30%. The TinyStories sample is 1,000 sequences in Section 3.3 and 500 prompts in Appendix A.1.

## Use in this repo

- **This repo prunes inside a neuron, and the paper prunes whole neurons.** `papers/pruning-keep/` zeroes the smallest input weights of each of Pythia-70m's 127 VERIFIED poles, and keeps the neuron. At 90%, 63 explanations keep against 17 under random pruning. The paper instead removes whole neurons and heads and reads circuits downstream. The two results do not conflict, because they change different things.
- **The paper's point applies to the repo's keep test.** A keep test re-runs a pole's own probes. It does not check what the pole then does downstream. `papers/pruning-keep/` says as much in its Discussion. In the paper's sign flip, a part keeps its weights but its effect on the answer reverses. A keep test would miss that case. #6400 would measure exactly this. For each pole that keeps after input pruning, it asks whether the pole's direct effect on the next token keeps its sign and size.
- **Read a flip in more than one measure.** The paper's sign flip shows in DLA, but not in activation patching. #6400 should report the pole-effects estimate and the direct logit effect together, and give a random-pruning null with several masks, which the paper lacks.
- **GPT-2 is the model of #6164.** #6164 would repeat the pruning-keep test on GPT-2, so its poles share a model with this paper. If a GPT-2 pole that keeps under input pruning sits upstream of the Name Movers, the two results could be compared directly. That is a note for #6164, not a new issue.
- **No new issue.** #6400 and #6164 already hold the follow-ups.

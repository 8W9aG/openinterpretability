# Characterizing Mechanisms for Factual Recall in Language Models

**Citation.** Qinan Yu, Jack Merullo and Ellie Pavlick (Department of Computer Science, Brown University). "Characterizing Mechanisms for Factual Recall in Language Models". arXiv 2310.15910v1, 24 October 2023. [Paper](https://arxiv.org/abs/2310.15910). In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (EMNLP 2023, Singapore), pages 9924–9959, [2023.emnlp-main.615](https://aclanthology.org/2023.emnlp-main.615/), DOI 10.18653/v1/2023.emnlp-main.615. We checked the venue on the ACL Anthology page. The paper gives no code or data link. The acknowledgements name no funding, only colleagues and hardware support.

**Status.** Planned. Issue #5015 asks why pythia-70m never reaches " months", and the paper's head-scaling test is a direct way to ask it. The repo has the hook that the test needs, but no run uses it yet.

**Read.** We read the arXiv abstract page and the v1 HTML on 2026-11-18. We downloaded the HTML and read its text ourselves, not through a summary. Sections 1–9, the limitations, the ethics statement and Appendices A–G are in the paper's own text. The equations came through only as their alt text. We did not see the figures, and Appendices A, B, C.3, D and F are mostly figures. So we have no numbers for Pythia-70m, the GPT-2 models or the COUNTERFACT run. We did not read the PDF.

## Summary

**Claim.** When a prompt states a fact that disagrees with pretraining, the training frequency of the terms predicts which answer a model gives. Single attention heads promote the memorized answer or the in-context answer. Scaling one head's value vectors at run time controls which answer wins.

**Method.**

- **Task.** 248 country–capital pairs and 6 aliases. Each country goes with each of the other 247 capitals, in this template: "The capital of {country} is {in-context city}. Q: What is the capital of {country}? A:". That gives 62,992 prompts. The model writes a full sentence. The answer is "in-context" if the substituted city is in it, and "memorized" if the true capital is.
- **Models.** Pythia (70m to 2.8b) and GPT-2 (small to xl). The frequency study uses the Pile for both, as an estimate for GPT-2.
- **Frequency.** Counts of countries, cities and their co-occurrence in a document, over the Pile. The prompts go into 10 percentile bins.
- **Head attribution.** Each head's output goes through its slice of the output matrix and then the unembedding. The logit of the memorized city minus the logit of the in-context city is the head's score. The data are 200 prompts: 10 per bin where the model gives the in-context answer, and 10 per bin where it gives the memorized one.
- **Intervention.** Multiply one head's value vectors by a factor α. α = 1 is no change. The authors pick the head and its α on the 200 prompts, then run them on all 62,992. The head analysis and the interventions use only Pythia-1.4b, Pythia-2.8b and GPT2-xl.
- **Weights.** An SVD of the memory head's OV matrix, with each right singular vector read through the unembedding.
- **Transfer.** The same heads and α on the COUNTERFACT paraphrase task, with no new tuning.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| All models, world-capital task | The model gives one of the two cities at least 80% of the time, "varying depending on model size" |
| Frequency, all models | A more frequent country gives more memorized answers. A more frequent in-context city gives fewer in-context answers |
| Model size, Pythia up to 2.8b | Larger models give the memorized answer more. The shift starts with the most frequent countries |
| Pythia-1.4b, 24 layers × 16 heads | Memory head 15.7. In-context head 19.12 (Table 2) |
| Pythia-2.8b | Memory head 17.17. In-context head 17.31 (Table 2) |
| GPT2-xl | Memory head 35.19. In-context head 25.20 (Table 2) |
| Pythia-1.4b, 200 tuning prompts, memory head at α = −0.7 | 86% of the memorized answers become in-context answers |
| Pythia-1.4b, all 62,992 prompts, no intervention | In-context 26%, memorized 43% |
| The same, memory head scaled down | In-context 86.2%, memorized 4% |
| The same, memory head scaled up | Memorized 50% |
| The same, in-context head scaled up | In-context up by 12%, memorized down by about 20% |
| Moves towards the memorized answer | "only increasing 6%" |
| By frequency bin, memory head scaled down | The frequency effect goes away: the lines are flat across bins |
| By frequency bin, in-context head scaled up | The frequency effect stays. A frequent in-context city still gets few in-context answers |
| SVD of head 15.7, top five vectors | Mostly place tokens (' Italy', ' Naples', ' Guatemala', ' Brazil'). The in-context head shows no clear clusters |
| COUNTERFACT, same heads and α | No transfer: memorized and in-context answers both fall, and invalid answers rise |
| Appendix E, Pythia-1.4b | Answers that start "The capital of `<country>` is" go from 7,961 to 57,440 after the memory-head intervention. 92% of them give the in-context city |
| Appendix E, answers with neither city | 77% repeat the country name. After the intervention, 83% of those give the memorized answer |

The paper's discussion says the heads hold domain-specific knowledge, "rather than a more abstract concept of truthfulness". It calls the result a proof of concept for control at run time.

## Where the claims go beyond the evidence

**"Downweighting" is a sign flip.** The best memory-head setting is α = −0.7. That reverses the head's write and does not only weaken it. The paper calls it "tuning down" and "downweighting" throughout. The result shows what a reversed head does, which differs from removal of the head.

**One template, one relation.** Every number comes from one prompt template about capitals. The heads do not transfer to COUNTERFACT. The authors state both limits, and the discussion calls the heads domain-specific.

**"New data" is the same distribution.** The abstract says the method works "on new data". The full-set run uses the same template and the same 248 countries as the 200 tuning prompts, and the tuning prompts are inside it. On the one truly new dataset, COUNTERFACT, it fails.

**The authors pick the head on the result.** Head attribution gives a list of candidates. The paper then keeps the head "that has the strongest effect in the intended direction", with its best α. There is no random-head baseline for the scaling, so we cannot tell how many heads would flip answers at a tuned α.

**No statistics.** There are no seeds, intervals or tests for any proportion. The frequency results are trends read from figures.

**Some numbers disagree.** The abstract gives 88% in-context, and Section 6.3 gives 86.2%. The in-context head is 19.14 in Figure 5, 19.12 in Table 2 and 19.4 in Section 6.4. The 0.00001% of parameters claim is about 140 weights of Pythia-1.4b. By our count, one head's value slice alone is 2,048 × 128 weights.

**Heads in the large models only.** The head analysis uses models of 1.4b parameters and more. The frequency study includes Pythia-70m, but its result is only a figure that we did not see.

**The authors state the main limits.** They say the heads are not the only ones with this role, and the result does not cover arbitrary contexts. They also say that a move towards the memorized answer may need the MLP layers.

## Use in this repo

**The #4781 failure is the same competition, in a weaker form.** On "A week has seven days and a year has twelve", pythia-70m gives " days" at 0.7257 and " months" at 0.0295, rank 4 (`docs/analyses/hallucination-shortlist.jsonl`). The VERIFIED answer route copies the first clause's unit. Copy heads L3/H3, L4/H0 and L4/H7 move " days" to the last position (`wiki/answers/EleutherAI--pythia-70m/970a91e11d8c/20261118T234629Z.json`). Those heads are the paper's in-context heads. No verified unit supplies " months", the other side.

**The prompt is not a counterfactual, so the analogy has a limit.** The paper's context states a false value for the same relation. In #4781, the context states a true fact about a different subject, and the model copies by analogy. The record's roles call L3/H3 "induction-style". So the in-context side here is a pattern copy, nearer the paper's Appendix E than its main task.

**What the paper predicts for #4781.** The paper finds that smaller Pythia models take the in-context answer more often. So a 70m model that copies " days" over a weak " months" recall is the expected case, not an odd one. By the paper's account, there is a weak memory path for "year → months" that the copy heads outvote. Scaling the copy heads down should then remove " days" but lift " months" only a little. The paper found moves towards the memorized answer hard ("only increasing 6%"), and it points at the MLP layers. That fits #5015's first question: the " months" source may be present but too weak, or in MLPs that no verified unit covers.

**A candidate memory head already shows in the data.** The pick's contrast in `docs/analyses/hallucination-shortlist.jsonl` is each unit's first-order attribution to logit(" days") − logit(" months"). It is a gradient method, not the paper's direct projection. L4/H1 favours " months" by 0.599 and is not in the route. L0/N1101+ and L1/N707+ favour " months" most, by 1.532 and 1.013. Three route heads (L0/H0, L1/H6, L1/H7) also come out on the " months" side. This needs a check, because a first-order score can't separate a head's direct write from its effect through later heads.

**The repo can run the paper's intervention now.** A head's output z is its attention pattern times its value vectors. So scaling the value vectors by α scales z by α. `NeuronHookRunner.run_with_interventions(..., head_interventions=...)` in `openinterpretability/harness/hooks.py` already rewrites a head's slice, as `ablate_head` does. A sweep of α over the copy heads and L4/H1, on the six prompts of the shortlist, would measure what the paper measured. The `neuron-steering-tests` skill covers the read-out of generated text.

**The paper's warnings apply to such a sweep.** Report α < 0 apart from 0 < α < 1, because a reversed head is not a removed head. Compare each head with random heads at the same α, as the answer review already does with its matched random draws (`openinterpretability/answer_review.py`). Pick α on some prompts and report it on others.

**This is evidence, not a fix.** The paper offers head scaling as a way to control answers at run time. The fix campaign is the maintainer's to design. This page only reports what the paper measured and what a sweep in this repo would measure.

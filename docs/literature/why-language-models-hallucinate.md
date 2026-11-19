# Why Language Models Hallucinate

**Citation.** Adam Tauman Kalai (OpenAI), Ofir Nachum (OpenAI), Santosh S. Vempala (Georgia Tech) and Edwin Zhang (OpenAI). "Why Language Models Hallucinate". arXiv 2509.04664v1, 4 September 2025. [Paper](https://arxiv.org/abs/2509.04664). Preprint, no venue given. Licence CC BY 4.0. The paper releases no code or data. NSF award CCF-2106444 and a Simons Investigator award support Vempala in part. The paper doesn't say whether an LLM helped write it.

**Status.** Planned. The paper is a statistical theory of errors, and it has no neurons, heads or circuits in it. Its error taxonomy can frame the paper that #5015 asks for, and it predicts a scale test for the copy-the-first-unit failure.

**Read.** The abstract page and the v1 HTML, on 2026-11-18. A model read the text in four passes and reported it to us, with short quotes. We have the source's own words for the abstract, the author block, Sections 1 to 3, and short quotes from Sections 4 to 6. Table 2 and the appendices came back only as a summary. We saw no figures and no PDF. The theorem statements came from text that the HTML extracted from the maths, so some symbols can differ from the typeset paper.

## Summary

**Claim.** Hallucinations "originate simply as errors in binary classification". Pretraining makes them through ordinary statistical pressure. Post-training keeps them, because most benchmarks give no credit for "I don't know", so guessing scores better than abstaining.

**Method.** The paper is theory, with a few examples from chat models.

- **The reduction.** A model that generates text also defines a classifier for the "Is-It-Valid" (IIV) problem: is this string valid or an error? The paper bounds the generative error rate below by the IIV misclassification rate.
- **Error factors.** Five factors make IIV hard. Statistical complexity is one, shown by arbitrary facts such as birthdays. Poor models are another, shown by letter counting and by trigram models. Computational hardness, distribution shift and "garbage in, garbage out" (GIGO) are the other three.
- **Calibration.** The bound has a term δ, a calibration error at one threshold. The paper argues that δ is small after pretraining, because cross-entropy is at a local minimum under a rescaling of the probabilities.
- **Evaluation.** Observation 1 proves that under binary grading, abstaining is never optimal. A survey of benchmarks (Table 2, Appendix F) finds that nearly all of them grade binary.
- **The proposal.** A benchmark states a confidence target t in its instructions. A mistake costs t/(1−t) points, a right answer earns 1 and "I don't know" earns 0. The paper calls the matching behaviour "behavioral calibration".
- **Examples.** DeepSeek-V3 gave three wrong birthdays in three tries. It also gave "2" or "3" as the count of Ds in DEEPSEEK, in ten trials. Three chat models gave three different wrong dissertation titles, in one query each on 2025-05-09 (Table 1).

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Generation error against IIV error, no prompts (Corollary 1) | err ≥ 2·err_iiv − \|V\|/\|E\| − δ |
| With prompts (Theorem 1) | err ≥ 2·err_iiv − max_c \|V_c\| / min_c \|E_c\| − δ |
| Arbitrary facts, lower bound (Theorem 2, probability ≥ 99%) | err ≥ sr − 2/min_c \|E_c\| − (35 + 6 ln N)/√N − δ |
| Arbitrary facts, upper bound for a calibrated learner (Theorem 2) | err ≤ sr − sr/(max_c \|E_c\| + 1) + 13/√N |
| Singleton rate sr | the fraction of prompts seen exactly once in training, without abstention |
| Birthday example | if 20% of birthday facts appear once, the paper expects base models to hallucinate on at least 20% of them |
| Pure multiple choice, one right answer (Theorem 3) | err ≥ 2(1 − 1/C)·opt(G), with C = min_c \|E_c\| + 1 |
| Trigram model (Corollary 2) | error ≥ 1/2 on "She lost it and was completely out of…" against "He lost it and was completely out of…" |
| DeepSeek-R1 on the same letter count | spells the letters in a 377-token chain of thought and counts one D correctly |
| GPT-4 calibration (Figure 2, from an OpenAI report) | well calibrated after pretraining, and less so after RL |

The trigram case is the paper's example of a poor model. Its context holds only the last two words. So it can't see "She" or "He", and it can't choose between "her mind" and "his mind".

## Where the claims go beyond the evidence

- **A theory, with anecdotes for data.** No experiment has a sample size above ten, an interval or a seed. The paper cites a later empirical study (Miao and Kearns) and doesn't report its own. The theorems are sound as theorems, but the paper doesn't measure how often each factor causes real errors.
- **The paper assumes calibration for base models, and doesn't measure it.** The δ argument is a local optimality argument. The only evidence is one figure for one model, taken from an OpenAI report. The authors say that δ at one threshold is weaker than integrated measures such as ECE.
- **The reduction doesn't name a cause.** The lower bound says that errors must happen when IIV is hard. It doesn't say which inputs fail, or what inside the model fails. "Poor model" covers any shortfall of the model family or of the fit, so it explains every residual error and predicts few in particular.
- **The post-training claim is an argument.** Table 2 shows that benchmarks grade binary. The paper doesn't show that this grading is what keeps hallucinations after training, for example with a training run under the proposed scoring.

The authors state many limits themselves. They assume noiseless training data and a finite sample space, and they exclude nonsense strings. They treat one phrasing per fact and one factual question at a time, and they leave out hedging and degrees of error. They say that search does not fix errors such as letter counting.

## Use in this repo

- **It studies no units.** The paper has no neurons, heads or patching. Its only internal claim is a citation that "internal model activations encode predictive signals" about accuracy. So nothing here bears on a pole's explanation, on the reviewer or on a verdict.
- **The copy failure is a poor-model error, not an arbitrary-fact error.** `scripts/hallucination_shortlist.py` scores 35 prompts with one right answer each (`docs/analyses/hallucination-shortlist.jsonl`). Pythia-70m answers 28 of them wrong. All 6 prompts that it keeps copy the first clause's unit. On "A week has seven days and a year has twelve", " days" has 0.7257 and " months" has 0.0295, at rank 4. "A year has twelve months" is a common fact, so the singleton rate of Theorem 2 can't explain the error. The paper's other factor is a poor model.
- **The route of #4781 fits a poor-model reading.** The VERIFIED answer route copies the first clause's unit (`changelog.d/4781.added.md`). Layer-0 count-word and time-unit neurons mark " seven", " days" and " twelve". The copy heads L3/H3, L4/H0 and L4/H7 then carry " days" to the answer position, and no verified unit supplies " months". This is like the trigram case. The word that settles the answer ("year") is in the prompt. But the route that the model uses reads past it, and copies the unit of the parallel clause.
- **A scale test follows.** A poor-model error should fall as the model family grows on the same data. Larger Pythia models share the tokenizer and the training data, so they are a clean test. A larger model that still copies the unit would point to distribution shift or to the data, not to capacity. The proposed follow-up below asks for this. It uses the larger models for comparison only, not as registry models.
- **Calibration on the shortlist is out of reach.** The shortlist keeps only confident wrong answers, so its probabilities can't measure δ. Thirty-five prompts are also too few for a calibration curve. We note this and don't propose it.
- **The fix campaign.** The paper's remedy is a change to benchmark scoring, not to a model's units. So it gives the campaign no method for a fix, only the framing that the model's capacity forces some errors. The maintainer designs that campaign, and this page reports evidence only.

**Proposed follow-up.** "Does scale fix the copy-the-first-unit failure? The shortlist across the Pythia suite." Area: research questions. Priority: medium. Run `scripts/hallucination_shortlist.py` on the larger Pythia models, and report for each one the rank of the right unit on the six kept prompts.

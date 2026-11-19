# Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?

**Citation.** Zorik Gekhman (Technion, intern at Google Research), Gal Yona, Roee Aharoni, Matan Eyal, Amir Feder, Jonathan Herzig (Google Research) and Roi Reichart (Technion). "Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?". arXiv 2405.05904. Versions: v1 of 9 May 2024, v2 of 13 May 2024 and v3 of 1 October 2024. [Paper](https://arxiv.org/abs/2405.05904). The arXiv comment says "Accepted as a long paper at EMNLP 2024". We did not check the proceedings page. The abstract page gives no code or data link for this paper.

**Status.** Planned. Issue #5015 asks why pythia-70m never reaches " months". The paper's knowledge categories (SliCK) give a measure for one part of that question: does the model know the fact at all?

**Read.** On 2026-11-18, a model read the abstract page and the v3 HTML for us, in three passes. The web reader limits quotes to 125 characters. So most of the text came back as a close paraphrase, with short exact quotes and the tables as printed. The pass covered the main text through Section 11 (Limitations), and Appendices A to K. We did not see the figures or the PDF. The numbers below are from the tables as the model extracted them. Table 2 came back flattened, and the model rebuilt its columns.

## Summary

**Claim.** The paper fine-tunes a model on closed-book questions. Some answers are facts that the model already knows, and some are new to it. The model fits the examples with new facts "substantially slower" than the known ones. When it does fit them, its test accuracy falls linearly, that is, it hallucinates more. The authors conclude that models "mostly acquire factual knowledge through pre-training, whereas fine-tuning teaches them to use it more efficiently."

**Method.**

- **Model.** PaLM 2-S, the base model. This is the only model.
- **Data.** EntityQuestions: 12 relations for training and in-distribution test, and 7 other relations for an out-of-distribution test. The train split has 81,700 annotated examples. The in-distribution test has 10,481 examples, and the out-of-distribution test has 5,848.
- **SliCK categories.** The model answers each question under 10 random 4-shot prompts. Each prompt gives one greedy answer (T = 0) and 16 sampled answers (T = 0.5, top-40). From these answers, P_Correct puts each question into one of four categories:
  - *HighlyKnown*: the greedy answer is always right.
  - *MaybeKnown*: the greedy answer is sometimes right.
  - *WeaklyKnown*: the greedy answer is never right, but a sampled answer sometimes is.
  - *Unknown*: no answer is ever right.
- **Fine-tuning sets.** Each set has exactly 6,142 examples. The sets differ in their mix of Known and Unknown examples, or they hold one category only. *D_Natural* keeps the train split's own mix.
- **Training.** 50 epochs, learning rate 1e-5, batch size 128, dropout 0.05. *early_stop* is the best dev epoch, about 5 to 10 epochs. *Convergence* is epoch 50, where train accuracy is 100%.
- **Measure.** Test accuracy, "as a proxy for hallucinations since we are in a closed-book QA setup".
- **Statistics.** Appendix J splits each test subset into 100 random parts, and runs a paired t-test across them at p < 0.05 and p < 0.01. The text we received reports no training seeds and no intervals.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| SliCK categories of EntityQuestions (footnote, from a paraphrase) | About 24% HighlyKnown, 23% MaybeKnown, 17% WeaklyKnown, 36% Unknown |
| Linear fit of test accuracy on the fraction of Known and Unknown train examples fitted, in-distribution (Table 1) | β0 = 36.9, β_kn = 7.3, β_unk = −8.3, R² = 0.86 |
| The same fit, out-of-distribution (Table 1) | β0 = 36.2, β_kn = 3.2, β_unk = −3.0, R² = 0.95 |
| Full test accuracy, D_Unknown, early_stop and Convergence (Table 2) | 37.5 and 25.8 |
| The same, D_WeaklyKnown | 39.2 and 35.4 |
| The same, D_HighlyKnown | 40.5 and 40.0 |
| The same, D_MaybeKnown | 43.6 and 43.2 |
| The same, D_Natural | 43.5 and 41.8 |
| HighlyKnown test questions, model fine-tuned on D_Unknown, early_stop and Convergence (Table 2) | 95.6 and 55.8 |
| Half Known and half Unknown, early_stop and Convergence (Table 3) | 43.0 and 38.8, with 100% of questions answered |
| The same set, Unknown answers relabelled "I don't know" | 61.8 at both points, with 58.7% and 55.6% of questions answered |

The authors read the Table 2 result as "MaybeKnown examples are essential": D_MaybeKnown is best on the full test set. They say that the effect holds on the held-out relations. In the text we received, the out-of-distribution drop is up to 6 points, against up to 14 in-distribution.

## Where the claims go beyond the evidence

**One model.** The authors say so in Section 11. The model is PaLM 2-S only, so the slope of the linear fit may not carry to other sizes or families.

**One task.** The setting is closed-book QA with short entity answers. The authors say that they did not test long-form generation, and that they did not mix in other tasks.

**Accuracy is a proxy for hallucination.** A wrong test answer counts as a hallucination. The paper does not separate a wrong fact from a refusal or a malformed answer. Appendix C has an error analysis, but our reading cut it short.

**No training seeds.** The significance tests resample the test set, not the training run. So they do not show how much the curves move from one fine-tuning run to another.

**"Mostly through pre-training" is wider than the experiments.** The study changes the fine-tuning mix of one model at one learning rate (with two more rates as a check). It does not compare against continued pre-training. The authors name continued pre-training and LoRA as future work.

**Credit.** The authors control the size of every fine-tuning set and report a held-out-relation test. They also give the cost of SliCK: over 15M inference steps, 170 per example.

## Use in this repo

**The repo does no fine-tuning yet.** The repo has no fine-tuning code. EleutherAI/pythia-70m is a base model with no fine-tuning stage. So the paper's mechanism, where fitting Unknown examples teaches the model to make up facts, cannot be the cause of the #4781 failure. The paper becomes relevant only if the fix campaign uses training. The maintainer designs that campaign, so this page gives evidence and no design.

**What the paper predicts for the copy failure.** SliCK asks one question first: does the model know the fact under a few-shot prompt? The #4781 prompt is "A week has seven days and a year has twelve". In `docs/analyses/hallucination-shortlist.jsonl`, the right answer " months" is rank 4 at p = 0.0295, and " days" is at p = 0.7257. The reversed prompt "A year has twelve months and a week has seven" gives " months" at p = 0.4549. So copying wins in both orders. The shortlist has no control that asks for the fact without a first clause to copy. Without that control, the repo cannot tell two cases apart. In one, the model knows "twelve months" and copies the wrong unit. In the other, it does not know the fact.

**A cheap measure would separate the two cases.** The paper's P_Correct needs only the model and a few-shot prompt, and pythia-70m runs in seconds per prompt. The measure would put each shortlisted right answer into a SliCK category. The MQuAKE evidence on #5015 already asks for a single-clause control, so this is the same check with a defined rule. The shortlist prompts that copying answers right give a contrast. For example, "A spider has eight legs and a dog has four" answers " legs" at p = 0.5494.

**For a fix by training, the category matters.** The paper's numbers say that a HighlyKnown or MaybeKnown fact costs little to fit, because fine-tuning then teaches the model to use what it knows. An Unknown fact fits slowly. Once the model fits it, test accuracy falls by about β_unk = −8.3 points per unit fraction of Unknown examples fitted, in-distribution. In Table 2, fitting D_Unknown to convergence takes HighlyKnown test accuracy from 95.6 to 55.8. That is the collateral damage that #4784 asks the campaign to measure. The paper's other two results are early stopping, and relabelling Unknown answers as "I don't know".

**The prompts that copying answers right are the natural hard negatives.** The pythia-70m route copies the first clause's unit through the copy heads L3/H3, L4/H0 and L4/H7. A change that fits " months" and weakens that route would match the paper's pattern: other answers that the model answered right would go wrong. The shortlist's " legs" and " wheels" prompts are the cases where copying is right.

**Question for the maintainer.** Should the fix campaign record a SliCK category for each target fact before it tries a fix by training? This page records it as a question, not an issue.

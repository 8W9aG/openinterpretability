# An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning

**Citation.** "An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning". arXiv:2308.08747 [cs.CL], version 5, 5 January 2025. [arXiv](https://arxiv.org/abs/2308.08747).

- **Authors.** Yun Luo, Yafu Li and Yue Zhang are at Westlake University. Yue Zhang is also at the Westlake Institute for Advanced Study. Zhen Yang, Fandong Meng and Jie Zhou are at WeChat AI, Tencent.
- **Versions.** Version 1 is from 17 August 2023 and version 2 from 21 August 2023. Version 3 is from 2 April 2024, and version 4 from 30 December 2024. Version 3 grew from 688 KB to 3,075 KB.
- **Venue.** Preprint, no venue given. The PDF's header says "Under review", and the arXiv page gives no journal reference.
- **Code and data.** The paper links no code. The tasks come from Scialom et al. (2022), and the evaluation uses lm-evaluation-harness.
- **Funding and tools.** The paper gives no funding statement. Training used 8 Tesla A100 40G GPUs.
- **Mismatches.** The PDF marks Yun Luo as an equal contributor, but gives no second author the mark. The PDF prints the Tencent e-mail domain as `tentent.com`.

**Status.** Planned. #4782 (catastrophic interference and weight-space programming) would use it, together with [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md). The paper measures benchmark scores, not units. So this page reports evidence and records questions, because the campaign to fix models is the maintainer's.

**Read.** On 2026-11-19 we downloaded the PDF of version 5 from arXiv with `curl`, and extracted its text with `pdftotext`. A model (claude-opus-5-5) read it, not a person. We read all 15 pages in the source's own text. That covers the abstract, sections 1 to 7, the references and Appendix A. Tables 1 to 7 came through as text. Figures 3 to 6 did not, so we rendered pages 8 to 10 as images and looked at them. This page takes no numbers from the figures. We did not read versions 1 to 4, so we can't say what changed between them.

## Summary

**Claim.** When a language model of 1 to 7 billion parameters is fine-tuned on a sequence of instruction tasks, it loses general knowledge. The larger models of the range lose more. A decoder-only model loses less than an encoder-decoder model of similar size. A general instruction tune before the sequence, or general instruction data mixed into it, reduces the loss.

**Method.**

- **Models.** BLOOMZ at 1.1b, 1.7b, 3b and 7.1b (decoder-only). mT0 at 1.2b and 3.7b (encoder-decoder, based on T5). BLOOM-7.1b, LLAMA-7b and ALPACA-7b, to compare models with and without a general instruction tune.
- **Sequence.** Five tasks from Scialom et al. (2022), in one order: text simplification, empathetic dialogue, inquisitive question generation, explanation generation and headline generation with a constraint. Each task uses 100,000 samples behind one shared prompt template.
- **Training.** Full fine-tuning with Adam at learning rate 2e-5, batch size 4 on each GPU, 3 epochs for each task and maximum input length 512. The scheduler is constant for BLOOMZ and mT0, and cosine for LLAMA and ALPACA. The authors evaluate the final checkpoint of each task.
- **General knowledge.** MMLU for domain knowledge (5-shot, four parts). BoolQ, PIQA, Winogrande, Hellaswag, MathQA and Mutual for reasoning (zero-shot accuracy). RACE for reading comprehension (zero-shot accuracy). CrowS-Pairs for bias, as the share of pairs where the model prefers the stereotypical sentence.
- **Metric.** The forgetting value FG of an evaluation set is the relative fall from the initial score. The paper averages it over the elements of the set and over the five stages. So FG is not the fall at the end of the sequence.
- **Mixing.** One arm mixes 10,000 ALPACA samples into the sequence for BLOOMZ-7.1b and LLAMA-7b.
- **Baselines.** The initial model is the only reference. There is no random or untrained baseline, and no other task order.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| BLOOMZ-7.1b, explanation task, before and after its own stage (Table 3) | 51.47 → 68.71 BERTScore |
| BLOOMZ-7.1b, all five stages (Table 4) | MMLU 33.08 → 25.61, FG 18.37. Reasoning 59.15 → 49.24, FG 13.62. RACE 48.79 → 33.05, FG 26.75 |
| BLOOMZ, domain knowledge FG by size (Table 4) | 9.54 (1.1b), 10.72 (1.7b), 14.63 (3b), 18.37 (7.1b) |
| BLOOMZ, final MMLU score by size (Table 4) | 23.84, 24.52, 24.29 and 25.61 |
| mT0, domain knowledge FG by size (Table 4) | 9.18 (1.2b), 20.15 (3.7b) |
| Reasoning FG, BLOOMZ-3b against mT0-3.7b (Table 4) | 11.09 against 16.73. The text calls the difference 5.64 |
| Bias FG (Table 5) | Above zero for all six models, from 2.97 (BLOOMZ-3b) to 13.10 (mT0-3.7b) |
| BLOOMZ-7.1b, physical-appearance pairs (section 5.1) | Preference for the stereotypical sentence 75.0% → 63.88% |
| With and without a general instruction tune (Table 6) | Domain knowledge FG 13.54 (BLOOM) against 18.37 (BLOOMZ). 34.57 (LLAMA) against 18.14 (ALPACA) |
| Reasoning and RACE FG, LLAMA against ALPACA (Table 6) | 31.33 against 7.56, and 31.72 against 10.31 |
| LLAMA-7b, MMLU humanities, with mixed data (section 5.4) | 34.72 initially. 26.8 after the plain sequence, 30 after the mixed sequence |

The authors read FG above zero on bias as a fall in bias. They explain the effect of size by the higher initial scores of the larger models, because the final scores are similar.

## Where the claims go beyond the evidence

- **"Larger models forget more" rests on a floor.** The final MMLU scores of the four BLOOMZ sizes lie between 23.84 and 25.61, which is about chance for four options. FG is relative to the initial score, so a model that starts higher must show a larger FG. The authors say this themselves in section 5.2. The abstract still states the effect of size as a finding.
- **One task order, one run, no spread.** Every number comes from one run of one order. The paper gives no seeds, no intervals and no tests. The authors state the single order as a limit.
- **The instruction-tuning claim holds for one pair of two.** BLOOMZ forgets more than BLOOM on every set in Table 6. Only LLAMA against ALPACA supports the claim. The authors explain BLOOMZ by its higher initial scores, but they don't apply the same argument against ALPACA's result.
- **The architecture claim rests on one comparison.** At 1.1b and 1.2b, BLOOMZ and mT0 have similar FG. Only the 3b against 3.7b pair separates them. The two families also differ in their pretraining, not only their architecture.
- **Knowledge loss is not separated from format loss.** Each stage trains on one prompt template, and the evaluation is plain zero-shot or 5-shot. A model can lose the evaluation format without losing the knowledge. The paper doesn't test this.
- **The bias result has another reading.** A CrowS-Pairs preference near 50% is also what a model with no preference shows. A model that loses its fluency could move toward 50% without any change in bias. The paper doesn't separate the two.
- **Inconsistencies within the source.** Section 5.2 cites "Figure ??" and points to Table 1 for results that are in Table 4. The text calls Table 3's metric "BLUEScore", but the caption says BERTScore. The captions of Figures 3 to 5 call the values FG, but the y-axes show raw scores. Section 5.1 gives the same start, 36.18%, for MMLU social science and MMLU other.
- **No code.** The paper names the training settings, but releases no code or checkpoints.
- **Credit.** The authors state the limits of size, of the single task order and of the choice of benchmarks. They also note that the hyperparameters come from the original works and "may still introduce some effects".

## Use in this repo

- **The registry's nearest measure finds that units keep.** [`base-instruct.md`](../analyses/base-instruct.md) tests VERIFIED explanations again after instruction tuning. On Qwen2.5-1.5B, 38 of 43 keep (88%, 95% CI 76–95%), and activations correlate at a median r of 0.983. On Qwen2.5-0.5B, 252 of 311 keep (81%, 95% CI 76–85%), at a median r of 0.974. The paper finds large falls in benchmark scores after fine-tuning. These two results don't contradict each other, for three reasons.
- **The fine-tunes differ.** Qwen's instruct model comes from a general instruction tune. The paper finds that general tuning, as in ALPACA, loses less than a sequence of narrow tasks. The repo has not tested a narrow fine-tune or a sequence of them.
- **The measures differ.** `base-instruct.md` measures units against their own probes, and has no benchmark score. The paper measures benchmarks and has no unit. So nothing in the repo links a unit that keeps to knowledge that keeps.
- **The format effect agrees with ours.** In `base-instruct.md`, probes inside the chat template lower the keep share. It falls to 70% on 1.5B and 72% on 0.5B. The fall is mostly in sensitivity. The paper's evaluation format differs from its training format, and it doesn't control for that. Our result says that format alone moves what we measure. So part of the paper's "forgetting" could be format, as #4829 (spurious forgetting) also proposes.
- **Size goes the other way here, but weakly.** The paper finds more forgetting at larger size. Qwen2.5-1.5B keeps 88% and Qwen2.5-0.5B keeps 81%. The intervals overlap, the measure is different, and the paper's size effect may be a floor. We draw no conclusion about size.
- **Units move when scores stay flat.** In [`papers/neuron-development/`](../../papers/neuron-development/README.md), Pythia-70m's 86 VERIFIED poles correlate with their final form at a median of 0.62 at step 16,000. That is after the loss is within 1% of its final value. Adjacent checkpoints correlate at a median of 0.78 to 0.82. So a flat benchmark score doesn't show that units are stable, and the paper's benchmark falls don't show which units moved.
- **The mixing result matches the interleaving argument.** [`complementary-learning-systems.md`](complementary-learning-systems.md) says that interleaving old material avoids interference. The paper's mixed-data arm agrees on one model and one split (34.72 → 26.8 plain, 30 mixed). Whether a fix of the #4781 " days" error should interleave old data is the maintainer's question.
- **No single-unit content.** The paper says nothing about neurons, explanations, probes or nulls.
- **Follow-up.** The repo can measure what the paper didn't. Do VERIFIED explanations keep through a sequence of narrow fine-tunes, in a base model and in its instruct model? That gathers evidence only, and fixes nothing. We propose one issue for it. #5118 (Fisher importance and keep share) and #4105 (a second family and more templates) stay as they are.

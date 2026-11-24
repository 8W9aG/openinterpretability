# Base and instruct: do verified neurons keep their hypothesis?

Issue #3375. Instruction tuning changes a model's behaviour. Does it change what its neurons compute? This page takes every neuron pole of Qwen2.5-1.5B that stands VERIFIED, and tests its hypothesis again on Qwen2.5-1.5B-Instruct. Run on 2026-11-14 with `scripts/base_instruct.py`. The rows are in `base-instruct.jsonl`. Reported, not gated. No record was written.

**Answer: nearly all keep, and the neurons barely move.** 38 of 43 hypotheses still pass their own probes on the instruct model: 88% (95% CI 76–95%). Each neuron's activation correlates with its instruct counterpart at r = 0.95 or more, median 0.983. The five that fail were near a gate on the base model too, and each one fails by a probe or two. Issue #3762 added Qwen2.5-0.5B, where 252 of 311 keep (81%). It also put the probes inside the instruct model's chat template, which lowers the keep share to 70–72% (section *More pairs*).

## Method

- **The pair.** Qwen2.5-1.5B and Qwen2.5-1.5B-Instruct have one architecture, so neuron *i* of layer *l* is the same position in both. The instruct model is not in the registry.
- **Units.** The 43 neuron poles of Qwen2.5-1.5B that stand VERIFIED, with errata applied, in 22 layers. Each one uses the probes of the record that its standing rests on.
- **Base.** Each record's probes are measured again on the base model, with the record's own scale. All 43 pass again, so the method reproduces the records.
- **Instruct.** The same probes, at the same layer, neuron and pole. The scale is the record's scale times the ratio of the pole's maximum in the instruct model to its maximum in the base model. Both maxima are over one 20,000-token sample of `corpus.jsonl`. So "active" means the same share of each model's own range. The median ratio is 0.98 (0.63–1.29).
- **Keeps.** A hypothesis keeps when the instruct readings pass the record's specificity and sensitivity gates (`review_metrics`: at least 0.80 and 0.60). The other gates concern the review, not the neuron. Counterfactuals, independence and the confound baseline aren't measured again.
- **Correlation.** The Pearson correlation of the neuron's activation in the two models, over every token of the sample, first tokens left out.

## Results

| | Poles | Keep | Median correlation |
| --- | --- | --- | --- |
| All | 43 | 38 (88%) | 0.983 |
| Depth below a third | 16 | 15 | |
| Depth a third to two thirds | 15 | 13 | |
| Depth above two thirds | 12 | 10 | |

The lowest correlation is 0.953. The correlation doesn't separate the poles that keep (median 0.983) from those that fail (median 0.986).

The five that fail:

| Pole | Fails | Base → instruct | Margin on base | r |
| --- | --- | --- | --- | --- |
| L7/N3197 positive (existence or containment clauses) | specificity | 0.80 → 0.75 | 0.00 | 0.989 |
| L11/N526 positive (formally attested collective procedure) | specificity | 0.85 → 0.77 | 0.05 | 0.986 |
| L17/N6653 positive (the count of what a thing contains) | sensitivity | 0.71 → 0.57 | 0.11 | 0.967 |
| L20/N6271 positive (a report's citing span) | sensitivity | 0.67 → 0.50 | 0.07 | 0.976 |
| L23/N873 positive (the digits of a year 1000–1999) | sensitivity | 0.71 → 0.43 | 0.11 | 0.988 |

The *margin* is how far the base readings sit above the nearer gate. Among the poles that keep, the median margin is 0.11, and 18 of 38 sit within 0.10 of a gate. So a failure here is a probe or two moving across the line, not a neuron that computes something else. The records have 12 to 20 hard negatives and 6 to 10 holdouts. So one probe moves specificity by 0.05 to 0.08, and sensitivity by 0.10 to 0.17. The sensitivity failures lean late: three of the five are in the last third, where an instruct model's tuning would act most.

## More pairs, and probes in the chat template

Issue #3762, run on 2026-11-15 on the CPU, with the same method and two changes to `scripts/base_instruct.py`:

- `--base` names the base model, and the instruct model is `<base>-Instruct`.
- `--chat` sends each probe to the instruct model as one user turn of its chat template. That is `apply_chat_template`, with no generation prompt. The pole is read only over the tokens that overlap the probe's own characters, so a template token can't fire it. The base readings stay plain text. The sample, the correlations and the scale ratio don't change: they come from plain text in both models.

| Pair | Probes on instruct | Poles | Keep | 95% CI | Rows |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-1.5B | plain (2026-11-14) | 43 | 38 (88%) | 76–95% | `base-instruct.jsonl` |
| Qwen2.5-1.5B | chat template | 43 | 30 (70%) | 55–81% | `base-instruct-chat.jsonl` |
| Qwen2.5-0.5B | plain | 311 | 252 (81%) | 76–85% | `base-instruct-0.5b.jsonl` |
| Qwen2.5-0.5B | chat template | 312 | 225 (72%) | 67–77% | `base-instruct-0.5b-chat.jsonl` |

Every pole reproduces on its base model in every run.

- **Qwen2.5-0.5B keeps most of its hypotheses too.** 252 of 311 keep with plain probes, 81%. Its activations correlate with the instruct model's at median r 0.974, and at least 0.927. The median scale ratio is 0.96, from 0.17 to 1.55. By depth, 104 of 112 keep in the first third, 63 of 92 in the middle third and 85 of 107 in the last third.
- **The chat template costs about 9 to 18 points.** The comparison below takes the poles that both runs share, and counts the poles that change outcome:

  | Pair | Shared poles | Keep in both | Plain only | Chat only | Neither | Exact McNemar p |
  | --- | --- | --- | --- | --- | --- | --- |
  | Qwen2.5-0.5B | 311 | 212 | 40 | 12 | 47 | 0.0001 |
  | Qwen2.5-1.5B | 42 | 28 | 9 | 1 | 4 | 0.02 |

- **It costs sensitivity, not specificity.** Inside the template, the holdouts fire less often. Mean sensitivity falls by 0.044 on 0.5B and by 0.073 on 1.5B, and mean specificity moves by less than 0.01. Of the 40 poles of 0.5B that keep only with plain probes, 27 fail on sensitivity, 12 on specificity and 1 on both. Of the 9 on 1.5B, 7 fail on sensitivity.
- **On 0.5B every depth loses some.** With the template, 92 of 112 keep in the first third. In the middle third 56 of 92 keep, and in the last third 76 of 107. On 1.5B the middle and last thirds lose the most (12 to 8 and 10 to 7), on 14 and 12 poles.

**Deviation from the plan.** The plan had a fourth run, Qwen2.5-1.5B with plain probes again, as a check of the changed script. It wrote no file, so the table keeps the 2026-11-14 run. The chat run on 1.5B gives the same base readings and correlations as that run, which checks the shared code.

**The pole sets differ by one.** Standings changed between runs. The 0.5B chat run has one more pole (L20/N3061 negative). The 1.5B chat run has L13/N4792 positive in place of L17/N7486 positive. The paired comparison uses only the shared poles.

## Layer-level CKA, and blocks of base layers (#5127)

Issue #5127 asks whether the change in Qwen2.5-0.5B's middle third changes the whole layer, or only some of its neurons. The run was on 2026-11-23, with `scripts/base_instruct_cka.py`, on an NVIDIA GPU. The rows are in `base-instruct-cka.jsonl`. The registry now holds 362 VERIFIED poles of Qwen2.5-0.5B, against 311 in the section above.

- **CKA.** Linear CKA (Kornblith et al., 2019, [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)) between the two models' MLP activations, over every neuron of the layer. The sample is the same 20,000 tokens as the correlations above. *Standardised* CKA first scales each neuron to unit variance.
- **Null.** The base model against the same architecture with random weights (seed 0).
- **Swaps.** The instruct model with one third of its blocks taken from the base model. Each pole is then correlated with the base model again.

**Answer: the whole layer changes a little, most in the middle third. The change comes partly from the middle blocks and partly from the blocks before them.**

| Layer | CKA | Standardised CKA | Null CKA | Poles | Median pole r |
| --- | --- | --- | --- | --- | --- |
| L0 | 0.999 | 0.997 | 0.171 | 27 | 0.994 |
| L1 | 0.994 | 0.992 | 0.137 | 14 | 0.989 |
| L2 | 0.171 | 0.988 | 0.018 | 14 | 0.985 |
| L3 | 0.033 | 0.983 | 0.004 | 14 | 0.981 |
| L4 | 0.984 | 0.982 | 0.182 | 22 | 0.979 |
| L5 | 0.976 | 0.982 | 0.236 | 15 | 0.977 |
| L6 | 0.980 | 0.980 | 0.209 | 16 | 0.974 |
| L7 | 0.980 | 0.979 | 0.165 | 10 | 0.979 |
| L8 | 0.977 | 0.978 | 0.156 | 12 | 0.974 |
| L9 | 0.969 | 0.972 | 0.169 | 14 | 0.961 |
| L10 | 0.970 | 0.973 | 0.244 | 11 | 0.963 |
| L11 | 0.967 | 0.967 | 0.096 | 11 | 0.958 |
| L12 | 0.964 | 0.970 | 0.220 | 12 | 0.964 |
| L13 | 0.965 | 0.972 | 0.183 | 16 | 0.961 |
| L14 | 0.973 | 0.979 | 0.260 | 13 | 0.962 |
| L15 | 0.971 | 0.979 | 0.231 | 12 | 0.964 |
| L16 | 0.976 | 0.984 | 0.168 | 22 | 0.971 |
| L17 | 0.980 | 0.984 | 0.232 | 13 | 0.975 |
| L18 | 0.980 | 0.983 | 0.214 | 14 | 0.970 |
| L19 | 0.980 | 0.985 | 0.175 | 13 | 0.974 |
| L20 | 0.986 | 0.987 | 0.243 | 15 | 0.977 |
| L21 | 0.695 | 0.986 | 0.171 | 22 | 0.975 |
| L22 | 0.979 | 0.985 | 0.142 | 17 | 0.973 |
| L23 | 0.981 | 0.984 | 0.101 | 13 | 0.971 |

- **The layers stay close.** Standardised CKA is 0.967 to 0.997 in every layer, against a null of at most 0.260.
- **The middle third moves most.** Its standardised CKA is 0.967 to 0.979, where its median pole r is lowest, at 0.958 to 0.965. The median over layers is 0.983 in the first third, 0.973 in the middle and 0.985 in the last. So the dip that the poles show is in the whole layer too. It is small, though.
- **Raw CKA fails on three layers.** At L2, L3 and L21, raw CKA is 0.171, 0.033 and 0.695, while their poles correlate at 0.975 or more. Standardised CKA there is 0.988, 0.983 and 0.986. So a few neurons with very large activations dominate raw CKA, and a change in their scale hides a layer that barely changed.

| Base blocks in the instruct model | Median r, first third | Middle third | Last third |
| --- | --- | --- | --- |
| None (the instruct model) | 0.982 | 0.962 | 0.974 |
| Layers 0–7 | 0.987 | 0.977 | 0.980 |
| Layers 8–15 | 0.982 | 0.976 | 0.980 |
| Layers 16–23 | 0.982 | 0.962 | 0.983 |

- **Each swap changes only its own blocks and those after them.** That is the check that the copy took. A first version of the script lost the copy on a full GPU, as #5127 records.
- **The middle third's change is not only local.** Base blocks 8–15 raise the middle third's median r from 0.962 to 0.976. Base blocks 0–7 raise it as far, to 0.977, with the middle blocks still the instruct model's. So part of the change reaches the middle third through its input from earlier layers. The two parts overlap, because either swap alone recovers most of the gap.
- **The early blocks don't fit the instruct model's embeddings.** With base blocks 0–7, the first third's poles correlate 0.0028 higher at the median, but some L0 poles fall. L0/N532+ goes from 0.990 to 0.978. The base blocks then read the instruct model's embeddings, which instruction tuning also changed.

## The weight change at each pole (#5133)

Issue #5133 asks whether the weight change of instruction tuning at a pole predicts whether its explanation keeps. Δ is instruct − base for each MLP matrix. The analysis reads the weights only, with `scripts/base_instruct_delta.py`, on 2026-11-24, and the rows are in `base-instruct-delta.jsonl`. The poles, and whether each keeps, come from the plain-probe runs above. Qwen2.5-0.5B has 311 poles, of which 59 fail, and Qwen2.5-1.5B has 43, of which 5 fail.

- **Measures.** Δ's norm in the pole's `gate_proj` and `up_proj` rows and `down_proj` column, over the base weight's norm there, for each matrix and for all three. Also the share of the pole's base weights in Δ's top-r singular directions. Here r is Δ's rank for 90% of its variance ([Biderman et al., 2024](../literature/lora-learns-less-forgets-less.md), §4.6, and [Hu et al., 2021](../literature/lora-low-rank-adaptation.md), Table 7).
- **Test.** Each measure as an AUROC for predicting a failure. The null shuffles the keep labels within each layer, 10,000 times. A measure that only tracks depth then gets no credit.
- **Rotation.** The share of Δ's energy on the top 5% of neurons. The null is the same share after a Haar-random rotation of the neuron axis ([Li et al., 2018](../literature/intrinsic-dimension-objective-landscapes.md)). A spread update gives the rotated share.

**Answer: no.** Neither measure predicts which explanations keep, beyond what depth predicts.

| Measure | AUROC, 0.5B | p, 0.5B | AUROC, 1.5B | p, 1.5B |
| --- | --- | --- | --- | --- |
| Δ's size in `gate_proj` | 0.60 | 0.82 | 0.58 | 0.25 |
| Δ's size in `up_proj` | 0.62 | 0.36 | 0.61 | 0.04 |
| Δ's size in `down_proj` | 0.61 | 0.32 | 0.42 | 0.46 |
| Δ's size, all three | 0.61 | 0.66 | 0.57 | 0.16 |
| Share of the weights in Δ's top-r directions | 0.44 | 0.76 | 0.42 | 0.19 |

- **The raw AUROCs are depth.** On Qwen2.5-0.5B, Δ's size gives AUROCs of 0.60 to 0.62. But shuffles within each layer reach them as often, with p from 0.32 to 0.82. Failures fall in the layers where Δ is larger overall, the middle third, and inside a layer Δ's size says nothing. The median relative change at a pole is 0.078 for poles that keep and 0.084 for poles that fail.
- **1.5B can't test it.** It has 5 failures. One measure, `up_proj`, reaches p = 0.04, but it is one of five measures. With five failures, that is not evidence.
- **Δ is high-rank and spread out.** Its 90% rank is 668 to 701 of 896 in 0.5B, and 679 to 1105 of 1,536 in 1.5B. The top 5% of neurons hold 6.6% of its energy in 0.5B, against 5.3% after a rotation. In 1.5B they hold 6.7% against 5.4%. So instruction tuning moves almost every neuron a little, not a few neurons a lot.
- **0.5B changed more.** Its weights moved about 8% at a pole, at the median, and 1.5B's about 1.2%.

## Reading it

Instruction tuning leaves these neurons in place. A neuron's activation pattern is almost the same after tuning, and its explanation still predicts its probes in eight or nine cases in ten. So a VERIFIED explanation of a base model is a good first guess for its instruct model, at the same index.

The chat template matters more than the tuning. Inside it, the same neurons fire less on the text that their explanations describe. The fall is in sensitivity, so the explanation still says when the pole is quiet, but it fires less often where it should. A base-model explanation is a weaker guess for an instruct model in use, which reads every prompt inside its template.

## Limits

- **Two pairs, one family.** Qwen2.5-0.5B adds 311 poles, and its interval is 76–85%. Both pairs are Qwen2.5, so another family could differ.
- **The base model's own probes.** The reviewer wrote them to break the base model's hypothesis. Probes written against the instruct model could find differences these miss. That would need a review of the instruct model itself, which would make it a registry model.
- **One template, one turn.** The chat runs put each probe in one user turn. A system prompt, earlier turns or the assistant's side of the template were not tested. The scale ratio comes from plain text, so a template that shifts a pole's whole range would show up as lost sensitivity here.
- **The weight change is one test.** Five measures, one null, and the 1.5B pair has only 5 failures.
- **CKA and swaps on one pair.** The CKA and the block swaps cover only Qwen2.5-0.5B, on one sample, and one null seed. The swaps read correlations, not the probes' gates.
- **Not a crosscoder.** This compares neuron to neuron at one index. A learned shared basis, the crosscoder of Lindsey et al. (2024), could also find features that moved between neurons. That needs training, and it is a follow-up.

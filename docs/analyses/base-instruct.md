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

## Reading it

Instruction tuning leaves these neurons in place. A neuron's activation pattern is almost the same after tuning, and its explanation still predicts its probes in eight or nine cases in ten. So a VERIFIED explanation of a base model is a good first guess for its instruct model, at the same index.

The chat template matters more than the tuning. Inside it, the same neurons fire less on the text that their explanations describe. The fall is in sensitivity, so the explanation still says when the pole is quiet, but it fires less often where it should. A base-model explanation is a weaker guess for an instruct model in use, which reads every prompt inside its template.

## Limits

- **Two pairs, one family.** Qwen2.5-0.5B adds 311 poles, and its interval is 76–85%. Both pairs are Qwen2.5, so another family could differ.
- **The base model's own probes.** The reviewer wrote them to break the base model's hypothesis. Probes written against the instruct model could find differences these miss. That would need a review of the instruct model itself, which would make it a registry model.
- **One template, one turn.** The chat runs put each probe in one user turn. A system prompt, earlier turns or the assistant's side of the template were not tested. The scale ratio comes from plain text, so a template that shifts a pole's whole range would show up as lost sensitivity here.
- **Not a crosscoder.** This compares neuron to neuron at one index. A learned shared basis, the crosscoder of Lindsey et al. (2024), could also find features that moved between neurons. That needs training, and it is a follow-up.

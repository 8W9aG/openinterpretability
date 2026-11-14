# Base and instruct: do verified neurons keep their hypothesis?

Issue #3375. Instruction tuning changes a model's behaviour. Does it change what its neurons compute? This page takes every neuron pole of Qwen2.5-1.5B that stands VERIFIED, and tests its hypothesis again on Qwen2.5-1.5B-Instruct. Run on 2026-11-14 with `scripts/base_instruct.py`. The rows are in `base-instruct.jsonl`. Reported, not gated. No record was written.

**Answer: nearly all keep, and the neurons barely move.** 38 of 43 hypotheses still pass their own probes on the instruct model: 88% (95% CI 76–95%). Each neuron's activation correlates with its instruct counterpart at r = 0.95 or more, median 0.983. The five that fail were near a gate on the base model too, and each one fails by a probe or two.

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

## Reading it

Instruction tuning leaves these neurons in place. A neuron's activation pattern is almost the same after tuning, and its explanation still predicts its probes in nearly nine cases in ten. So a VERIFIED explanation of a base model is a good first guess for its instruct model, at the same index.

## Limits

- **One pair and 43 poles.** The interval on the keep share is wide (76–95%). Qwen2.5-0.5B and its instruct model are a follow-up.
- **The base model's own probes.** The reviewer wrote them to break the base model's hypothesis. Probes written against the instruct model could find differences these miss. That would need a review of the instruct model itself, which would make it a registry model.
- **Prompts in the base format.** The probes are plain text, not chat-formatted. An instruct model may compute differently inside its chat template.
- **Not a crosscoder.** This compares neuron to neuron at one index. A learned shared basis, the crosscoder of Lindsey et al. (2024), could also find features that moved between neurons. That needs training, and it is a follow-up.

# The untrained-model null

Run on 2026-11-14 for issue #3202, with `scripts/untrained_null.py`. The rows are in `untrained-null.jsonl`, one per unit, and the summary is in `untrained-null.json`. Reported, not gated.

The question: does the protocol verify units of a model that learned nothing? The null calibrations (`null-calibration.md`) vary the hypotheses and the basis, but not the weights. A random network still has structure, because token and position embeddings give its neurons something to respond to. So some random units may verify, and zero isn't the expected rate. A rate near the trained model's would mean the protocol rewards something other than what the model learned.

## Method

- **Units.** 20 GPT-2 small positive poles, with the layer drawn uniformly and then the neuron (seed 3202). Each pole is audited twice: on the trained weights, and on a copy of the same architecture with fresh weights (`from_config`, seed 0). Negative poles are left out, because in GPT-2 most of them are default-on (`position-zero.md`).
- **Blinding.** The 40 units are shuffled under opaque names, and every prompt names GPT-2. Proposers measured their texts only through `scripts/measure_unit.py`, which runs the unit's own weights. Reviewers saw only the submission. Two deviations, both on the issue:
  - The first 9 proposers could tell what the study tested, so their answers were discarded and all 40 were proposed again under neutral names.
  - Two units each got a second proposer run by mistake. The first, submitted proposal was kept for both.
- **Pipeline.** The scan used the registry corpus, the 200,000-token sample, 128-token windows and float32. Each unit then got:
  - one `neuron-proposer` (Claude Opus 5.5).
  - one `neuron-reviewer` (Claude Sonnet 5.5), writing 30 probes: 20 hard negatives and 10 holdouts.
  - today's decision rule, applied to the measured probes and counterfactuals, as `null_calibration.review_probes` does. Causal metrics and wiring are left out, because they aren't gated.

## Result

| Arm | Units | VERIFIED | Passes without the proposer's gates | Mean balanced accuracy | REJECTED |
| --- | --- | --- | --- | --- | --- |
| Trained GPT-2 | 20 | 9 (45%, 95% CI 26–66%) | 11 | 0.87 | 0 |
| Random-init GPT-2 | 20 | 2 (10%, 95% CI 3–30%) | 2 | 0.62 | 8 |

- **The protocol rewards what the model learned.** Trained units verify 4.5 times as often (one-sided Fisher exact p = 0.015). No scan stopped in either arm.
- **Random units fail on specificity.** The gate failed on 17 of the 20 untrained units, and `beats_confound_baseline` on 11. Their hypotheses are broad, because their top examples share no theme, and the reviewers' hard negatives fire too.
- **Pairs agree with the totals.** In 9 of the 20 pairs the trained unit verified and its untrained twin did not. In 2 the reverse held, and in 9 neither verified.

## What the two random units that verified claim

Both claims are about context length, not about a token or a meaning:

- **u01 (L11/N2129+).** "A diffuse, context-wide dial". It is silent at the start of a text, then 0.1–0.5 of max on many tokens after about 20. It peaks on paragraph breaks and word pieces.
- **u09 (L10/N2881+).** "Silent on short texts (roughly the first 20–30 tokens)" and sporadic later, on determiners after prepositions and on technical word pieces.

A random GPT-2 still adds a position embedding to every token, random but fixed. So its neurons can respond to how far into a text a token sits, and these two do. The reviewers tested the claims as stated: short hard negatives and long holdouts. In u01 the holdouts averaged 35 words against 14 for the negatives, and in u09 90 against 37. A rule that reads only probe length would have scored about as well as the hypothesis. The confound baseline can't catch this, because it tries token-identity, position-profile and never-active rules, but no rule of probe length.

## Limits

- **20 pairs, one random seed, one model.** The intervals are wide. The untrained rate could be as high as 30%.
- **One proposer and one reviewer model.** Both are Claude models.
- **No causal metrics or wiring.** These aren't gated, but a random unit might differ on them.
- **Blinding was partial.** Proposers and reviewers could see that the untrained units' top examples share no theme, and that may have shaped how much effort a hypothesis got.

## What this means

- **The protocol passes the sanity test.** It does not verify random units at the trained rate.
- **Length-based claims need a length control.** Both random passes were "fires once the text is long enough" hypotheses, tested with short negatives and long holdouts. Issue #3525 holds adding probe length to the confound baseline. That would change the decision rule, so it needs a protocol bump.
- **Run it again on a second model or seed** before quoting the 10% as a false-positive rate.

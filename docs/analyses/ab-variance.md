# Less noise in prompt A/B tests

Run on 2026-11-21 for issue #3198, with `scripts/ab_variance.py`. The rows are in `ab-variance.jsonl`, one per submission and review, and the summary is in `ab-variance.json`. Reported, not gated.

The question: a prompt A/B test compares two arms on the same units, and its noise sets how many units it needs. In `prompt-trim-ab.md`, both the proposer and the reviewer varied, and the paired Δ balanced accuracy had an SD of about 0.14. Detecting a shift of 0.05 then needs about 62 units. Which changes to the design cut that number, and by how much?

## Method

- **Data.** The 30 submissions of `reviewer-crossover.md` and `verdict-stability.md`. Each has three Sonnet 5.5 reviews at 20 probes and two at 30 probes, all on one prompt. Each also has one review by Opus 5.5, by Haiku 5.5 and by the original reviewer. No new agent ran for this note.
- **A null A/B test.** Two reviews of one submission on one prompt differ only by chance. So their paired difference is the noise of a reviewer-prompt A/B test in which the hypotheses are held fixed.
- **Units needed.** The units for a two-sided 5% test at 80% power: n = ((1.96 + 0.84) × SD / shift)². The shift is 0.05 in balanced accuracy, or 10 points in the VERIFIED rate.
- **More reviews.** The pooled variance of one review around its submission's mean gives the SD when each arm averages k reviews: √(2σ²/k). This assumes that reviews are independent. The single-review value matches the measured paired SD (0.105 against 0.103).

## Results

| Design | SD of one unit's difference | Units to detect the shift |
| --- | --- | --- |
| Both sides vary, 1 review (`prompt-trim-ab.md`) | 0.14 | 62 |
| Proposer fixed, 1 review of 20 probes | 0.103 | 34 |
| Proposer fixed, 1 review of 30 probes | 0.081 | 21 |
| Proposer fixed, mean of 2 reviews of 20 probes | 0.074 | 18 |
| Proposer fixed, mean of 3 reviews of 20 probes | 0.061 | 12 |
| VERIFIED or not, 1 review of 20 probes, 10-point shift | 0.41 | 132 |
| Unpaired arms, 1 review each, per arm | 0.24 | 180 |
| Unpaired arms, adjusted for an earlier review (CUPED) | 0.15 | 72 |

- **Hold the proposer fixed.** Reusing the same hypotheses in both arms cuts the SD from 0.14 to 0.10 and the units from 62 to 34. The proposer side is a large part of the noise.
- **Use 30 probes.** One review of 30 probes is about as good as two reviews of 20, at less cost. The harness already asks for 30 (#3509).
- **Replicate the reviewer.** Averaging k reviews shrinks the SD by about √k, so three reviews of 20 probes need 12 units.
- **Compare balanced accuracy, not the verdict.** VERIFIED or not flips between two reviews of the same submission in 18% of pairs. 29% of reviews have a specificity within 0.05 of its 0.80 gate. Detecting a 10-point shift in the VERIFIED rate needs 132 units.
- **Pair the arms.** Units differ much more than reviews do. Arms on different units need 180 units each. CUPED adjusts for an earlier review of each unit, here the Opus review (ρ = 0.77). It removes 60% of the variance, which still leaves 72 units per arm. Pairing removes the unit's effect outright, so CUPED adds little to a paired test.

## A procedure for prompt A/B tests

1. **Pair the arms.** Run both arms on the same units.
2. **Change one side at a time.** For a reviewer-prompt change, reuse frozen hypotheses in both arms. For a proposer-prompt change, measure both arms' hypotheses on one shared probe set.
3. **Use 30 probes and compare balanced accuracy.** Report the verdicts too, but don't test on them.
4. **Plan the units from the table.** With the proposer fixed, 21 units detect a 0.05 shift with one 30-probe review per arm. Add reviews before adding units if units are scarce.

## Limits

- **The reviewer side only.** No study has repeated proposals of one unit. So the proposer's own noise is known only as the gap between 0.14 and 0.10, from 8 units in `prompt-trim-ab.md`. #5766 would measure it.
- **One reviewer and one prompt.** The replicates are all Sonnet 5.5 on the protocol-0.8 prompt. Another reviewer may be noisier.
- **30 submissions.** Each SD has an interval of roughly ±25%, so the unit counts are estimates, not exact numbers.
- **The √k rule assumes independent reviews.** The three 20-probe reviews fit it, but more reviews may share a reviewer's habits.

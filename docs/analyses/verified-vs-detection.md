# Does VERIFIED predict held-out detection?

Run on 2026-11-13 for issue #3239, with `scripts/verified_vs_detection.py`. The numbers are in `verified-vs-detection.json`. A VERIFIED verdict says that an explanation survived probes that a reviewer chose. Detection and fuzzing scores say how well a judge, given only the explanation, finds where the pole fires in ordinary held-out text (`openinterp score`). If VERIFIED explanations score much higher, the two measures agree, and a score could become a gate. If they score about the same, the verdict measures something the scores don't.

The null-explanation control (`null-explanation-control.md`) showed that with a Haiku judge a score above 0.5 comes from the explanation. So the scores here can be read as the explanation's own.

## Method

- **Records.** Every neuron-pole record with explanation scores: 2,288 for detection (581 VERIFIED) and 1,967 for fuzzing (501 VERIFIED). Protocols 0.5 to 0.8, on GPT-2, Qwen2.5-0.5B and Qwen2.5-1.5B.
- **Measure.** Balanced accuracy of the judge's answers on held-out windows, for VERIFIED against all other records.
- **Within judge.** Judges differ in how strict they are, and the judge mix differs between arms. So the main number is the difference within each judge, pooled over judges with weights n₁n₂/(n₁+n₂). A judge needs at least 10 records in each arm to count. The 95% intervals come from 2,000 bootstrap samples, drawn within each judge.
- **One record per unit.** The records of one unit are not independent. So the same numbers are also given for each unit's latest record only (1,868 units).
- **AUROC.** How well the score alone separates VERIFIED records from the rest.

## Results

| | Detection | Fuzzing |
| --- | --- | --- |
| VERIFIED mean | 0.647 | 0.617 |
| Other mean | 0.600 | 0.574 |
| Difference within judge | +0.040 (95% CI 0.024 to 0.057) | +0.030 (0.015 to 0.046) |
| Same, latest record per unit | +0.041 (0.024 to 0.059) | +0.034 (0.017 to 0.051) |
| AUROC | 0.59 | 0.61 |

- **VERIFIED explanations score higher, but only a little.** The difference is real (its interval excludes zero), and it holds for each record and for each unit. It is about 4 points of balanced accuracy on a scale where chance is 50.
- **The order follows the verdict.** Mean detection is 0.647 for VERIFIED, 0.612 for NEEDS_REVISION and 0.580 for REJECTED. Fuzzing gives 0.617, 0.584 and 0.558.
- **A score separates the verdicts poorly.** An AUROC of 0.59 means that a VERIFIED record outscores a random other record only 59% of the time.
- **It holds for every judge.** Detection differences are +0.055 with Haiku, +0.033 with DeepSeek Flash and +0.050 with Muse Spark. In the registry the judge is almost always the reviewer model, so judge and reviewer can't be told apart here.
- **It differs by model.** On Qwen2.5-0.5B and Qwen2.5-1.5B, VERIFIED detection is 7–8 points higher. On GPT-2 it is 1 point higher. GPT-2's fuzzing difference is larger (+6), and Qwen2.5-1.5B's is negative, but on 28 VERIFIED records.
- **Protocol 0.8 narrows the gap.** The detection difference is +0.072 under 0.7 and +0.034 under 0.8. Protocol 0.8's changes (default-on poles stopped, ties with the confound baseline pass) changed which records verify. This run can't say which change did it.

## Limits

- **230 records have no scores.** 116 of them are VERIFIED. They come from protocols 0.5 to 0.8 and from many dates, not from one group, so the comparison should not shift much. They are left for a follow-up issue.
- **Judge and reviewer are mostly the same model.** A strict reviewer can also be a strict judge, and then both arms score lower together. The within-judge pooling removes the level, not that link.
- **Scores are noisy.** Each uses about 50 detection windows and 37 fuzzing windows, from one seed. Much of the spread within each arm is window noise, and that noise also lowers the AUROC.
- **The windows sample every activation bin.** A VERIFIED explanation can be right about the top activations and miss the weak ones (`recall_by_quantile` in each record). The probes test the claim where the reviewer looked, and the windows test where the pole fires in ordinary text.

## What this means

- **Don't make detection or fuzzing a gate now.** The scores agree with the verdict only weakly. A gate at any one threshold would reject many VERIFIED explanations and pass many that failed review. A gate also needs a fixed judge, since judges differ by 5 to 8 points.
- **The verdict and the scores measure different things.** Keep both, and report both on each record, as now. An explanation that is VERIFIED but scores near chance is narrower than where the pole fires. That is a revision, not a failed review.
- **Before a gate:** score every record with one judge model, and with more windows to cut noise. Score the 230 records that have none. Then repeat this comparison.

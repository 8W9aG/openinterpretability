# Does VERIFIED predict held-out detection?

Run on 2026-11-13 for issue #3239, and again on 2026-11-15 for issue #3499 after the score backfill, with `scripts/verified_vs_detection.py`. The numbers are in `verified-vs-detection.json`. A VERIFIED verdict says that an explanation survived probes that a reviewer chose. Detection and fuzzing scores say how well a judge, given only the explanation, finds where the pole fires in ordinary held-out text (`open-interpretability score`). If VERIFIED explanations score much higher, the two measures agree, and a score could become a gate. If they score about the same, the verdict measures something the scores don't.

The null-explanation control (`null-explanation-control.md`) showed that with a Haiku judge a score above 0.5 comes from the explanation. So the scores here can be read as the explanation's own.

## Method

- **Records.** Every neuron-pole record with explanation scores: 2,623 for detection (718 VERIFIED) and 2,297 for fuzzing (636 VERIFIED). Protocols 0.5 to 0.8, on GPT-2, Qwen2.5-0.5B and Qwen2.5-1.5B.
- **Backfill.** 197 of these records had no scores of their own. Issue #3499 scored them afterwards, with `scripts/score_backfill.py` and a Claude Haiku 5.5 judge run from Claude Code. Their scores are in `explanation-scores-backfill.jsonl`, and the records stay as they are. The other 2,426 records hold their own scores.
- **Measure.** Balanced accuracy of the judge's answers on held-out windows, for VERIFIED against all other records.
- **Within judge.** Judges differ in how strict they are, and the judge mix differs between arms. So the main number is the difference within each judge, pooled over judges with weights n₁n₂/(n₁+n₂). A judge needs at least 10 records in each arm to count. The 95% intervals come from 2,000 bootstrap samples, drawn within each judge.
- **One record per unit.** The records of one unit are not independent. So the same numbers are also given for each unit's latest record only (1,999 units).
- **AUROC.** How well the score alone separates VERIFIED records from the rest.

## Results

| | Detection | Fuzzing |
| --- | --- | --- |
| VERIFIED mean | 0.640 | 0.617 |
| Other mean | 0.602 | 0.576 |
| Difference within judge | +0.033 (95% CI 0.020 to 0.048) | +0.028 (0.015 to 0.042) |
| Same, latest record per unit | +0.035 (0.019 to 0.051) | +0.033 (0.017 to 0.049) |
| AUROC | 0.57 | 0.60 |

- **VERIFIED explanations score higher, but only a little.** The difference is real (its interval excludes zero), and it holds for each record and for each unit. It is about 3 points of balanced accuracy on a scale where chance is 50. Before the backfill it was 4 points (+0.040 for detection, +0.030 for fuzzing).
- **The order follows the verdict.** Mean detection is 0.640 for VERIFIED, 0.614 for NEEDS_REVISION and 0.581 for REJECTED. Fuzzing gives 0.617, 0.586 and 0.559.
- **A score separates the verdicts poorly.** An AUROC of 0.57 means that a VERIFIED record outscores a random other record only 57% of the time.
- **It holds for every judge, but least with Haiku.** Detection differences are +0.014 with Haiku, +0.032 with DeepSeek Flash and +0.046 with Muse Spark. Haiku's difference was +0.055 before the backfill, on 137 records. The backfill added 197 Haiku-judged records. On those, VERIFIED records score 0.596 for detection (98 records) and the others 0.609 (99 records), so the gap reverses. In the registry the judge is almost always the reviewer model, so judge and reviewer can't be told apart here.
- **It differs by model.** On Qwen2.5-0.5B and Qwen2.5-1.5B, VERIFIED detection is 6–7 points higher. On GPT-2 it is 1 point higher. GPT-2's fuzzing difference is larger (+6), and Qwen2.5-1.5B's is negative (−6), but on 32 VERIFIED records.
- **Protocol 0.8 narrows the gap.** The detection difference is +0.070 under 0.7 and +0.028 under 0.8. Protocol 0.8's changes (default-on poles stopped, ties with the confound baseline pass) changed which records verify. This run can't say which change did it.

## Limits

- **48 records still have no scores.** 24 of them are VERIFIED. The backfill could not score them:
  - 26 poles fire in at most 5 of the 4,000 held-out windows, and 14 in none. They have no windows to fuzz, and detection would rest on 0 to 10 windows, so a score would mean nothing.
  - 15 records don't say how to rebuild their scan corpus.
  - 6 records are newer than the backfill.
  - 1 record's task could not be written.
- **The backfill used one judge, Haiku.** Its records are older ones that the harness had not scored, so they are not a random sample. The within-judge pooling compares them only with other Haiku-judged records.
- **Judge and reviewer are mostly the same model.** A strict reviewer can also be a strict judge, and then both arms score lower together. The within-judge pooling removes the level, not that link.
- **Scores are noisy.** Each uses about 50 detection windows and 37 fuzzing windows, from one seed. Much of the spread within each arm is window noise, and that noise also lowers the AUROC.
- **The windows sample every activation bin.** A VERIFIED explanation can be right about the top activations and miss the weak ones (`recall_by_quantile` in each record). The probes test the claim where the reviewer looked, and the windows test where the pole fires in ordinary text.

## What this means

- **Don't make detection or fuzzing a gate now.** The scores agree with the verdict only weakly. A gate at any one threshold would reject many VERIFIED explanations and pass many that failed review. A gate also needs a fixed judge, since judges differ by 5 to 8 points.
- **The verdict and the scores measure different things.** Keep both, and report both on each record, as now. An explanation that is VERIFIED but scores near chance is narrower than where the pole fires. That is a revision, not a failed review.
- **Before a gate:** score every record with one judge model, and with more windows to cut noise. The backfill has scored all but 48 records (issue #3499). Then repeat this comparison.

# A single-judge rescore pilot

Run on 2026-11-16 for issue #4102, with `scripts/rescore_pilot.py`. The rows are in `rescore-pilot.jsonl` and the numbers in `rescore-pilot.json`.

The registry's detection and fuzzing scores separate VERIFIED records from the rest poorly, with an AUROC of 0.57 (`verified-vs-detection.md`). Two kinds of noise could hide a stronger link. The scores mix judges that differ by 5 to 8 points, and each uses about 50 windows from one seed. This pilot removes both from 200 records. If the AUROC then rose a lot, a full single-judge rescore would be worth its cost, and a score might become a gate (#3209).

## Method

- **Records.** The latest record of each neuron-pole unit: 100 VERIFIED and 100 others (NEEDS_REVISION or REJECTED). Each arm is split over the models in proportion to the registry, with a seeded draw. The VERIFIED arm has 57 records of Qwen2.5-0.5B, 33 of GPT-2, 8 of Qwen2.5-1.5B and 2 of Pythia-70m. The other arm has 41, 42, 13 and 4.
- **Corpus.** Each record's own scan corpus, rebuilt from its recorded source. 34 records of the first draw were scanned on the built-in corpus of 40 short texts, which is too small to draw held-out windows from. Each was replaced by an unchosen record of the same model and arm.
- **Windows.** Twice the default: 10 active windows per activation bin (`--per-quantile 10`) and up to 8,000 held-out windows. The corpus sets the real count, with a median of 7,076 (range 1,192 to 8,000). Two seeds per record.
- **Judge.** Claude Haiku 5.5 only, run from Claude Code with one `explanation-judge` subagent per prompt. No judge saw both prompts of a unit.
- **Measures.**
  - The AUROC of the score for VERIFIED against the rest, and the difference in means. Both have 95% intervals from 2,000 bootstrap samples.
  - The correlation between the two seeds' scores of each record, and the Spearman–Brown projection to more seeds.
  - The same records' existing scores, from mixed judges and one seed, for comparison.

## Results

| | Detection | Fuzzing |
| --- | --- | --- |
| Records scored | 200 | 188 |
| AUROC, Haiku, mean of 2 seeds | 0.54 (95% CI 0.45 to 0.62) | 0.58 (0.50 to 0.66) |
| AUROC, Haiku, 1 seed | 0.52 (0.43 to 0.59) | 0.61 (0.53 to 0.68) |
| AUROC, existing scores of the same records | 0.52 (0.44 to 0.60) | 0.53 (0.44 to 0.62) |
| VERIFIED minus other, mean of 2 seeds | +0.014 (−0.023 to +0.051) | +0.030 (−0.001 to +0.063) |
| Correlation of the two seeds | 0.93 | 0.76 |
| Projected reliability of 4 seeds | 0.98 | 0.93 |
| Correlation with the existing scores | 0.77 | 0.58 |

- **One judge and twice the windows don't separate the verdicts.** The detection AUROC is 0.54, and its interval includes 0.5. Fuzzing does a little better, at 0.58.
- **Window noise is small.** Two seeds of the same record agree closely for detection (0.93) and well for fuzzing (0.76). More seeds would add little. Four seeds would raise detection's reliability from 0.93 to 0.98, and the AUROC can't rise much from that.
- **The mixed judges were not what hid a link.** On these 200 records, the existing scores separate the verdicts no better (0.52 and 0.53). The new scores agree with them (0.77 for detection), so the judge mix changed the level of a score more than its order.
- **Mean scores by verdict.** Detection gives 0.622 for VERIFIED, 0.615 for NEEDS_REVISION and 0.592 for REJECTED. Fuzzing gives 0.623, 0.592 and 0.597. The order follows the verdict, as in `verified-vs-detection.md`, but the gaps are about 1 to 3 points.
- **Rare poles now get a detection score.** The 9 sampled records that had no score now have one. 12 records still get no fuzzing score, because their poles fire in too few windows to mark.

## Limits

- **200 records.** The interval on each AUROC is about ±0.08. It can rule out a large effect, not a small one.
- **The two seeds share a corpus.** They draw different windows from the same pool, with a median of 7,076. For a pole with few active windows, both seeds see most of the same ones. So the seed correlation is an upper bound on how well a fresh corpus would repeat.
- **One judge.** Haiku was the judge here and in #3499, and `null-explanation-control.md` checked it. Another judge could order records differently.
- **The other arm mixes two verdicts.** NEEDS_REVISION records are often close to VERIFIED, and they make up two thirds of the other arm. A gate that only had to reject REJECTED records would face an easier task.

## What this means

- **A full single-judge rescore is not worth its cost for this question.** The pilot removed the two kinds of noise that might have hidden a link, and the AUROC stayed between 0.54 and 0.58.
- **#3209 can be decided: don't make detection or fuzzing a gate.** The verdict and the scores measure different things. Keep both and report both, as `verified-vs-detection.md` says.
- **Detection scores repeat well across seeds,** so one seed of about 100 windows is enough to report a score.

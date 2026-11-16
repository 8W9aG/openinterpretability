# Explanation scores across the activation range

Issue #3238. Oikarinen et al. (arXiv 2506.07985, `docs/literature/beyond-top-activations.md`) show that a judge that sees only a unit's strongest inputs overrates its explanation. This analysis asks which part of a pole's activation range `openinterp score` reads, and whether the judge's recall depends on it. Run on 2026-11-16 with `scripts/score_by_range.py`, over every record whose scores keep recall by bin. The rows are in `score-by-range.jsonl`. Reported, not gated.

## Method

- **The windows.** `sample_heldout` (`openinterp/harness/explanation_scores.py`) cuts held-out text into 32-token windows. A window is active when it peaks at or above the pole's 0.25×max line, the reviewer's threshold. Active windows are sorted by peak and cut into 5 bins of equal count. Detection and fuzzing each draw 5 windows from each bin. Each record keeps the 4 bin edges and the judge's recall in each bin.
- **Coverage.** The bin edges, as a share of the pole's maximum activation in its scan.
- **Recall by bin.** The mean recall in each bin over records, pooled and for each judge with at least 20 records. Intervals are 95% bootstrap intervals over records. A bin with no labelled window is left out of that bin.
- **Top-only bias.** For each record with all 5 bins, the top bin's detection recall minus the mean over all 5 bins. A score drawn only from the strongest windows would read the top bin.

## Results

The 2,394 records with recall by bin hold 2,114 with bin edges.

| Bin | Median lower edge, × max | Detection recall | Fuzzing recall |
| --- | --- | --- | --- |
| 1 (weakest) | 0.25 (the line) | 0.38 (0.37–0.40) | 0.20 (0.18–0.21) |
| 2 | 0.28 | 0.39 (0.37–0.41) | 0.20 (0.19–0.22) |
| 3 | 0.32 | 0.41 (0.40–0.43) | 0.22 (0.21–0.24) |
| 4 | 0.36 | 0.45 (0.43–0.47) | 0.24 (0.22–0.25) |
| 5 (strongest) | 0.44 | 0.51 (0.50–0.53) | 0.28 (0.27–0.30) |

- **The scores look mostly at weak activations, not at the top.** Four fifths of the active windows peak between 0.25 and 0.44 of the maximum. The fifth bin spans the rest of the range, up to 1.0. Its lower edge is below 0.39×max for a quarter of records and above 0.53×max for another quarter. So the score is not drawn from top activations. If anything, it leans the other way.
- **Recall rises with strength.** Detection recall goes from 0.38 in the weakest bin to 0.51 in the strongest, and fuzzing from 0.20 to 0.28. The intervals of the first and the last bin don't overlap. The rise holds for each judge with enough records. Detection rises from 0.38 to 0.53 for deepseek-flash, from 0.41 to 0.52 for muse-spark and from 0.25 to 0.37 for Haiku. Gemini has 24 records, too few to tell.
- **A score from the top bin alone would read 8.5 points higher.** Over 2,069 records with all five bins, top-bin detection recall is 0.51, against 0.43 over all bins. The gap is 0.085 (0.076–0.094). That is the bias that the paper describes. The registry's equal-count bins already avoid it.

## What this means

The concern behind #3238 does not hold for the registry's active windows. Its scores already sample the whole active range, weighted towards its low end. One part of the range is still thin. Windows below the 0.25×max line count as inactive and are drawn uniformly, so detection rarely meets a weak but real activation. That is #3595, which importance-samples the inactive windows.

## Limits

- **Recall only.** The table counts the active windows that the judge found. Each record's false positive rate comes from its inactive windows, which this analysis doesn't split by strength.
- **Five windows per bin.** Each record gives each bin a recall of 0, 0.2, 0.4, 0.6, 0.8 or 1, so a single record says little. The means are over about 2,000 records each.
- **Mixed judges and protocols.** The pooled means mix judges and protocol versions. The per-judge rows show the same trend, but their levels differ by up to 0.16.
- **The maximum comes from the scan.** Edges are a share of the scan corpus's maximum. The held-out text is another sample, so a held-out window can peak above 1.0×max.

# The Winner's Curse in LLM Self-Improvement Loops

**Citation.** Litao Hu (Meta) and Yutong Tang (Microsoft). "The Winner's Curse in LLM Self-Improvement Loops: Selection Noise, Lock-in, and Acceptance Rules". [arXiv:2610.09239](https://arxiv.org/abs/2610.09239) [cs.AI, cs.LG], v1 of 7 October 2026, the only version, 34 pages. Preprint, no venue given. It is under CC BY 4.0. The code and the 939 run records are arXiv ancillary files. The paper states no funding and no use of AI in the writing. The compute was one RTX 3080 Ti and about 28 rented GPU-hours. The README calls the pre-registration "dated local written records, not an independently timestamped public registration". The plan isn't in the package, and the paper never states its five hypotheses in full.

**Status.** Not applicable. The repo runs no self-improvement loop. Its verdicts are a selection under noise, and this page estimates that effect from the paper's model. The repo already has both of the paper's remedies, scores on held-out text and fresh re-reviews, so no new issue is opened.

**Read.** On 2026-11-27 a subagent (Claude) downloaded the PDF and turned it into text. It read all 34 pages and all 18 tables, and checked each equation it quotes against the HTML's LaTeX. It also read the ancillary file list and the README. It saw the figures only as captions, and it didn't run the code. The reader of this page was a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** "Keep it if it scores better" is selection under noise. A loop that picks the best of several rewrites on a small selection set, and reuses that set, overstates its own gain. The overstatement shrinks with more selection items, but no acceptance rule tested beat greedy acceptance. A few items that are never used for selection give an unbiased reading of the final gain.

**Method.**

- **The loop.** A model rewrites its own system message. Each generation, it proposes K = 4 rewrites from four failures. Each rewrite is scored against the incumbent on n selection items. Greedy acceptance keeps the best one if its measured gain is above 0.
- **Models and tasks.** Qwen2.5-1.5B and 7B Instruct, as both solver and proposer, with Qwen2.5-14B as a stronger proposer in one arm. The tasks are TREC-50, Banking77 and GSM8K. Each task has 600 gold items that no loop sees.
- **The pre-registered study.** Four settings, nine arms and seeds 11 to 18, which gives 336 runs. A separate pool of 256 items supplies the failures, so n changes only the scoring.
- **Rules tested.** Greedy acceptance at n = 16, 64 and 256, and a one-sided McNemar test. Also PACE, an e-process, and a rule that picks on one half of the items and commits on the other. Also Bayes gates, with a prior from a pilot run or from the loop's history.
- **Theory.** Candidates share an error c from their common incumbent and items, and each has its own error. Proposition 1 gives the overstatement of the best of K as (1 − h²)σκ_K + E[c]. κ_K is the expected maximum of K standard normals, about 1.03 for K = 4.

**Findings, as the source reports them.**

| Result | Setting |
| --- | --- |
| After the first rewrite, only 4–39% of rewrites improve on their incumbent | Exploratory runs, judged on the 600 gold items |
| The final selection score overstates held-out accuracy by 13–20 points at n = 16, and by 1–5 points at n = 256 | Pre-registered greedy runs. The drop is significant after Holm correction in three settings, and Banking77 just misses (p = 0.058) |
| The pilot-calibrated Bayes gate scores −2.1 ± 1.1 points against greedy (p = 0.06). The split-half rule scores −1.6 ± 1.2 | n = 16, over whole runs |
| McNemar at α = 0.05 loses 7.4 and 12.3 points | TREC |
| Scoring the seed and the final instruction on 64 unused items cuts the bias of the reported gain to about 0. Its RMSE falls from 15.5 to 5.7 points | 50 random halvings of the gold set |
| GEPA and MIPROv2 overstate their gains by up to 26 and 7 points | 16 validation items |

## Where the claims go beyond the evidence

- **A pre-registration that no one else can check.** The plan is a local record, not in the package, with clock times and no date. The analysis script was written when 16 of 336 runs had finished. The authors say so, to their credit.
- **Small models and one kind of artifact.** Every loop rewrites a system message for a model of 1.5B to 27B parameters, on three tasks. Whether the size of the bias carries over to programs or tools is untested.
- **Lock-in is measured, not modelled.** The theory assumes a fresh selection set, but every loop reuses its set. So the propositions describe a case that the experiments do not run.
- **"A few hundred items" has no calculation.** Section 6 cites Appendix C for the number of items needed to resolve small gains, and the subagent found no such calculation there.
- **Model revisions not pinned.** The authors say they "did not pin their revisions" of the Qwen models.

## Use in this repo

- **Registry verdicts are a selection under noise.** A unit can take several tries, and the search stops at the first VERIFIED one. Answer claims stop at the first VERIFIED attempt too, after at most 5 records (`MAX_ATTEMPTS`, `docs/PROTOCOL.md` §4). The paper's setting differs in two ways. A try is a review on fresh probes, not a reused selection set, so there is no lock-in. And the search stops at the first pass, which the paper doesn't analyse.
- **An estimate of the effect.** This is the page's own arithmetic from Proposition 1, not the paper's. Take the worst case, in which each try's hypothesis is as good as the others, and its probes are independent.
  - The noise is the same-reviewer SD of balanced accuracy at 30 probes, √0.0032 ≈ 0.057 (`docs/analyses/verdict-stability.md`).
  - The best of k tries then overstates by about κ_k × 0.057. κ_2 ≈ 0.56 and κ_3 ≈ 0.85, so that is about 0.03 and 0.05.
  - In `docs/analyses/retries.md`, 16% of VERIFIED units took two independent tries and 2% took three. Averaged over all VERIFIED units, the overstatement is about 0.006 in balanced accuracy.
  - The size for units that took more tries agrees with what the repo measured. Those units score 0.043 lower on held-out detection (95% interval +0.008 to +0.080). The two numbers are on different scales, so this is an agreement in size, not a test.
- **The paper's remedy is already here.** Its fix is a score on items never used for selection. `open-interpretability score` gives each VERIFIED record a detection and fuzzing score on held-out text, which the review never sees. `retries.md` uses exactly those scores to find the 0.043 gap.
- **A fresh re-test after VERIFIED already runs.** The planner draws `replicate` for a unit VERIFIED by one reviewer model, and `challenge` for one that is replicated (`openinterpretability/planner.py`). Each is a new review with fresh probes. At 30 probes, one review in ten flips VERIFIED-or-not (`verdict-stability.md`). So a replication demotes some claims, and the standing rule settles the split (`docs/PROTOCOL.md` §3).
- **No new gate.** A rule that every VERIFIED claim needs a second VERIFIED review would cut the noise further. It would change the decision rule and need a `PROTOCOL_VERSION` bump. The paper doesn't support it: its stricter rules, such as McNemar and the split-half rule, did not beat greedy over whole runs. So this page records no question.

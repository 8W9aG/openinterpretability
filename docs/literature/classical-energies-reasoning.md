# Classical Energies Learn to Reason 1,000x Faster than Looped Models

**Citation.** Benjamin Hoover, Hugh Blayney and James C. R. Whittington, all of Prefrontal, London. "Classical Energies Learn to Reason 1,000x Faster than Looped Models". [arXiv:2611.17309](https://arxiv.org/abs/2611.17309), version 2 of 19 November 2026. Version 1 is of 12 November 2026. Preprint, no venue given. The licence is CC BY 4.0. The code, data preparation and run records are at [github.com/Prefrontal-AI/CRAM](https://github.com/Prefrontal-AI/CRAM). The text gives no funding and no disclosure of tool or LLM use.

**Mismatch.** The headline says "at least 1,000x faster" training. Table 2's own Sudoku fit times give a much larger ratio, and the text doesn't say which ratio the headline uses. Two figure captions call 10,000 puzzles the test set. Appendix C.3.1 calls them the development set, and gives 412,786 test puzzles.

**Status.** Planned. It is evidence for #5144, which counts the ceiling on weight-only checks, and for #4782's strand on weight-space programming. No code, skill or protocol here depends on it. This page proposes no new issue.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 2 for this page's author, a model (Claude Opus 5.5). It read the source's own text in four fetches: Sections 1 to 5, the acknowledgements, the references and Appendices A to F. A few words of the Figure 2 caption may be missing at a fetch boundary. We didn't see any figure as an image, so the learned weight maps appear here only as the text describes them. We didn't read version 1 or the code.

## Summary

**Claim.** A small pairwise energy over the cells of a puzzle, trained by pseudolikelihood with no backpropagation through time, solves Sudoku, mazes, N-Queens and graph colouring. It matches or beats looped models, with far less training time and far fewer inference FLOPs. Its learned weights are the puzzle's constraints.

**Method.**

- **The energy.** The model is a "Compact Reasoning Associative Memory" (CRAM), a mean-field free energy over each cell's belief $p_i$. It has a $C \times C$ cost matrix $K_{ij}$ for each pair of cells and a bias $B_i$ for each cell. The paper calls it a special case of modern Hopfield energies.
- **Inference.** One Euler step of the neuron dynamics is $v_i \leftarrow -\sum_j W_{ij} p_j - B_i$, which the authors liken to one loop of a looped model. In practice a Gibbs search with several chains stops at the first board whose energy is at or below a threshold fitted from training solutions.
- **Training.** Pseudolikelihood, with SGD, a cosine schedule, a start at zero and an L1 soft threshold. It avoids "differentiating through relaxation."
- **Tasks.** Sudoku-Extreme has 1,000 training puzzles and 412,786 test puzzles. Maze-Unique has 1,000 test mazes of 30 × 30. N-Queens uses 8 × 8 and 10 × 10 boards, and graph colouring uses 8 and 10 vertices. Maze-Hard is in Appendix F only.
- **Baselines.** HRM, TRM, GRAM, EqR, FRM, EBR and TRMP, each retrained on one H200 with its authors' code. TRMP's code is not released, so the authors wrote their own version of it.
- **Metrics.** Exact solve rate, accuracy and Cov@20, inference FLOPs by a stated rule, and the wall-clock time of one fit.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Sudoku-Extreme, accuracy, fit time, parameters (Table 2) | CRAM 100.00 ± 0.00, 0.04 s, 7.7K nonzero of 263,169. TRMP 99.87 ± 0.08 in 9.0 h. EBR 99.85 ± 0.05 in 2.8 h. HRM 54.62 ± 4.18 in 1.2 h |
| Maze-Unique, same (Table 2) | CRAM 100.00 ± 0.00, 0.13 s, 18 parameters. TRMP 99.23 ± 0.31 in 7.6 h. EBR 89.80 ± 6.35 in 2.0 h |
| Hand-set energies (Table 2) | 100.00 on all four tasks, with 1, 3, 2 and 1 distinct weights |
| N-Queens 10 × 10, 20 chains (Table 3) | CRAM 99.99 ± 0.0 |
| Graph colouring 10, Cov@20 (Table 3) | CRAM 52.6 ± 0.6 against EBR 54.8 ± 0.2 |
| Inference FLOPs (Section 4.1) | "between 282x (against EBR) and 128,000x (against EqR)" fewer |
| Wrong stops, pooled over 3 runs (Table 4) | CRAM 0.00% on Sudoku and Maze. TRM 2.37% and 3.54% |
| Learned Sudoku costs (Section 4.2) | "positive equal-digit costs on exactly the 810 cell pairs that share a row, column or box, leaving all other costs zero" |

## Where the claims go beyond the evidence

- **The weights match the rules only as a set.** The 810-pair finding compares the nonzero pattern with the Sudoku constraint graph. It gives no metric and no count of runs beyond "every training run." The Maze reading, that each move "turn[s] off one dead-end cell," has no ablation or probe behind it.
- **"In no task is the energy told the rules" is wider than the setup.** Table 1 lists priors for each task, such as locality and translation invariance for mazes and relative position for N-Queens. Sudoku training uses digit-relabelling augmentation. The pairwise form itself fits these puzzles. The authors say this themselves: a high score "only shows that a model has learned a few pairwise rules."
- **The speed ratios are not from one rule.** Fit times are seed 0 only, on a shared cluster. CRAM's time is a warm median of seven fits with start-up left out, and each baseline's time is one cold fit with compilation. From Table 2, the FLOP ratios are about 167x against EBR and 104,000x against EqR. The text's 282x uses another operating point, and the 128,000x figure doesn't follow from Table 2.
- **Three seeds and no intervals.** Tables give mean ± SD over 3 seeds, with no confidence interval or test. Each CRAM seed also uses a different training split, while most baselines share one.
- **No untrained or random control.** Every fit starts at zero. No random-weight energy is scored against the search.

The authors state limits themselves. These puzzles don't show that a model scales to harder tasks such as ARC-AGI-1. Learning latent quantities without knowing the task's design is still open. And their energy-descent result doesn't prove that a minimum is a correct solution.

## Use in this repo

- **It is a case where weights can be read, and it shows why.** Each weight has one stated meaning: the cost of two digits in two cells. That is the condition the [Ok and Lee page](weight-space-programming-without-transformers.md) names too. A transformer neuron's input and output weights have no such set meaning. This page is evidence for #5144's ceiling, not against it.
- **The repo's own weight-only evidence is weak.** The [weight-kurtosis analysis](../analyses/weight-kurtosis.md) predicts a VERIFIED verdict with an AUC of 0.51 to 0.54 on 518 test units. Every interval includes 0.5. CRAM's clean reading comes from its hand-made structure, so it doesn't predict a better result for ROTATE (#3446) on Qwen or GPT-2.
- **The neurons here aren't registry units.** CRAM's "neurons" are one belief vector for each puzzle cell, set by the task. The registry's units are MLP neuron poles and attention heads that training found. No harness model can load CRAM.
- **The test of a reading is still a forward pass.** The paper checks its weight reading against the known rules, and its accuracy against a search. The registry has no known rules for its units, so a weight reading still needs probes and ablation before a verdict.
- **Related pages.** [Fast weights](fast-weights-recent-past.md) and [Titans](titans-test-time-memory.md) are the other memory views in this section. [`alta.md`](alta.md) and [`tracr.md`](tracr.md) give units whose role is known by construction, as CRAM's costs are.

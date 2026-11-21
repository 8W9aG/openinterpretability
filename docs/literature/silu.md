# Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning

**Citation.** Stefan Elfwing, Eiji Uchibe and Kenji Doya. The paper lists two affiliations: ATR Computational Neuroscience Laboratories, Kyoto, and the Okinawa Institute of Science and Technology. "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning". [arXiv:1702.03118](https://arxiv.org/abs/1702.03118), version 1 of 10 February 2017, last revised as version 3 on 2 November 2017. The arXiv page gives no journal, so this page treats it as a preprint. NEDO, JSPS KAKENHI grant 16K12504 and the Okinawa Institute of Science and Technology funded the work. The paper links no code.

**Mismatch.** None found.

**Status.** Used. `papers/why-neurons-rest` trains a SiLU condition (`scripts/train_resting.py`), and credits the function to this paper.

**Read.** On 2026-11-21 a `web-fetch` subagent read the abstract page and the HTML of version 3 for this page's author, a model (Claude Opus 5.5). It had the abstract and Sections 1 to 4.1 in the source's own words. The PDF's text didn't extract, so the page's author read its pages 15 to 18 as images. They hold the end of Section 4.2, the conclusions, the acknowledgements and the references. This page uses nothing from the part of Section 4 that neither read.

## Summary

**Claim.** The sigmoid-weighted linear unit (SiLU), and its derivative (dSiLU), make good hidden units for value-function approximation in reinforcement learning. With them, an on-policy agent with softmax action selection matches or beats DQN.

**Method.**

- **The units.** The SiLU is zσ(z), with σ the logistic sigmoid. The paper calls it a "continuous and 'undershooting' version" of ReLU. It has "a global minimum value of approximately −0.28 for z_k ≈ −1.28". The dSiLU is its derivative, σ(z)(1 + z(1 − σ(z))).
- **Agents.** TD(λ) and Sarsa(λ), with softmax action selection whose temperature falls over training.
- **Tasks.** Stochastic SZ-Tetris, 10 × 10 Tetris, and 12 Atari 2600 games whose names start with A or B.
- **Baselines.** ReLU and sigmoid units in the Tetris agents. For Atari, DQN, Gorila DQN and double DQN results from earlier papers, not run again.
- **Runs.** 10 runs per unit for shallow SZ-Tetris, 5 for deep SZ-Tetris and 10 × 10 Tetris, and 2 per Atari game.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Shallow SZ-Tetris, final mean score ± SD (Table 1) | SiLU 214 ± 74, ReLU 191 ± 58, dSiLU 263 ± 80, sigmoid 232 ± 75 |
| Deep SZ-Tetris (Table 1) | SiLU–dSiLU 229 ± 55, SiLU–SiLU 217 ± 53, ReLU–ReLU 215 ± 54 (SiLU–SiLU against ReLU–ReLU, p = 0.32) |
| 10 × 10 Tetris, dSiLU agent (Section 3.2) | Mean 4,900 points over the last 10,000 episodes, against a previous best of 4,200 |
| Atari, 12 games, mean and median DQN-normalised score (Table 2) | Best SiLU–dSiLU agent 332% and 125%. Double DQN 127% and 105% |

## Where the claims go beyond the evidence

- **Reinforcement learning only.** The units are tested as value-function approximators. The paper makes no test in supervised learning or language models.
- **Few runs.** Atari has two runs per game, and the scores of other methods come from their papers, under other protocols.
- **The deep SiLU gain is not significant.** Against ReLU in deep SZ-Tetris, the paper reports p = 0.32.
- **Best and final scores mixed.** The Atari headline uses each game's best agent. The final agents have a median below DQN's, 78% against 100%.

## Use in this repo

- **Our citations check out.** `papers/why-neurons-rest` writes SiLU as $h \cdot \sigma(h)$, which is the paper's definition. It says SiLU dips below 0, and the paper gives the minimum, about −0.28 at z ≈ −1.28. The three authors, 2017 and the arXiv link match.
- **The credit is right on the arXiv record.** The GELU paper ([`gelus.md`](gelus.md)) also names xσ(x) the SiLU. A `web-fetch` subagent read the HTML of its first three versions. Versions 1 and 2, of 2016, don't contain xσ(x). It first appears, as the "Sigmoid Linear Unit (SiLU)", in version 3 of November 2018. So this paper is the first on arXiv to define and name the SiLU. It doesn't cite the GELU paper, and versions 1 to 3 of the GELU paper don't cite it.
- **The minimum doesn't set the rest.** The paper says the SiLU's minimum works as a "soft floor" on the weights. `papers/why-neurons-rest` measures where each activation leaves a neuron's pre-activation after training. Its SiLU neurons rest at −0.24 to −0.49 in layers L0 to L3 (Table 3), well above the minimum at −1.28. That fits its finding that "No dip and no floor sets the depth".
- **No question for the maintainer.**
- **Related pages.** [`gelus.md`](gelus.md) reviews GELU, and [`softplus-option-pricing.md`](softplus-option-pricing.md) reviews softplus, the other activations in the same paper.

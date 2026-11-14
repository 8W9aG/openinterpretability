# Predicting which neurons verify

Issue #3372. Can the scan and the weights tell, before any agent writes a hypothesis, whether a random neuron pole will be VERIFIED? If so, the planner could steer explore draws toward poles likely to verify. Run on 2026-11-14 with `scripts/verify_predictor.py`, on the registry as `open-interpretability build` exported it. The rows are in `verify-predictor.jsonl`. Reported, not gated.

**Answer: not yet.** The neuron's features rank poles better than chance, also within one proposer. But their forecasts are no better than the base rate of VERIFIED in earlier units. So this result doesn't justify steering the planner.

## Method

- **Units.** The 1,712 random MLP neuron poles. Each has one row, from its first reviewed experiment. That is the unit set of `registry-stats.md`. The outcome is VERIFIED on that first reading.
- **Features.** Only what is known before review:
  - from the scan: `token_r2`, `position_r2` and the log of the maximum activation
  - from the scan's histogram: the share of tokens above 0.25 × the pole's maximum, and the share in the lowest bin
  - relative depth
  - the vocabulary lenses: `jacobian_kurtosis`, `lens_overlap`, the strongest promoted token's z and `current_token_share`
  - the top partner's lift
  - the output norm, from the weights: the norm of the neuron's row of the down projection
  - the model and the pole

  A missing value gets the training mean and an indicator. The `neuron-family-triage` families are built from these same statistics.
- **Split by time.** The first 70% of units by first-reading time train (1,198), and the rest test (514, first read from 2026-11-05). No unit is on both sides.
- **Model.** L2-regularised logistic regression on standardised features. The penalty is chosen by time-ordered validation inside the training set.
- **Rival.** The training units' base rate of VERIFIED, 26.7%, given to every test unit.
- **Scores.** The Brier and log scores on the test units (lower is better), with a bootstrap 95% interval for each gain over the base rate. AUC is the chance that a VERIFIED unit gets a higher forecast than one that isn't.

## Results

| Model | Test units | Brier (base rate) | Log score (base rate) | Brier gain, 95% CI | AUC, 95% CI |
| --- | --- | --- | --- | --- | --- |
| Neuron features | 514 | 0.176 (0.171) | 0.536 (0.527) | −0.005 (−0.015 to 0.005) | 0.64 (0.58–0.70) |
| Neuron features, deepseek-flash only | 242 | 0.208 (0.214) | 0.607 (0.620) | 0.007 (−0.009 to 0.022) | 0.64 (0.57–0.71) |
| Neuron features and the proposer | 514 | 0.152 (0.171) | 0.472 (0.527) | 0.019 (0.009 to 0.029) | 0.72 (0.67–0.77) |

- **The features rank, but they don't forecast.** An AUC of 0.64 means the features put a VERIFIED pole above one that isn't about two times in three. But the forecasts score no better than the base rate on either proper score. Part of this is drift: the test units verified at 21.4%, against 26.7% in training. So any forecast that carries the training rate starts behind.
- **The ranking is not just the proposer.** Within deepseek-flash alone, the proposer with the most units in both halves, the AUC is the same, 0.64. Its forecasts gain a little over that proposer's base rate (Brier 0.007, log 0.013), but both intervals include zero.
- **The proposer forecasts.** Adding the proposer gives a clear gain on both scores, and AUC 0.72. That repeats `registry-stats.md`: the agents move the share VERIFIED more than the neuron does. But the proposer isn't a property of the neuron, so it can't choose which neuron to draw.
- **What the weights say.** These are standardised log-odds weights, under a strong penalty of 100. Poles verify more often when:
  - their maximum activation is larger (+0.23)
  - their top partner's lift is higher (+0.22)
  - more of their tokens sit in the lowest bin, so they are sparser (+0.20)

  They verify less often when token identity (−0.11) or position (−0.10) explains more of their activity. The output norm, the depth and the model each carry little weight (|w| ≤ 0.06).

## For the planner

The issue's condition for steering was a forecast that beats the earlier base rate, and this one doesn't. So `open-interpretability next` doesn't change.

A ranking alone could still steer draws: draw more often from the poles ranked high, and keep a share of draws uniform. That would raise the share VERIFIED by the AUC's margin at most. It would also skew any statistic over the steered draws, so the registry's random-unit statistics would have to use only the uniform ones. Whether that trade is worth it is the maintainer's call.

## Limits

- **One split and one model class.** A tree model or other features could rank better. The interval on the AUC is wide (0.58–0.70).
- **The maximum activation is partly the corpus.** Units scanned on the 552-token built-in sample have smaller maxima and verify less often (`registry-stats.md`). They are all in the training half. Within deepseek-flash, about 90% of units scanned a real corpus, and the ranking holds there.
- **First readings are noisy.** Two readings of one hypothesis agree on VERIFIED or not 71% of the time (`registry-stats.md`). So no forecast can rank much better than a reading repeats.

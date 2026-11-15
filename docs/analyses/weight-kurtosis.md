# A weight-only kurtosis signal as a prior for VERIFIED

Issue #3226. ROTATE (Avrahamy, Gur-Arieh and Geva, 2026, [arXiv 2604.06005](https://arxiv.org/abs/2604.06005), §3) expects monosemantic neurons to have a high kurtosis of their weights' vocabulary projection. That needs no forward pass. If it held for this registry, the kurtosis would be a free prior for which neurons to draw. Run on 2026-11-14 with `scripts/weight_kurtosis.py`, on the registry as `open-interpretability build` exported it. The rows are in `weight-kurtosis.jsonl`. Reported, not gated.

**Answer: no.** Neither kurtosis predicts VERIFIED, alone or added to the features of `verify-predictor.md`. Poles in the top 10% of their layer verify as often as the rest. So the signal doesn't go into the predictor or into `neuron-family-triage`.

## Method

- **The signal.** For each neuron of every layer of the three configured models, two numbers come from the weights alone:
  - the *output kurtosis*: the neuron's `w_out` through the final norm's gain, then the unembedding `U`. This is the `logit_kurtosis` that each experiment's vocabulary lens records (`openinterp/harness/jlens.py`).
  - the *input kurtosis*: the neuron's read vector through `U`. The read vector is `gate` for Qwen and `in` for GPT-2, with the pre-MLP norm folded in (`wiring.mlp_reads`). ROTATE also uses `w_gate` for a gated MLP.

  The kurtosis is the mean of z⁴ over the vocabulary, where z standardises the neuron's scores for all tokens. A Gaussian scores 3. ROTATE subtracts 3, which changes no ranking. The kurtosis is the same for both poles of a neuron.
- **Percentiles.** Each pole also gets its neuron's percentile within its layer. That is the scale of ROTATE's own test: the median known concept vector is at the 90th to 95th percentile of random neurons. The paper gives no threshold and no AUC.
- **Units and split.** These are the units of `verify-predictor.md`: the 1,724 random MLP neuron poles at their first reading, on this build. The split is the same, by time: the first 70% train (1,206, 26.7% VERIFIED) and the rest test (518, 22.0%).
- **Tests.**
  - the AUC of each signal alone on the test units, with bootstrap 95% intervals
  - the predictor's features with and without the four kurtosis columns (raw and percentile, both sides), scored on the test units. Each gain is paired per unit, with a bootstrap interval.
  - the share VERIFIED in the top decile of each layer against the rest, on all units

## Results

The kurtosis has a heavy tail only in later layers. In most layers the median neuron scores 3.1 to 3.6, near a Gaussian. The 99th percentile of the output kurtosis passes 4 only in later layers. Those start at layer 6 of 12 in GPT-2 small, 13 of 24 in Qwen2.5-0.5B and 16 of 28 in Qwen2.5-1.5B. That percentile reaches 14, 39 and 89 in those models.

| Signal, test units (518) | AUC | 95% CI |
| --- | --- | --- |
| Output kurtosis | 0.51 | 0.45–0.57 |
| Output kurtosis, percentile in layer | 0.52 | 0.46–0.58 |
| Input kurtosis | 0.54 | 0.48–0.60 |
| Input kurtosis, percentile in layer | 0.54 | 0.49–0.60 |
| Output skewness, signed by the pole | 0.52 | 0.46–0.59 |
| Jacobian-lens kurtosis (already a predictor feature) | 0.51 | 0.45–0.56 |

Every interval includes 0.5, the AUC of a coin. ROTATE finds its strongest effect in mid-to-late layers. On all units at relative depth 0.5 or more (892), the output kurtosis scores 0.49 (0.45–0.54) and the input kurtosis 0.53 (0.48–0.57).

| Predictor on the test units | Brier | Log score | AUC |
| --- | --- | --- | --- |
| The #3372 features | 0.1781 | 0.5395 | 0.639 |
| With the four kurtosis columns | 0.1781 | 0.5393 | 0.638 |
| Gain, 95% CI | −0.0001 (−0.0007 to 0.0006) | 0.0002 (−0.0013 to 0.0018) | −0.008 to 0.008 |

The validation chose the same penalty, 100, with and without the kurtoses. The kurtosis weights are small (|w| ≤ 0.05).

| Top 10% of the layer against the rest, share VERIFIED | Output: top | Output: rest | Input: top | Input: rest |
| --- | --- | --- | --- | --- |
| All models | 25.7% (171) | 25.2% (1,553) | 24.2% (165) | 25.4% (1,559) |
| GPT-2 small | 23.7% (76) | 21.1% (692) | 27.4% (73) | 20.7% (695) |
| Qwen2.5-0.5B | 29.1% (79) | 30.3% (715) | 24.3% (74) | 30.8% (720) |
| Qwen2.5-1.5B | 18.8% (16) | 19.9% (146) | 11.1% (18) | 20.8% (144) |

The model rows point different ways, and each top group is small. GPT-2's input side is the largest difference, 6.7 points on 73 poles. It isn't a pattern across models.

## Consistency with the lenses

- **It is the lens's number.** For all 1,724 units, the output kurtosis matches the recorded `logit_kurtosis` within 0.0001.
- **It overlaps the Jacobian lens.** The output kurtosis correlates with the Jacobian-lens kurtosis at r = 0.91, and with the input kurtosis at r = 0.88. So the predictor already had most of this signal, and neither number forecast VERIFIED there either.

## Why it may not carry over

- **A different outcome.** ROTATE's test is whether a known concept vector sits high among random neurons. VERIFIED asks whether one explanation survives adversarial probes. A pole can verify on a context or syntax explanation that no token set describes. The vocabulary projection can't see that, and ROTATE says that features orthogonal to the vocabulary are out of its scope.
- **The tail is late.** Most studied poles are in layers where nearly every neuron scores near a Gaussian. There the kurtosis mostly ranks noise.
- **Not ROTATE itself.** This tests only the cheap statistic of the paper's §3. ROTATE's decomposition optimises vocabulary channels per neuron, over thousands of steps. It is a way to explain a neuron, not to choose one, and this page doesn't test it.

## Limits

- **One split.** It is the time split of `verify-predictor.md`. The test units verify less often (22.0%) than the training units (26.7%).
- **First readings are noisy.** Two readings of one hypothesis agree on VERIFIED 71% of the time (`registry-stats.md`). That caps how well any prior can rank.

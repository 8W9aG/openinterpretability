# How far float32 moves a head's strength

Run on 2026-11-15 for issue #3956, with `scripts/head_tolerance.py`. The rows are in `head-reproduce-tolerance.jsonl`. A head record from PR #3952 (Pythia-70m L4/H4, float32, scanned on an Intel CPU) failed `reproduce`. Some per-token strengths differed by about 2% from the recorded ones. This page measures how much float32 rounding alone moves a head's strength, and sets the head tolerance from it.

## Method

- **Records.** Every head record in the registry (25 on GPT-2, 2 on Pythia-70m), and the record of PR #3952.
- **Measure.** Each top example is measured again on one machine (Apple M1 Max, CPU), in float32 and in float64. The strength is `|W_O (z − z̄)|`, as the scan computes it. Each number is the largest difference over all tokens of all top examples, as a fraction of the record's scale (its top peak).
- **Columns.**
  - *re-measured vs recorded* is what `reproduce` compares.
  - *recorded vs float64* and *float32 vs float64* say how far each float32 result is from a more exact one.

## Results

| Records | Re-measured vs recorded | Recorded vs float64 | float32 vs float64 |
| --- | --- | --- | --- |
| GPT-2, 25 heads, largest | 0.0012% | 0.0012% | 0.0009% |
| Pythia-70m L1/H6 | 0.0009% | 0.0002% | 0.0010% |
| Pythia-70m L5/H1 | 0.34% | 0.25% | 0.16% |
| Pythia-70m L4/H4 (PR #3952) | 0.70% | 0.41% | 0.73% |

- **It is rounding, not a bug.** On L4/H4 the recorded values and the values measured here are both up to about 0.7% of the scale from float64, in different directions. Measuring again on the same machine gives the same numbers. One thread gives the same numbers, and so does an example alone or in a padded batch.
- **It depends on the head.** GPT-2's heads move by about 0.001%. Pythia-70m's heads after layer 1 move by up to 0.7%.
- **The old tolerance was set for neurons.** It allows `0.001 × scale + 0.01 × |recorded|`. On L4/H4 that is less than the rounding at low strengths.

## The head tolerance

`reproduce.HEAD_TOLERANCE` is `atol 0.015 × scale + rtol 0.01 × |recorded|`, about twice the largest move above. `reproduce` uses it for head records only. At a peak it allows 2.5%, so a 10% edit of a peak still fails. Neurons and dictionary features keep the old tolerance.

## Limits

- One machine re-measured. Another CPU or GPU can round differently. The float64 column bounds how far a float32 result can stray, and the tolerance doubles it.
- Only three Pythia-70m heads exist. A head that moves more than 1.5% of its scale would still fail, and its record would show how far it moved.
- This page does not say why Pythia's later heads are sensitive. Large activations in its residual stream are one possible cause, and this run did not test it.

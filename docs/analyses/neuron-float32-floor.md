# How far float32 moves a neuron's activations

Run on 2026-11-17 for PR #4187, with `scripts/neuron_float32_floor.py`. The rows are in `neuron-float32-floor.jsonl`. The record of PR #4187 (Pythia-70m L5/N389, negative pole, float32, scanned on an Intel CPU) failed `reproduce` in CI. Twelve tokens of its top examples differed by up to 1.1% of the scale. Its peaks, results, gates and status agreed. This page measures how much float32 rounding moves a neuron pole's activations, and sets an absolute floor for the tolerance from it.

## Method

- **Records.** Every Pythia-70m neuron record (120 positive poles, 15 negative poles, with the record of PR #4187), and every GPT-2 negative-pole record (148).
- **Measure.** Each top example is measured again on one machine (Apple M1 Max, CPU), as `reproduce` measures it. It is measured three times: in float32 with one thread, in float32 with the default threads, and in float64. Each number is the largest difference over all tokens of all top examples, in activation units.
- **Columns.**
  - *re-measured vs recorded* is what `reproduce` compares. It is the larger of the two float32 runs.
  - *one vs default threads* is float32 against float32, with another reduction order.
  - *float32 vs float64* says how far a float32 result is from a more exact one.

## Results

| Records | Scale (smallest) | Re-measured vs recorded | One vs default threads | float32 vs float64 | Peaks vs recorded |
| --- | --- | --- | --- | --- | --- |
| Pythia-70m, 120 positive poles | 0.95 | 0.0045 | 0.0049 | 0.0039 | 0.0017 |
| Pythia-70m, 15 negative poles | 0.17 | 0.0028 | 0.0019 | 0.0033 | 0.0000 |
| GPT-2, 146 negative poles | 0.064 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

The GPT-2 row leaves out the 2 records under "Limits". Its numbers are all below 0.00001.

- **It is rounding, not an edit.** On Pythia-70m, float32 is up to 0.0039 from float64 on this machine. One thread and the default threads differ by up to 0.0049. The record of PR #4187 is 0.0028 from the values measured here, and 0.0023 from float64.
- **It is absolute, not a share of the scale.** Pythia-70m's moves are of the same size on positive poles (scale 0.95 to 7) and on negative poles (scale 0.17). On a positive pole, 0.0045 is at most 0.14% of the scale. On a negative pole, 0.0028 is 1.6% of the scale.
- **Negative poles show it.** A negative pole of a GELU model has a scale of at most 0.17, because GELU is never less than −0.17. The tolerance `0.001 × scale + 0.01 × |recorded|` then allows 0.0002 near 0. That is less than the rounding.
- **It depends on the model.** GPT-2's activations move by less than 0.00001 in float32. Pythia-70m's move by up to 0.005. The head study found the same split (`head-reproduce-tolerance.md`).

## The floor

`[tool.openinterpretability.reproduce_floor]` in `pyproject.toml` gives a model an absolute floor. `reproduce` adds it to the tolerance of each neuron pole of that model: `0.001 × scale + 0.01 × |recorded| + floor`. Pythia-70m's floor is 0.01, about twice the largest move above. Other models have no floor. Heads keep `HEAD_TOLERANCE`.

At the peak of a negative pole (0.17), the tolerance is then 0.012, or 7%. So a 10% edit of the peak still fails. At a positive pole's peak of 1, it is 2.1%.

## Limits

- One machine re-measured. Another CPU or GPU can round differently. The float64 column bounds how far a float32 result can stray, and the floor doubles the largest move.
- Only 15 Pythia-70m negative poles exist. A pole that moves by more than 0.01 would still fail, and its record would show how far it moved.
- Two GPT-2 records move for other reasons, and the floor does not cover them:
  - L3/N2838 (negative) was recorded on MPS. It is 0.0013 from float64, but float32 on this CPU is within 0.000001 of float64. So the recording, not the CPU, moved.
  - L11/N1071 (negative, protocol 0.8, signed) records activations of up to 3.7. A negative GELU pole can't be more than 0.17. Its activations were measured under another rule, and none agree with the values measured here.
- This page does not say why Pythia-70m is sensitive. Large activations in its residual stream are one possible cause, and this run did not test it.

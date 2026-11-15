# Reproducing 16-bit records

Run on 2026-11-13 on an M1 Max (64 GB) with torch 2.14, for issue #2995. The script is `scripts/half_tolerance.py`. One summary row per record is in `half-precision-reproduce.jsonl`.

`open-interpretability reproduce` re-runs a record in the dtype it pins, and compares each number within `0.001 × scale + 0.01 × |recorded|`. That tolerance suits float32 differences between devices. A record made in bfloat16 on one device drifts by a few percent on another, so no 16-bit record could pass. On the CPU, bfloat16 is also about 10 times slower than float32, and the CI job ran out of time on Qwen2.5-1.5B records.

## Method

`scripts/half_tolerance.py` reproduces each record with the model in another dtype. It keeps every number that `reproduce` compares, with its kind and the scale it is judged against. It also keeps each exact check that differs. Two directions measure the same rounding:

- **The 8 bfloat16 records, reproduced in float32 on the CPU.** These are the 6 bfloat16 records in the registry and the records of pull requests #2899 and #2879. Six were made on CUDA and two on the CPU. This is what `reproduce` now does with a 16-bit record.
- **24 float32 records, reproduced in bfloat16 on MPS.** These are 8 records each of GPT-2, Qwen2.5-0.5B and Qwen2.5-1.5B, drawn with seed 2995 from protocol 0.8. They give more models, layers and records.

The drift of a number is |reproduced − recorded| as a share of the neuron's scale (`max_activation`).

## Results

| Numbers | Count | 99% under | 99.9% under | Largest |
| --- | --- | --- | --- | --- |
| Top examples, per token and peak | 214,318 | 1.0% | 2.8% | 21.7% |
| Probe activations | 681 | 3.2% | 5.1% | 6.0% |
| Counterfactual Δ | 202 | 2.8% | 3.5% | 3.6% |
| Explanation windows (not gated) | 42,616 | 1.7% | 4.1% | 10.1% |

The table has the 24 float32 records and 2 of the bfloat16 records, whose numbers carry their kind. The other 6 bfloat16 records ran before the script kept kinds. No number of theirs drifted by more than 3.1%.

- **Most numbers drift by about 1% of the scale.** A few drift far more. The largest, 21.7%, is in Qwen2.5-0.5B L23 N541, a bfloat16 record made on the CPU. On a run of `////` characters, a token recorded as 8.0 is 5.90 in float32, with a scale of 9.69. Only 4 of its 11,439 tokens drift by more than 10%. Its peaks agree to 0.7%.
- **Peaks drift less than the tokens between them.** No peak, probe activation or counterfactual Δ drifted by more than `0.05 × scale + 0.02 × |recorded|`. That is 7% at a peak.
- **With that tolerance, at most 3 tokens of any top example fall outside it.** That was at most 1.6% of the example's tokens.
- **A pass/fail can flip at its threshold.** In 2 of the 24 float32 records reproduced in bfloat16, a positive holdout moved across the threshold and its result flipped. That changed the balanced accuracy from 0.643 to 0.714. None of the 8 bfloat16 records flipped in float32.

## The rules

A record in a 16-bit dtype is reproduced in float32 (`openinterp/half_precision.py`):

- Numbers must agree within `0.05 × scale + 0.02 × |recorded|` (`reproduce.HALF_TOLERANCE`). A 10% edit of a peak still fails.
- Up to 4 tokens of each top example may differ by more, as a warning (`HALF_STRAY_TOKENS`). The peaks may not.
- A pass/fail that flips is a warning only if the numbers agree and the recorded pass/fail follows from the recorded number. The recorded gates, balanced accuracy and status must then follow exactly from the recorded results.
- Every recorded pass/fail must follow from its own number, so a pass can't be changed without its number.

float32 records keep the float32 tolerance and exact checks. Records signed by a maintainer are skipped as before.

## With the rules

`open-interpretability reproduce --device cpu`, with the rules above, reproduces all 8 bfloat16 records:

- **The 6 in the registry** took 28 minutes together, while other jobs used the machine. Only Qwen2.5-0.5B L23 N541 needed the stray tokens: 3, 1 and 1 tokens of three top examples. No result flipped.
- **The records of #2899 and #2879**, both Qwen2.5-1.5B, took 11 minutes together. In bfloat16 on the CPU, the first of them had run for more than an hour and failed on 5% drift.

## Caveats

- **The sample is small.** It has 8 bfloat16 records, and 24 float32 records reproduced the other way. A record with more stray tokens, or a peak that drifts more than 7%, would fail. Its warnings and mismatches show the numbers to widen the rules with.
- **A 16-bit record checks less than a float32 one.** An edit of a token by less than 5% of the scale, or of up to 4 tokens of an example, is not caught. A pass/fail at its threshold is checked against the record's own number, not a new measurement.
- **float16 was not measured.** It has more precision than bfloat16 and less range, so the same rules are used for it.

# Two ways to find superpositions in Qwen2.5-0.5B

Issue #3686. A superposition is a sparse direction over one layer's MLP neurons (`docs/PROTOCOL.md` §1). This note compares two ways to find one, on the subject–verb agreement task of `sva-circuit.md`. The definitions are in `data/superpositions/Qwen--Qwen2.5-0.5B.jsonl`. The code is `openinterp/superposition_finders.py`. Run on 2026-11-14 on the CPU. Reported, not gated.

## Method

- **Circuit.** The top 100 neurons of the agreement circuit (`sva-circuit.jsonl`) are grouped by layer. Each layer with at least 2 of them gives one direction. A member's weight is its attribution. 15 layers have 2 or more, with 95 of the 100 neurons.
- **Sparse probe.** At each of those 15 layers, a probe predicts whether the noun that the verb agrees with is plural. It reads the MLP activations at the last token, where the verb comes next. There are 400 pairs from the same four templates, 100 for each template. Both prompts of a pair count, so there are 400 singular and 400 plural examples. An L1 logistic regression chooses 16 neurons. Then an L2 logistic regression is fitted again on them. The accuracy is on a held-out 20% of the pairs, with seed 0. The split is by pair, so both prompts of a pair are always on the same side.
- **Comparison.** The shared neurons are in both definitions. The cosine is between the two weight vectors in the layer's 4,864-neuron space.

## Results

| Layer | Circuit members | Probe accuracy | Shared neurons | Cosine |
| --- | --- | --- | --- | --- |
| 0 | 20 | 0.78 | 10: 255, 430, 561, 706, 2432, 2615, 3528, 3679, 4163, 4371 | −0.12 |
| 4 | 2 | 0.81 | 2: 2514, 4858 | 0.10 |
| 6 | 3 | 0.85 | 1: 3717 | 0.12 |
| 7 | 2 | 0.88 | 2: 727, 870 | 0.08 |
| 9 | 4 | 0.93 | 1: 4128 | 0.04 |
| 14 | 2 | 0.86 | 1: 4219 | −0.01 |
| 15 | 3 | 0.88 | 1: 3954 | 0.04 |
| 16 | 8 | 0.87 | 2: 3891, 4422 | 0.15 |
| 17 | 7 | 0.88 | 3: 1449, 1479, 2220 | −0.07 |
| 18 | 8 | 0.88 | 5: 429, 444, 3643, 3864, 4771 | −0.05 |
| 19 | 4 | 0.99 | 4: 267, 1274, 4551, 4688 | 0.03 |
| 20 | 5 | 0.99 | 4: 3855, 4213, 4662, 4712 | −0.04 |
| 21 | 7 | 0.98 | 3: 1653, 3515, 4413 | 0.07 |
| 22 | 8 | 0.99 | 6: 1019, 1532, 1684, 4334, 4567, 4762 | −0.06 |
| 23 | 12 | 0.98 | 6: 26, 729, 765, 2195, 2458, 4102 | 0.08 |

The two finders choose the same neurons much more often than chance. They share 51 of the 95 circuit neurons. Two random sets of 16 and 20 neurons in one layer share 0.07 neurons on average. The probe reads number with at least 0.86 accuracy from layer 9 on, and almost perfectly from layer 19. The cosines are near zero, so the two weight vectors are different directions on the same neurons. A circuit weight is a signed attribution to the logit difference, and a probe weight is in activation units. So the cosine compares two different scales, and the shared neurons are the better measure of agreement.

## Reproduce

```sh
uv run open-interpretability superpositions from-circuit --rows docs/analyses/sva-circuit.jsonl \
    --model Qwen/Qwen2.5-0.5B --top 100 --min-members 2 --out circuit.jsonl
uv run open-interpretability superpositions grammatical-number --model Qwen/Qwen2.5-0.5B \
    --layers 0,4,6,7,9,14,15,16,17,18,19,20,21,22,23 --k 16 --n 400 --seed 0 --device cpu \
    --out probe.jsonl
cat circuit.jsonl probe.jsonl > data/superpositions/Qwen--Qwen2.5-0.5B.jsonl
```

The probe run takes about 15 seconds. `compare` in `openinterp/superposition_finders.py` gives the shared neurons and the cosine of two definitions.

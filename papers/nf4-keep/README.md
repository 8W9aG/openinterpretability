# Do neuron explanations survive 4-bit NF4 rounding?

QLoRA fine-tunes a language model on top of base weights stored in 4-bit NormalFloat (NF4), and it reports that this rounding costs little on benchmarks. Whether the rounding keeps what single neurons do is a separate question. We rounded every linear weight of Qwen2.5-0.5B and Qwen2.5-1.5B to NF4, with QLoRA's 16 values and a scale for each block of 64 weights. Then we re-tested each neuron explanation that had passed an adversarial review. On Qwen2.5-0.5B, 281 of 362 explanations keep (77.6%), and random changes with NF4's error norm in every matrix keep 283 to 294 (paired McNemar $p \geq 0.18$). On Qwen2.5-1.5B, 66 of 75 keep (88.0%), against 58 to 62 under the random changes. The median peak moves by 1.7% of the pole's scale on Qwen2.5-0.5B and 1.5% on Qwen2.5-1.5B. The 99% points are 33% and 21%, against 3.2% for bfloat16 in this repo's earlier test. The poles that NF4 flips are mostly those near one of the review's thresholds, and they overlap with those that instruction tuning flips. Half or more of the poles have weights that fail QLoRA's normality test, but failing it does not reliably predict a lost explanation. NF4 is therefore a change of about the size of instruction tuning, for explanations, and a QLoRA run starts from a model whose explanations have already moved.

**Status:** Preprint. Work left: #6878 (QLoRA's bfloat16 compute and double quantization).

## Introduction

QLoRA makes fine-tuning cheap by storing the frozen base model in 4 bits (Dettmers et al., 2023). Each weight is rounded to one of 16 NF4 values, scaled per block of 64 weights. The model turns the weights back into 16 bits before it computes. The paper shows that a model fine-tuned this way matches 16-bit LoRA on benchmarks, on average. Benchmarks measure the whole model, however, and a mechanistic explanation is a claim about one neuron. A rounding that leaves a benchmark score unchanged can still move a neuron's activations enough that its explanation no longer passes its test prompts.

The question matters for this repo in two ways. An adapter trained with QLoRA trains against the rounded weights, so any later test of the adapter's effect on explanations inherits the rounding's effect too. And NF4 rounding is a weight change of known form and size. It is a second reference, beside the matched random change of an earlier paper here ([`papers/random-change-keep/`](../random-change-keep/README.md)), for how many explanations a change of a given size breaks.

The paper's contributions are these:

- A keep test of every passing explanation of two Qwen2.5 models under NF4 rounding, with Wilson intervals.
- A matched random null with NF4's own error norm in every matrix, and paired tests against it.
- The drift of every test prompt's peak, on the same scale as this repo's earlier test of bfloat16.
- QLoRA's own normality test applied to each pole's weights, and a test of whether it predicts which explanations break.

The paper bears on research question 4, how far an explanation carries to a changed model ([research questions](../../docs/research-questions.md)).

## Related work

**QLoRA.** Dettmers et al. (2023) introduce the NF4 data type, whose 16 values are quantiles of a normal distribution, so that each value holds the same expected share of normally distributed weights. They scale each block of 64 weights by its largest absolute value. On LLaMA 7B to 65B, NF4 with double quantization matches 16-bit LoRA on 5-shot MMLU, at a mean of 53.1 against 53.0 over eight cells. Their Appendix F applies a Shapiro-Wilk test to the weights of each hidden unit of LLaMA 7B, and finds 7.5% not normal at the 5% level. Apart from that test, they measure benchmarks and perplexity, and they test no neuron's behaviour or explanation. Our test is at the level of single explanations.

**Weight changes in this repo.** The paper on matched random changes adds to each Qwen2.5 base model a Gaussian change with the instruct change's norm in every matrix ([`papers/random-change-keep/`](../random-change-keep/README.md)). On Qwen2.5-0.5B, such a change breaks as many explanations as instruction tuning. On Qwen2.5-1.5B, it breaks fewer, because a change of that norm is small beside the larger model's weights. Instruction tuning keeps 298 of 361 explanations of Qwen2.5-0.5B and 65 of 75 of Qwen2.5-1.5B in that paper's run. The present paper applies the same test to a different change, a day later. Our set of tested explanations then holds one more passing pole of Qwen2.5-0.5B.

**Precision in this repo.** An earlier analysis reproduced 24 float32 records in bfloat16 ([`docs/analyses/half-precision-reproduce.md`](../../docs/analyses/half-precision-reproduce.md)). Over 681 test-prompt activations of those records and of 2 bfloat16 records reproduced in float32, 99% moved by less than 3.2% of the pole's scale, and none by more than 6.0%. In 2 of the 24 records, a positive test prompt crossed its threshold. That run reproduced the records in bfloat16, which we read as rounding the activations as well as the weights. Here the model computes in float32, and only the weights are rounded.

## Method

**Models and units.** The models are Qwen2.5-0.5B and Qwen2.5-1.5B (Yang et al., 2024). Following Cammarata (2026), a *pole* is one sign of an MLP neuron's activation, positive or negative. A pole is *passing* when its explanation, a sentence that says when the pole fires, passed our adversarial review. In that review, a second model wrote test prompts to break the explanation. Hard negatives are near-miss texts that the explanation says should not fire, and positive test prompts are new texts that it says should fire. The pole fires on a text when its peak activation reaches the record's line, a quarter of the pole's maximum on its scan. The explanation passed when its specificity on the hard negatives reached the threshold of 0.80, and its sensitivity on the positive test prompts reached 0.60. Our set of tested explanations holds 363 passing poles of Qwen2.5-0.5B and 75 of Qwen2.5-1.5B. On the base models, 362 and 75 of them pass both tests again.

**NF4 rounding.** We round the seven linear weight matrices of each block: the attention's query, key, value and output projections, and the MLP's gate, up and down projections. We take each matrix's weights in row order, in blocks of 64. For a block $w$ with largest absolute value $s$, each weight becomes

$$
\hat{w}_i = s \cdot q\!\left(\frac{w_i}{s}\right),
$$

where $q$ maps a value in $[-1, 1]$ to the nearest of the 16 values in Appendix E of Dettmers et al. (2023). A block of zeros stays zero. We do not quantize the block scales a second time, as QLoRA's double quantization does. Biases, norms, the embeddings and the output head stay as they are. The model then computes in float32.

**The other changes.** *bfloat16 rounding* rounds the same weights to bfloat16. The *random change* adds to each of the same matrices Gaussian noise $\varepsilon_W$ with the Frobenius norm of NF4's own error in that matrix:

$$
\varepsilon_W = \lVert \hat{W} - W \rVert_F \, \frac{Z}{\lVert Z \rVert_F}, \qquad Z_{ij} \sim \mathcal{N}(0, 1).
$$

It has three seeds. It matches NF4's size in each matrix, but not its form: NF4's error is bounded within each block, and it scales with the block's largest weight.

**The keep test.** This is the test of the base-instruct analysis, as the paper on random changes runs it. Each pole's test prompts run on the base model and on the changed model. On the changed model, the line is the record's line times the ratio of the pole's maximum after the change to its maximum before. Both maxima come from the same 20,000 tokens of our scan corpus. So "fires" means the same share of the pole's own range on both models. An explanation *reproduces* when it passes both tests on the base model, and it *keeps* when it passes them on the changed model too. A pole *flips* when it reproduces and does not keep. The keep share has a 95% Wilson interval, and the exact McNemar test compares two changes on the same poles.

**Drift.** For each test prompt, the drift is the absolute change in the pole's peak activation on the prompt, as a share of the pole's scale, its maximum on its scan. This is the measure of the bfloat16 analysis. We also give Pearson's $r$ between each pole's activations before and after the change, over the 20,000 tokens.

**Margin.** A pole's margin is how far its base readings sit above the nearer gate: its specificity less 0.80, or its sensitivity less 0.60, whichever is smaller. We cut the poles into quarters by margin.

**Normality.** We apply Appendix F's Shapiro-Wilk test, at the 5% level, to three weight vectors of each pole's neuron: its row of the gate projection, its row of the up projection, and its column of the down projection. Fisher's exact test compares how often poles with and without a non-normal vector lose their explanation under NF4.

**Motion.** Figure 2 steps through the four changes of all weights, one frame each, and marks the poles that flip in each. The frames show no state between two changes.

## Does an explanation keep when the weights are rounded to NF4?

Most do. On Qwen2.5-0.5B, 281 of the 362 explanations that reproduce keep under NF4, 77.6% (95% interval 73.1–81.6%). On Qwen2.5-1.5B, 66 of 75 keep, 88.0% (78.7–93.6%) (Figure 1, Table 1). These shares are close to those of instruction tuning on the same models, 82.6% and 86.7% in the paper on random changes. On Qwen2.5-0.5B, NF4's error even has about the norm of the instruct change, 36.38 against 36.34. On Qwen2.5-1.5B, NF4's error is almost six times the instruct change's norm, 95.88 against 16.88, yet it breaks no more explanations.

![NF4 rounding keeps most explanations, and on Qwen2.5-1.5B more than a random change of the same size.](media/keep.svg)

**Figure 1.** NF4 rounding keeps most explanations, and on Qwen2.5-1.5B more than a random change of the same size. The share of each model's passing explanations that still pass their test prompts under each change, with 95% Wilson intervals. Circles are NF4 rounding, triangles are random changes with NF4's error norm in every matrix, one for each seed, and the green circle is bfloat16 rounding.

**Table 1.** The explanations that keep under each change, out of those that reproduce on the base model, with the norm of the change and the median correlation of each pole's activations with its base activations. A random row gives the range over its three seeds.

| Model | Change | Norm | Keep | Share | Median $r$ |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | NF4 | 36.38 | 281 of 362 | 77.6% | 0.947 |
| Qwen2.5-0.5B | random, NF4's norm | 36.38 | 283–294 | 78.2–81.2% | 0.935–0.937 |
| Qwen2.5-0.5B | bfloat16 | 0 | 362 of 362 | 100% | 1.000 |
| Qwen2.5-1.5B | NF4 | 95.88 | 66 of 75 | 88.0% | 0.954 |
| Qwen2.5-1.5B | random, NF4's norm | 95.89 | 58–62 | 77.3–82.7% | 0.941–0.945 |
| Qwen2.5-1.5B | bfloat16 | 0 | 75 of 75 | 100% | 1.000 |

The bfloat16 rounding changes nothing. Both models are stored in bfloat16, so their float32 weights are already bfloat16 values, and rounding them again is exact. Its row is a check of the method, not a measurement. The bfloat16 comparison that the issue asked for therefore rests on the earlier analysis, which also rounded the activations.

## Is NF4 gentler than a random change of its size?

On Qwen2.5-1.5B it seems to be, and on Qwen2.5-0.5B it is not. On Qwen2.5-0.5B, NF4 flips 81 poles and the three random changes flip 68, 76 and 79, with paired McNemar $p$ of 0.18, 0.65 and 0.91 (Table 4 in Appendix B). On Qwen2.5-1.5B, NF4 flips 9 poles and the random changes flip 16, 17 and 13. Each seed's paired test alone is not significant ($p$ = 0.092, 0.077 and 0.42), but all three point the same way. NF4 also moves the activations a little less than its random change on both models, to a median correlation of 0.947 against 0.935 to 0.937 on Qwen2.5-0.5B, and 0.954 against 0.941 to 0.945 on Qwen2.5-1.5B.

A reason is in NF4's form. Its error in a block is bounded by the block's own scale, and a block with small weights gets a small error. The random change spreads the same norm evenly. So NF4 puts less error on the weights that are already small. Whether that is what spares the explanations of Qwen2.5-1.5B is a hypothesis that this paper does not test.

## How far do the test prompts' readings drift?

Much further than under bfloat16, at the tail. The median test prompt's reading moves by 1.7% of the pole's scale on Qwen2.5-0.5B and by 1.5% on Qwen2.5-1.5B. On Qwen2.5-0.5B, 37.4% of the 6,953 test prompts move by less than 1% of the pole's scale, 75.3% by less than 5%, and 82.3% by less than 7%. The 99% point is 33.5% of the scale, and the 99.9% point is 121.8% (Table 2). On Qwen2.5-1.5B, of 1,910 test prompts, 42.0% move by less than 1%, and the 99% point is 21.4%. In the bfloat16 analysis, 99% of 681 test-prompt activations moved by less than 3.2%, and the largest moved by 6.0%. NF4's 99% point is therefore 6.7 to 10.5 times bfloat16's.

**Table 2.** The drift of each test prompt's peak, as a share of the pole's scale, over the test prompts of the poles that reproduce. The random rows give the range over seeds. The bfloat16 row is from the earlier analysis of 24 float32 records and 2 bfloat16 records, which rounded the activations too.

| Model | Change | Test prompts | Under 1% | Under 5% | 99% point | 99.9% point | Largest |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | NF4 | 6,953 | 37.4% | 75.3% | 33.5% | 121.8% | 154.8% |
| Qwen2.5-0.5B | random | 6,953 | 34.1–34.8% | 72.1–73.3% | 32.4–36.0% | 79.6–102.7% | 127.7–191.0% |
| Qwen2.5-1.5B | NF4 | 1,910 | 42.0% | 76.7% | 21.4% | 35.4% | 78.3% |
| Qwen2.5-1.5B | random | 1,910 | 39.5–40.6% | 72.7–74.9% | 26.7–29.1% | 38.8–48.2% | 73.4–77.1% |
| three models | bfloat16 | 681 | – | – | 3.2% | 5.1% | 6.0% |

The drift is large, and yet most explanations keep. The keep test rescales each pole's line by the change in its maximum, so a pole whose whole range shrinks still fires on the same texts. Most drift is of this kind. The median scale ratio is 0.964 on Qwen2.5-0.5B and 0.971 on Qwen2.5-1.5B. The test prompts that cross a line are fewer than the drift suggests.

## Which poles flip?

The poles that flip are mostly those near a gate. On Qwen2.5-0.5B, cut into quarters of 90 or 91 poles by margin, NF4 flips 39 poles in the quarter nearest a gate, 18 in the next, 15 in the third and 9 in the furthest (Figure 2). The random changes give 27 to 34, 19 to 29, 13 to 17 and 4 to 6. On Qwen2.5-1.5B, NF4 flips 4, 3, 2 and 0, in quarters of 18 or 19.

![On Qwen2.5-0.5B, NF4 and random changes of its size flip many of the same poles, near a gate.](media/flips.svg)

**Figure 2.** On Qwen2.5-0.5B, NF4 and random changes of its size flip many of the same poles: those near a gate. Each cell is one of the 362 passing poles of Qwen2.5-0.5B that reproduce on the base model, in order of how far its base readings sit above the nearer gate, nearest first. A filled cell is a pole that flips under the change that the frame names: NF4 rounding, then each random change of its norm. Each frame is one measured condition.

The same poles flip under different changes. On Qwen2.5-0.5B, NF4 and the first random change share 34 flips, where independent flips would share 15.2 (hypergeometric $p = 1.5 \times 10^{-8}$). NF4 and instruction tuning, from the paper on random changes, share 38 flips of 361 poles, against 14.0 by chance. On Qwen2.5-1.5B, they share 4 flips of 75 poles, against 1.2. Both changes fail mostly on sensitivity. Of NF4's 81 flips on Qwen2.5-0.5B, 52 fail sensitivity alone, 27 specificity alone and 2 both.

## Do poles with non-normal weights lose their explanation more often?

Not clearly. Many more of these weights fail the normality test than QLoRA found for LLaMA 7B. On Qwen2.5-0.5B, the gate row fails for 93 of 363 poles (25.6%), the up row for 103 (28.4%) and the down column for 186 (51.2%). On Qwen2.5-1.5B, the shares are 20.0%, 21.3% and 42.7% of 75. Appendix F found 7.5% of LLaMA 7B's hidden units non-normal.

Failing the test barely predicts a lost explanation (Table 3). On Qwen2.5-0.5B, 30.4% of the poles with a non-normal gate row flip under NF4, against 19.6% of the others (Fisher $p$ = 0.042). The up row and the down column show no difference ($p$ = 0.78 and 0.45), and nor does a non-normal vector of any kind ($p$ = 0.23). On Qwen2.5-1.5B, no test reaches 0.05. One test of eight reaches $p < 0.05$, which is about what chance gives, so the result does not support normality as a predictor.

**Table 3.** The poles that reproduce, by whether their weight vector fails the Shapiro-Wilk test at 5%, and how many of each group flip under NF4. "Any" counts a pole when any of its three vectors fails.

| Model | Vector | Non-normal: flip / all | Normal: flip / all | Fisher $p$ |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | gate row | 28 / 92 | 53 / 270 | 0.042 |
| Qwen2.5-0.5B | up row | 24 / 103 | 57 / 259 | 0.78 |
| Qwen2.5-0.5B | down column | 45 / 186 | 36 / 176 | 0.45 |
| Qwen2.5-0.5B | any | 59 / 243 | 22 / 119 | 0.23 |
| Qwen2.5-1.5B | gate row | 4 / 15 | 5 / 60 | 0.073 |
| Qwen2.5-1.5B | up row | 3 / 16 | 6 / 59 | 0.39 |
| Qwen2.5-1.5B | down column | 6 / 32 | 3 / 43 | 0.16 |
| Qwen2.5-1.5B | any | 6 / 41 | 3 / 34 | 0.50 |

## Discussion

NF4 rounding is not a small change for explanations. It keeps about four in five explanations of Qwen2.5-0.5B and nine in ten of Qwen2.5-1.5B, which is about what instruction tuning keeps. At the 99% point, it moves the test prompts' peaks 6.7 to 10.5 times as far as bfloat16 does. A QLoRA run therefore starts from a model whose explanations have already moved by about as much as a full instruction tune moves them. Any test of what a QLoRA adapter does to explanations should compare the adapted model with the rounded base, not with the 16-bit base, or it will count the rounding's flips as the adapter's.

The flips say more about the poles than about the change. Under NF4, under a random change and under instruction tuning, the poles that flip are largely the same ones, and they sit near a gate on the base model. An explanation that barely passed its review is fragile under any change of this size. This agrees with the paper on random changes, which found the same overlap between instruction tuning and noise on Qwen2.5-0.5B. It suggests that a keep test should report the base margin of the poles it loses, because a loss near a gate is weak evidence that the change did something specific.

On Qwen2.5-1.5B, NF4 seems gentler than random noise of its size, although no single seed's test is significant. Its error is bounded by each block's scale, so it puts little error on small weights. This agrees with QLoRA's design: NF4 aims to spend its 16 values where the weights lie. It does not mean that the weights' normality matters for explanations. Half or more of the poles here have a non-normal weight vector, far more than in LLaMA 7B, and that barely predicts which explanations break.

## Limitations

- **Two models of one family.** Both are Qwen2.5, and both are stored in bfloat16. Another family could round differently.
- **Rounding only, with float32 compute.** QLoRA computes in bfloat16 after it dequantizes, and it quantizes the block scales again. Here the model computes in float32 and has no double quantization. So this measures the rounding of the weights alone.
- **bfloat16 was a check, not a measure.** The models' weights are already bfloat16, so rounding them to bfloat16 changed nothing. The comparison with bfloat16 rests on an earlier analysis, of fewer and different records, that also rounded the activations.
- **Few random seeds.** The random change has three seeds. On Qwen2.5-1.5B, they range from 58 to 62 keeps, and no single paired test is significant.
- **A rescaled line.** The keep test rescales the line by the change in each pole's maximum. A rounding that shifts a pole's whole range does not count against it.
- **Eight normality tests.** The one test below 0.05 is about what chance gives among eight. QLoRA's authors note that the test's $p$-value may not be accurate at large sample sizes, and the vectors here hold 896 to 8,960 weights.
- **Our poles.** The poles are our passing poles, not a random sample of the models' neurons.

## Conclusion

Most passing explanations keep through NF4 rounding: 281 of 362 on Qwen2.5-0.5B and 66 of 75 on Qwen2.5-1.5B. That is about as many as instruction tuning keeps. The rounding moves the test prompts' readings much further than bfloat16 does, and the explanations it breaks are mostly the fragile ones near a gate, which other changes break too. On Qwen2.5-1.5B, NF4 keeps more explanations than a random change of its size. Its weights' normality does not tell which explanations break.

## References

- Cammarata, N. (2026). *Neural networks are already clean*. Goodfire research essay. [already-clean.goodfire.com](https://already-clean.goodfire.com/).
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient finetuning of quantized LLMs*. NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314).
- Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., . . . Qiu, Z. (2024). *Qwen2.5 technical report*. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115).

## Appendices

### A. Reproduce

Each run reads every VERIFIED pole of the model from `wiki/`, with errata applied, and loads the model in float32 on the GPU. A random change's noise comes from one generator, seeded with the seed of the row, drawn over the weights in the order of their names. The runs took about 12 minutes for Qwen2.5-0.5B and 15 minutes for Qwen2.5-1.5B, on one GPU with 12 GB. The normality test reads the weights on the CPU, with SciPy 1.16.3.

    uv run python scripts/nf4_keep.py --base Qwen/Qwen2.5-0.5B --device cuda --out 0.5b.jsonl
    uv run python scripts/nf4_keep.py --base Qwen/Qwen2.5-1.5B --device cuda --out 1.5b.jsonl
    cat 0.5b.jsonl 1.5b.jsonl > runs.jsonl
    uv run --with scipy==1.16.3 python scripts/nf4_keep_stats.py normality runs.jsonl --out normality.jsonl
    cat runs.jsonl normality.jsonl > papers/nf4-keep/results.jsonl
    uv run python scripts/nf4_keep_stats.py drift papers/nf4-keep/results.jsonl
    uv run python scripts/nf4_keep_stats.py paired papers/nf4-keep/results.jsonl \
        --instruct papers/random-change-keep/results.jsonl
    uv run python -c "import sys; sys.path.insert(0, 'scripts'); import json, random_change_stats as s; rows = [json.loads(l) for l in open('papers/nf4-keep/results.jsonl')]; [print(r) for m in ('Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B') for r in s.by_margin(rows, m) + s.activity(rows, m)]"
    uv run python scripts/paper_figures.py

`results.jsonl` holds a `base` row for each pole, with its readings on the base model. It holds a `summary` row for each model, condition and seed, with the keep share, its interval, the poles that flip and the norm of the change. Each `pole` row gives a pole's readings under one condition and seed, whether it keeps, its scale ratio, its correlation and the drift of each probe. Each `normality` row gives the Shapiro-Wilk $p$-values of a pole's three weight vectors, and the `normality_keep` rows give Table 3. The poles that a run selects depend on the registry on the day it runs.

### B. Paired tests for each seed

**Table 4.** NF4 against each random change on the same poles that reproduce. *Only NF4* and *only random* are the poles that flip under one change and not the other, and $p$ is the exact McNemar test. *Both* is the poles that flip under both, *expected* the overlap that independent flips would give, and the last column the one-sided hypergeometric test of the overlap.

| Model | Seed | NF4 flips | Random flips | Only NF4 | Only random | McNemar $p$ | Both | Expected | Overlap $p$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | 0 | 81 | 68 | 47 | 34 | 0.18 | 34 | 15.22 | $1.5 \times 10^{-8}$ |
| Qwen2.5-0.5B | 1 | 81 | 76 | 42 | 37 | 0.65 | 39 | 17.01 | $1.8 \times 10^{-10}$ |
| Qwen2.5-0.5B | 2 | 81 | 79 | 42 | 40 | 0.91 | 39 | 17.68 | $9.3 \times 10^{-10}$ |
| Qwen2.5-1.5B | 0 | 9 | 16 | 3 | 10 | 0.092 | 6 | 1.92 | 0.0022 |
| Qwen2.5-1.5B | 1 | 9 | 17 | 4 | 12 | 0.077 | 5 | 2.04 | 0.024 |
| Qwen2.5-1.5B | 2 | 9 | 13 | 5 | 9 | 0.42 | 4 | 1.56 | 0.043 |

### C. Who did the work

A model, Claude Opus 5.5 in Claude Code, wrote the plan, the code, the analysis and the text. The registry's agents wrote and reviewed the explanations and their probes.

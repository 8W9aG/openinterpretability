# Does pruning a few instruct-model neurons bring back the explanations that instruction tuning breaks?

Jain et al. (2024) report that fine-tuning often adds a thin wrapper on top of a pretrained capability, and that pruning a few neurons, chosen by gradient times weight on the old task's loss, brings the capability back. We asked whether the same pruning brings back the neuron explanations that instruction tuning breaks. On Qwen2.5-0.5B-Instruct, we scored each MLP neuron by gradient times weight of the KL to the base model. Then we pruned the top 5 to 500 neurons' writes to the residual stream, and tested every explanation that had passed the adversarial review again. Of the 63 explanations that fail on the instruct model, pruning the top 5 neurons brings back 16, against 0 to 4 for random neurons of the same layers (McNemar $p \leq 0.002$). But it also breaks 26 of the 299 that kept, against 0 or 1, and larger prunes break far more: 158 at 500 neurons. The KL to the base model rises, from 0.075 to 0.19 at 5 neurons and to 2.37 at 500, where the score predicted a fall. In this model, gradient times weight does not find a wrapper whose removal restores the base model. The explanations it brings back come at a larger cost to others. Pruning input rows, or scoring on the failed explanations' texts, brings back no more.

**Status:** Preprint. Work left: #6892 (a prune ranked by each neuron's measured KL drop), #7164 (whether neuron 671 of layer 1 alone breaks about 30 explanations).

## Introduction

Instruction tuning breaks about one in six of Qwen2.5-0.5B's neuron explanations that passed an adversarial review: they pass their test prompts on the base model and fail on the instruct model ([`papers/random-change-keep/`](../random-change-keep/README.md)). Whether those explanations are gone, or only hidden, is open. Jain et al. (2024) study fine-tuning on procedurally defined tasks, and find that it often leaves the pretrained capability in place. Fine-tuning learns a small transform on top of the capability, which they call a wrapper. In their grammar-counting task, pruning 5 to 15 neurons by gradient times weight on the pretraining loss removes the wrapper, and the pretraining task's accuracy comes back.

If instruction tuning works the same way on the neurons that we explain, a small set of instruct neurons should hold the change, and pruning them should bring the base model's explanations back. The test needs no training. It also differs from an earlier paper here, which pruned a neuron's own input weights ([`papers/pruning-keep/`](../pruning-keep/README.md)).

The paper's contributions are these:

- A test of Jain et al.'s pruning step on a real base-instruct pair, with the base model's next-token distribution as the old task.
- A random null of the same size and the same layers, with paired tests, at five sizes of the prune.
- Counts of both the explanations that the prune brings back and those that it breaks, beside the KL to the base model.
- A check of the score's first-order prediction, one neuron at a time.

The paper bears on research question 4, how far an explanation carries to a changed model ([research questions](../../docs/research-questions.md)).

## Related work

**Wrappers.** Jain et al. (2024) study small transformers: compiled Tracr programs, and models pretrained on a probabilistic grammar or on TinyStories. They fine-tune each on a related task. With a small learning rate, the pretraining capability stays readable by linear probes, and pruning a few neurons brings the pretraining accuracy back. In their counter task, pruning 5 to 15 neurons was enough. They prune weights or neurons in one step, by gradient times weight on the pretraining task's loss, and they test no random set of neurons. A neuron there is one row of a weight matrix. Their largest model has 91 million parameters, and most of their tasks are synthetic. We apply the step to a released instruct model, with a null.

**Explanations under instruction tuning in this repo.** On Qwen2.5-0.5B, the explanations that instruction tuning breaks are mostly those that sat near one of the review's thresholds on the base model, and a random change of the instruct change's size breaks about as many, largely the same ones ([`papers/random-change-keep/`](../random-change-keep/README.md)). Neither a neuron's diagonal Fisher nor the size of the instruct change in its own weights predicts which explanations break ([`papers/fisher-keep/`](../fisher-keep/README.md), [`docs/analyses/base-instruct.md`](../../docs/analyses/base-instruct.md)). And the passing poles carry little of the change in the instruct model's output ([`papers/instruct-difference/`](../instruct-difference/README.md)). These results make a wrapper on the explained neurons unlikely, but none of them tests one.

## Method

**Model and units.** The pair is Qwen2.5-0.5B and Qwen2.5-0.5B-Instruct (Yang et al., 2024), which share one architecture. Following Cammarata (2026), a *pole* is one sign of an MLP neuron's activation, positive or negative. A pole is *passing* when its explanation, a sentence that says when the pole fires, passed our adversarial review. In that review, a second model wrote test prompts to break the explanation. Hard negatives are near-miss texts that the explanation says should not fire, and positive test prompts are new texts that it says should fire. The explanation passed when its specificity on the hard negatives reached 0.80, and its sensitivity on the positive test prompts reached 0.60. Our set of tested explanations holds 363 passing poles of Qwen2.5-0.5B, and 362 of them pass both tests again on the base model.

**The score.** Let $p_{\text{base}}$ and $p_{\text{inst}}$ be the two models' next-token distributions, on the about 20,000 tokens of our scan corpus that the keep test reads, in whole windows of 512. The loss is their mean token KL,

$$
L = \frac{1}{T} \sum_{t=1}^{T} \mathrm{KL}\left(p_{\text{base}}(\cdot \mid x_{<t}) \,\middle\|\, p_{\text{inst}}(\cdot \mid x_{<t})\right).
$$

For an MLP neuron $j$ of layer $\ell$, let $W_\ell$ be the layer's down projection and $w_{\ell j}$ its column $j$, the neuron's write to the residual stream. The neuron's score is

$$
s_{\ell j} = \sum_i \frac{\partial L}{\partial W_{\ell, ij}} W_{\ell, ij},
$$

which is the first-order drop in $L$ when $w_{\ell j}$ is set to zero. This is Jain et al.'s gradient times weight, with the base model's distribution as the old task. Their unit is a row of a weight matrix, and ours is the column that holds an MLP neuron's write, for the reason that the next paragraph gives.

**Pruning.** To prune a neuron is to zero its write column in the instruct model. Its own activation stays, so the test prompts of a passing pole still read the pole where it fires. Zeroing the neuron's input rows would switch the pole off, which could not count as a recovery. We prune the top $K$ neurons by score over all 24 layers, for $K$ = 5, 15, 50, 150 and 500. The null prunes $K$ random neurons with the same count in each layer as the top $K$, and none of them, with three seeds for each $K$.

**The keep test.** This is the test of the paper on random changes. Each pole's test prompts run on the base model and on the pruned instruct model. On the pruned model, the line is the record's line times the ratio of the pole's maximum after the change to its maximum before, over the same 20,000 tokens. An explanation *keeps* when it passes both tests on both models. Against the unpruned instruct model, an explanation is *recovered* when it fails there and keeps after the prune. It is *lost* when it keeps there and fails after the prune. The exact McNemar test compares top and random pruning of the same $K$ on the same poles.

**The KL.** After each prune, we read $L$ again on the next windows of the corpus, about 20,000 tokens that the score never saw. As a check of the first order, we also prune each of the top 10 neurons alone, and read $L$ on the scoring tokens.

**Motion.** Figure 2 steps through the five values of $K$ for top pruning, one frame each. The frames show no state between two values.

## Does pruning by gradient times weight bring the failed explanations back?

It brings a few back, more than chance. Of the 63 explanations that fail on the instruct model, pruning the top 5 neurons brings back 16 (Figure 1, Table 1). Pruning 5 random neurons of the same layers brings back 1, 0 and 4 for the three seeds, and the paired tests give $p$ = $6.1 \times 10^{-5}$, $3.0 \times 10^{-5}$ and 0.0018. The top prune recovers 13 at 15 neurons and 15 at 50, still more than random ($p$ from $2.4 \times 10^{-4}$ to 0.049). At 150 and 500 neurons, the random prunes recover as many as the top prune, 6 to 15 against 11.

![Pruning by gradient times weight brings back a few failed explanations, and breaks many more.](media/recovery.svg)

**Figure 1.** Pruning by gradient times weight brings back a few failed explanations, and breaks many more. Qwen2.5-0.5B's instruct model with the writes of $K$ MLP neurons pruned. Left: of the 63 explanations that fail on it, those that pass their test prompts again. Right: of the 299 that keep on it, those that fail. Circles prune the top $K$ by gradient times weight, and triangles $K$ random neurons of the same layers, one for each seed.

**Table 1.** The explanations recovered and lost at each $K$, the keep share over the 362 that reproduce, and the KL to the base model on held-out tokens. The unpruned instruct model keeps 299 (82.6%) at a KL of 0.075. Random rows give the range over three seeds.

| $K$ | Prune | Recovered, of 63 | Lost, of 299 | Keep share | Held-out KL |
| --- | --- | --- | --- | --- | --- |
| 5 | top | 16 | 26 | 79.8% | 0.188 |
| 5 | random | 0–4 | 0–1 | 82.6–83.7% | 0.076 |
| 15 | top | 13 | 60 | 69.6% | 0.402 |
| 15 | random | 0–2 | 1–3 | 81.8–82.9% | 0.076–0.077 |
| 50 | top | 15 | 69 | 67.7% | 0.585 |
| 50 | random | 3–6 | 2–4 | 82.3–83.7% | 0.079–0.084 |
| 150 | top | 11 | 123 | 51.7% | 1.305 |
| 150 | random | 6–9 | 5–9 | 82.0–83.7% | 0.081–0.084 |
| 500 | top | 11 | 158 | 42.0% | 2.373 |
| 500 | random | 13–15 | 14–21 | 80.4–82.9% | 0.094–0.104 |

## What does the prune cost?

It costs more explanations than it brings back, at every $K$. Pruning the top 5 neurons breaks 26 of the 299 explanations that keep on the instruct model, so the keep share falls from 82.6% to 79.8%. At 500 neurons it breaks 158, and the keep share falls to 42.0%. Random prunes of the same layers break 0 to 21. On the lost explanations, top pruning differs from every random prune with $p < 10^{-7}$ (Table 4 in Appendix B).

The prune also moves the instruct model away from the base model, not towards it. On held-out tokens, the KL to the base model rises from 0.075 to 0.188 at 5 neurons and to 2.373 at 500, while random prunes leave it between 0.076 and 0.104. The score predicted the opposite: the top 5 neurons' scores sum to a first-order drop of 0.050 on the scoring tokens.

Figure 2 shows where the losses fall. At small $K$ they fall near a gate, like the instruct model's own failures, and at large $K$ they fall everywhere. We cut the poles into quarters of 90 or 91 by margin, the distance of a pole's base readings above the nearer gate. From the quarter nearest a gate to the furthest, the top 5 prune loses 8, 12, 5 and 1 explanations. The instruct model keeps 59, 76, 80 and 84 poles in those quarters, so 20 of the 26 losses are in the two quarters nearest a gate. At 500 neurons it loses 45, 32, 46 and 35, at every margin. The instruct model's own failures sit near a gate: 31, 15, 11 and 6. So a large top prune breaks explanations far from a gate, which instruction tuning does not.

![As more neurons go, top pruning breaks explanations at every margin.](media/grid.svg)

**Figure 2.** As more neurons go, top pruning breaks explanations at every margin. Each cell is one of the 362 passing poles of Qwen2.5-0.5B that reproduce on the base model, nearest a gate first. Circles are explanations that fail on the instruct model and pass again after top pruning. Triangles are explanations that keep on the instruct model and fail after it. Dark cells still fail. Each frame is one measured value of $K$.

## Why does the score fail?

Because the first order does not hold for the neurons it ranks highest. Pruning each of the top 10 neurons alone, on the scoring tokens, raises the KL for 5 of them, where each score predicted a fall (Table 2). Neuron 813 of layer 2 has a score of 0.0033 and raises the KL by 0.112 when pruned, from 0.0717, which more than doubles it. Neuron 567 of layer 21 has the highest score, 0.0288, and raises the KL by 0.012. For the five neurons with small effects, the measured drop has the score's sign and a size within a factor of about two.

**Table 2.** The top 10 neurons by score, each pruned alone. The measured drop is the KL before the prune, 0.0717 on the scoring tokens, less the KL after it. A negative drop is a rise. The column norm is the norm of the neuron's write, beside its layer's median.

| Layer | Neuron | Score | Measured drop | Column norm | Layer median |
| --- | --- | --- | --- | --- | --- |
| 21 | 567 | 0.0288 | −0.0119 | 0.636 | 0.597 |
| 21 | 3486 | 0.0077 | −0.0083 | 0.610 | 0.597 |
| 21 | 3435 | 0.0069 | −0.0073 | 0.604 | 0.597 |
| 6 | 2765 | 0.0036 | 0.0029 | 0.436 | 0.534 |
| 2 | 813 | 0.0033 | −0.1121 | 0.449 | 0.509 |
| 21 | 2930 | 0.0032 | −0.0071 | 0.350 | 0.597 |
| 23 | 4512 | 0.0025 | 0.0020 | 0.383 | 0.560 |
| 5 | 419 | 0.0020 | 0.0023 | 0.399 | 0.495 |
| 6 | 4648 | 0.0010 | 0.0007 | 0.607 | 0.534 |
| 7 | 505 | 0.0009 | 0.0004 | 0.535 | 0.532 |

The failures are not large writes. No neuron's column norm is much above its layer's median, and two are well below it. So the harm comes from what the neuron's write does downstream, which the gradient at the unpruned model does not see. Linear estimates of this kind are known to fail for large changes: Syed et al. (2023) find that attribution patching, a first-order estimate of a component's effect, fits exact patching with $R^2 = 0.27$ on one task. In this repo, a first-order estimate of a single pole's effect ranks its exact effect well, at a Spearman of 0.85 on Qwen2.5-0.5B ([`docs/analyses/pole-effects.md`](../../docs/analyses/pole-effects.md)), but it reads small changes to one activation, not the removal of a write. We suggest, as a hypothesis we did not test, that the rest of the network depends on these writes in a way that is flat for small changes and steep for their removal.

## Do the two variants do better?

No. We ran the two variants that the first version of this paper left out (#7071). Each changes one part of the method, and keeps the rest.

- **Prune the input rows.** We zero the neuron's rows of `gate_proj` and `up_proj`, in place of its write column. Its activation is then 0 everywhere. The score sums gradient times weight over both rows.
- **Score on the failed explanations' texts.** The old task is the KL to the base model on the test prompts of the 63 failed explanations, in place of the corpus. The prune is the paper's write prune. The texts could fit the score to the test prompts that the keep test reads. So only a random half of the failed explanations, 32, give their texts to the score, with seed 0. The other 31 are a held-out check.

Our set of tested explanations has grown to 379 passing poles of Qwen2.5-0.5B that reproduce since the paper's run. Table 3 counts only the paper's 362, so that it compares with Table 1. The 17 new poles all keep on the instruct model.

![Neither variant recovers more explanations, and the failed-text loss loses more.](media/variants.svg)

**Figure 3.** Neither variant recovers more explanations, and the failed-text loss loses more. The top $K$ prune of the paper and of each variant, on the paper's 362 poles. Left: of the 63 explanations that fail on the instruct model, those that pass again. Right: of the 299 that keep on it, those that fail. Circles are the paper's prune, triangles the input-row prune and squares the failed-text prune.

**Table 3.** The top $K$ prune of each variant against the paper's, on the paper's 362 poles. Each cell gives the explanations recovered, of 63, and lost, of 299, with the held-out KL. An input prune switches the pruned neurons' own poles off, so they count as neither: 1 pole for $K$ up to 150 and 6 at 500. For the failed-text prune, *held out* counts the recovered explanations among the 31 whose texts the score never read.

| $K$ | Paper: recovered, lost, KL | Input rows: recovered, lost, KL | Failed texts: recovered, lost, KL | Failed texts: held out, of 31 |
| --- | --- | --- | --- | --- |
| 5 | 16, 26, 0.188 | 16, 26, 0.188 | 16, 26, 0.190 | 6 |
| 15 | 13, 60, 0.402 | 16, 28, 0.200 | 18, 81, 0.547 | 7 |
| 50 | 15, 69, 0.585 | 13, 65, 0.597 | 14, 119, 1.079 | 8 |
| 150 | 11, 123, 1.305 | 9, 85, 0.848 | 9, 157, 1.712 | 3 |
| 500 | 11, 158, 2.373 | 11, 112, 1.200 | 6, 185, 1.985 | 1 |

**An input prune is the same change as a write prune, except for the pruned neuron's own poles.** A neuron whose activation is 0 writes nothing, so the rest of the model sees the same change either way. At $K$ = 5 the two scores choose the same five neurons, and the results match exactly: 16 recovered, 26 lost and a KL of 0.188. Random prunes match too, since they copy the top prune's layers. At larger $K$ the two scores rank neurons a little differently, and so the counts differ. At $K$ = 15 the sets differ by one neuron. The input score takes neuron 765 of layer 12 in place of neuron 671 of layer 1, and the losses fall from 60 to 28. So the first version's worry, that an input prune "would test a different question", holds only for the 1 to 6 poles whose own neuron goes.

**A closer old task finds the same few explanations and breaks more** (Figure 3). Scored on the failed explanations' texts, the top 5 neurons differ from the paper's by one: neuron 2116 of layer 3 in place of neuron 2765 of layer 6. They recover the same 16 explanations, and lose the same 26. From $K$ = 15 on, this prune loses more than the paper's at every $K$, 81 against 60 at 15 and 185 against 158 at 500. The KL to the base model rises faster too, to 0.547 at 15 against 0.402. The held-out half shows little fitting. At 5 neurons the held-out half recovers 6 of 31, against 10 of the 32 whose texts were scored, and the random prunes recover 0 or 1 held-out explanations ($p$ = 0.031 to 0.063). At 500 neurons the random prunes recover more held-out explanations than the top prune, 8 to 12 against 1. `scripts/wrapper_prune_stats.py` prints the tests against each random prune, as for Table 4.

## Discussion

In Qwen2.5-0.5B-Instruct, gradient times weight on the KL to the base model does not find a wrapper whose removal restores the base model. It finds neurons whose removal brings back some explanations, more than chance at small $K$, but it breaks more explanations than it brings back, and it moves the model further from the base. Jain et al. (2024) found the opposite in small models fine-tuned at a small learning rate, mostly on synthetic tasks. Our setting differs in three ways that could explain it. The model is larger and trained on natural text. Instruction tuning is a long fine-tune on broad data, not a short one on a related synthetic task. And our old task is the base model's whole next-token distribution, not one capability.

The 16 explanations that come back at 5 neurons are a hint, not a wrapper. Random prunes recover almost none at that size, so the top neurons carry something that these explanations depend on. But the same prune breaks 26 others, and the first-order score is wrong for half of its own top 10. A score that reads the measured effect of each prune, or a second-order score, could find a cleaner set. A closer old task does not: scored on the failed explanations' own texts, the top 5 neurons bring back the same 16 (Table 3). Without such a set, the result agrees with the earlier finding that the instruct change is spread over almost every neuron ([`docs/analyses/base-instruct.md`](../../docs/analyses/base-instruct.md)): no small set of neurons holds it.

The failure of the first order is itself a warning for a pruning step that ranks neurons by gradient times weight, as Jain et al.'s does. In this model, a few neurons with ordinary weights and small scores do great harm when pruned. One neuron of layer 2 more than doubles the KL alone. A pruning study that ranks by such a score, with no check of each prune's measured effect, can remove exactly the neurons that the model depends on most.

## Limitations

- **One pair.** Qwen2.5-0.5B is the only pair with enough failed explanations. Qwen2.5-1.5B has 10, too few for this test.
- **One score, in one step.** We use Jain et al.'s one-step gradient times weight. An iterative prune, a second-order score or a score from each prune's measured effect could find a different set.
- **The base model's whole distribution as the old task.** Jain et al. use one task's loss. A loss built from the failed explanations' own texts recovers no more (Table 3), but other task losses were not tried.
- **Writes, not activations.** Pruning a neuron's write keeps its activation, so a pruned passing pole is still read. Pruning its input rows is the same change for every other pole (Table 3).
- **Three seeds.** The random null has three seeds at each $K$.
- **A rescaled line.** The keep test rescales each pole's line by the change in its maximum.

## Conclusion

Pruning the top 5 instruct neurons by gradient times weight on the KL to the base model brings back 16 of the 63 explanations that instruction tuning breaks, more than random pruning. But it breaks 26 others, raises the KL that it was meant to lower, and does more harm as more neurons go. In Qwen2.5-0.5B, instruction tuning's effect on the explained neurons is not a thin wrapper that this score can find.

## References

- Cammarata, N. (2026). *Neural networks are already clean*. Goodfire research essay. [already-clean.goodfire.com](https://already-clean.goodfire.com/).
- Jain, S., Kirk, R., Lubana, E. S., Dick, R. P., Tanaka, H., Grefenstette, E., Rocktäschel, T., & Krueger, D. (2024). *Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks*. ICLR 2024. [arXiv:2311.12786](https://arxiv.org/abs/2311.12786).
- Syed, A., Rager, C., & Conmy, A. (2023). *Attribution patching outperforms automated circuit discovery*. [arXiv:2310.10348](https://arxiv.org/abs/2310.10348).
- Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., . . . Qiu, Z. (2024). *Qwen2.5 technical report*. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115).

## Appendices

### A. Reproduce

The run reads every VERIFIED pole of Qwen2.5-0.5B from `wiki/`, with errata applied. It loads the base model in float32 on the GPU, and the instruct model on the CPU, only to take the instruct change. The base model's log-probabilities on the 40,000 tokens are kept on the CPU in float16. A random prune's neurons come from one generator per seed. The run took about 45 minutes on one GPU with 12 GB.

    uv run python scripts/wrapper_prune.py --base Qwen/Qwen2.5-0.5B --device cuda --out runs.jsonl
    uv run python scripts/wrapper_prune_check.py --base Qwen/Qwen2.5-0.5B --device cuda \
        --results runs.jsonl --out check.jsonl
    cat runs.jsonl check.jsonl > papers/wrapper-prune/results.jsonl
    uv run python scripts/wrapper_prune_stats.py papers/wrapper-prune/results.jsonl
    uv run python scripts/paper_figures.py

The two variants of Table 3 (#7071) ran on the same GPU, in about 20 minutes each. Each writes its own file, and every row names its `unit` and `loss`. The paper's own rows have neither field, and count as `write` and `base`. The variant files were written before the base rows got these fields, so a short script added them when the files were merged, with each file's values:

    uv run python scripts/wrapper_prune.py --base Qwen/Qwen2.5-0.5B --device cuda --unit input \
        --out variant1.jsonl
    uv run python scripts/wrapper_prune.py --base Qwen/Qwen2.5-0.5B --device cuda --loss failed \
        --out variant2.jsonl
    cat variant1.jsonl variant2.jsonl >> papers/wrapper-prune/results.jsonl
    uv run python scripts/wrapper_prune_stats.py papers/wrapper-prune/results.jsonl --unit input --paper-poles
    uv run python scripts/wrapper_prune_stats.py papers/wrapper-prune/results.jsonl --loss failed --paper-poles

`results.jsonl` holds a `base` row for each pole and a `scores` row with every neuron's score. It holds a `summary` row for each condition, with the neurons pruned, the keep share and the held-out KL. It holds a `pole` row for each pole and condition, and a `single` row for each of the top 10 neurons pruned alone.

### B. Paired tests for each seed

**Table 4.** Top pruning against each random prune of the same $K$, on the same poles. *Recovered p* is the exact McNemar test on the 63 explanations that fail on the instruct model, and *lost p* the test on the 299 that keep.

| $K$ | Seed | Random recovered | Random lost | Recovered $p$ | Lost $p$ |
| --- | --- | --- | --- | --- | --- |
| 5 | 0 | 1 | 1 | $6.1 \times 10^{-5}$ | $6.0 \times 10^{-8}$ |
| 5 | 1 | 0 | 0 | $3.0 \times 10^{-5}$ | $3.0 \times 10^{-8}$ |
| 5 | 2 | 4 | 0 | 0.0018 | $3.0 \times 10^{-8}$ |
| 15 | 0 | 1 | 2 | $4.9 \times 10^{-4}$ | $6.9 \times 10^{-18}$ |
| 15 | 1 | 0 | 3 | $2.4 \times 10^{-4}$ | $1.6 \times 10^{-15}$ |
| 15 | 2 | 2 | 1 | 0.0074 | $5.4 \times 10^{-17}$ |
| 50 | 0 | 5 | 2 | 0.021 | $2.4 \times 10^{-19}$ |
| 50 | 1 | 3 | 4 | 0.0018 | $5.1 \times 10^{-17}$ |
| 50 | 2 | 6 | 2 | 0.049 | $1.4 \times 10^{-20}$ |
| 150 | 0 | 9 | 5 | 0.8 | $1.8 \times 10^{-34}$ |
| 150 | 1 | 7 | 9 | 0.45 | $4.2 \times 10^{-32}$ |
| 150 | 2 | 6 | 8 | 0.3 | $2.1 \times 10^{-32}$ |
| 500 | 0 | 13 | 21 | 0.8 | $1.1 \times 10^{-33}$ |
| 500 | 1 | 14 | 14 | 0.66 | $6.1 \times 10^{-38}$ |
| 500 | 2 | 15 | 14 | 0.5 | $6.1 \times 10^{-38}$ |

### C. Who did the work

A model, Claude Opus 5.5 in Claude Code, wrote the plan, the code, the analysis and the text. The registry's agents wrote and reviewed the explanations and their probes.

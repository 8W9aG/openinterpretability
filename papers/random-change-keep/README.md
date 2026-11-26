# Tuning or size? A matched random change for neuron explanations under instruction tuning

When a model is fine-tuned, some explanations of its neurons stop being true, and a keep test counts them. Instruction tuning keeps about four in five of the VERIFIED explanations of Qwen2.5-0.5B, but that share alone cannot say whether the tuning or merely the size of the change breaks the rest. We added to each base model a random Gaussian change with the instruct change's Frobenius norm in every weight matrix, with three seeds for all weights and five for the MLP weights alone. Then we ran each explanation's own probes again. On Qwen2.5-0.5B, the instruct change keeps 298 of 361 explanations, and the random changes keep 296 to 301 (paired McNemar $p \geq 0.79$). The two changes flip largely the same poles, which sit near a review gate on the base model. On Qwen2.5-1.5B, the instruct change keeps 65 of 75, and the random changes keep 72 or 73 ($p$ = 0.02 to 0.04). There, the random changes barely move the activations, whereas on Qwen2.5-0.5B they move them more than the tuning does. A random change of the same norm is therefore not a change of the same effect. A keep share needs such a matched random change beside it before it is read as an effect of the tuning.

**Status:** Preprint. Work left: #6814 (a low-rank random change of the same norm), #6842 (a random change of the same effect on Qwen2.5-1.5B).

## Introduction

When a model is fine-tuned, an explanation of one of its neurons can stop being true. A keep test measures this. It runs the explanation's own probes again on the tuned model, and asks whether the explanation still passes. An earlier analysis in this repo found that instruction tuning keeps 81% of the VERIFIED explanations of Qwen2.5-0.5B and 88% of those of Qwen2.5-1.5B ([`docs/analyses/base-instruct.md`](../../docs/analyses/base-instruct.md)). Such a share is hard to read alone. A share of 81% could mean that instruction tuning rewrites one neuron in five, or that one explanation in five sits so close to a gate that any change would flip it.

A null separates the two readings. Fine-tuning moves the weights by a definite amount, and a random change of the same amount has no structure that could target a neuron. If the random change breaks as many explanations, then the keep share measures the size of the change and the fragility of the explanations, not what the tuning did. Aghajanyan et al. (2021) show that fine-tuning can reach 90% of a full fine-tune's score inside a random subspace of as few as a few hundred dimensions, but they never compare a fine-tune with a random update of its size. The repo's review of their paper proposed this test ([`docs/literature/intrinsic-dimensionality-fine-tuning.md`](../../docs/literature/intrinsic-dimensionality-fine-tuning.md)).

This paper runs that null on both Qwen2.5 pairs (Yang et al., 2024). For each weight of the base model, we take its change under instruction tuning and draw Gaussian noise with the same shape and Frobenius norm. We then run the keep test on every VERIFIED pole under the instruct change and under the matched random changes, on all weights and on the MLP weights alone.

The paper's contributions are these:

- A matched random null for the keep test under instruction tuning, on two model sizes, with several seeds and Wilson intervals.
- The same comparison on the MLP weights alone, where a pole's own weights sit.
- An analysis of which poles flip, which shows that instruction tuning and random changes flip largely the same poles, and that those poles were near a gate on the base model.
- The rows for each pole, condition and seed, so that other analyses can use the same outcomes.

The paper bears on research question 1, *Are native MLP neurons clean?*, and on the question of how far an explanation survives a change to the model ([research questions](../../docs/research-questions.md)).

## Related work

**Fine-tuning in a small subspace.** Li et al. (2018) train a network only inside a random subspace of its weights, and raise the subspace's dimension until a solution appears. Aghajanyan et al. (2021) apply this to pre-trained language models, and find that as few as a few hundred parameters, projected into the full weight space, can reach 90% of a full fine-tune's score. That holds for RoBERTa-Large on MRPC, whereas BERT-Base on QQP needs about 8,000. Both papers measure task performance. Neither measures a unit, and neither compares a fine-tune with a random update of the same size. Our null is that comparison, at the level of single explanations.

**Random baselines for weight changes.** Ilharco et al. (2023) edit models by adding or subtracting a task vector, the weight change of a fine-tune. As a control, they add a random vector scaled per layer to the task vector's norm. On GPT-2 Large, the random vector leaves the toxic share, the mean toxicity and the perplexity where they were, whereas the negated task vector lowers the toxic share from 4.8% to 0.8%. Our null is the same idea, scaled per weight matrix, and our measure is whether each VERIFIED explanation keeps.

**The size of a fine-tune's change.** Biderman et al. (2024) find that full fine-tuning makes weight changes of a much higher rank than usual LoRA ranks, and that LoRA forgets less of the base model's skills. The Qwen2.5 instruct change is of the high-rank kind. Its rank for 90% of its variance is 668 to 701 of 896 in Qwen2.5-0.5B, and the top 5% of neurons hold 6.6% of its energy, against 5.3% after a random rotation ([`docs/analyses/base-instruct.md`](../../docs/analyses/base-instruct.md)). A dense random change is therefore a fair null for its direction. A low-rank random change, of the kind a LoRA adapter makes (Hu et al., 2022), is a different null, and #6814 holds it. Shuttleworth et al. (2025) find that LoRA adds new high-ranking singular vectors to a weight matrix, which they call intruder dimensions, and that full fine-tuning almost never does. A dense Gaussian change spreads its norm over all directions, so we expect it to add none either. That inference is ours, not theirs.

**Keep tests in this repo.** The interference report compares a changed model with its base against a random floor of the same size ([`docs/analyses/interference.md`](../../docs/analyses/interference.md)). On Qwen2.5-1.5B, one random draw flipped 3 of 59 VERIFIED poles, and the instruct model flipped 9. That report used one seed and one model. Other work here asks what predicts which explanations keep under instruction tuning. Neither the diagonal Fisher of a neuron's weights ([`papers/fisher-keep/`](../fisher-keep/README.md)) nor the size of the instruct change in a pole's own weights predicts it beyond depth ([`docs/analyses/base-instruct.md`](../../docs/analyses/base-instruct.md)). And the VERIFIED poles carry little of the change in the instruct model's output ([`papers/instruct-difference/`](../instruct-difference/README.md)). Magnitude pruning of a neuron's own input weights keeps more explanations than random pruning of as many weights ([`papers/pruning-keep/`](../pruning-keep/README.md)). That test removes weights, whereas the present test adds a change to all of them.

## Method

**Models and units.** The models are Qwen2.5-0.5B and Qwen2.5-1.5B and their instruct models (Yang et al., 2024). Each pair shares one architecture, so a weight has the same position in both. A *pole* is one sign of an MLP neuron's activation, positive or negative. A pole is *VERIFIED* when its explanation, a sentence that says when the pole fires, passed the registry's adversarial review. In that review, a reviewer model wrote hard negatives, near-miss texts that the explanation says should not fire, and positive holdouts, new texts that it says should fire. The pole fires on a text when its peak activation there reaches the record's line, a quarter of the pole's maximum on its scan. The explanation passed when its specificity on the hard negatives reached the gate of 0.80, and its sensitivity on the holdouts reached 0.60. The registry holds 362 VERIFIED poles of Qwen2.5-0.5B and 75 of Qwen2.5-1.5B, with errata applied. On the base models, 361 and 75 of them pass both gates again.

**The changes.** For each weight matrix or vector $W$ of the base model, let $\Delta_W$ be the instruct model's $W$ minus the base model's. We add one of four changes to the base model's weights:

- *Instruct*: $\Delta_W$ on every weight. This is the instruct model.
- *Instruct, MLP only*: $\Delta_W$ on the three MLP matrices of each block (the gate, up and down projections), and no change elsewhere.
- *Random, all weights*: on every weight, Gaussian noise $\varepsilon_W$ with the shape of $W$, rescaled so that its Frobenius norm is that of $\Delta_W$.
- *Random, MLP only*: the same noise on the three MLP matrices of each block, and no change elsewhere.

$$
\varepsilon_W = \lVert \Delta_W \rVert_F \, \frac{Z}{\lVert Z \rVert_F}, \qquad Z_{ij} \sim \mathcal{N}(0, 1).
$$

Each random change has the norm of its instruct change in every matrix, so the two differ only in direction. We draw 3 seeds for the change on all weights and 5 for the change on the MLP weights. The change on all weights is the random floor of the interference report.

**The keep test.** This is the test of the base-instruct analysis. Each pole's probes run on the base model and on the changed model. On the changed model, the line is the record's line times the ratio of the pole's maximum after the change to its maximum before. Both maxima come from the same 20,000 tokens of the registry's scan corpus. So "fires" means the same share of the pole's own range on both models. An explanation *reproduces* when it passes both gates on the base model, and it *keeps* when it passes them on the changed model too. The keep share is over the poles that reproduce, and it has a 95% Wilson interval. A pole *flips* when it reproduces and does not keep.

**Activation change.** For each pole and change, we also give Pearson's $r$ between the pole's activations before and after the change, over the same 20,000 tokens, with each window's first token left out.

**Overlap.** Two changes flip sets of poles $A$ and $B$ out of $n$. If the flips were independent, the expected overlap would be $|A||B|/n$. We compare the observed overlap with that value, with a one-sided hypergeometric test.

**Motion.** Figure 2 steps through the conditions, one frame each, and marks the poles that flip in each. The frames show no state between two conditions.

## Does a random change of the same size break as many explanations?

On Qwen2.5-0.5B, a random change of the same size breaks as many explanations as instruction tuning. The instruct change keeps 298 of the 361 explanations that reproduce, 82.6% (95% interval 78.3–86.1%). The three random changes of all weights keep 296, 301 and 300 (Figure 1, Table 1). On the same poles, the paired McNemar test finds no difference for any seed ($p$ = 0.79 to 0.90, Table 2 in Appendix B). The MLP weights alone give the same answer. The instruct change on the MLP weights keeps 312 explanations (86.4%), and the five random changes of the MLP weights keep 302 to 309 ($p$ = 0.25 to 0.79).

On Qwen2.5-1.5B, the random changes break far fewer. The instruct change keeps 65 of the 75 explanations that reproduce, 86.7% (77.2–92.6%). The three random changes of all weights keep 73, 73 and 72, from 96.0% to 97.3%. Instruction tuning flips 10 poles, and a random change of its size flips 2 or 3. The paired test gives $p$ = 0.022, 0.022 and 0.039 for the three seeds. On the MLP weights alone, the instruct change keeps 66 and the random changes keep 71 to 75, with $p$ from 0.0039 to 0.13.

![On Qwen2.5-0.5B, a random change of the same size breaks as many explanations as instruction tuning. On Qwen2.5-1.5B, it breaks far fewer.](media/keep.svg)

**Figure 1.** The share of each model's VERIFIED explanations that still pass their probes under each change, with 95% Wilson intervals. Circles are the instruct change, and triangles are random changes with the instruct change's Frobenius norm in every matrix, one for each seed. The top two rows change every weight, and the bottom two the MLP weights only.

**Table 1.** The explanations that keep under each change, out of those that reproduce on the base, with the median correlation of each pole's activations with its base activations. A random row gives the range over its seeds.

| Model | Change | Weights | Keep | Share | Median $r$ |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | instruct | all | 298 of 361 | 82.6% | 0.974 |
| Qwen2.5-0.5B | random, 3 seeds | all | 296–301 | 82.0–83.4% | 0.960 |
| Qwen2.5-0.5B | instruct | MLP | 312 of 361 | 86.4% | 0.980 |
| Qwen2.5-0.5B | random, 5 seeds | MLP | 302–309 | 83.7–85.6% | 0.973–0.974 |
| Qwen2.5-1.5B | instruct | all | 65 of 75 | 86.7% | 0.987 |
| Qwen2.5-1.5B | random, 3 seeds | all | 72–73 | 96.0–97.3% | 0.999 |
| Qwen2.5-1.5B | instruct | MLP | 66 of 75 | 88.0% | 0.993 |
| Qwen2.5-1.5B | random, 5 seeds | MLP | 71–75 | 94.7–100% | 0.999 |

The activations explain the difference between the two models. On Qwen2.5-0.5B, the random change of all weights moves each pole's activations more than the instruct change does, to a median correlation of 0.960 against 0.974. On Qwen2.5-1.5B, it barely moves them, to 0.999 against 0.987. A random change with the norm of the instruct change is therefore not a change with its effect. Its effect depends on how large the change is beside the weights, and the instruct change is about 8% of the weights at a median pole in Qwen2.5-0.5B, against 1.2% in Qwen2.5-1.5B ([`docs/analyses/base-instruct.md`](../../docs/analyses/base-instruct.md)).

## Do instruction tuning and a random change break the same explanations?

On Qwen2.5-0.5B, they largely do. The instruct change flips 63 poles, and the first random change of all weights flips 65. Of these, 32 flip under both, where independent flips would share 11.3 (hypergeometric $p = 1.9 \times 10^{-11}$). The other two seeds share 32 and 30 poles with the instruct change, against 10.5 and 10.6 by chance. Of the 63 poles that instruction tuning flips, 50 also flip under at least one of the three random changes. Two random changes overlap with each other about as much as with the instruct change: seeds 0 and 1 share 29 flips, against 10.8 by chance. So a flip on Qwen2.5-0.5B says less about the tuning than about the pole.

The poles that flip are those near a gate. We ordered the 361 poles by their *margin*, the distance of their base readings above the nearer gate, and cut them into quarters of 90 or 91. The instruct change flips 31 poles in the quarter nearest a gate, 15 in the next, 11 in the third and 6 in the quarter furthest from a gate (Figure 2). The three random changes of all weights give 24 to 29, 15 to 19, 11 to 14 and 4 or 5. So both kinds of change flip mostly the poles whose explanations barely passed on the base model, and they rarely flip one with room to spare. A pole near a gate can cross it under any change that moves its activations a little, because each explanation rests on a few tens of probes.

![Instruction tuning and random changes flip many of the same poles of Qwen2.5-0.5B, and most of them sit near a gate.](media/flips.svg)

**Figure 2.** Each cell is one of the 361 VERIFIED poles of Qwen2.5-0.5B that reproduce on the base model, in order of how far its base readings sit above the nearer gate, nearest first. A filled cell is a pole that flips under the change that the frame names: the instruct change, then each random change of all weights. Each frame is one measured condition.

Both kinds of change fail the same gates. On Qwen2.5-0.5B, 38 of the instruct change's 63 flips fail sensitivity alone and 25 fail specificity alone. The first random change gives 37, 27, and 1 that fails both. So neither change makes a pole fire on near-misses more than the other does.

On Qwen2.5-1.5B, the flips do not share a cause with the random changes. Instruction tuning flips 10 poles, 6 on sensitivity and 4 on specificity. The three random changes of all weights flip 4 poles in all, and only 2 of the 10 are among them. These flips also sit near a gate: 7 of the 10 lie in the quarter of poles nearest a gate, of 19 there. The random changes flip at most 1 pole in that quarter. So on Qwen2.5-1.5B, instruction tuning moves the fragile poles across a gate, whereas noise of its size does not move them far enough. With so few flips, the overlap of the two sets cannot be tested well.

## Discussion

The matched random change answers the question differently for the two models, and the activations say why. On Qwen2.5-0.5B, the instruct change is large beside the weights, and a random change of the same norm disturbs the poles at least as much as the tuning does. It then flips as many explanations, and largely the same ones. So the 82.6% keep share of Qwen2.5-0.5B measures how far its explanations sit from a gate under a change of that size. It is not evidence that instruction tuning targets those neurons. On Qwen2.5-1.5B, the instruct change is small beside the weights, and noise of its size leaves the activations almost unchanged. Instruction tuning then flips about seven more explanations than size alone accounts for. So on Qwen2.5-1.5B the tuning's direction matters, and its flips are a real if small effect of the tuning.

This agrees with Ilharco et al. (2023), whose random vector of a task vector's norm leaves GPT-2 Large's behaviour as it was, whereas the task vector itself changes it. Their scale is chosen on held-out data to protect a control score, so that change is small by selection. On CLIP ViT-L/14, the same kind of random vector is not inert: it lowers the mean target accuracy from 64.8 to 60.9, and ImageNet accuracy from 75.5 to 72.9. Meral et al. (2026) report another such case, in which random steering vectors of a learned vector's norm lower an image model's reward. Our Qwen2.5-1.5B pair is that case at the level of single explanations. Our Qwen2.5-0.5B pair shows the same failure of the control for explanations. When a change is large beside the weights, a random change of the same norm is not inert, and it can do as much damage to explanations as the structured change.

The result also bears on the low intrinsic dimension of fine-tuning (Aghajanyan et al., 2021, after Li et al., 2018). A fine-tune that needs few directions could still move every weight, and the instruct change of Qwen2.5 is high-rank and spread over almost every neuron ([`docs/analyses/base-instruct.md`](../../docs/analyses/base-instruct.md)). Per unit of norm, it moves the poles' activations more than noise does on Qwen2.5-1.5B, and less than noise does on Qwen2.5-0.5B. We do not know why the two pairs differ. One reading is that a change of a few percent of the weights reaches the poles mostly through its size, whatever its direction, whereas a change of about one percent reaches them only through the directions that the tuning chose.

For the keep tests in this repo, the result suggests a rule. A keep share needs a matched random change beside it, with more than one seed, before it is read as an effect of the change. The interference report has such a floor with one seed. Here the floors of three seeds agree to within 5 poles of 361, and to within 1 pole of 75, so a few seeds suffice.

A null matched on norm is one choice. A null matched on the activation change, or on the output change, would compare the instruct change with a random change of the same effect. On Qwen2.5-0.5B, the random change already has the larger effect and the same keep share. On Qwen2.5-1.5B, a larger random change would be needed, and #6842 holds that test.

## Limitations

- **Two models of one family.** Both pairs are Qwen2.5, tuned by one recipe. Another family, or a narrow fine-tune, could make a change with more structure at the level of neurons.
- **Dense Gaussian noise.** The null matches each matrix's norm, not its rank or its spectrum. A low-rank random change, as a LoRA adapter makes (Hu et al., 2022), is a different null, and #6814 holds it.
- **Few seeds.** The random changes have 3 or 5 seeds. Their keep shares agree to within a few poles, so more seeds would narrow the range a little, but would not move it far.
- **A rescaled line.** The keep test rescales the line by the change in the pole's maximum. So it asks whether the pattern at the top of the pole's range survives, not whether its absolute size does.
- **Plain text only.** The probes run as plain text, not inside the instruct model's chat template, which lowers the instruct keep share in the base-instruct analysis.
- **The registry's poles.** The poles are the registry's VERIFIED poles, not a random sample of the model's neurons.

## Conclusion

Instruction tuning does not break more VERIFIED explanations of Qwen2.5-0.5B than a random change of its size: 298 of 361 keep, against 296 to 301, and the two changes largely flip the same poles near a gate. It does break more on Qwen2.5-1.5B: 65 of 75 keep, against 72 to 73 under random changes of its size, which barely move the activations there. A keep share under a change therefore needs a matched random change beside it before it can be read as an effect of that change.

## References

- Aghajanyan, A., Gupta, S., & Zettlemoyer, L. (2021). *Intrinsic dimensionality explains the effectiveness of language model fine-tuning*. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers, pp. 7319–7328). [DOI 10.18653/v1/2021.acl-long.568](https://doi.org/10.18653/v1/2021.acl-long.568).
- Biderman, D., Portes, J., Gonzalez Ortiz, J. J., Paul, M., Greengard, P., Jennings, C., King, D., Havens, S., Chiley, V., Frankle, J., Blakeney, C., & Cunningham, J. P. (2024). *LoRA learns less and forgets less*. Transactions on Machine Learning Research. [arXiv:2405.09673](https://arxiv.org/abs/2405.09673).
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). *LoRA: Low-rank adaptation of large language models*. ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685).
- Ilharco, G., Ribeiro, M. T., Wortsman, M., Gururangan, S., Schmidt, L., Hajishirzi, H., & Farhadi, A. (2023). *Editing models with task arithmetic*. ICLR 2023. [arXiv:2212.04089](https://arxiv.org/abs/2212.04089).
- Li, C., Farkhoor, H., Liu, R., & Yosinski, J. (2018). *Measuring the intrinsic dimension of objective landscapes*. ICLR 2018. [arXiv:1804.08838](https://arxiv.org/abs/1804.08838).
- Meral, T. H. S., Oktay, K., Helbling, A., Kara, A., Chau, D. H., & Yanardag, P. (2026). *RL post-training in diffusion transformers is activation steering*. Preprint. [arXiv:2611.26627](https://arxiv.org/abs/2611.26627).
- Shuttleworth, R., Andreas, J., Torralba, A., & Sharma, P. (2025). *LoRA vs full fine-tuning: An illusion of equivalence*. NeurIPS 2025. [arXiv:2410.21228](https://arxiv.org/abs/2410.21228).
- Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., . . . Qiu, Z. (2024). *Qwen2.5 technical report*. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115).

## Appendices

### A. Reproduce

Each run reads every VERIFIED pole of the base model from `wiki/`, with errata applied. It loads the base model in float32 on the GPU, and the instruct model in float32 on the CPU, only to take $\Delta$. A random change's noise comes from one generator, seeded with the seed of the row, drawn over the weights in the order of their names. The runs took about 25 minutes for Qwen2.5-0.5B and 20 minutes for Qwen2.5-1.5B, on one GPU with 12 GB.

    uv run python scripts/random_delta_keep.py --base Qwen/Qwen2.5-0.5B --device cuda --out 0.5b.jsonl
    uv run python scripts/random_delta_keep.py --base Qwen/Qwen2.5-1.5B --device cuda --out 1.5b.jsonl
    cat 0.5b.jsonl 1.5b.jsonl > papers/random-change-keep/results.jsonl
    uv run python scripts/random_change_stats.py papers/random-change-keep/results.jsonl
    uv run python scripts/paper_figures.py

`results.jsonl` holds a `base` row for each pole, with its readings on the base model. It holds a `summary` row for each model, condition and seed, with the keep share, its interval, the poles that flip and the norm of the change. It also holds a `pole` row for each pole, condition and seed, with the readings on the changed model, whether the pole keeps, the scale ratio and the correlation. `scripts/random_change_stats.py` prints the paired tests, the overlaps and the flips by margin from these rows. The poles that a run selects depend on the registry on the day it runs.

The script reads each pole's probes in batches, and reads the first-token check for all poles in one pass over the vocabulary. On the instruct change and on one random change of Qwen2.5-0.5B, it gives the same readings for every pole as reading each probe alone.

### B. Paired tests for each seed

**Table 2.** Each random change against the instruct change on the same weights, over the poles that reproduce. *Only instruct* and *only random* are the poles that flip under one change and not the other, and $p$ is the exact McNemar test. *Both* is the poles that flip under both, *expected* the overlap that independent flips would give, and the last column the one-sided hypergeometric test of the overlap.

| Model | Weights | Seed | Instruct flips | Random flips | Only instruct | Only random | McNemar $p$ | Both | Expected | Overlap $p$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | all | 0 | 63 | 65 | 31 | 33 | 0.90 | 32 | 11.34 | $1.9 \times 10^{-11}$ |
| Qwen2.5-0.5B | all | 1 | 63 | 60 | 31 | 28 | 0.79 | 32 | 10.47 | $9.1 \times 10^{-13}$ |
| Qwen2.5-0.5B | all | 2 | 63 | 61 | 33 | 31 | 0.90 | 30 | 10.65 | $1.3 \times 10^{-10}$ |
| Qwen2.5-0.5B | MLP | 0 | 49 | 59 | 26 | 36 | 0.25 | 23 | 8.01 | $3.7 \times 10^{-8}$ |
| Qwen2.5-0.5B | MLP | 1 | 49 | 56 | 24 | 31 | 0.42 | 25 | 7.6 | $1.5 \times 10^{-10}$ |
| Qwen2.5-0.5B | MLP | 2 | 49 | 55 | 23 | 29 | 0.49 | 26 | 7.47 | $8.5 \times 10^{-12}$ |
| Qwen2.5-0.5B | MLP | 3 | 49 | 52 | 26 | 29 | 0.79 | 23 | 7.06 | $1.6 \times 10^{-9}$ |
| Qwen2.5-0.5B | MLP | 4 | 49 | 52 | 28 | 31 | 0.79 | 21 | 7.06 | $9.6 \times 10^{-8}$ |
| Qwen2.5-1.5B | all | 0 | 10 | 2 | 9 | 1 | 0.022 | 1 | 0.27 | 0.25 |
| Qwen2.5-1.5B | all | 1 | 10 | 2 | 9 | 1 | 0.022 | 1 | 0.27 | 0.25 |
| Qwen2.5-1.5B | all | 2 | 10 | 3 | 8 | 1 | 0.039 | 2 | 0.4 | 0.045 |
| Qwen2.5-1.5B | MLP | 0 | 9 | 4 | 6 | 1 | 0.13 | 3 | 0.48 | 0.0047 |
| Qwen2.5-1.5B | MLP | 1 | 9 | 0 | 9 | 0 | 0.0039 | 0 | 0.0 | 1.0 |
| Qwen2.5-1.5B | MLP | 2 | 9 | 1 | 9 | 1 | 0.022 | 0 | 0.12 | 1.0 |
| Qwen2.5-1.5B | MLP | 3 | 9 | 2 | 7 | 0 | 0.016 | 2 | 0.24 | 0.013 |
| Qwen2.5-1.5B | MLP | 4 | 9 | 2 | 9 | 2 | 0.065 | 0 | 0.24 | 1.0 |

### C. Who did the work

A model, Claude Opus 5.5 in Claude Code, wrote the plan, the code, the analysis and the text. The registry's agents wrote and reviewed the explanations and their probes.

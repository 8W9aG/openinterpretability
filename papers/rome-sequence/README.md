# How do neuron explanations fail as ROME edits pile up at one layer?

Model editors such as ROME insert one fact with one small change, but a useful editor must make many edits in turn, and sequential edits are known to end in collapse. We asked what happens inside the model on the way: which explanations of single neurons stop being true, and whether they fail gradually or all at once. We made 100 ROME edits of unrelated facts in turn at layer 2 of pythia-70m. After 0 to 100 edits, we ran the adversarial probes of all 180 VERIFIED neuron explanations again, and we compared the sequence with random rank-one edits of the same norms. All 92 explanations at or below the edited layer kept at every step, as the edit's form requires. Above it, the 88 explanations faded steadily with the edited layer's weight distance from its start. Their number fits $88 \, e^{-0.24 d}$ with $R^2$ = 0.98, so the risk of each explanation failing is about constant per unit of distance, with no threshold. The random edits erased them faster, at a rate of 0.34. So in this small model, the explanations above an edit fail in step with the accumulated weight change, and a sequence of ROME edits is gentler on them than random changes of its size.

**Status:** Preprint. Work left: #6987 (MEMIT and AlphaEdit, other layers, seeds and orders).

## Introduction

A model editor changes one fact that a language model has learned, with a change small enough to leave the rest of the model alone. ROME does this with one rank-one change to one MLP's output matrix (Meng et al., 2022). A single edit can be precise, but an editor that is useful in practice must make many edits, one after another. Gupta et al. (2024) found that under such sequences ROME and MEMIT first forget slowly and then collapse, losing almost all earlier edits and their downstream skill at once. They attribute both phases to the edited layer drifting away from its starting weights. Their evidence is that the drift and the forgetting coincide, which is a correlation, not a causal test (`docs/literature/editing-at-scale-forgetting.md`).

Benchmarks of editing, such as those of Gupta et al. (2024), measure behaviour: whether edited facts are recalled, whether neighbouring facts change, and how downstream tasks score. They do not say what happens to the units inside the model. A registry of verified explanations of single neurons allows a finer test. An explanation is a claim about what makes one neuron fire, checked against probes that a reviewer wrote to break it. When the model changes, each explanation can be tested again on its own probes. This gives a count of explained structure that survives, unit by unit, as edits accumulate.

Our earlier paper made 10 single ROME edits to pythia-70m, each on the base model, and found that one edit breaks no more explanations than a random edit of its size (`papers/rome-keep/`). The present paper asks what happens when the edits pile up. Its contributions are these:

- A sequence of 100 ROME edits at one layer of pythia-70m, with all 180 VERIFIED explanations tested again at 19 checkpoints.
- A matched null: a sequence of random rank-one edits, each with the norm of the corresponding ROME edit.
- A test of the shape of the decline: whether explanations fail gradually, at a jump in weight change, or not at all.
- A record of how the edits themselves change along the sequence, with the loss and the recall of the edited facts.

The question belongs to research question 4 of `docs/research-questions.md`, which asks when an explanation carries across a change of the model.

## Related work

**Sequential editing.** Gupta et al. (2024) edit GPT2-XL and GPT-J with ROME and MEMIT, one fact at a time from CounterFact. They report two phases: gradual forgetting of earlier edits and a gradual fall in downstream scores, then a sudden collapse at one edit. In their data the collapse comes after about 100 to 1,000 ROME edits, and the normalised weight distance of a collapsing edit is far larger than that of a normal one. Our test uses their design at a much smaller scale, and it adds the unit-level measure that their benchmarks lack.

**Rank-one editing.** ROME locates a fact in a middle MLP and writes it with a rank-one change $W \to W + \Lambda u^\top$ (Meng et al., 2022). Here $u = C^{-1} k^*$ is set by the subject's key $k^*$ and the second moment $C$ of the layer's hidden activations. MEMIT spreads the change over several layers (Meng et al., 2023). Our ROME paper found that the explanations which flip under a single edit tend to sit near a review gate, and that their input weights tend to face the edit's write (`papers/rome-keep/`).

**Random changes as a null.** A random change of the same norm is the repo's standard comparison for a keep test. On Qwen2.5-0.5B, a random change as large as instruction tuning flips about as many explanations as the tuning does (`papers/random-change-keep/`). Our traffic map found that the damage of a random change depends strongly on where it is placed (`papers/traffic-map/`). Here the place is fixed, and the comparison is between structured and random directions at that place.

## Method

**Model and explanations.** The model is pythia-70m (Biderman et al., 2023), with 6 layers and 2,048 MLP neurons in each. The registry holds 180 VERIFIED explanations of its MLP neurons, 4 more than when `papers/rome-keep/` was run. Each explains one *pole*, the positive or the negative side of one neuron's activation. An explanation is VERIFIED when it passed an adversarial review (`docs/PROTOCOL.md`). An agent that did not write the explanation wrote probes to break it. Among other gates, the pole then fired on at least 60% of the positive probes and stayed quiet on at least 80% of the negative probes. A pole *fires* when its activation reaches the activation line, a quarter of the largest activation that its scan recorded.

**The edits.** We made 100 counterfactual edits, each of the form "{city} is located in the country of" followed by a wrong country of one token, such as Baghdad to Turkey. Each of 100 cities is edited once. The cities' order and their new countries are drawn with seed 5362, and the new country is never the true one. Every edit is at layer 2, the layer of `papers/rome-keep/`, with that paper's fit. There, $k^*$ is the mean hidden activation at the subject's last token after 10 corpus prefixes. $v^*$ is fitted by Adam, with a KL term that keeps the subject's other predictions. Each edit is fitted on the model as the earlier edits left it. $C$ is fixed at the base model's second moment, which ROME computes once as a constant (Meng et al., 2022). Here it comes from the registry's scan corpus, not from Wikipedia.

**The null.** The $n$-th random edit is $\Lambda' u'^\top$, with $\Lambda'$ and $u'$ Gaussian and scaled to the norms of the $n$-th ROME edit's $\Lambda$ and $u$. The random edits are summed in the same way.

**Measures.** We test both sequences after 0, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50, 60, 80 and 100 edits. At each checkpoint we measure:

- **Keeps.** We run the keep test of `papers/random-change-keep/` on every pole. An explanation *keeps* when it passes its gates on the base model and on the edited model. On the edited model, the activation line is scaled by the ratio of the pole's largest activation after the change to its largest before, both on a 20,000-token sample of the scan corpus.
- **Weight distance.** The distance $d_n$ of layer 2's output matrix from its start:

$$
d_n = \frac{\lVert W_n - W_0 \rVert_F}{\lVert W_0 \rVert_F} .
$$

- **Edits.** Retention, the share of the edits so far whose new country is the model's top prediction, and whether the latest edit succeeded.
- **Loss.** The model's mean next-token loss on the 20,000-token sample.

**Statistics.** For each checkpoint we give the keep count with a 95% Wilson interval, and an exact McNemar test of ROME against random over the same poles. To ask whether the decline has a threshold, we fit one rate $h$ to each sequence,

$$
K(d) = N \, e^{-h d} ,
$$

by least squares on the log scale through the base point, where $K$ is the number of explanations above the layer that keep and $N$ = 88. A constant $h$ means that each explanation's risk of failing is the same for each unit of distance, at any point in the sequence. A threshold would show as a poor fit, with a step that no constant rate explains. The interval on $h$ comes from 2,000 bootstrap draws of the poles.

**Figures.** Figure 1 moves: it adds the checkpoints in order of edits, so that a reader sees the decline unfold.

## Do the explanations below the edit stay as they were?

The question is a check of the method. A ROME edit changes only the output matrix of one MLP, so no neuron at or below that layer can change its activation. All 92 explanations at layers 0 to 2 kept at every checkpoint of both sequences, from the first edit to the hundredth. The keep test therefore reads no change where none is possible, and every flip in the rest of the paper is above the edit.

## How do the explanations above the edit fail?

The question is whether the 88 explanations above layer 2 fail gradually, at a jump in weight change, or not at all. They fail gradually, in step with the weight distance (Figure 1 and Table 1). After one ROME edit 83 keep, after 10 edits 57, after 30 edits 31 and after 100 edits 4. A single rate fits the decline well: $h$ = 0.24 per unit of distance (95% interval 0.20 to 0.31), with $R^2$ = 0.98 on the log scale. So the net loss of explanations is about the same for each unit of weight change, early and late, and the decline has no threshold that these checkpoints resolve.

![As ROME edits pile up, the explanations above the layer fade steadily, with no threshold, and more slowly than under random edits of the same size.](media/decay.svg)

**Figure 1.** As ROME edits pile up, the explanations above the layer fade steadily, with no threshold, and more slowly than under random edits of the same size. The VERIFIED poles above layer 2 of pythia-70m that still pass their probes, against the weight distance of layer 2, at each checkpoint of 0 to 100 edits. Circles are the ROME sequence, and triangles the random sequence of the same norms. The curves are $88 \, e^{-h d}$, fitted to each sequence. The marks appear in order of edits.

**Table 1.** The explanations above layer 2 that keep, out of 88, with a 95% Wilson interval, at selected checkpoints, with the weight distance and the loss. The last column is the exact McNemar $p$ of ROME against random over the same poles.

| Edits | ROME keeps | Distance | Loss | Random keeps | Distance | Loss | McNemar $p$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 88 | 0 | 3.45 | 88 | 0 | 3.45 | 1 |
| 1 | 83 [0.87, 0.98] | 0.20 | 3.46 | 85 [0.90, 0.99] | 0.20 | 3.47 | 0.73 |
| 3 | 78 [0.80, 0.94] | 0.56 | 3.51 | 63 [0.61, 0.80] | 0.54 | 3.77 | 0.0026 |
| 5 | 77 [0.79, 0.93] | 0.66 | 3.53 | 63 [0.61, 0.80] | 0.64 | 3.82 | 0.016 |
| 10 | 57 [0.54, 0.74] | 2.04 | 4.23 | 41 [0.37, 0.57] | 1.93 | 5.53 | 0.011 |
| 20 | 47 [0.43, 0.63] | 3.66 | 5.66 | 21 [0.16, 0.34] | 3.57 | 8.05 | 0.0001 |
| 30 | 31 [0.26, 0.46] | 4.84 | 7.14 | 11 [0.07, 0.21] | 4.80 | 9.53 | 0.0005 |
| 50 | 16 [0.12, 0.28] | 6.95 | 9.33 | 7 [0.04, 0.16] | 6.85 | 11.31 | 0.064 |
| 100 | 4 [0.02, 0.11] | 11.98 | 12.78 | 2 [0.01, 0.08] | 11.79 | 14.40 | 0.63 |

The fit is not perfect, and 18 checkpoints with gaps of up to 20 edits can resolve only a coarse shape. In the middle of the sequence, from about 10 to 30 edits, ROME keeps a few more explanations than the fitted curve, which suggests a slightly slower decline there. But no interval of the sequence loses explanations much faster than the rest for its step in distance. Between checkpoints, the loss per unit of distance is highest in the first edits, at about 20 to 30 explanations per unit, and it falls as fewer explanations remain, as a constant rate predicts.

The failures are not one-way. 40 of the 88 explanations flip at some checkpoint and pass again at a later one, and 46 under the random sequence. The rate $h$ therefore describes the net decline, not a risk of failing for good. A flip is a pole crossing a gate, and our earlier keep tests found that the poles which flip under a change mostly sit near a gate on the base model (`papers/random-change-keep/`, `papers/rome-keep/`). As edits accumulate, such a pole can cross back and forth before it fails for good.

## Is a ROME sequence gentler than random edits of its size?

The question is whether the direction of the edits matters, or only their size. It matters. The random sequence of the same norms erases explanations faster, at a rate of 0.34 (0.29 to 0.41, $R^2$ = 0.94), against 0.24 for ROME. The two intervals overlap a little, so the paired test at each checkpoint carries the comparison. At every checkpoint from 3 to 30 edits, ROME keeps more explanations over the same poles, by 13 to 26, with exact McNemar $p$ from 0.00002 to 0.019 (Table 1). After 40 edits the gap is no longer significant, because few explanations remain under either sequence. The two sequences reach almost the same weight distance, 11.98 and 11.79 after 100 edits, so the difference is in the directions, not in the size.

The loss tells the same story (Figure 2). It rises from 3.45 to 12.78 under ROME and to 14.40 under the random edits. At 10 edits it is 4.23 under ROME and 5.53 under random edits. This agrees with our single-edit result, where ROME edits flipped fewer explanations in total than random edits of their norms, although not significantly (`papers/rome-keep/`). Here the gap holds while the edits pile up.

## How do the edits change along the sequence?

The question is what drives the decline in the number of edits. Later edits are larger (Figure 2). The first five edits have a mean norm of 6.2, and edits 6 to 100 a mean of 26.3, with a largest edit of 42.6. Layer 2's output matrix has a norm of about 22.6, so even the first edit, at 4.6, is about a fifth of the layer's norm. The weight distance therefore grows fast: past 1 between 6 and 8 edits, and to 12 after 100.

![Later edits are larger, and the loss rises with them.](media/edits.svg)

**Figure 2.** Later edits are larger, and the loss rises with them. Left: the Frobenius norm of each of the 100 ROME edits, in the order made. Right: the language-model loss on 20,000 corpus tokens after each checkpoint, for the ROME sequence (circles) and the random sequence of the same norms (triangles).

This accounts for the shape in the number of edits. One of the steepest falls in explanations comes between edits 5 and 10, from 77 to 57, which is where the edits grow from about 6 to 15 to 24 in norm. Counted in weight distance, however, that interval loses no more than the rest. So what looks like an acceleration over the number of edits is the growth of the edits themselves.

The edits also stop holding. Retention of the edited facts falls to 0.4 after 10 edits, 0.17 after 30 and 0.09 after 100. The latest edit is the top prediction at 12 of the 18 checkpoints after the base. It fails at 2, 4, 6, 40, 50 and 60 edits. That is the forgetting of earlier edits that Gupta et al. (2024) describe, here alongside the loss of the model's own explained structure.

## Discussion

In this small model, the explanations above a sequence of ROME edits fail in step with the edited layer's weight distance, with about the same risk for each unit of distance. That is the gradual phase of Gupta et al. (2024), but we see no separate catastrophic phase in the explanations. The model as a whole does degrade badly: by 100 edits its loss is almost four times the base model's. One reading is that pythia-70m has no gradual phase to speak of, because each edit is already about a fifth of the layer's norm. In Gupta et al.'s Table 2, a single disabling ROME edit made first lies about 400 times further from the original weights than a normal one, by our ratio of their numbers. Every edit here is large, so the model may be in the regime of their collapse from the first few edits.

The comparison with random edits shows that the direction of the change matters at a fixed size. ROME's edits are built to change one fact's output, with $u = C^{-1} k^*$. Our reading, not a result of Meng et al. (2022), is that this weights the hidden directions by how rarely they occur, and so keeps each edit away from the directions that the layer's typical inputs use (`docs/literature/locating-editing-factual-associations.md`). A random rank-one edit has no such constraint. That ROME is gentler on the explanations, and on the loss, is consistent with this design, although we did not test the mechanism. Hu et al. (2025) give a related account: in their analysis, a sequential edit disturbs other facts by the overlap of their keys, which is small but not zero in real models. That would give a steady loss with each edit, as we see, rather than a threshold. AlphaEdit projects each update onto the null space of preserved keys to cancel that overlap (Fang et al., 2025). Whether it slows the decline of explanations is a question for #6987.

The result also bears on the repo's plan for fixes (#5220). A fix is one change, or a few, so the single-edit result of `papers/rome-keep/` is the closer guide. But if fixes accumulate on one model, as the patch format allows (#5229), the explanations above them will fade with the weight change at the fixed sites. The interference report (#5230) should then be read against the accumulated change, not against each fix alone.

## Limitations

- **One model, one layer, one order.** Every result is on pythia-70m, at layer 2, with one order of 100 facts and one random sequence. #6987 holds other layers, orders and draws.
- **The rate's interval is over poles, not over sequences.** The bootstrap resamples poles. It does not show how much another order of facts would change the rate.
- **Large edits from the start.** Each edit is about a fifth of the layer's norm or more. A larger model, where single edits are small, may show the two phases that Gupta et al. report.
- **ROME only.** MEMIT, which is designed for batches of many edits (Meng et al., 2023), and AlphaEdit, which is designed for sequences (Fang et al., 2025), are not tested here. #6987 holds them.
- **Flips cross gates both ways.** 40 explanations flip and come back. The keep count at one checkpoint is therefore noisier than a monotone decline would suggest.
- **The edited facts are of one form.** All edits change a city's country. Other relations may give other edit norms.

## Conclusion

As ROME edits pile up at one layer of pythia-70m, the VERIFIED explanations above the layer fade steadily, about as $88 \, e^{-0.24 d}$ in the layer's weight distance, with no threshold. Random edits of the same norms erase them faster. The apparent speed-up over the first edits comes from the edits growing larger, not from a tipping point in the explanations.

## References

- Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S. S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). *Pythia: A suite for analyzing large language models across training and scaling*. ICML 2023. [arXiv:2304.01373](https://arxiv.org/abs/2304.01373).
- Fang, J., Jiang, H., Wang, K., Ma, Y., Jie, S., Wang, X., He, X., & Chua, T.-S. (2025). *AlphaEdit: Null-space constrained knowledge editing for language models*. ICLR 2025. [arXiv:2410.02355](https://arxiv.org/abs/2410.02355).
- Gupta, A., Rao, A., & Anumanchipalli, G. (2024). *Model editing at scale leads to gradual and catastrophic forgetting*. Findings of ACL 2024. [arXiv:2401.07453](https://arxiv.org/abs/2401.07453).
- Hu, C., Cao, P., Chen, Y., Liu, K., & Zhao, J. (2025). *Knowledge in superposition: Unveiling the failures of lifelong knowledge editing for large language models*. AAAI 2025. [arXiv:2408.07413](https://arxiv.org/abs/2408.07413).
- Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). *Locating and editing factual associations in GPT*. NeurIPS 2022. [arXiv:2202.05262](https://arxiv.org/abs/2202.05262).
- Meng, K., Sharma, A. S., Andonian, A., Belinkov, Y., & Bau, D. (2023). *Mass-editing memory in a transformer*. ICLR 2023. [arXiv:2210.07229](https://arxiv.org/abs/2210.07229).

## Appendices

### A. Reproduce

The run reads every VERIFIED pole of pythia-70m from `wiki/`, with errata applied, and loads the model in float32 on the GPU. The facts, their order and the random sequence are seeded with 5362, and the $n$-th random edit with 5362 + $n$. The run took about 8 minutes on one GPU with 12 GB, and gives the same rows when it runs again.

    uv run python scripts/rome_sequence.py --device cuda --out papers/rome-sequence/results.jsonl
    uv run python scripts/rome_sequence_stats.py papers/rome-sequence/results.jsonl
    uv run python scripts/paper_figures.py

`results.jsonl` holds a `base` row for each pole, with its readings on the base model, and a `fact` row for each edit, with its city, its new country and the norm of its ROME edit. It holds a `checkpoint` row for each sequence and checkpoint, with the keeps above and below the layer, the weight distance, the loss and the retention. It holds a `pole` row for each pole, sequence and checkpoint, with its readings on the edited model.

### B. Keeps by layer

**Table 2.** The explanations that keep at each layer above the edit, under ROME and under random edits, at selected checkpoints.

| Edits | Layer 3 (28) | Layer 4 (36) | Layer 5 (24) |
| --- | --- | --- | --- |
| 5, ROME / random | 26 / 21 | 29 / 26 | 22 / 16 |
| 10, ROME / random | 21 / 16 | 22 / 17 | 14 / 8 |
| 20, ROME / random | 17 / 8 | 20 / 9 | 10 / 4 |
| 50, ROME / random | 7 / 2 | 6 / 2 | 3 / 3 |

### C. Who did the work

A model, Claude Opus 5.5 in Claude Code, wrote the plan, the code, the analysis and the text. The registry's agents wrote and reviewed the explanations and their probes.

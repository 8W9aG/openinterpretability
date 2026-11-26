# Which neuron explanations survive a ROME edit?

ROME inserts a new fact into a language model with one rank-one change to one MLP's output matrix. The change rewrites every neuron's write at that layer by a known amount, and leaves every input weight as it was. We asked which of pythia-70m's 176 VERIFIED neuron explanations still pass their adversarial probes after such an edit. We edited 10 unrelated facts at layer 2, one at a time, and set each edit beside a random rank-one change of the same norm. Nine of the ten edits work, raising the new object's probability above 0.70. As the edit's form requires, no reading of the 89 poles at layers 0 to 2 changed, across all 20 edits. Above layer 2, ROME flips 25 of 870 pole readings, against 40 for the random edits, and it flips fewer in 7 of the 10 facts. The poles that flip sit near a review gate: ROME's flips fall 16, 9, 0 and 0 across quarters of the poles by margin. A pole's alignment with the edit's write direction predicts its flip under ROME (AUROC 0.67, $p = 0.0009$ under a within-layer shuffle), but not under the random edits (0.55, $p = 0.14$). ROME's key does not single out the VERIFIED neurons of the edit layer. A ROME edit of one fact breaks no more explanations than a random change of its size, and fewer in total, although no test we ran makes that difference significant. The few it breaks are fragile, and their input weights tend to face its write.

**Status:** Preprint. Work left: #6949 (other layers, sequential edits and the poles' write claims).

## Introduction

Model editing changes a model's behaviour on one fact without retraining it. ROME does this with one rank-one change to the output matrix of one MLP (Meng et al., 2022). If edits are to fix models, it matters whether they keep what the model's units already do, because an explanation of a neuron is a claim that an edit could silently break. The registry holds VERIFIED explanations of single neuron poles, each with probes written to break it. So it can test, pole by pole, what an edit costs.

A ROME edit is a clean test for two reasons. It changes the write of every neuron at one layer by an amount that the method gives in closed form, and it changes no neuron's input weights. So every pole at or below the edit layer must fire exactly as before: the edit leaves its input weights alone, and nothing before the edited MLP's output reads that output. That gives the test a built-in check. In pythia-70m, attention and the MLP read the same input in parallel (Biderman et al., 2023), so the edit reaches the next layer only through the residual stream. Above the edit layer, a pole changes only through what the edit adds to the residual stream.

The paper's contributions are these:

- A ROME implementation for pythia-70m, and 10 edits of unrelated facts at one layer, each with its efficacy.
- A keep test of every VERIFIED pole after each edit, beside a random rank-one edit of the same norm.
- The built-in check: no pole at or below the edit layer changes, across 1,780 readings.
- Two predictors of which poles flip: the edit's loading on a pole's own neuron, and a pole's alignment with the edit's write.

The paper bears on research question 4, how far an explanation carries to a changed model ([research questions](../../docs/research-questions.md)).

## Related work

**ROME.** Meng et al. (2022) treat an MLP's output matrix $W$ as a linear memory, and insert a fact with a rank-one update. The update makes the matrix map a key $k^*$, the hidden activation at the subject's last token, to a value $v^*$ that makes the model predict the new object. On GPT-2 XL, ROME reaches 100.0% efficacy and 96.4% paraphrase success on CounterFact, and it lowers the neighbourhood score from 78.1 to 75.4. The paper measures whole-model behaviour on facts. It does not test single units. This repo's review of the paper proposed the present test ([`docs/literature/locating-editing-factual-associations.md`](../../docs/literature/locating-editing-factual-associations.md)).

**Weight changes in this repo.** Two papers here run the same keep test on other changes. A random change of the instruct change's size breaks as many explanations of Qwen2.5-0.5B as instruction tuning ([`papers/random-change-keep/`](../random-change-keep/README.md)). Rounding the weights to NF4 breaks about as many as instruction tuning ([`papers/nf4-keep/`](../nf4-keep/README.md)). Both find that the poles that flip sit near a review gate. An earlier analysis asked whether the size of the instruct change in a pole's own weights predicts its flip, and found that it does not, beyond depth ([`docs/analyses/base-instruct.md`](../../docs/analyses/base-instruct.md)). A ROME edit is the clean case of that question, since its change at each pole is known exactly.

## Method

**Model and units.** The model is pythia-70m (Biderman et al., 2023), with 6 layers and a parallel residual. Its config gives each MLP 2,048 neurons. A *pole* is one sign of an MLP neuron's activation, positive or negative. A pole is *VERIFIED* when its explanation, a sentence that says when the pole fires, passed the registry's adversarial review. In that review, a reviewer model wrote hard negatives, near-miss texts that the explanation says should not fire, and positive holdouts, new texts that it says should fire. The explanation passed when its specificity on the hard negatives reached 0.80, and its sensitivity on the holdouts reached 0.60. The registry holds 176 VERIFIED poles of pythia-70m: 32, 36 and 21 at layers 0 to 2, and 28, 35 and 24 at layers 3 to 5. All 176 pass both gates again on the base model.

**The edit.** We edit layer 2's output matrix $W$, of shape $512 \times 2{,}048$, with bias $b$. Let $C$ be the uncentred second moment of the layer's hidden activations over about 100,000 tokens of the registry's scan corpus, with a ridge of $10^{-4}$ of its mean diagonal for a stable solve. The key $k^*$ is the mean hidden activation at the subject's last token, after 10 corpus prefixes of 5 or 10 tokens. The paper uses 20 random prefixes, and ours are fixed window starts. The value $v^*$ is the MLP's mean output there plus a vector $\delta$. Adam fits $\delta$ over 40 steps at a rate of 0.5, so that the model predicts the new object after each prefix and the relation. A KL term, weighted 0.0625, keeps the model's prediction after "{subject} is a", as in the paper. $\delta$ is clamped to 4 times the output's norm. The rate, the weight and the clamp are our settings. The edit is

$$
\hat{W} = W + \Lambda u^\top, \qquad u = C^{-1} k^*, \qquad \Lambda = \frac{v^* - W k^* - b}{u^\top k^*}.
$$

Column $j$ of $\hat{W}$, neuron $j$'s write, is $w_j + u_j \Lambda$. No input weight changes.

**Facts.** We chose 10 unrelated counterfactual facts with one-token objects, such as "The Eiffel Tower is located in the city of" with the new object " Rome". The script lists them all. Each edit is applied alone, and the weights are restored after it. Efficacy here is the new object's probability after "{subject}{relation}", with no prefix. It differs from the paper's CounterFact efficacy, which asks whether the new object beats the true one.

**The random edit.** For each fact, a random rank-one change $\Lambda' u'^\top$ has Gaussian $\Lambda'$ and $u'$ with the norms of $\Lambda$ and $u$. So its Frobenius norm equals the ROME edit's, at the same layer.

**The keep test.** This is the test of the paper on random changes. Each pole's probes run on the base model and on the edited model. On the edited model, the line is the record's line times the ratio of the pole's maximum after the edit to its maximum before, over 20,000 tokens. An explanation *keeps* when it passes both gates on both models, and a pole *flips* when it reproduces and does not keep.

**Predictors.** For the VERIFIED neurons of layer 2, $|u_j|$ is the edit's loading on the neuron's write, and we give its percentile among the layer's 2,048 neurons. For a pole above layer 2, the *alignment* is the absolute cosine of the pole's input weights with $\gamma \odot \Lambda$, where $\gamma$ is the gain of the layer norm that feeds its MLP. A large alignment means that what the edit adds to the residual stream moves the pole's input. We test alignment with an AUROC for a flip. The null shuffles the outcomes 10,000 times within each fact and layer, so that depth and the edit's size get no credit. A pole's *margin* is how far its base readings sit above the nearer gate.

**Motion.** Figure 2 steps through the 10 facts, one frame each, and marks the poles that flip under each fact's ROME edit and its random edit. The frames show no state between two facts.

## Do the edits work, and do the poles below them stay as they were?

Nine of the ten edits insert their fact. The new object's probability rises from at most 0.24 to between 0.71 and 0.9998 (Table 1 in Appendix B). The edit of "Toyota is a company based in the country of Canada" fails, rising only from 0.0013 to 0.0039. The fitted contexts read the subject as one token, " Toyota". The prompt starts with the subject, with no space, which the tokenizer splits into "T", "oy" and "ota". So the edit's key never meets the prompt. For the other subjects, the last token is the same either way. We keep it in every count, because it is still a rank-one change of ROME's form. The random edits leave the new object's probability near where it was, from 0.0003 to 0.13.

No pole at or below the edit layer changed. Across the 20 edits, the 89 poles at layers 0 to 2 give 1,780 readings. In every one, the scale ratio is exactly 1, the correlation with the base activations is 1, and the pass or fail is as on the base model. This is the check that the edit's form requires, since it changes no input weight, and it shows that the test reads the edited model as intended.

## Does a ROME edit break more explanations than a random edit of its size?

It breaks fewer in total, although the difference is not significant. Above layer 2, each edit reads 87 poles, so 10 edits give 870 readings of each kind. ROME flips 25 of them, 2.9%, and 12 distinct poles (Figure 1). The random edits flip 40, 4.6%, and 21 distinct poles. ROME flips fewer than its random edit in 7 of the 10 facts, more in 2 and as many in 1. That sign test is not significant ($p = 0.18$). The totals and the distinct poles point the same way, but the same 87 poles are read under every fact, so their readings are not independent, and we did not test them.

![A ROME edit at layer 2 leaves every pole below it intact, and breaks fewer above it than a random edit of its size.](media/keep.svg)

**Figure 1.** The share of pythia-70m's VERIFIED explanations at each layer that still pass their probes, over 10 ROME edits at layer 2 and 10 random rank-one edits of the same norm. Circles are ROME and triangles the random edits. Every pole at layers 0 to 2 keeps under every edit.

ROME also moves the activations above it less. The median correlation of a pole's activations with its base activations is 0.9988 under ROME, against 0.9962 under the random edits. By layer, ROME keeps 97.1%, 96.0% and 98.8% at layers 3 to 5, and the random edits 96.8%, 93.1% and 97.1%. One reason may be the edit's form. A ROME edit moves another input's output in proportion to $(C^{-1}k^*)^\top k$, as Hu et al. (2025) write for edits of facts, and the second moment makes that small on most corpus tokens. A random edit of the same norm writes on every token whose hidden state has a component along $u'$. We did not test this account.

## Which poles flip?

The poles that flip sit near a gate. We cut the 87 poles above layer 2 into quarters of 21 or 22 by margin. ROME's 25 flips fall 16, 9, 0 and 0, from the quarter nearest a gate to the furthest. Figure 2 shows each flip in the grid's order, by layer and then by margin. The random edits' 40 flips fall 24, 13, 3 and 0. No edit of either kind flipped a pole in the quarter furthest from a gate.

![Each edit flips a few poles above layer 2, and none at or below it.](media/flips.svg)

**Figure 2.** Each cell is one of pythia-70m's 176 VERIFIED poles, by layer, nearest a gate first within each layer. The first 89 cells, shaded, are layers 0 to 2. A circle marks a pole that flips under the frame's ROME edit, and a triangle one that flips under its random edit. Each frame is one measured fact.

Under ROME, a pole's alignment with the edit's write predicts its flip. The AUROC is 0.67, and a shuffle of the outcomes within each fact and layer reaches it in fewer than 1 of 1,000 draws ($p = 0.0009$). Under the random edits the AUROC is 0.55 ($p = 0.14$). So under ROME, the poles whose input weights align with the edit's write flip more often. A random edit's write has no direction of its own, so its flips depend less on alignment, although they sit near a gate just as ROME's do. We did not test margin and alignment together.

## Does the edit's key single out the VERIFIED neurons?

It does not. For each edit, we ranked the VERIFIED neurons of layer 2 by $|u_j|$ among the layer's 2,048 neurons. Their median percentile ranges from 0.31 to 0.69 across the 10 edits, with a mean of 0.52 (Table 1 in Appendix B). It is above one half in 6 of the 10. A set of random neurons would sit at 0.5, and the random edits' $u'$ gives a mean of 0.47. So the edit loads on the explained neurons about as much as on any neurons of the layer. The review page suggested that $C^{-1}$ might aim edits at sparse, selective neurons such as the registry explains. In these edits it does not. The issue's measure, $|u_j|$, describes how much a neuron's write changes. Since the edit changes no input weight, it cannot change that neuron's activation or break its explanation, as the check above shows.

## Discussion

A ROME edit of one fact is a small change for explanations. It leaves every pole at and below its layer exactly as it was. Above its layer it breaks fewer explanations in total than a random change of the same norm, and moves the activations less, although we did not test these differences. This agrees with ROME's design: the update is the smallest change, under the key's second moment, that stores the new value. A random direction of the same size has no such constraint.

The explanations that ROME breaks share two traits. They sat near a gate on the base model, as the poles that instruction tuning, NF4 rounding and random changes break do in the papers on those changes. And under ROME, their input weights tend to face the edit's write. The first trait says that a keep test mostly finds fragile explanations, whatever the change. The second is specific to ROME: an edit has a direction, and the poles downstream that read along it are the ones it moves. That gives a cheap screen for an editor. Once $\Lambda$ is fitted, and before the edit is applied, the alignment of each downstream pole with it can be read from the weights. The fragile, aligned poles are the ones to test.

The issue's measure, $|u_j|$, turned out to be the wrong one for activation claims, for a reason that follows from the edit's form. It measures how much a neuron's write changes. A pole's explanation here is a claim about when it fires, which depends on its inputs, not its write. $|u_j|$ would bear on claims about what a pole does downstream, such as its steering or lookahead claims, which this test does not read.

## Limitations

- **One model and one layer.** pythia-70m is small, and every edit is at layer 2. ROME's own paper edits much larger models at a middle layer.
- **Ten facts, one at a time.** Each edit is alone. ROME degrades under many sequential edits: after 1,000 edits its reliability falls to near 0 (Yang et al., 2025). This paper does not test that.
- **One failed edit.** The Toyota edit did not insert its fact. It is still a rank-one change of ROME's form, and keeping it changes the totals by at most a few flips.
- **Our own facts and settings.** The facts are ours, not CounterFact's. The prefixes, the rate, the clamp and the 40 steps are our settings, and C comes from 100,000 tokens of the scan corpus, not Wikipedia. Layer 2 was chosen so that half the poles serve as the check, not for the edits' success.
- **Activation claims only.** The keep test reads when a pole fires. An edit can change what a pole does downstream without changing that, which this test cannot see.
- **The registry's poles.** The poles are the registry's VERIFIED poles, not a random sample of the model's neurons.

## Conclusion

After a ROME edit at layer 2 of pythia-70m, every explanation at or below the layer keeps, and above it ROME breaks 25 of 870 readings, against 40 for a random edit of its size. The poles it breaks sat near a gate, and their input weights tend to face the edit's write. The edit's loading on a pole's own neuron cannot break that pole's explanation, and the edit loads on the explained neurons no more than on others.

## References

- Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S. S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). *Pythia: A suite for analyzing large language models across training and scaling*. ICML 2023. [arXiv:2304.01373](https://arxiv.org/abs/2304.01373).
- Hu, C., Cao, P., Chen, Y., Liu, K., & Zhao, J. (2025). *Knowledge in superposition: Unveiling the failures of lifelong knowledge editing for large language models*. AAAI 2025. [arXiv:2408.07413](https://arxiv.org/abs/2408.07413).
- Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). *Locating and editing factual associations in GPT*. NeurIPS 2022. [arXiv:2202.05262](https://arxiv.org/abs/2202.05262).
- Yang, W., Sun, F., Tan, J., Ma, X., Cao, Q., Yin, D., Shen, H., & Cheng, X. (2025). *The mirage of model editing: Revisiting evaluation in the wild*. ACL 2025. [arXiv:2502.11177](https://arxiv.org/abs/2502.11177).

## Appendices

### A. Reproduce

The run reads every VERIFIED pole of pythia-70m from `wiki/`, with errata applied, and loads the model in float32 on the GPU. The facts, the edit layer and the fitting settings are constants of the script. A random edit's draw is seeded with its fact's index. The run took about 4 minutes on one GPU with 12 GB, and gives the same rows when it runs again.

    uv run python scripts/rome_keep.py --device cuda --out papers/rome-keep/results.jsonl
    uv run python scripts/rome_keep_stats.py papers/rome-keep/results.jsonl
    uv run python scripts/paper_figures.py

`results.jsonl` holds a `base` row for each pole, with its readings on the base model. It holds a `summary` row for each fact and kind of edit, with the keep share, the poles that flip and the edit's norm. The row also gives the new object's probability before and after, and the percentiles of $|u_j|$ of the VERIFIED neurons of layer 2. It holds a `pole` row for each pole, fact and kind, with its readings on the edited model, whether it keeps, its scale ratio, its correlation and, above layer 2, its alignment.

### B. The edits

**Table 1.** Each fact's ROME edit and its random edit: the new object's probability before and after, the explanations that keep out of 176, and the edit's Frobenius norm. The last column is the median percentile of the VERIFIED neurons of layer 2 among the layer's 2,048 neurons, by $|u_j|$ of the ROME edit. 0.5 is what a random set of neurons gives.

| Fact | Subject and new object | ROME: before | ROME: after | ROME: keep | Random: after | Random: keep | Norm | Median percentile of $\|u_j\|$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | The Eiffel Tower: Rome | 0.0013 | 0.9998 | 174 | 0.0014 | 169 | 4.52 | 0.69 |
| 2 | Barack Obama: France | 0.0003 | 0.9998 | 172 | 0.0003 | 172 | 3.95 | 0.31 |
| 3 | The Amazon River: Europe | 0.2404 | 0.9997 | 174 | 0.1313 | 172 | 3.97 | 0.31 |
| 4 | Mount Everest: Alps | 0.0062 | 0.7079 | 174 | 0.0063 | 175 | 3.84 | 0.68 |
| 5 | Albert Einstein: music | 0.0025 | 0.9938 | 172 | 0.0032 | 170 | 4.75 | 0.68 |
| 6 | The Great Wall: Brazil | 0.0002 | 0.9680 | 172 | 0.0003 | 173 | 3.80 | 0.57 |
| 7 | William Shakespeare: German | 0.0012 | 0.9884 | 175 | 0.0011 | 173 | 3.69 | 0.37 |
| 8 | The Pacific Ocean: desert | 0.0016 | 0.9875 | 173 | 0.0013 | 172 | 4.11 | 0.56 |
| 9 | Toyota: Canada | 0.0013 | 0.0039 | 173 | 0.0015 | 171 | 4.05 | 0.61 |
| 10 | The Moon: Mars | 0.0003 | 0.9956 | 176 | 0.0003 | 173 | 3.99 | 0.45 |

### C. Who did the work

A model, Claude Opus 5.5 in Claude Code, wrote the plan, the code, the analysis and the text. The registry's agents wrote and reviewed the explanations and their probes.

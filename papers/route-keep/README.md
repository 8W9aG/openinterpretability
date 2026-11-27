# Does a verified answer route keep on the instruct model?

An answer route says which units of a language model produce one answer to one prompt. Prakash et al. (2024) found that a circuit found in a base model stays faithful after fine-tuning, but their transfer test asked only whether the circuit suffices. We asked whether a route that passed our full answer review on Qwen2.5-0.5B still passes it on Qwen2.5-0.5B-Instruct, with the same units, the same prompts and no new search. The route copies " ball" in "Mary has a red ball and John has a blue kite. Mary has a red", through 33 attention heads and neuron poles. On the instruct model, the answer is unchanged, the route passes or fails on the same test prompts, and it still suffices: with only the route live, it brings back 0.57 of the answer. But it is no longer necessary. Ablating it removes only 0.35 of what ablating every unit does, against 0.59 on the base model, and patching from the route's corrupted prompt falls from 0.81 to 0.37. So the route fails the review on the instruct model with all five seeds of the random sets. In this case, tuning keeps the old route working, but other paths to the same answer now carry much of it.

**Status:** Preprint. Work left: #7063 (more routes and more model pairs).

## Introduction

Fine-tuning changes a model's weights, and an explanation of the base model may no longer hold for the tuned one. For single neurons, our keep tests ask this one explanation at a time (`papers/random-change-keep/`, `papers/weight-average-keep/`). An answer route is a larger claim: a set of heads and neuron poles that together produce one answer, which a review tests by ablation, by keeping only the route, and by patching.

Prakash et al. (2024) asked the same question of a circuit for entity tracking. They found that the circuit of Llama-7B stays faithful in its fine-tuned versions: with every head outside it mean-ablated, the circuit keeps 0.88 to 0.97 of each tuned model's accuracy, and 1.00 in Llama-7B itself (`docs/literature/fine-tuning-enhances-entity-tracking.md`). That transfer test measures sufficiency. It does not ask whether the tuned model still needs the transferred circuit.

Our contributions are these:

- A keep test of one answer route of Qwen2.5-0.5B on its instruct model, with the route's own claim and the testing model's own prompts, and no new search.
- The full answer review on both models, with five seeds of the matched random sets, so that no result rests on one draw.
- A split of what keeps: the route's sufficiency and the model's behaviour keep, whereas its necessity and patching do not.

The question belongs to research question 4 of `docs/research-questions.md`, which asks when an explanation carries across a change of the model.

## Related work

**Circuits under fine-tuning.** Prakash et al. (2024) find that fine-tuning, on arithmetic or on chat (Vicuna-7B), enhances an existing entity-tracking mechanism: the same circuit carries the task in the base and the tuned models, and the tuned models' heads do the same jobs better. Their transfer test keeps only the circuit and measures the accuracy that is left, which is our sufficiency test. Their own circuit search also finds that the tuned models need additional components to reach their full performance, so tuning adds to the circuit. Their removal tests run on each model's own circuit. Our test measures the necessity of the transferred route directly.

**Answer routes in this repo.** Our earlier routes are on pythia-70m, which has no tuned partner in the repo. The route tested here is the first route on Qwen2.5-0.5B to pass the review (`wiki/answers/`, #6941). Its first attempt, on "The sun rises in the", failed the patching and patching-null gates: its 42-unit route moved under half the gap on two corrupted prompts, and random sets of its size moved up to 0.19 (#6941).

**Neuron explanations under tuning.** Instruction tuning breaks about one explanation of a single neuron in five on Qwen2.5-0.5B, and a random change of its size breaks as many (`papers/random-change-keep/`). On Qwen2.5-1.5B the explanations fail a little at a time along the line from the base to the instruct model (`papers/weight-average-keep/`).

## Method

**Model and route.** The models are Qwen2.5-0.5B and Qwen2.5-0.5B-Instruct (Yang et al., 2024), which share one architecture. The route is the passing record `wiki/answers/Qwen--Qwen2.5-0.5B/15385436b27b/20261127T002913Z.json`. Its prompt is "Mary has a red ball and John has a blue kite. Mary has a red", and its answer is " ball". The route has 33 units: 24 attention heads and 9 neuron poles. Following Cammarata (2026), a *pole* is one sign of a neuron's activation. The route's claim names a previous-token head that marks " ball" as the word after " red", heads that link the final " red" back to the first, and heads that copy " ball" to the output.

**The review.** The answer review measures the route with each unit replaced by its mean over 20,000 corpus tokens, a mean ablation. Its gates are these:

- **Necessity.** Ablating the route removes at least half as much of the answer's log-probability as ablating every unit.
- **Beats the random sets.** The route's cut is at least three times the mean cut of 8 random sets with the route's layers and kinds.
- **Sufficiency.** With only the route live and every other unit ablated, at least half of the answer comes back.
- **Patching.** Copying the route's activations from a corrupted prompt, which swaps " ball" for another noun, moves at least half of the gap between the two prompts. This holds on the route's own corrupted prompt and on each of the testing model's.
- **Patching null.** The same random sets move at most a tenth, at their 75th percentile.
- **Controls and test prompts.** The model answers as the claim predicts on all 8 controls and on at least 80% of the testing model's 22 test prompts.

**The keep test.** We ran the review unchanged on both models, with the record's claim and prompts. Each model has its own unit means, from the same corpus tokens. Each model ran with the record's seed of the random sets and with 4 more, so 5 seeds in all. We wrote no new record to our dataset of tested routes.

**Figures.** Figure 1 is animated: the base model's marks appear first and the instruct model's after them.

## Does the route keep?

The question is whether the route that passed on the base model still passes on the instruct model. Table 1 gives each gate, and Figure 1 plots three of them.

![The route stays sufficient on the instruct model, but it is no longer necessary](media/gates.svg)

**Figure 1.** The route stays sufficient on the instruct model, but it is no longer necessary. Each mark is one gate's measure for the 33-unit route, on each model, and the grey line is the threshold of 0.5. The base model's marks appear first.

**Table 1.** Each gate's measure on both models, with the record's seed of the random sets. The patching null is the largest 75th percentile over the corrupted prompts.

| Gate | Threshold | Base model | Instruct model |
| --- | --- | ---: | ---: |
| Necessity | $\geq 0.5$ | 0.59 | 0.35 |
| Beats the random sets | $\geq 3\times$ | passes | passes |
| Sufficiency | $\geq 0.5$ | 0.55 | 0.57 |
| Patching, the route's own prompt | $\geq 0.5$ | 0.81 | 0.37 |
| Patching null | $\leq 0.1$ | 0.02 | 0.01 |
| Controls | 8 of 8 | 8 | 8 |
| Test prompts | $\geq 80\%$ | 19 of 22 | 19 of 22 |
| Verdict | | passed | failed |

The route keeps on the base model, where all five seeds pass, as the original review did. On the instruct model it fails the necessity and patching gates with every seed, because those two measures do not depend on the random sets. The model itself has not changed its answer: it still gives " ball", and it answers 19 of the testing model's 22 test prompts as the claim predicts, the same 19 as the base model. Ablating every unit removes about as much on both models, 14.1 nats on the base model and 14.8 on the instruct model. So the answer is not weaker after tuning. What changes is how much of it the route carries.

## Where does the answer go?

Sufficiency and necessity move apart. Keeping only the route live brings back 0.57 of the answer on the instruct model, so the route can still produce it. But ablating only the route removes 5.2 of the 14.8 nats, against 8.4 of 14.1 on the base model. The rest of the network now makes up for the lost route. That could be self-repair, in which later layers compensate when a component is ablated (McGrath et al., 2023), or paths that tuning added. Prakash et al. (2024) report the second kind for their circuits. Figure 2 shows the same split under patching.

![On the instruct model, patching falls below the gate on 2 of the 4 corrupted prompts](media/patching.svg)

**Figure 2.** On the instruct model, patching falls below the gate on 2 of the 4 corrupted prompts. Each mark is the share of the gap between the clean and the corrupted prompt that the route's activations from the corrupted prompt move. Each prompt swaps " ball" for another noun.

Patching from the route's own corrupted prompt, which swaps in " book", moves 0.81 of the gap on the base model and 0.37 on the instruct model. On the testing model's three prompts the route still moves 0.66 and 0.71 for " hat" and " pen", but only 0.32 for " bag". The gap between the clean and the corrupted prompt also shrinks, from 5.1 to 3.4 nats on the route's own prompt. Both changes fit one reading. On the instruct model, more of the copying runs outside the route, so changing the route's activations moves less of the answer, and the answer depends less on the copied noun.

## Discussion

The result agrees with Prakash et al. (2024) on what they measured: the route is still sufficient after tuning. It also measures directly what their circuit search suggests: the tuned model no longer needs the route, because other units can produce the same answer. A test that keeps only the circuit would call this route intact, whereas a test that removes it would call it broken. Both are true, and a keep test of a route should report both.

This matters for any change to a model that a route guides. An edit of the route's units on the base model would act on the main path to the answer. On the instruct model, the same edit would leave other paths that can still produce the old answer. Work that moves between a base model and its tuned version should therefore check necessity on the model that it changes, not only sufficiency.

Other explanations remain. Self-repair could inflate the drop in necessity if the instruct model compensates more strongly when the route is ablated (McGrath et al., 2023). A test that freezes the later layers would separate it from added paths. The route has 33 units, chosen to pass the gates with small margins on the base model: necessity was 0.59 against a gate of 0.5. A route with a larger margin might keep. The unit means differ between the models, and a mean ablation on the instruct model removes its own average, which is the right comparison but not the only one. And one prompt cannot show how often routes keep.

## Limitations

- **One route.** One prompt, one answer and one route, so this is a case, not a rate (#7063).
- **One model pair.** Qwen2.5-0.5B and its instruct model only (#7063).
- **Small margins.** The route passed necessity on the base model by 0.09 and sufficiency by 0.05, so a smaller change could have failed it.
- **Plain text.** The prompt is plain text, not inside the instruct model's chat template.
- **Mean ablation.** Every gate ablates units to their corpus mean. Other ablations could give other shares.

## Conclusion

A route that passed the full answer review on Qwen2.5-0.5B does not pass it on Qwen2.5-0.5B-Instruct. The instruct model still gives the same answer, and the route still suffices to produce it. But the model no longer needs the route: ablating it removes 0.35 of the answer against 0.59 before, and patching through it falls from 0.81 to 0.37. Tuning kept the old route, and other paths now carry much of the answer. More routes and more model pairs would show how general this is (#7063).

## References

- Cammarata, N. (2026). *Neural networks are already clean*. Goodfire research essay. [already-clean.goodfire.com](https://already-clean.goodfire.com/).
- McGrath, T., Rahtz, M., Kramár, J., Mikulik, V., & Legg, S. (2023). *The Hydra effect: Emergent self-repair in language model computations*. Preprint. [arXiv:2307.15771](https://arxiv.org/abs/2307.15771).
- Prakash, N., Rott Shaham, T., Haklay, T., Belinkov, Y., & Bau, D. (2024). *Fine-tuning enhances existing mechanisms: A case study on entity tracking*. ICLR 2024. [arXiv:2402.14811](https://arxiv.org/abs/2402.14811).
- Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., . . . Qiu, Z. (2024). *Qwen2.5 technical report*. Preprint. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115).

## Appendices

### A. Reproduce

The run takes about 90 minutes on one 12 GB GPU, most of it in the route's readings on every prompt:

```bash
uv run python scripts/route_keep.py \
    --record wiki/answers/Qwen--Qwen2.5-0.5B/15385436b27b/20261127T002913Z.json \
    --device cuda --out papers/route-keep/results.jsonl
uv run python scripts/paper_figures.py
```

The seeds of the random sets are the record's own (2112625968) and 1 to 4. The unit means use the first 20,000 tokens of `corpus.jsonl`.

### B. Who did the work

Claude Opus 5.5 proposed the route, ran the keep test and wrote this paper. Claude Sonnet 5.5 wrote the reviewer's prompts. The maintainer set the question in #5164.

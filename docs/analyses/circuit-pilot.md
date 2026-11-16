# A circuit pilot: cluster identity, and one circuit study

Run on 2026-11-16 for issue #4283, step 1 of `docs/circuits-and-hubs.md`. The design makes a circuit a cluster of the registry's wiring graph, which keeps its id from build to build by member overlap. This pilot asks two questions before the circuit unit is built (#4284). Do circuit ids survive as the registry grows? And does a circuit found this way carry a task when it is measured as the design says? The rows are in `circuit-identity.jsonl` and `circuit-pilot.jsonl`.

## Do circuit ids survive?

`scripts/circuit_identity.py` replays the registry one day at a time, from its first record to 2026-11-16 (2,924 records). Each day it builds the tree from the records that existed then. It clusters each model's wiring graph with Leiden (networkx, seed 0), with edges weighted by their mean absolute weight and hub edges left out. Each cluster of at least 3 units is a circuit. It then matches the day's circuits to the day before by Jaccard overlap, with a maximum-weight matching. An id survives at an overlap of 0.5 or more.

| Model | Circuits on 2026-11-16 | Continue | Grow or shrink | New | Split | Merge | Gone |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-2 | 4 | 29 | 7 | 4 | 0 | 0 | 0 |
| Qwen2.5-0.5B | 1 | 1 | 0 | 1 | 0 | 0 | 0 |
| Qwen2.5-1.5B | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Pythia-70m | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

- **Ids were stable, but the test is small.** No circuit split, merged or was lost, and each one that changed kept its id. But the wiring graph is sparse. The registry has only 5 circuits of 3 or more units, with 3 to 7 members each. That is far too few to measure churn with confidence.
- **Clusters of the wiring graph are small.** An edge joins two units only when both are studied. So the clusters will likely stay small until the registry covers more of each layer.

## One circuit study

The largest circuit is GPT-2's **C1**: five neurons (L4/N2999, L5/N1208, L5/N2567, L5/N616, L6/N37) and two heads (L7/H10, L7/H2). Their neurons' explanations mostly describe completing a unit that the previous tokens began, such as a split word or a collocation. None of the five neurons is VERIFIED. So the task hypothesis is **completing a two-token word**.

- **Task.** 120 pairs. A clean prompt ends on the first piece of a two-token word, and its pair ends on another word's first piece. The metric is logit(the clean word's second piece) − logit(the other word's). The 20 words are drawn from `/usr/share/dict/words` and kept only if GPT-2 ranks their second piece in its top 10 (`circuit-pilot.jsonl` lists them). The clean metric is 9.05.
- **Hard negatives.** Subject–verb agreement and indirect-object name copying, 120 pairs each. The clean metrics are 4.44 and 6.10.
- **Measures.** Mean ablation, as in `scripts/sva_circuit.py`, relative to ablating every neuron. The two heads are not ablated, because the tools ablate neurons only. The null is 64 random sets of five neurons, with the circuit's layers.

| | Word completion | Agreement | Names |
| --- | --- | --- | --- |
| Faithfulness (circuit alone) | 0.0007 | 0.0001 | 0.0009 |
| Necessity (loss when ablated) | 0.0001 | 0.0014 | 0.0006 |

Against the null, on word completion, random five-neuron sets have a median necessity of 0.0000, with a 95th percentile of 0.0006. No single member's necessity exceeds 0.0003.

- **C1 does not carry word completion.** Its necessity, 0.0001, lies below the null's 95th percentile. Alone, it keeps less than 0.1% of the metric. It does no more on the two other tasks.
- **Five neurons are too few to test this way.** In Qwen2.5-0.5B, the subject–verb agreement circuit needs about 100 neurons, ranked by attribution, to keep the whole metric. Even its 10 best neurons keep only 0.07 of it (`sva-circuit.md`). A wiring cluster of a few studied units is not chosen for any task, and it is much smaller.

## What this means for the design

- **Absolute gates would reject every wiring circuit today.** A faithfulness of 0.6 or a necessity of 0.5 can't be met by 3 to 7 units. Gates should instead compare the circuit with matched random sets: the circuit's necessity must beat the null's 95th percentile. An absolute floor should only say that the effect is large enough to matter, for example a necessity of at least 0.02.
- **A task hypothesis should be able to grow its members.** A wiring cluster is a good place to start, because it says where in the network to look. But the hypothesis should name the members that carry the task, with attribution (as for subject–verb agreement) seeded by the cluster. Then the experiment records those members, and staleness follows as designed.
- **Ids are stable so far, on too few circuits to be sure.** The identity rule should go ahead, and the replay should run again when the registry has more circuits.

These proposals go to #4284. The pilot measured one circuit, on one task hypothesis, with neurons only.

## Grown by attribution

Issue #4284 (part B) added `openinterp/harness/circuit_measures.py`, which ablates heads as well as neurons and ranks every unit by attribution patching. `scripts/circuit_grown.py` uses it on the same task. It measures C1 whole, with its two heads, and then C1 grown to larger sizes by attribution. The null is 32 random sets with the same layers and kinds, so its 95th percentile is coarse.

| Members | Heads among them | Necessity | Null, 95th percentile | Faithfulness | Beats the null |
| --- | --- | --- | --- | --- | --- |
| 7 (C1 as clustered) | 2 | 0.0001 | 0.0033 | −0.00 | no |
| 25 | 7 | 0.075 | 0.019 | 0.056 | yes |
| 50 | 10 | 0.153 | 0.024 | 0.060 | yes |
| 100 | 16 | 0.198 | 0.033 | 0.072 | yes |
| 200 | 21 | 0.281 | 0.042 | 0.103 | yes |

- **C1 with its heads still carries nothing.** Ablating the heads too changes nothing.
- **Grown by attribution, the circuit carries the task.** From 25 members, its necessity is 4 to 7 times the null's 95th percentile. It adds heads as it grows.
- **Word completion is spread out.** At 200 members the circuit alone keeps only 10% of the metric. Subject–verb agreement, by contrast, needs only about 100 neurons to keep all of it.
- **The proposed gates work.** The circuit's necessity must beat the null's 95th percentile, with a floor of 0.02. The bare cluster fails, and each grown circuit passes.

## Reproduce

```bash
uv run python scripts/circuit_identity.py --out docs/analyses/circuit-identity.jsonl
uv run python scripts/circuit_study.py --device auto --out docs/analyses/circuit-pilot.jsonl
uv run python scripts/circuit_grown.py --out docs/analyses/circuit-grown.jsonl
```

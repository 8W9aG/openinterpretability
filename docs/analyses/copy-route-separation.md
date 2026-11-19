# Do the copy route's units tell a wrong copy from a right one?

Run on 2026-11-19 for issue #5074, with `scripts/copy_route_separation.py`. The rows are in `copy-route-separation.jsonl`. pythia-70m answers "A week has seven days and a year has twelve" with " days". It copies the first clause's unit, by a route that #4781 VERIFIED. On "A dog has four legs and a cat has four", the same copy is right. Before anyone repairs the wrong copies, it helps to know whether the model holds the information that tells the two apart. This page measures it in the route's units, and in the residual stream as a direction. The tests follow the control check of Goyal and Daumé ([`steering-specificity.md`](../literature/steering-specificity.md)) and the difference-in-means direction of Arditi et al. ([`refusal-single-direction.md`](../literature/refusal-single-direction.md)).

**Answer.** The route's own units barely do: none of its 32 units separates the groups beyond a null of units like it. A direction in the residual stream does, modestly. It is strongest after layer 2, before the copy heads, with a balanced accuracy of 0.73 against 0.62 for random directions.

## Method

- **Prompts.** One template: `A <subject> has <count> <unit> and a <subject> has <count>`. The facts are 29 objects with four units (legs, wheels, sides, strings) and 7 units of time, plus two shortlist prompts. Every ordered pair of facts gives a prompt, 1,262 in all with single-token units.
- **Copies.** A prompt is kept when pythia-70m's top token is the first clause's unit, so every kept prompt is a copy. 956 are kept. In 216 the copy is right, because both subjects share the unit. In 740 it is wrong. *Objects only* repeats each test on the 738 copies with no unit of time, 216 right and 522 wrong.
- **Units.** The 32 units of the VERIFIED route (`wiki/answers/EleutherAI--pythia-70m/970a91e11d8c/20261118T234629Z.json`), at the answer position. A neuron pole's value is its activation on its side. A head's value is its strength |W_O(z − z̄)|, with z̄ its mean on 20,000 corpus tokens. Separation is the AUROC between the groups. The null for a neuron is every other neuron of its layer and side that is active on at least 10% of the prompts. A head's null is the other 7 heads of its layer.
- **Directions.** After each layer, the residual stream at the answer position. The direction is the difference of the group means, scored by leave-one-out balanced accuracy. Its threshold is the midpoint of the other prompts' group means. The null is 200 seeded random directions, scored the same way.

## Results

**Units.** No route unit's AUROC is beyond its null's 95th percentile on all the copies. On objects only, one is: L0/N1944+, with an AUROC of 0.44. Among the neurons, the furthest from 0.5 is L5/N2036−, at 0.37 (77th percentile). Among the heads, three are the most separating of their layers:

| Head | AUROC (right copy above wrong) | Its rank in its layer |
| --- | --- | --- |
| L4/H0 | 0.69 | 1 of 8 |
| L1/H3 | 0.67 | 1 of 8 |
| L5/H7 | 0.33 | 1 of 8 |
| L3/H3 | 0.63 | |

**Table 1.** The route's heads with the most separation. With 8 heads in a layer, a 95th-percentile line can't be met, so the table gives the rank. L4/H0 and L3/H3 are copy heads of the route. They are a little stronger when the copy is right.

**Directions.**

| Layer | Balanced accuracy, all copies | Random 95th percentile | Objects only | Random 95th percentile |
| --- | --- | --- | --- | --- |
| 0 | 0.58 | 0.58 | 0.55 | 0.57 |
| 1 | 0.62 | 0.59 | 0.60 | 0.57 |
| 2 | 0.73 | 0.62 | 0.66 | 0.60 |
| 3 | 0.69 | 0.64 | 0.64 | 0.62 |
| 4 | 0.67 | 0.65 | 0.64 | 0.64 |
| 5 | 0.67 | 0.66 | 0.64 | 0.65 |

**Table 2.** Leave-one-out balanced accuracy of the difference-in-means direction, against random directions.

The direction beats random directions at every layer on all copies, and at layers 1 to 4 on objects only. Its margin is largest after layer 2. It shrinks at layers 4 and 5, after the copy heads of layers 3 and 4 have written the copied unit.

## What this means

- **The route doesn't know whether its copy is right.** Its units fire alike on right and wrong copies, which fits a route that copies by matching the clauses' form, not their meaning.
- **The residual stream holds some of the difference, early.** After layer 2 a direction separates the groups at 0.73 balanced accuracy. The separation is modest, and part of it may be the category of the second subject, not the rightness of the copy. The objects-only test lowers it to 0.66, though it still beats random directions.
- **For a repair, a direction is a better handle than a route unit.** A change aimed at the route's units would act on right and wrong copies alike, and the right copies are the hard negatives of #5015. A direction at layer 2 is a candidate to condition a change on. Whether to use it is the maintainer's question.

## Limits

- **One template, and a confound.** In the right-copy group, both subjects share a category. In most wrong copies they don't. A direction that finds "same category" would score well without tracking rightness. The objects-only test reduces the confound but doesn't remove it.
- **First-order separation only.** AUROC and a linear direction can't see a combination of units that separates the groups.
- **One model**, pythia-70m.

## Reproduce

```bash
uv run python scripts/copy_route_separation.py --out docs/analyses/copy-route-separation.jsonl
```

It runs in about 10 minutes on a CPU, and reads `corpus.jsonl` for the heads' mean outputs.

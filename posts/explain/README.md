# Why did the model say that? How an explanation is built

`open-interpretability explain` takes one prompt and asks why the model gave its answer. This post follows two explanations of EleutherAI/pythia-70m through to the end, and shows how each part of an explanation is built and tested. One of the answers is wrong, and the explanation says how the model gets it wrong.

Every number here comes from a record in the repository. Each record was written by `explain --record` or by an answer experiment. The records name the package version that wrote them, 0.1.0 or 0.2.0, and the post gives the record with each number.

## Two prompts

- **"The sun rises in the" → " sky".** The record is `wiki/explanations/EleutherAI--pythia-70m/6c78ac114096/20261119T013311Z.json`.
- **"A week has seven days and a year has twelve" → " days".** The right answer is " months". The record is `wiki/explanations/EleutherAI--pythia-70m/970a91e11d8c/20261118T235000Z.json`, and its answer experiments are in `wiki/answers/EleutherAI--pythia-70m/970a91e11d8c/`.

## Attribution is a first guess

`explain` first measures each unit's attribution. Each MLP neuron pole and each attention head gets a first-order estimate of how much it pushes the answer's logit, against the unit's mean on a corpus. Then it asks the registry which of those units have a VERIFIED explanation.

For " days", the units with verified explanations hold 6.4% of the total attribution (`explained_share` 0.064). For " sky" they hold 10.1%. Read alone, that looks like a model that is barely explained.

But attribution is first-order. Many units push the answer up and many push it down, and the pushes cancel. A unit's share of the total says little about what the answer needs. So `explain` doesn't use attribution as its target.

## Causal coverage is the target

Coverage asks a direct question. Keep the explained units as they are, set every other unit to its corpus mean, and run the model again. How much of the answer's log-probability comes back?

- **" days":** the explained units bring back 93.8% (`prompt_coverage` 0.938).
- **" sky":** they bring back 81.9% (0.819).

So a set of units that holds 6.4% of the attribution brings back 93.8% of the answer. Coverage is the number that `explain --target` aims at, and the one an explanation is judged by (`openinterpretability/explain_coverage.py`).

`explain` also prunes the explained set to a minimal set that keeps at least 90% of the coverage. For " days" that is 23 units, which keep 84.8%. For " sky" it is 21 units, which keep 75.7%.

## The random baseline

High coverage could mean that any units of the same kinds would do. So every record also measures random sets of units, with the same number of heads and of neuron poles in each layer as the explained set.

- **" days":** random sets bring back 10.1% (`random_coverage` 0.101), against 93.8%.
- **" sky":** random sets bring back 13.0%, against 81.9%.

Even so, both records set `coverage_caution`. It is true when coverage is within 0.2 of the random baseline, or when verified units hold less than a quarter of the attribution. Here the second applies, with 6.4% and 10.1%. Then coverage shows that the units are sufficient on this prompt, not how they make the answer. The page and the narrator both say so.

## Heads and neuron poles

The record also splits coverage by kind of unit:

| Prompt | Heads only | Poles only | Both |
| --- | --- | --- | --- |
| " days" | 0.422 | 0.034 | 0.938 |
| " sky" | 0.234 | 0.019 | 0.819 |

Neither kind alone brings back much. Together they bring back most of the answer. The heads and the poles work as one route, which is what the answer experiment then tests.

## Coverage changes as the registry grows

The " sky" explanation has seven records, each building on the one before (`previous`):

| Record | Package | Coverage |
| --- | --- | --- |
| `20261117T191922Z` | 0.1.0 | 0.316 |
| `20261117T200746Z` | 0.1.0 | 0.519 |
| `20261117T224241Z` | 0.1.0 | 0.751 |
| `20261118T053532Z` | 0.1.0 | 1.0 |
| `20261118T151956Z` | 0.2.0 | 0.745 |
| `20261118T155641Z` | 0.2.0 | 1.0 |
| `20261119T013311Z` | 0.2.0 | 0.819 |

The first four rise as unit experiments VERIFY more of the units on the route. The fall to 0.745 came when coverage was measured again against the matched random baseline (#4777, #4780), and more experiments brought it back to 1.0.

The last record measures it again with its inputs stored, so that CI can reproduce it (#4793). By then the explained set had grown to 86 units, with poles VERIFIED in the meantime. Coverage is not additive: keeping more units live can bring back less, because each kept unit also keeps its pushes the wrong way. The 100% record stays on the prompt's page.

## The wrong answer: " days" for " months"

The model completes "A week has seven days and a year has twelve" with " days". The answer experiment tests a claim about how it does that.

**The claim.** It starts from three source tokens, " seven", " days" and " twelve". The VERIFIED claim's route has 32 units:

- **Count-word neurons** in layer 0, such as L0/N1877+ and L0/N1277+. They fire on " seven" and on the final " twelve", and mark each as a bare count word that wants a counted noun.
- **Time-unit neurons**, such as L0/N1101+. They write the identity of " days".
- **Early heads** in layers 0 and 1. They line up the two parallel clauses of the form "a X has" and a count.
- **A copy step.** At " twelve", the induction-style head L3/H3, with L4/H0 and L4/H7, attends to " days", the noun after the earlier count word. It copies " days" to the last position.
- **Layer-5 neurons** push a counted noun up and layout tokens down. They don't choose between " days" and " months".

So the copied " days" wins over the fact that a year has twelve months. The model completes a pattern from the first clause rather than recalling a fact.

**The test.** A claim is VERIFIED only when every gate passes:

| Gate | Rule | First claim | Second claim |
| --- | --- | --- | --- |
| necessity | ablating the route cuts ≥ 50% of what ablating everything cuts | 0.667 | 0.683 |
| beats null | the cut is ≥ 3 × the mean cut of 8 matched random sets | 51 × | 123.9 × |
| sufficiency | with only the route live, ≥ 50% of the answer comes back | 0.567 | 0.624 |
| patching | the route's activations from a corrupted prompt move the answer ≥ 50% of the gap | 0.653 | 0.749 |
| patching null | the random sets, patched, move it ≤ 0.1 on the mean (rule 1.0) | 0.160 | −0.004 |
| controls | every proposer prompt comes out as claimed | 8 of 8 | 7 of 7 |
| probes | ≥ 80% of the reviewer's prompts come out as claimed | 17 of 20 | 20 of 24 |

The ratios in the necessity and beats-null rows come from the records' measured cuts. For example, the second claim's route cuts 6.227 against 9.122 for every unit, and 0.050 on the mean for random sets.

**Why the first claim failed.** The first attempt (`20261118T194721Z`) passed six gates and failed `patching_null`. Its corrupted prompt changed " days" to " hours": "A week has seven hours and a year has twelve". Random sets of units, patched from that prompt, moved the answer by 0.16 on the mean, more than the 0.1 that rule 1.0 allows. So the patching result didn't belong to the route. It was REJECTED.

**What the second claim changed.** The second attempt (`20261118T234629Z`) changed the count word instead: "A week has seven days and a year has the". That turns off the count-word neurons, so the copy doesn't fire, and the answer leaves " days". The random sets, patched from it, moved the answer by −0.004 on the mean. Its route added count and quantity neurons, and more layer-1 heads. It passed every gate and is VERIFIED.

**What the reviewer tested.** The reviewer writes prompts that try to break the claim, without seeing any measure. They include the order swapped ("A year has twelve months and a week has seven"), paraphrases, and other counted pairs. The second claim got 20 of 24 right. Among the four it got wrong is "A week has seven days and a year has twenty". Another is "There are seven days in a week and twelve", where the clauses aren't parallel.

## What this shows, and what it doesn't

It shows a mechanism for one wrong answer on one prompt: a copy route that the gates can't explain away by chance. It names the units, and it tests what happens when the count word or the noun changes.

It doesn't show that pythia-70m always copies in this way, or why it doesn't recall that a year has twelve months. The answer experiment is VERIFIED for this prompt only. And `coverage_caution` holds: the units with verified explanations hold 6.4% of the attribution, so most units on this prompt still have no explanation of their own.

## How unit experiments add to it

An explanation counts a unit as explained when the registry holds a VERIFIED experiment for it, or for a superposition or circuit that holds it. `explain` lists the experiments that would explain the rest, ranked by the coverage each would add. `explain --run N` runs them through the experiment loop, and `explain --record` writes a new record. That is how the " sky" chain above went from 0.316 to 1.0.

`open-interpretability explain --help` lists every option. A test checks that each option this post names still exists.

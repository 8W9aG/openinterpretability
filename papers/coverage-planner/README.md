# Planning the experiments that explain an answer

To explain one answer of a language model, we choose unit experiments in turn until the units they would explain bring back the whole answer when every other unit is mean-ablated. Our earlier planner chose greedily, but it scored only the 120 candidates with the largest first-order attribution. On eight prompts of pythia-70m it stopped between 0.30 and 0.64 of the answer, and reached 0.9 on none. We show that the window, not the interaction between units, caused the stall. A greedy step that scores all of the 18,448 to 19,631 candidates once, and then rescores only the best 256 at each pick, reaches the whole answer on every prompt in 35 to 73 picks. A greedy step that scores every candidate at every pick needs 32 to 58 picks, but takes 216 to 444 seconds per prompt against 9 to 14, 24 to 32 times as long. Keeping every unit and then removing what the answer does not need leaves 103 to 1,028 experiments. Planned together, so that one experiment counts for every prompt it helps, eight prompts reach 0.9 in 118 experiments, against 160 when each prompt is planned alone.

**Status:** Preprint. Work left: #7005 (the copy family's prompts, explained with this planner).

## Introduction

An explanation of one answer of a language model names the units that produce it. One way to check such an explanation is causal: mean-ablate every unit that the explanation does not name, and ask how much of the answer is left. Circuit studies call this share faithfulness (Marks et al., 2025). We call it the prompt's coverage when it is measured on one prompt and capped at 1. A full explanation brings back the whole answer, and a coverage of 1.0 is the target that our explanations of single answers aim at.

Each unit that an explanation names needs its own experiment, in which a model writes an explanation of the unit and a second model writes test prompts to break it. These experiments cost time and model calls, so the order in which they run matters. A planner chooses that order. It should choose the experiments that bring back the most of the answer, and stop when the target is reached.

Our earlier planner did not reach the target. When we planned the copy family of pythia-70m, a set of prompts on which the model copies the wrong unit word (#7005), the plan stopped below 0.8 on seven of eight prompts. The issue that started this work suggested two causes. Coverage is not additive, because units push the answer both ways, so a pair can help where neither unit helps alone. On one earlier pythia-70m prompt, heads alone kept 0.42 of the answer and poles alone 0.03, but together they kept 0.94 (`docs/literature/compressed-sensing-localization.md`). The planner also saw only a window of 120 candidates, chosen by first-order attribution. This paper separates the two causes.

Our contributions are these:

- A measure of each plan's coverage after each pick, on eight prompts, for the earlier planner, a new planner and two references.
- Evidence that the attribution window, not the interaction between units, made the earlier planner stall.
- A planner that reaches the target on every prompt, within a few picks of a full rescan, and 24 to 32 times faster.
- A joint planner for many prompts, which needs 26% fewer experiments than separate plans on eight prompts.

The question belongs to research question 0 of `docs/research-questions.md`, which asks how much of each model our explanations cover.

## Related work

**Attribution as a guide to causal effects.** Attribution patching estimates the effect of patching a unit by a first-order expansion, the change in the unit times the gradient of the metric (Syed et al., 2023). It needs two forward passes and one backward pass, so it can rank every unit of a model at once. Averaged over three tasks, Syed et al. (2023) find that it recovers known circuits better than ACDC, which prunes edges by activation patching one at a time, although ACDC with a KL metric does better on one task. Our earlier planner used the same kind of estimate to choose its window. Our results show where the estimate fails: it ranks a unit by its effect alone, whereas the planner needs the gain of a unit on top of the units already chosen.

**Self-repair.** McGrath et al. (2023) show that when one attention layer of a language model is ablated, a later attention layer can take over part of its function, an effect they call the Hydra effect. In our reading, such compensation is one reason that the effect of a set of units is not the sum of their single effects. Our measure ablates every unit outside the kept set at once, so it does not rely on additivity. Our planner tests whether the lack of additivity stops a greedy search.

**Finding small sufficient sets.** Bair et al. (2026) locate the components behind a capability with random masks and a sparse regression, at a cost of a few hundred forward passes whatever the number of units. Their top five heads agree poorly with those of a one-shot greedy search, with a recall of 0.33 to 0.53. That agrees with our finding that many sets of units can carry one answer. Their regression is first-order, however, and so it shares the weakness of attribution that we measure here.

**The model.** Pythia-70m is the smallest model of the original Pythia suite, trained on the Pile with public checkpoints (Biderman et al., 2023). It has six layers of eight attention heads and a width of 512 (Biderman et al., 2023). Its MLP layers are four times as wide, 2,048 neurons each, by the model's published configuration.

## Method

**Units and coverage.** A unit is a neuron pole or an attention head. A neuron pole is one sign of one MLP neuron's activation: the positive pole is the neuron when it fires above zero, and the negative pole when it fires below. For one prompt and its answer token, let $\ell_{\mathrm{clean}}$ be the answer's log-probability in the model as it is. Let $\ell_{\mathrm{abl}}$ be the log-probability when every neuron pole and every head is mean-ablated at every position after the first. For a kept set $S$ of units, let $\ell(S)$ be the log-probability when every unit outside $S$ is mean-ablated, and the units in $S$ keep their real values. The coverage of $S$ is

$$
c(S) = \min\left(1, \frac{\ell(S) - \ell_{\mathrm{abl}}}{\ell_{\mathrm{clean}} - \ell_{\mathrm{abl}}}\right).
$$

The raw ratio can exceed 1, because some units hold the answer down. The cap gives such a set full credit and no more. Every kept set also contains the units whose explanations have already passed our adversarial review, 146 to 173 of them for each prompt, so $c(\varnothing)$ is the coverage before any new experiment. The means come from the first 20,000 tokens of our scan corpus.

**Experiments and plans.** Each candidate experiment studies one unit, or a group of units, and would add its units to the kept set if its explanation passed. A plan is an ordered list of experiments. A greedy plan picks, at each step, the experiment $e$ that maximises $c(S \cup e)$, where $S$ holds the units of the experiments already picked. It stops when $c(S)$ reaches the target $\tau$, when no candidate raises the coverage by more than $10^{-6}$, or after 400 picks. A coverage within $10^{-3}$ of the target counts as reaching it, because the batched forward passes in float32 leave a complete set at 0.9999.

**The four plans.** We compare four plans on each prompt.

- **The old plan.** The greedy step over the first 120 candidates, in the order that the analysis gives them, which is mostly the order of first-order attribution. It picks at most 60 experiments. This was our planner before this work.
- **The lazy plan.** The greedy step, with every candidate scored once at the start. At each pick it rescores only the 256 open candidates that ranked best in the last full scoring. If none of them helps, it scores every candidate again, and stops if none helps then.
- **The full scan.** The greedy step that scores every open candidate at every pick. It is the reference that the lazy plan approximates.
- **Pruning.** All candidates kept, then experiments removed one at a time, the one whose removal costs least first, while the coverage stays at the target. It asks how small a set a search from the other end finds.

**The joint plan.** For prompts $p_1, \ldots, p_k$, each with its own coverage $c_i$, the joint plan picks greedily by

$$
V(S) = \sum_{i=1}^{k} \min\left(c_i(S), \tau\right),
$$

so that an experiment counts for every prompt it helps, and a prompt that has reached the target adds nothing more. It stops when every prompt reaches $\tau$. We compare it with the plans of each prompt alone, cut at the pick where each reaches $\tau$. Their union counts a shared experiment once, and their sum counts it for each prompt.

**Batched scoring.** One coverage value needs one forward pass. A full scoring of about 19,000 candidates would take minutes with one pass at a time. We run 256 kept sets in one batch instead, and each row of the batch has its own boolean masks of the kept poles and heads. In our tests on small models, the batched values agree with the one-at-a-time values to $10^{-4}$.

**Data.** We sampled eight prompts of the copy family with seed 7061: six wrong copies from its train split, and two right copies from its retain split. On a wrong copy such as "A square has four sides and a foot has five", pythia-70m continues with the first clause's unit word, " sides", although a foot has toes. A right copy is a prompt on which that word is also right. Each prompt has 18,448 to 19,631 candidate experiments, one for each pole or head that the analysis plans. No language model is called. The plans only measure what each experiment's units would bring back if its explanation passed.

The figures show measured values only. Figure 1 animates the order of the picks, and each frame is a measured prefix of the plans.

## Why did the old plan stall?

The old plan's window held the candidates with the largest attribution, and the question is whether better candidates lay outside it. Figure 1 shows one prompt, "A banjo has five strings and a horse has four", on which the old plan does worst. In its first ten picks it rises with the other two plans. Then it flattens, and it stops after 47 picks at 0.30, because no candidate left in its window adds coverage. The lazy plan and the full scan keep rising, and both reach 1.0.

![Coverage after each pick on one prompt: the old plan flattens near 0.3, whereas the lazy plan and the full scan rise to 1.0.](media/trajectory.svg)

**Figure 1.** Scoring every candidate once lets the greedy plan reach full coverage, whereas the old plan stalls. The coverage of one prompt after each pick, for the old plan (grey), the lazy plan (blue) and the full scan (orange). The grey horizontal line is the target of 1.0. Each segment appears when its pick is made, so the animation draws the three plans in step.

The pattern holds on every prompt (Table 1). The old plan stops between 0.30 and 0.64, and it reaches neither 1.0 nor 0.9 on any prompt. The lazy plan and the full scan reach 1.0 on all eight. Both use the same greedy step as the old plan, and they differ from it only in which candidates they score. So the window was the cause of the stall. Had interaction between units been the cause, a greedy step over every candidate would also stall, and it does not.

| Prompt | Coverage before | Old plan stops at | Lazy plan, picks to 1.0 | Full scan, picks to 1.0 |
| --- | --- | --- | --- | --- |
| square / foot | 0.13 | 0.48 | 47 | 43 |
| ant / bike | 0.24 | 0.63 | 35 | 32 |
| tricycle / bird | 0.20 | 0.56 | 53 | 49 |
| wagon / spider | 0.15 | 0.50 | 54 | 52 |
| banjo / horse | −0.01 | 0.30 | 73 | 58 |
| chicken / violin | 0.18 | 0.61 | 35 | 34 |
| bus / bike (right copy) | 0.16 | 0.55 | 52 | 45 |
| dog / duck (right copy) | 0.25 | 0.64 | 44 | 43 |

**Table 1.** Each prompt's coverage before any pick, the coverage at which the old plan stops, and the picks that the lazy plan and the full scan need to reach 1.0. The first two nouns of each prompt name it.

The attribution window fails because attribution ranks a unit by its own effect, whereas the plan needs the unit's gain on top of the units already kept. After the first ten picks, the units that add the most are those that complete what the kept set has started. Such a unit can have a small attribution of its own. McGrath et al. (2023) show such dependence for whole attention layers of a 7B model. That the same holds for single poles and heads of pythia-70m is our reading of these results, not their claim. The interaction does not stop the greedy search, however. It only moves the useful candidates out of a window that is ranked by single effects.

## How close is the lazy plan to the full scan?

The lazy plan scores every candidate once and then rescores only 256 of them at each pick, so it may miss a candidate whose gain has grown since the first scoring. Figure 2 and Table 1 show what that costs. On seven prompts the lazy plan needs 1 to 7 more picks than the full scan to reach 1.0, a mean of 49.1 against 44.5 over all eight. On the banjo prompt it needs 15 more. At a target of 0.9 the gap is smaller: 26 to 58 picks for the lazy plan against 25 to 48 for the full scan, and it is 10 picks only on the banjo prompt. At 0.8 the two plans need the same picks on six prompts.

![For each prompt, a dot plot of the experiments that the lazy plan, the full scan and pruning need for full coverage, on a log scale.](media/picks.svg)

**Figure 2.** The lazy plan needs a few more picks than the full scan, and far fewer experiments than pruning every unit. The experiments that each plan needs to reach a coverage of 1.0, for each prompt, on a logarithmic axis. The lazy plan (blue) and the full scan (orange) lie close together, and pruning (green) keeps 3 to 22 times as many experiments. The old plan reaches 1.0 on no prompt, so it has no mark.

The lazy plan is much faster. It takes 9 to 14 seconds per prompt on one GPU, against 216 to 444 seconds for the full scan, because the full scan scores about 19,000 candidates at each pick. In no run did the pool of 256 run out: the lazy plan scored the full set only once on each prompt. So its rescan rule was never needed on these prompts.

Pruning from the other end does much worse. With every candidate kept, the coverage is 0.99 to 1.03. Removing experiments one at a time while the coverage holds at 1.0 leaves 103 to 1,028 of them. That is 3 to 22 times what the lazy plan needs. Removal in order of the smallest single cost keeps many units that a greedy addition never needs, which suggests that much of the answer can be carried by several different sets of units.

## Does planning prompts together save experiments?

When the planner explains many prompts of one family, an experiment that helps several prompts should count once. Figure 3 compares the joint plan with plans for each prompt alone, at a target of 0.9, for the first 2, 4 and 8 prompts of the sample. The joint plan reaches the target on every prompt in each case.

![Bars of the experiments that 2, 4 and 8 prompts need, planned together, planned apart with each unit once, and planned apart and summed.](media/joint.svg)

**Figure 3.** Planned together, 8 prompts need 118 experiments, against 160 when each prompt is planned alone. The experiments needed to bring each of the first $k$ prompts to a coverage of 0.9. The blue bar plans the prompts together. The green bar plans each prompt alone and counts an experiment that two plans share once. The grey bar counts it once for each plan.

The saving grows with the number of prompts. Two prompts need 44 experiments together against 52 apart, four need 62 against 93, and eight need 118 against 160. Separate plans already share experiments: of the 160 distinct experiments that the eight plans need, 62 are in two or more plans. The joint plan saves more than that sharing, because it prefers an experiment that helps several prompts at once, whereas separate plans take each prompt's best experiment. From four to eight prompts, each new prompt adds about 14 experiments to the joint plan, against 22 for each of the first two.

The joint plan's cost grows with the prompts, because each pick rescores the pool on every prompt. It took 21 seconds for two prompts, 52 for four and 161 for eight.

## Discussion

The results separate the two causes that we started from. The interaction between units is real, since the gain of a unit depends on the units already kept, and the old plan's window failed for that reason. But the interaction did not stop a greedy search that could see every candidate. Neither a lookahead of two picks nor a search over groups of units was needed on these prompts, so we did not add them.

A first-order ranking is a good start and a poor guide later. In the first ten picks the three plans rise together, which suggests that the units with the largest single effects are also the best first picks. After that, the useful units are those that complete the kept set, and a ranking of single effects does not find them. A planner should therefore rank by the measured gain of the whole set, not by attribution, once the first units are chosen.

The lazy plan's shortcut works because the ranking changes slowly. A candidate that was in the best 256 at the first scoring tends to stay among the best candidates later. The banjo prompt is the exception, where the lazy plan needs 15 more picks than the full scan. It is also the prompt that starts lowest, at −0.01, and so it needs the most change from its first ranking. A wider pool, or a full rescoring every few picks, would close the gap at some cost in time.

Pruning shows that the answer has many sufficient sets. The full set of candidates brings back the answer, and so do sets of 35 to 73 experiments. A removal order by single cost finds sets of 103 to 1,028, which are far from the smallest. So "the units that produce this answer" is not a unique set, and an explanation should say how its units were chosen.

Joint planning matters for a family of prompts. If each new prompt of the copy family adds about 14 experiments, as from four to eight prompts here, the 252 train and retain prompts of the family would need about 3,500 experiments at a target of 0.9. That is a linear extrapolation from eight prompts. If sharing keeps growing with more prompts, it is an overestimate.

## Limitations

- **One model and one family.** All eight prompts are copy-family prompts on pythia-70m. A larger model has more candidates and more heads, and its plans may need more picks.
- **Eight prompts.** The sample is small, and we report ranges, not intervals. The joint plan is measured only for nested sets of the first 2, 4 and 8 prompts.
- **Coverage, not explanations.** The plans assume that each experiment's explanation passes. In practice some fail, and a failed experiment adds nothing. The plans count experiments, not the cost of each, which depends on the unit.
- **One pool size.** We tested a pool of 256 only. The rescan rule never ran, so these prompts do not test it.
- **Pruning without the tolerance.** Pruning holds the coverage at the target with no tolerance, whereas the greedy plans accept a coverage within $10^{-3}$. On prompts where the full set falls just short of 1.0, pruning could hold 1.0 only after removals raised the coverage, which the cap allows.
- **No link to the optimisation literature.** The greedy step and its pool resemble known methods for maximising set functions, whose guarantees need the function to grow with the set and to show diminishing returns. Coverage has neither property, since pairs of units interact. We have not reviewed that literature (#7155), and we make no claim that the method is new.
- **Sufficiency only.** Coverage asks whether the kept units suffice. It does not ask whether the answer needs them, which the completeness score of Marks et al. (2025) measures.

## Conclusion

The earlier planner stalled because it looked at too few candidates, not because units interact. A greedy plan that scores every candidate once and rescores the best 256 at each pick reaches the whole answer on all eight prompts, in 35 to 73 picks. That is within a few picks of a full rescan, and 24 to 32 times faster. Planned together, eight prompts need 118 experiments to reach 0.9, against 160 when each prompt is planned alone. This planner can now plan the copy family's prompts (#7005).

## References

- Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S. S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). *Pythia: A suite for analyzing large language models across training and scaling*. In *Proceedings of the 40th International Conference on Machine Learning* (Vol. 202, pp. 2397–2430). PMLR. [arXiv:2304.01373](https://arxiv.org/abs/2304.01373).
- Bair, A., Xu, Y. E., Sun, M., & Kolter, J. Z. (2026). *Compressed sensing for capability localization in large language models*. Preprint. [arXiv:2603.03335](https://arxiv.org/abs/2603.03335).
- Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., & Mueller, A. (2025). *Sparse feature circuits: Discovering and editing interpretable causal graphs in language models*. ICLR 2025. [arXiv:2403.19647](https://arxiv.org/abs/2403.19647).
- McGrath, T., Rahtz, M., Kramár, J., Mikulik, V., & Legg, S. (2023). *The Hydra effect: Emergent self-repair in language model computations*. Preprint. [arXiv:2307.15771](https://arxiv.org/abs/2307.15771).
- Syed, A., Rager, C., & Conmy, A. (2023). *Attribution patching outperforms automated circuit discovery*. NeurIPS 2023 ATTRIB Workshop. [arXiv:2310.10348](https://arxiv.org/abs/2310.10348).

## Appendices

### A. Reproduce

The run takes about 70 minutes on one 12 GB GPU, most of it in the full scan and in pruning:

```bash
uv run python scripts/coverage_planner.py --device cuda \
    --out papers/coverage-planner/results.jsonl
uv run python scripts/paper_figures.py
```

The sample's seed is 7061, and the prompts come from `docs/analyses/copy-family.jsonl`. The planner is `openinterpretability/explain_joint.py`, and the batched coverage is `openinterpretability/explain_batched.py`. The pool is 256 candidates, the step limit 400 picks, the smallest gain $10^{-6}$ and the tolerance $10^{-3}$.

### B. Every plan, at three targets

| Prompt | Lazy, 1.0 / 0.9 / 0.8 | Full scan, 1.0 / 0.9 / 0.8 | Pruned, 1.0 / 0.9 / 0.8 |
| --- | --- | --- | --- |
| square / foot | 47 / 36 / 29 | 43 / 35 / 29 | 1,028 / 850 / 748 |
| ant / bike | 35 / 26 / 20 | 32 / 25 / 20 | 103 / 86 / 70 |
| tricycle / bird | 53 / 42 / 34 | 49 / 40 / 33 | 174 / 150 / 116 |
| wagon / spider | 54 / 43 / 35 | 52 / 42 / 35 | 184 / 161 / 115 |
| banjo / horse | 73 / 58 / 46 | 58 / 48 / 39 | 730 / 652 / 591 |
| chicken / violin | 35 / 27 / 21 | 34 / 26 / 21 | 126 / 98 / 76 |
| bus / bike | 52 / 40 / 31 | 45 / 37 / 31 | 140 / 93 / 81 |
| dog / duck | 44 / 31 / 24 | 43 / 30 / 24 | 122 / 83 / 69 |

**Table 2.** The experiments that each plan needs to reach coverages of 1.0, 0.9 and 0.8.

### C. Who did the work

Claude Opus 5.5 wrote the planner, ran the measurements and wrote this paper. The maintainer set the question in #7061.

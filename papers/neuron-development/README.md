# When Pythia-70m's VERIFIED neurons form

Explanations of single neurons are usually written for a finished model, but it is not known whether they describe the model for most of its training. We followed 86 neurons of Pythia-70m back through 20 public training checkpoints. Each one's explanation had passed an adversarial review. The training loss reaches within 1% of its final value by step 16,000. At that step, the neurons' activations correlate with their final form at a median of only 0.62, and they keep changing until the end of training. Between adjacent checkpoints after step 16,000, the median correlation is 0.78 to 0.82, so the late change is real movement and not only an effect of comparing with the end. 38 of the 86 explanations hold without a break from step 16,000 or earlier, and 38 only from step 64,000 or later. A first version of this paper studied 9 of these neurons. Its main results hold on the 86. Its claim that the two measures of formation order the neurons in reverse does not hold: they are only weakly related. A neuron's explanation describes one checkpoint, and the loss curve does not show when it becomes true.

**Status:** Preprint. Work left: #4156 (more checkpoints between steps 64,000 and 143,000, OLMo and the local learning coefficient).

## Introduction

Interpretability research often explains a trained network one unit at a time (Bau et al., 2017). An explanation of this kind is written for the final weights. A network changes during training, though. If a unit's function forms early and then stays, one explanation describes most of the run. If it keeps changing, an explanation is true only of the checkpoint that it was written for.

Pythia (Biderman et al., 2023) makes this question testable, because its training runs are public at many steps. Earlier work on training dynamics has tracked circuits and global measures. Examples are the formation of induction heads (Olsson et al., 2022) and the stages of a whole model (Hoogland et al., 2025). Less is known about single neurons whose explanations have passed adversarial tests. This paper asks when such neurons form, and whether their explanations hold from then on. It also asks whether their formation lines up with the plateau of the training loss. It answers research question 7, *When do neurons form during training?* ([research questions](../../docs/research-questions.md)).

The first version of this paper, on 2026-11-16, studied the 9 neurons that the registry had VERIFIED then. By 2026-11-19 the registry held 86. This version reports all 86, with the first 9 beside them, so that a reader can see which claims held.

The paper's contributions are these:

- A measurement of 86 reviewed neuron explanations in Pythia-70m, in all six layers, at 20 checkpoints from initialisation to the end of training. It uses the review's own tests.
- Evidence that the neurons keep changing long after the loss settles. Correlations between adjacent checkpoints show that this is movement, not only an effect of the measure.
- Evidence that the shape of a neuron's activations and the success of its explanation are two weakly related views of its formation.

## Related work

Network Dissection measured how well individual units of vision networks align with semantic concepts. It also followed that alignment across training iterations (Bau et al., 2017). We ask the same kind of question of language-model neurons, with explanations in place of fixed concept labels.

The Pythia suite was built to study how language models develop across training and scale (Biderman et al., 2023). It publishes checkpoints of each model through training. PolyPythias adds nine more training runs, with other seeds, at five of its sizes (van der Wal et al., 2025). One of those sizes is the model that we study. We use one of the smallest models, so that every checkpoint can be read on one machine.

Olsson et al. (2022) found that induction heads form in a phase change early in training, and that this change is visible as a bump in the training loss. Their result links a mechanism's formation to a feature of the loss curve. Our neurons behave differently. Their formation lines up with no feature of the loss that we measured, and much of it happens after the loss has flattened.

Hoogland et al. (2025) divide the training of a small attention-only language model, of about 3 million parameters, into approximate developmental stages. They find the stage boundaries with the local learning coefficient, a measure of the degeneracy of the loss landscape (Lau et al., 2023), not with the loss. Their approach suggests a measure that may show the late changes that we find. We leave that test to further work (*Conclusion*).

## Method

**Terms.** A neuron in a transformer's MLP layer can carry a different meaning when it is strongly positive and when it is strongly negative. Following Cammarata (2026), we study one side at a time, and call it a *pole*. The registry that we draw on stores an explanation for each pole that it has audited. An explanation is *VERIFIED* when it survives an adversarial review. In that review, a second model writes probes meant to break the explanation. *Hard negatives* look relevant but should not activate the pole, and *holdouts* should activate it. A probe *activates* the pole when the pole's peak activation on it reaches 0.25 of the pole's maximum, which we call the *activation line*. The explanation passes when at least 80% of the hard negatives and 60% of the holdouts behave as expected. We call these two thresholds the review's *gates*. We write L5/N655+ for the positive pole of neuron 655 in layer 5.

**Model and checkpoints.** We read Pythia-70m (Biderman et al., 2023) at 20 checkpoints, in float32. Eleven are early: steps 0, 1, 2, 4, 8, 16, 32, 64, 128, 256 and 512. Nine are later: steps 1,000, 2,000, 4,000, 8,000, 16,000, 32,000, 64,000, 128,000 and 143,000, the last of which is the end of training.

**Poles.** We take each Pythia-70m pole whose newest record in the registry is VERIFIED, on 2026-11-19. There are 86, in all six layers: 16 in layer 0, 18 in layer 1, 9 in layer 2, 15 in layer 3, 15 in layer 4 and 13 in layer 5. A pole whose newest record is not VERIFIED would be left out, even with an older VERIFIED record, but there were none. The *first 9* are the poles of this paper's first version, chosen on 2026-11-16 as the newest VERIFIED record of each pole.

**One negative pole.** One of the 86, L3/N866−, is a negative pole. Its peak is 0.170 at every checkpoint, which is the depth of GELU's dip. So its peaks on the windows are almost all tied, and its correlation with the final form is near 0 until the last step, where it is 1 by construction. We count it among the explanations that hold, and leave it out of every correlation. The correlation results are therefore over 85 positive poles.

**Windows.** We use 300 windows of 64 tokens from the last 200 documents of the corpus that the registry scans, and the same windows at every step.

**Measures.** For each pole and step we measure four things.

- *The correlation with the final form*, written r, is the Pearson correlation between the pole's peak activations on the 300 windows at that step and at step 143,000.
- *The correlation with the checkpoint before* is the same correlation with the previous checkpoint in the list.
- *Whether the explanation holds*: the recorded probes are run again, and the explanation holds when both pass rates reach the gates. The activation line at each step is 0.25 of a scaled maximum: the recorded maximum, times the step's largest window activation divided by the last step's.
- *The loss* is the mean next-token loss on the 300 windows.

**Reproduction of the first run.** We ran the first 9 poles again with the same code before the full run. No pass or fail, pass rate or formation step changed. The largest numeric change was 2.5×10⁻⁴ in a step's maximum, which is float rounding.

## Does the loss settle before the poles?

The first question is whether the poles have taken their final form by the time the loss stops falling. On the 300 windows, the loss falls from 11.09 at step 0 to 3.62 at step 16,000. That is within 1% of its final 3.59, and the loss changes little after step 32,000 (Figure 1, right). At step 16,000, however, the median correlation of the 85 poles with their final activations is 0.62. The middle half of the poles lie between 0.48 and 0.73, and the range is 0.26 to 0.98. The median rises to 0.71 at step 32,000, 0.80 at step 64,000 and 0.94 at step 128,000 (Figure 1, left). The first 9 had a median of 0.48 at step 16,000.

![Training loss settles by step 16,000, but the VERIFIED poles keep changing](media/development.svg)

**Figure 1.** Left: the median correlation of the 85 positive poles with their final activations, by training step on a log scale. The band holds the middle half of the poles, and the dashed line is the median of the first 9. Right: the mean next-token loss on the same 300 windows. In each panel a moving dot steps through the 20 checkpoints, and a counter names the step. So the poles can be seen still moving after the loss has stopped. The vertical line marks step 16,000, by which the loss is within 1% of its final value.

A correlation with the final checkpoint must reach 1 at the end, so a pole that drifts steadily looks as if it forms late. Correlations between adjacent checkpoints test this (Figure 2, right). Between steps 8,000 and 16,000 the median is 0.79. It is 0.78 between steps 16,000 and 32,000, and again between 32,000 and 64,000. It is 0.82 between steps 64,000 and 128,000. Each doubling of steps after the loss settles thus moves the poles about as much as the doubling before it. The poles keep moving through the second half of training, so the late change is not only an effect of the measure.

## When does each explanation start to hold?

The second question is whether each pole's explanation holds from some step onwards. We record the first step after which the recorded probes pass the gates at every later checkpoint. For 38 of the 86 poles this step is 16,000 or earlier. For 10 it is step 32,000, and for 38 it is step 64,000 or later (Figure 2, left). The first 9 split in the same proportion, 4 and 5 (Table 1). 51 of the 86 explanations pass at step 16,000 itself, but 13 of those fail again at a later checkpoint.

![Half the explanations hold only after the loss settles, and the poles keep moving](media/holding.svg)

**Figure 2.** Left: the share of the 86 explanations that hold without a break from each step on. The dashed line is the first 9. Right: each positive pole's correlation with the checkpoint before, as the median and a band for the middle half of the poles. The vertical line marks step 16,000.

| Pole | Holds without a break from | Correlation 0.5 first reached | Correlation at step 16,000 |
| --- | --- | --- | --- |
| L5/N655+ | 2,000 | 32,000 | 0.38 |
| L5/N674+ | 8,000 | 32,000 | 0.37 |
| L5/N1256+ | 16,000 | 8,000 | 0.63 |
| L1/N535+ | 16,000 | 32,000 | 0.45 |
| L4/N566+ | 64,000 | 4,000 | 0.73 |
| L2/N1181+ | 128,000 | 64,000 | 0.48 |
| L4/N751+ | 128,000 | 64,000 | 0.29 |
| L4/N216+ | 143,000 | 4,000 | 0.74 |
| L4/N812+ | 143,000 | 1,000 | 0.73 |

**Table 1.** The first 9 poles, as the first version reported them. The re-run gives the same values. The first column gives the first step from which the pole's recorded probes pass the review's gates at every later checkpoint. The second gives the first step at which its correlation with its final activations reaches 0.5. The third gives that correlation at step 16,000, where the loss has settled.

Explanations therefore settle at very different times. At the step where the loss has settled, fewer than half of them describe the model.

## Does formation line up with the loss plateau?

The third question is whether the poles form when the loss flattens. If they did, most poles would reach a high correlation with their final form near step 16,000. Of the 85 positive poles, 16 reach a correlation of 0.8 by step 16,000, 25 by step 32,000 and 43 by step 64,000. The median pole reaches it at step 64,000, four times as many steps as the loss needs to settle. Formation, measured this way, does not line up with the loss plateau. The first 9 gave the same answer: only two of them reached 0.8 before step 128,000, both at step 64,000.

## Formation by layer

The 86 poles cover all six layers, where the first 9 covered four. Table 2 gives each layer's poles. Layer 0's poles take their final shape earliest, with a median correlation of 0.90 at step 16,000. Its median pole reaches 0.8 at step 8,000. Layer 2's poles are the latest by both measures: one of its 9 explanations holds from step 16,000 or earlier. Layer 1 has the most explanations that hold early, 9 of 18, and layer 3 the highest share, 8 of 14. Yet both layers' shapes settle late.

| Layer | Positive poles | Median correlation at step 16,000 | Median first step with correlation 0.8 | Explanations that hold from step 16,000 or earlier | Median step from which the explanation holds |
| --- | --- | --- | --- | --- | --- |
| 0 | 16 | 0.90 | 8,000 | 8 | 40,000 |
| 1 | 18 | 0.60 | 64,000 | 9 | 24,000 |
| 2 | 9 | 0.46 | 128,000 | 1 | 128,000 |
| 3 | 14 | 0.61 | 128,000 | 8 | 10,000 |
| 4 | 15 | 0.72 | 128,000 | 6 | 64,000 |
| 5 | 13 | 0.53 | 128,000 | 6 | 32,000 |

**Table 2.** Formation by layer, over the 85 positive poles. A median of two middle values is their mean. Layer 3's negative pole, left out here, holds from step 128,000.

## Discussion

The first version found that the two measures of formation order its 9 poles in reverse (Table 1). The poles whose activations take their final shape early, L4/N812+, L4/N216+ and L4/N566+, were the ones whose explanations settled last. The poles whose explanations held early, L5/N655+ and L5/N674+, took their final shape late. Over the 9, the Spearman correlation between the step from which an explanation holds and the first step with r ≥ 0.8 is −0.73.

That reversal does not hold for the 85 positive poles. There, the same correlation is 0.24, and with the first step with r ≥ 0.5 it is 0.25. The two measures are thus weakly related, not opposed, and the 9 were an unusual sample. The weaker claim still stands. Passing the probes and matching the final activations measure different things. A pole can do what its explanation says while much of the rest of its activation still changes. Equally, it can keep its overall shape while the detail that the probes test still moves. Table 2 shows the same at the level of layers. Layer 0's poles take their shape earliest, but layer 3's explanations hold earliest.

This result differs from the induction heads of Olsson et al. (2022), whose formation is visible as a bump in the loss. The poles studied here change without any matching feature in the loss. This suggests that the loss curve is a poor guide to when single neurons form. A measure of the model's local geometry may show these late changes where the loss does not. An example is the local learning coefficient (Lau et al., 2023), as Hoogland et al. (2025) use it.

For a registry of explanations, the result means that an explanation describes the checkpoint that it was written for. An explanation of the final Pythia-70m is not, in general, an explanation of the model at step 16,000, although the two models have almost the same loss.

## Limitations

- **One model and one run.** We read only the original training run. PolyPythias offers nine more runs of the same model with other seeds (van der Wal et al., 2025). Its neurons are not the same units, though, so a second run needs its own audited poles.
- **Not a random sample.** The 86 poles are the ones that the registry has VERIFIED, and many were drawn by follow-ups and targeted draws. Layers 2 and 5 have fewer poles than the others.
- **One negative pole, and it saturates.** The only negative pole sits at GELU's dip at every checkpoint, so it gives no correlation. The result says nothing about negative poles in general.
- **Probe noise.** Each explanation is tested with a few dozen probes. So a pass rate near a gate can change from one checkpoint to the next by chance alone (see the review of Yan, Oikarinen and Weng in [`faithful-stable-explanations.md`](../../docs/literature/faithful-stable-explanations.md)). A single failed checkpoint therefore resets the step from which an explanation holds. 13 explanations pass at step 16,000 and fail later, which may be noise.
- **The explanations were written for the last checkpoint.** An earlier form of a pole may have a different good explanation. So the failure of the recorded one does not mean that the pole had no clean function.
- **The scale is borrowed.** The activation line at each step scales the recorded maximum by the change in the window maximum. A pole whose top activations lie outside the 300 windows gets a less exact line.
- **Few late checkpoints.** Between steps 64,000 and 143,000 we read only one checkpoint, so the timing of late changes is coarse. The adjacent correlations double their gap at each step, so they compare unequal intervals.

## Conclusion

The 86 reviewed neurons of Pythia-70m keep changing for most of training, long after the loss has settled. Adjacent checkpoints show that they keep moving. Fewer than half of their explanations hold at the step where the loss stops improving. Neuron explanations should therefore be read as descriptions of one checkpoint. Three tests would sharpen this result:

- More checkpoints between steps 64,000 and 143,000, to time the late changes.
- The same measurement on OLMo (Groeneveld et al., 2024), whose training run is also public, once it has VERIFIED poles. Also on the PolyPythias runs of this model.
- The local learning coefficient at the same checkpoints, to test whether the late changes line up with a change in it, not in the loss.

## References

- Bau, D., Zhou, B., Khosla, A., Oliva, A., & Torralba, A. (2017). *Network dissection: Quantifying interpretability of deep visual representations*. CVPR 2017. [arXiv:1704.05796](https://arxiv.org/abs/1704.05796).
- Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S. S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). *Pythia: A suite for analyzing large language models across training and scaling*. In Proceedings of the 40th International Conference on Machine Learning (PMLR 202, pp. 2397–2430). [PMLR](https://proceedings.mlr.press/v202/biderman23a.html).
- Cammarata, N. (2026). *Neural networks are already clean*. Goodfire research essay. [already-clean.goodfire.com](https://already-clean.goodfire.com/).
- Groeneveld, D., Beltagy, I., Walsh, P., Bhagia, A., Kinney, R., Tafjord, O., Jha, A. H., Ivison, H., Magnusson, I., Wang, Y., Arora, S., Atkinson, D., Authur, R., Chandu, K. R., Cohan, A., Dumas, J., Elazar, Y., Gu, Y., Hessel, J., . . . Hajishirzi, H. (2024). *OLMo: Accelerating the science of language models*. ACL 2024. [ACL Anthology](https://aclanthology.org/2024.acl-long.841/).
- Hoogland, J., Wang, G., Farrugia-Roberts, M., Carroll, L., Wei, S., & Murfet, D. (2025). *Loss landscape degeneracy and stagewise development in transformers*. Transactions on Machine Learning Research. [OpenReview](https://openreview.net/forum?id=45qJyBG8Oj).
- Lau, E., Furman, Z., Wang, G., Murfet, D., & Wei, S. (2023). *The local learning coefficient: A singularity-aware complexity measure*. Preprint. [arXiv:2308.12108](https://arxiv.org/abs/2308.12108).
- Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Johnston, S., Jones, A., Kernion, J., Lovitt, L., . . . Olah, C. (2022). *In-context learning and induction heads*. Transformer Circuits Thread. [transformer-circuits.pub](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).
- van der Wal, O., Lesci, P., Müller-Eberstein, M., Saphra, N., Schoelkopf, H., Zuidema, W., & Biderman, S. (2025). *PolyPythias: Stability and outliers across fifty language model pre-training runs*. ICLR 2025. [arXiv:2503.09543](https://arxiv.org/abs/2503.09543).

## Appendices

### Reproduce

The script downloads the 20 checkpoints from Hugging Face. It ran on an M1 Max in 57 minutes. It measures each pole with the review's own code (`submission_unit` and `peak_strength`). It reads the newest record of each pole from `wiki/`, and keeps the poles where that record is VERIFIED. It cuts the windows from the last 200 documents of `corpus.jsonl`. With `--pole`, it measures only the poles named, as the check of the first 9 did.

```bash
uv run python scripts/neuron_development.py --out papers/neuron-development/results.jsonl
uv run python scripts/paper_figures.py
```

The poles that a run selects depend on the registry on the day it runs. The `selection` row of `results.jsonl` gives the date and the count.

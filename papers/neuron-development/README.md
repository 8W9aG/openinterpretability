# When Pythia-70m's VERIFIED neurons form

Explanations of single neurons are usually written for a finished model, but it is not known whether they describe the model for most of its training. We followed 9 neurons of Pythia-70m, whose explanations had passed an adversarial review, back through 20 public training checkpoints. The training loss reaches within 1% of its final value by step 16,000, yet at that step the neurons' activations correlate with their final form at a median of only 0.48, and they keep changing until the end of training. Four explanations hold without a break from step 16,000 or earlier, whereas the other five hold only from step 64,000 or later. A neuron's explanation is therefore a description of one checkpoint, and the loss curve does not show when it becomes true.

## Introduction

Interpretability research often explains a trained network one unit at a time (Bau et al., 2017). An explanation of this kind is written for the final weights. However, a network is not fixed during training, and whether a unit's function forms early and then stays, or keeps changing, decides how far an explanation of the final model reaches back into training. If units settle early, one explanation describes most of the run. If they keep changing, an explanation is true only of the checkpoint that it was written for.

Pythia (Biderman et al., 2023) makes this question testable, because its training runs are public at many steps. Earlier work on training dynamics has tracked circuits and global measures, such as the formation of induction heads (Olsson et al., 2022) or the stages of a whole model (Hoogland et al., 2025). Less is known about single neurons whose explanations have been checked against adversarial tests. This paper asks when such neurons form, whether their explanations hold from then on, and whether their formation lines up with the plateau of the training loss. It opens research question 7, *When do neurons form during training?* ([research questions](../../docs/research-questions.md)).

The paper's contributions are these:

- A measurement of 9 reviewed neuron explanations in Pythia-70m at 20 checkpoints, from initialisation to the end of training, with the review's own tests.
- Evidence that the neurons keep changing long after the loss settles, so that their formation does not line up with the loss plateau.
- Evidence that two views of formation, the shape of the activations and the success of the explanation, give different orders over the neurons.

## Related work

Network Dissection measured how well individual units of vision networks align with semantic concepts, and it also followed that alignment across training iterations (Bau et al., 2017). We ask the same kind of question of language-model neurons, with explanations in place of fixed concept labels.

The Pythia suite was built to study how language models develop across training and scale, and it publishes checkpoints of each model through training (Biderman et al., 2023). PolyPythias adds nine more training runs, with other seeds, at five of its sizes, including the one that we study (van der Wal et al., 2025). We use one of the smallest models, so that every checkpoint can be read on one machine.

Olsson et al. (2022) found that induction heads form in a phase change early in training, and that this change is visible as a bump in the training loss. Their result links a mechanism's formation to a feature of the loss curve. Our neurons behave differently: their formation lines up with no feature of the loss that we measured, and most of it happens after the loss has flattened.

Hoogland et al. (2025) divide the training of a small attention-only language model, of about 3 million parameters, into approximate developmental stages. They find the stage boundaries with the local learning coefficient, a measure of the degeneracy of the loss landscape (Lau et al., 2023), rather than with the loss. Their approach suggests a measure that may show the late changes that we find, and we leave that test to further work (*Conclusion*).

## Method

**Terms.** A neuron in a transformer's MLP layer can carry a different meaning when it is strongly positive and when it is strongly negative. Following Cammarata (2026), we study one side at a time, and call it a *pole*. All 9 poles here are positive. The registry that we draw on stores an explanation for each pole that it has audited. An explanation is *VERIFIED* when it survives an adversarial review. In that review, a second model writes probes meant to break the explanation: *hard negatives*, which look relevant but should not activate the pole, and *holdouts*, which should activate it. A probe *activates* the pole when the pole's peak activation on it reaches 0.25 of the pole's maximum, which we call the *activation line*. The explanation passes when at least 80% of the hard negatives and 60% of the holdouts behave as expected. We call these two thresholds the review's *gates*. We write L5/N655+ for the positive pole of neuron 655 in layer 5.

**Model and checkpoints.** We read Pythia-70m (Biderman et al., 2023) at 20 checkpoints, in float32. Eleven are early: steps 0, 1, 2, 4, 8, 16, 32, 64, 128, 256 and 512. Nine are later: steps 1,000, 2,000, 4,000, 8,000, 16,000, 32,000, 64,000, 128,000 and 143,000, the last of which is the end of training.

**Poles.** We take the newest VERIFIED explanation of each Pythia-70m pole in the registry on 2026-11-16. There are 9, in layers 1, 2, 4 and 5.

**Windows.** We use 300 windows of 64 tokens from the last 200 documents of the corpus that the registry scans, and the same windows at every step.

**Measures.** For each pole and step we measure three things.

- *The correlation with the final form*, written r, is the Pearson correlation between the pole's peak activations on the 300 windows at that step and at step 143,000.
- *Whether the explanation holds*: the recorded probes are rerun, and the explanation holds when both pass rates reach the gates. The activation line at each step is 0.25 of a scaled maximum: the recorded maximum, times the step's largest window activation divided by the last step's.
- *The loss* is the mean next-token loss on the 300 windows.

## Does the loss settle before the poles?

The first question is whether the poles have taken their final form by the time the loss stops falling. On the 300 windows, the loss falls from 11.09 at step 0 to 3.62 at step 16,000, which is within 1% of its final 3.59, and it changes little after step 32,000 (Figure 1, right). At step 16,000, however, the poles' activations correlate with their final activations at a median of only 0.48, with a range of 0.29 to 0.74. The median rises to 0.72 at step 64,000 and to 0.93 at step 128,000 (Figure 1, left).

![Training loss settles by step 16,000, but the VERIFIED poles keep changing](media/development.svg)

**Figure 1.** Left: the correlation of each of the 9 poles' activations with their final activations, by training step on a log scale. A filled dot marks a step at which the pole's recorded probes pass the review's gates, and an open dot a step at which they fail. Right: the mean next-token loss on the same 300 windows. In each panel a moving dot steps through the 20 checkpoints, and a counter names the step, so the poles can be seen still moving after the loss has stopped. The vertical line marks step 16,000, by which the loss is within 1% of its final value.

The loss therefore settles long before the poles do. A part of the late rise comes from the measure itself, because the correlation with the final checkpoint must reach 1 at the end (*Limitations*). Even so, the poles are far from their final form at the step where the loss stops improving.

## When does each explanation start to hold?

The second question is whether each pole's explanation holds from some step onwards. We record the first step after which the recorded probes pass the gates at every later checkpoint. For four poles this step is 16,000 or earlier: L5/N655+ from 2,000, L5/N674+ from 8,000, and L5/N1256+ and L1/N535+ from 16,000 (Table 1). For the other five it is 64,000 or later: L4/N566+ from 64,000, L2/N1181+ and L4/N751+ from 128,000, and L4/N216+ and L4/N812+ only at the last checkpoint.

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

**Table 1.** When each pole forms, by two measures. The first column gives the first step from which the pole's recorded probes pass the review's gates at every later checkpoint. The second gives the first step at which its correlation with its final activations reaches 0.5, and the third gives that correlation at step 16,000, where the loss has settled.

Explanations therefore settle at very different times, and only four of the nine describe the model at the step where the loss has settled.

## Does formation line up with the loss plateau?

The third question is whether the poles form when the loss flattens. If they did, most poles would reach a high correlation with their final form near step 16,000. Only two poles, however, reach a correlation of 0.8 before step 128,000, and both reach it at step 64,000, four times as many steps as the loss needs to settle. Formation, measured this way, does not line up with the loss plateau.

## Discussion

The two measures of formation order the poles differently (Table 1). The poles whose activations take their final shape early, L4/N812+, L4/N216+ and L4/N566+, are the ones whose explanations settle last, whereas the poles whose explanations hold early, L5/N655+ and L5/N674+, take their final shape late. Passing the probes and matching the final activations therefore measure different things. A pole can do what its explanation says while much of the rest of its activation still changes. Equally, it can keep its overall shape while the detail that the probes test is still moving.

This result differs from the induction heads of Olsson et al. (2022), whose formation is visible as a bump in the loss. The poles studied here change without any matching feature in the loss, which suggests that the loss curve is a poor guide to when single neurons form. A measure of the model's local geometry, such as the local learning coefficient (Lau et al., 2023) as Hoogland et al. (2025) use it, may show these late changes where the loss does not.

For a registry of explanations, the result means that an explanation describes the checkpoint that it was written for. An explanation of the final Pythia-70m is not, in general, an explanation of the model at step 16,000, even though the two models have almost the same loss.

## Limitations

- **One model and one run.** We read only the original training run. PolyPythias offers nine more runs of the same model with other seeds (van der Wal et al., 2025), but its neurons are not the same units, so a second run needs its own audited poles. The 9 poles are the ones that the registry has VERIFIED, not a random sample.
- **Probe noise.** Each explanation is tested with a few dozen probes, so a pass rate near a gate can change from one checkpoint to the next by chance alone (see the review of Yan, Oikarinen and Weng in [`faithful-stable-explanations.md`](../../docs/literature/faithful-stable-explanations.md)). A single failed checkpoint therefore resets the step from which an explanation holds, and Table 1's first column may be later than the true one.
- **The correlation is with the end.** A pole's correlation with its final form reaches 1 at the last step by construction, so a pole that drifts steadily looks as if it forms late. Correlations between adjacent checkpoints would separate drift from formation, but we did not measure them.
- **The explanations were written for the last checkpoint.** An earlier form of a pole may have a different good explanation, so the failure of the recorded one does not mean that the pole had no clean function.
- **The scale is borrowed.** The activation line at each step scales the recorded maximum by the change in the window maximum. A pole whose top activations lie outside the 300 windows gets a less exact line.
- **Few late checkpoints.** Between steps 64,000 and 143,000 we read only one checkpoint, so the timing of late changes is coarse.

## Conclusion

The 9 reviewed neurons of Pythia-70m keep changing for most of training, long after the loss has settled, and only four of their nine explanations hold at the step where the loss stops improving. Neuron explanations should therefore be read as descriptions of one checkpoint. Three tests would sharpen this result. Correlations between adjacent checkpoints would separate a pole's steady drift from its formation. The same measurement on OLMo, whose training run is also public (Groeneveld et al., 2024), would test a second model once it has VERIFIED poles, and on the PolyPythias runs it would test other seeds of this one. Finally, the local learning coefficient at the same checkpoints would test whether the late changes line up with a change in it rather than in the loss.

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

The script downloads the 20 checkpoints from Hugging Face. It ran on an M1 Max. It measures each pole with the review's own code (`submission_unit` and `peak_strength`), reads the newest VERIFIED record of each pole from `wiki/`, and cuts the windows from the last 200 documents of `corpus.jsonl`.

```bash
uv run python scripts/neuron_development.py --out papers/neuron-development/results.jsonl
uv run python scripts/paper_figures.py
```

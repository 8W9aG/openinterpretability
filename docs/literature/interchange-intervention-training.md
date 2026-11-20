# Inducing Causal Structure for Interpretable Neural Networks

**Citation.** Atticus Geiger and Zhengxuan Wu (equal contribution), Hanson Lu, Josh Rozner, Elisa Kreiss, Thomas Icard, Noah D. Goodman and Christopher Potts, all of Stanford University. "Inducing Causal Structure for Interpretable Neural Networks". [arXiv:2112.00826](https://arxiv.org/abs/2112.00826), version 2 of 20 July 2022. Version 1 is of 1 December 2021. The arXiv page gives no venue, so we report it as a preprint. The code is at [github.com/frankaging/Interchange-Intervention-Training](https://github.com/frankaging/Interchange-Intervention-Training). The text gives no funding statement and no disclosure.

**Mismatch.** The abstract says "In all our experiments, IIT achieves the best results." In Table 1, IIT combined with multi-task training does better than IIT alone. In Table 3(b), multi-task training alone does better on novel colours. The abstract's guarantee also needs an assumption on the loss that only Appendix A states.

**Status.** Planned. #5220 names it as background for phase 5(b), a compiled core with read and write adapters. The fix campaign is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 2 for this page's author, a model (Claude Opus 5.5). It read the source's own text in full, in three fetches: Sections 1 to 7, the references and Appendices A to C. We didn't see any figure as an image, so the MQNLI results appear here only as the text describes them. We didn't read version 1.

## Summary

**Claim.** Interchange intervention training (IIT) makes chosen parts of a network carry the variables of a stated causal model. Such a network matches the causal model under interventions, and it often generalises better to held-out cases.

**Method.**

- **The alignment.** A hand-chosen map sends each variable of the causal model to a set of neurons. The paper says the map "is not unique" and "substantially affects model performance and generalizability."
- **The interchange intervention.** Run a source input, copy the aligned neurons' values into a run on a base input, and read the output.
- **The loss.** The intervened network's output must match the causal model's output under the same intervention. The total is the IIT loss plus the task loss, each with weight 1.0 where the paper states weights.
- **What trains.** All of the network's parameters. The causal model is fixed, and the alignment adds no parameters.
- **The measure.** Interchange intervention accuracy (IIA): the share of sampled base and source pairs where the intervened network agrees with the intervened causal model.
- **Tasks.** MNIST-PVR with a ResNet-18, ReaSCAN with a CNN-LSTM, and MQNLI with BERT. Each has a held-out split.
- **Baselines.** Standard training, multi-task training with probes on the aligned neurons, data augmentation by the causal model, and IIT with multi-task training.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| MNIST-PVR, held-out test, behaviour / IIA (Table 1) | standard 0.00 / 20.60. IIT 93.93 / 94.85. IIT with multi-task 96.01 / 96.64. Augmentation 90.90 / 92.00 |
| ReaSCAN novel size, behaviour, mean (sd) of 3 seeds (Table 3b) | standard 41.67 (6.24). Multi-task 39.46 (7.68). IIT 65.65 (4.26). Both 74.84 (0.04) |
| ReaSCAN novel direction, same | standard 0.00. IIT 0.26 (0.14). Both 14.72 (3.54) |
| ReaSCAN novel length, same | standard 5.72 (3.44). IIT 10.20 (6.08). Both 25.82 (0.37) |
| ReaSCAN novel colour, same | multi-task 76.91 (5.02), above IIT's 74.12 (6.00). Both 80.37 (0.88) |
| MQNLI (Section 6) | IIT reaches "near perfect accuracy and IIT accuracy (≈100%)". Augmentation is perfect on behaviour, with "very low" IIA |

## Where the claims go beyond the evidence

- **The alignment is chosen by hand, with tries.** For the ResNet, the layer came from "initial experimentation," and layers near the head were "ineffective." For MQNLI, the paper reports the layers that work. No null alignment, such as random neurons, is trained and scored.
- **Statistics for one task only.** ReaSCAN has three seeds and deviations. MNIST-PVR and MQNLI give one number each. No test backs the word "significant."
- **The best results are mostly the combination.** IIT alone is not best on MNIST-PVR or on novel colour. Some of the text's comparisons are within one deviation.
- **Large gains from low baselines.** Novel direction goes from 0.00 to 14.72, and novel length from 5.72 to 25.82. Both are still far from solved.
- **Simple ReaSCAN commands only.** The authors keep "the simplest command structure," "For simplicity."

The authors state limits themselves. The alignment matters and isn't unique. How base and source inputs are paired changes results, which they leave to later work. And zero loss is enough for abstraction, but not needed.

## Use in this repo

- **The issue's two questions, as the source answers them.**
  - *Can it train only an interface around a fixed core:* the paper doesn't test this. IIT trains every parameter. The causal model is fixed, but it is an oracle used for labels, not a part of the network. Nothing in the method stops it from training only some parameters, since the loss reaches them through the intervened runs. That step is ours, not the paper's.
  - *How it generalises off the training distribution:* better than standard training on held-out splits in all three tasks. But it is best in combination with probe training, and it stays far from solved on novel direction and length.
- **For phase 5(b), the core is the causal model.** In #5220's compiled-addition plan, the core is written or compiled, so it is exact by construction. IIT's role would be the read adapter. It would train the model's own number units, or a small layer, to carry the core's input variables under interchange. The prediction in #5220 is that the core generalises and the interfaces fail. Our reading, not the paper's: the ReaSCAN results fit that. The held-out splits for size and colour come close, and those for direction and length stay low.
- **IIA is the registry's patching test with a model attached.** The protocol's causal checks patch a unit's value from one prompt into another. IIA does the same for a group of units, and scores the result against a stated algorithm. A registry circuit with a written algorithm could be scored by IIA directly, with no training.
- **The alignment null is missing, and the registry has one.** The registry compares explanations with matched random units. An IIT study here would need the same: train the same objective on random neurons and report its IIA.
- **Questions for the maintainer, recorded on this page only.**
  - In phase 5(b), should the read adapter be trained by IIT against the compiled core, with the base model frozen?
  - Should a written circuit in the registry report IIA against its algorithm, beside coverage?
- **Related pages.** [`causally-grounded-explanations.md`](causally-grounded-explanations.md) scores explanations by intervention. [`tracr.md`](tracr.md) and [`alta.md`](alta.md) compile programs into weights, the core that phase 5(b) would write.

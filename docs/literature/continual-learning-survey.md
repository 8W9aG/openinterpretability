# A Continual Learning Survey: Defying Forgetting in Classification Tasks

**Citation.** Matthias De Lange, Rahaf Aljundi and Tinne Tuytelaars are at KU Leuven, and Marc Masana at the Computer Vision Center, UAB. Sarah Parisot, Xu Jia, Aleš Leonardis and Gregory Slabaugh are at Huawei. "A continual learning survey: Defying forgetting in classification tasks". *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2021, [doi:10.1109/TPAMI.2021.3057446](http://dx.doi.org/10.1109/TPAMI.2021.3057446). [arXiv:1909.08383](https://arxiv.org/abs/1909.08383), version 3 of 16 April 2021 (version 1: 18 September 2019, version 2: 26 May 2020). Version 3 is the authors' copy of the journal article, with the IEEE notice that the publisher changed it before print. The journal's dates are 6 September 2019 for receipt and 28 July 2020 for revision. The paper gives a code link, `github.com/mattdl/clsurvey`, which we did not open. Funding: Huawei funded the research. Masana has a Generalitat de Catalunya grant (2019-FI B2-00189), and Aljundi an FWO scholarship.

**Status.** Planned. #4782 (catastrophic interference, weight-space programming and LoRA) would use it. The maintainer designs any campaign to fix a model, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we downloaded the arXiv PDF (version 3, 29 pages) and read its text, extracted with `pdftotext`. We read all of it in the source's own text, except the reference list. That is the abstract, Sections 1 to 9, Tables 1 to 10, and Appendices A and B with Tables 11 to 20. We did not see the figures as images. The text gives the legends of Figures 2, 3, 4, 6 and 9, with each method's average accuracy and forgetting. We take numbers from those legends. We have the per-task curves only as the text describes them. No model read the source for us.

## Summary

**Claim.** The survey sorts continual-learning methods into three families: replay, regularization and parameter isolation. It sets every method's stability-plasticity trade-off with a rule that uses only the current task's data. Under that rule, it compares 11 methods and 4 baselines on three image benchmarks. Parameter isolation, PackNet above all, does best. Methods that work on a balanced benchmark often fail when the tasks differ a lot.

**Method.**

- **Scope.** Task-incremental classification only. Tasks arrive one at a time with clear boundaries, and training on each task runs offline to convergence. Each task has its own output head, and the model has the task label at test time. The authors note that this multi-head setting "hides the true difficulty of the problem", and they keep it so that existing methods need no changes.
- **The taxonomy (Figure 1).**
  - *Replay methods* keep old samples or make pseudo-samples. They divide into rehearsal (iCaRL, ER), pseudo-rehearsal (deep generative replay) and constrained optimization ([GEM](gradient-episodic-memory.md), A-GEM).
  - *Regularization methods* add a penalty term and keep no raw data. Data-focused ones distill the old model's outputs ([LwF](learning-without-forgetting.md), EBLL). Prior-focused ones put a penalty on changes to important weights ([EWC](elastic-weight-consolidation.md), [SI](synaptic-intelligence.md), MAS, IMM).
  - *Parameter isolation methods* give each task its own parameters. Fixed networks mask parts of one network ([PackNet](packnet.md), [HAT](hard-attention-to-the-task.md), PathNet). Dynamic architectures grow new parts ([progressive networks](progressive-neural-networks.md), Expert Gate).
- **The hyperparameter rule (Section 4, Algorithm 1).** Hyperparameters chosen on validation data from all tasks break the continual setting, and the authors say this "may lead to overoptimistic results". Their rule has two phases for each new task:
  - *Maximal plasticity search.* Plain fine-tuning on the new task, over a coarse grid of learning rates, gives the best new-task validation accuracy A\*.
  - *Stability decay.* The method starts at its most stable setting. The rule multiplies its forgetting hyperparameters by α = 0.5 until new-task accuracy reaches (1 − p)·A\*, with p = 0.2.
- **Methods compared.** Replay: iCaRL and GEM. Regularization: LwF, EBLL, EWC, SI, MAS, mean-IMM and mode-IMM. Parameter isolation: PackNet and HAT. Baselines: fine-tuning (the floor), joint training on all data (the "soft upper bound"), and two plain rehearsal baselines, R-PM and R-FM. Replay buffers hold 4.5k exemplars, the memory of the BASE model, or 9k.
- **Benchmarks.**
  - *Tiny Imagenet*: 10 balanced tasks of 20 classes each, on four VGG-style networks trained from scratch (SMALL, BASE, WIDE and DEEP).
  - *iNaturalist*: 10 unbalanced tasks, one for each super-category, with 5 to 314 classes per task.
  - *RecogSeq*: 8 different datasets in order, from flowers, scenes, birds and cars to aircraft, actions, letters and digits (SVHN).
  - iNaturalist and RecogSeq use AlexNet pretrained on ImageNet.
- **Metrics.** Average accuracy over all tasks on the final model, and average forgetting. Forgetting is a task's accuracy when it was first learned minus its accuracy at the end.
- **Further tests.** Model size and depth, dropout and weight decay, three task orders, and memory and compute cost (Table 9). Also a 40-task sequence followed by SVHN (Appendix B.6). Training uses SGD with momentum 0.9, batch 200, and at most 70 epochs with early stopping. GEM uses 5 epochs.

**Findings, as the source reports them.** Values are average accuracy, with average forgetting in brackets, in percent.

| Setting | Result |
| --- | --- |
| Tiny Imagenet, BASE, random order (Figure 2, Tables 4 and 5) | Joint 55.70. Fine-tuning 21.30 (26.90). PackNet 49.13 (0.00), iCaRL 4.5k 47.27 (−1.11), MAS 46.90 (1.58), EBLL 45.34 (1.44), GEM 4.5k 45.13 (4.96) |
| The same, lower half | HAT 43.57 (0.00), EWC 42.43 (7.51), LwF 41.91 (3.08), mode-IMM 36.89 (0.98), SI 33.93 (15.77) |
| Tiny Imagenet, buffer size | iCaRL 9k reaches 48.76. GEM 9k (41.75) "doesn't perform better" than GEM 4.5k (45.13) |
| Tiny Imagenet, model size (Section 6.3) | WIDE beats DEEP by "on average 11%" over all methods. SMALL has "a modest 0.89% more forgetting" than BASE |
| Tiny Imagenet, dropout (Table 6) | PackNet reaches 55.96 on WIDE with dropout, "almost 56%". Dropout often raises forgetting, for example LwF on SMALL from −0.44 to 18.38 |
| Tiny Imagenet, SI overfitting (Table 14) | On BASE without regularization, SI's train-validation gap averages 48.8 points. Dropout cuts it to 20.6 |
| iNaturalist, random order (Figure 4a) | Joint 63.90. PackNet 60.61 (0.00), mode-IMM 55.61 (−0.29), MAS 54.59 (4.97), EWC 54.02 (6.67), EBLL 53.30 (7.51), SI 47.54, fine-tuning 45.59 (17.02), LwF 45.39 (13.77) |
| RecogSeq (Figure 3) | Joint 65.82. PackNet 64.88 (0.00), MAS 45.72 (13.08), SI 43.40 (18.23), EWC 42.01 (19.84), EBLL 33.82 (29.26), LwF 30.59 (32.76), fine-tuning 23.11 (43.74) |
| RecogSeq with low learning rates chosen for the setup (Appendix B.4, Figure 6) | Joint 60.13. EWC 51.09 (2.40), MAS 50.57 (0.91), SI 49.96 (4.33), PackNet 47.23 (0.00). The authors call this "cherry-picking" that a real continual learner cannot do |
| Task order (Section 6.6, Table 4) | "Overall, the impact of the task order seems insignificant". On BASE, EWC scores 42.43 (random), 34.09 (easy to hard) and 41.83 (hard to easy) |
| 40 Tiny Imagenet tasks, then SVHN (Table 20) | PackNet reaches 26.62 on SVHN, against 81.40 to 93.68 for the others. MAS learns SVHN (88.41), but its average falls from 60.43 to 35.91 |
| Summary (Section 6.8, Table 10) | PackNet is "the best performer by high margin on all three datasets". MAS leads the regularization methods. LwF and EBLL "collapse" under the shifts of RecogSeq |

The authors list ten desiderata for general continual learning in Section 7. Examples are constant memory, no task boundaries, no task label at test time, and "graceful forgetting".

## Where the claims go beyond the evidence

- **One run per setting, with no spread.** The paper reports no seeds, no standard deviations and no intervals. Several rankings rest on gaps of 1 to 2 points, such as PackNet 49.13 against iCaRL 47.27. The text says "significantly" and "insignificant" but names no test.
- **"Insignificant" task order is not shown.** EWC on BASE moves by 8.3 points between random and easy-to-hard order. SI and EWC do better on hard-to-easy than on easy-to-hard, which the authors call "unexpected". Without repeated runs, order effects and run-to-run noise cannot be told apart.
- **"Best on all three datasets" compares fewer methods on two of them.** The authors leave HAT and the replay methods out of iNaturalist and RecogSeq. HAT had capacity problems, and replay "lack[s] a policy" for unbalanced data. So iNaturalist and RecogSeq compare PackNet with the regularization methods only.
- **The ranking depends on the learning-rate grid.** With the low learning rates of the original RecogSeq papers, EWC and MAS beat PackNet (51.09 and 50.57 against 47.23). The authors give this as a reason for their rule. It also shows that the main ranking holds for one grid only.
- **The rule's own settings came from earlier runs.** Appendix A.3 says the starting hyperparameters are "the observed upper bound" of earlier decays, plus one step. The paper keeps p = 0.2 and α = 0.5 and tests no other values.
- **Multi-head image classification only.** All tasks are image classification with a separate head for each task. The networks are a small VGG and AlexNet. The authors state this limit and say most results come "from a highly confined setup".
- **Small inconsistencies.** Table 4 gives PackNet on BASE 49.13, but the "Plain" row of Table 6 gives 47.67. The SMALL, WIDE and DEEP rows differ too. Figure 9 gives fine-tuning 24.20 (49.78), but Table 20 gives 24.15 (49.89). Table 3 gives the BASE classifier "2.36k" parameters, but the total of 3.51m needs 2.36m.

The authors give their own reasons where a method failed, for example SI's overfitting and HAT's capacity in early layers. They test these with extra tables, not by assertion only.

## Use in this repo

- **The survey measures outputs, and the repo measures units.** Its forgetting is task accuracy at first learning minus accuracy at the end. [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) measures the unit-level form: whether a VERIFIED explanation still passes its own probes after a fine-tune. 38 of 43 keep on Qwen2.5-1.5B (88%), and 252 of 311 on Qwen2.5-0.5B (81%). The survey gives no unit-level measure, so it cannot say whether its methods keep explanations true.
- **Its hyperparameter warning applies to any test of a fix.** The survey's main protocol point is that settings chosen with held-out data from old tasks give "overoptimistic" results. Appendix B.4 shows the size of that effect: the ranking of methods turns over. For #4781, the hard negatives in [`fixing-models.md`](fixing-models.md) are the prompts where copying is right. If a fix's settings were chosen while those prompts were in view, they would no longer be a held-out test. This is evidence for the maintainer, not a design.
- **The setting differs from a language model's.** Each task in the survey has its own head and a known task label. A language model has one shared output over the vocabulary and no task label. The survey says the shared-head setting is harder. So its rankings, PackNet first in particular, need not carry over to a fine-tune of Pythia-70m or Qwen.
- **Parameter isolation holds only while there is capacity.** PackNet keeps 0.00 forgetting by fixing old weights, but after 40 tasks it learns SVHN to 26.62 only. MAS learns SVHN but loses 24.5 points of average accuracy. That is the same trade-off a fix would face: fix the weights of VERIFIED units, or let them move. The [PackNet](packnet.md) and [HAT](hard-attention-to-the-task.md) pages already record this question. This survey adds the measured cost at saturation.
- **Fixed weights need not keep a pole fixed.** [`papers/neuron-development/`](../../papers/neuron-development/README.md) found that Pythia-70m's 86 VERIFIED poles keep changing after the loss settles. Their median correlation with the final form is 0.62 at step 16,000, and adjacent checkpoints correlate at a median of 0.78 to 0.82. A pole changes when any weight upstream of it changes. Isolation of a pole's own weights would therefore not keep its explanation true, unless every weight upstream stays fixed too. This is our inference, and the survey does not test units.
- **Replay is the interleaving of complementary learning systems.** [`complementary-learning-systems.md`](complementary-learning-systems.md) reports that focused learning of one fact damages related knowledge, and that interleaving avoids it. The survey's replay methods are that idea with a fixed buffer. It finds them bounded by joint training and short of PackNet on Tiny Imagenet. Replay was not tested on the unbalanced benchmarks.
- **No bearing on sister neurons or superposition.** The survey works inside one network and never compares units across models. So it says nothing about `open-interpretability sisters` ([`docs/analyses/sister-neurons.md`](../analyses/sister-neurons.md)). It also does not study interference between units, the subject of `openinterpretability/explain_groups.py`.
- **A question for the maintainer.** Should every test of a #4781 fix set its hyperparameters without the hard negatives, as the survey's rule does with old-task data? That is a rule for the fix campaign, so this page records it as a question and opens no issue.
- **No single-unit content.** The survey studies tasks, weights and masks. It says nothing about single neurons, explanations, probes or random baselines for them.

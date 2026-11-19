# Orthogonal Gradient Descent for Continual Learning

**Citation.** Mehrdad Farajtabar (DeepMind), Navid Azizan (Caltech, an internship at DeepMind), Alex Mott (DeepMind) and Ang Li (DeepMind). "Orthogonal Gradient Descent for Continual Learning". *Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics* (AISTATS 2020), PMLR 108, pages 3762–3773, dated 3 June 2020 on [proceedings.mlr.press](https://proceedings.mlr.press/v108/farajtabar20a.html). [arXiv:1910.07104](https://arxiv.org/abs/1910.07104), version 1 of 15 October 2019, the only version. The arXiv copy names no venue. The paper gives no code or data link and no funding statement. The acknowledgements thank Dilan Gorur, Jonathan Schwarz, Jiachen Yang and Yee Whye Teh for comments.

**Status.** Planned. #4782 (catastrophic interference, weight-space programming and LoRA) would use it. The maintainer designs any campaign to fix a model, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we downloaded the arXiv PDF (version 1, 13 pages) and read its text, extracted with `pdftotext`. We read in the source's own text the abstract, sections 1 to 6, Algorithm 1, Lemma 3.1 and its proof, and Tables 1 to 3. We also read Appendix A.1 (Table 4), A.2 and A.3 (Tables 5 to 7), and skimmed the reference list. Figures 1 to 5 came back as axis labels only. So we have no numbers for the rotated-MNIST bars (Figure 3), the two-task split-MNIST bars (Figure 4) or the epoch study (Figure 5). We did not read the PMLR copy, which may differ from arXiv version 1. No model read the source for us.

## Summary

**Claim.** A network trained on one task after another forgets the earlier tasks. OGD stores the gradients of the network's outputs on some earlier examples. It then projects each new-task gradient to be orthogonal to them. To first order the earlier outputs do not change, and the step still lowers the new task's loss. OGD stores no data.

**Method.**

- **Model gradients, not loss gradients.** The paper separates the gradient of the loss from the gradient of the model, ∇f, with one vector for each logit. The loss gradient is ∇f times the loss derivative (Equation 4). For a well-fitted example that derivative is near zero. So the authors argue that loss gradients carry little information about old tasks, and that the model gradient is what must be kept.
- **The constraint.** Task A first converges at weights w*_A. Then each update for task B must be orthogonal to ∇f_j at w*_A, for each stored x and logit j (Equation 8). The gradients at w*_A stand in for the gradients at the current weights. The authors justify this by overparameterisation: a good point for both tasks lies near w*_A.
- **The projection.** Gram–Schmidt turns the stored gradients into an orthogonal basis S. Each new gradient g becomes g̃ = g minus its projections onto S (Equation 9). Lemma 3.1 proves that −g̃ is still a descent direction, because ⟨−g̃, g⟩ = −‖g̃‖².
- **Three variants.** OGD-ALL stores the gradient of every logit. OGD-AVE stores the gradient of the mean logit. OGD-GTL stores only the logit of the true label. All experiments use OGD-GTL.
- **Memory.** OGD stores 200 gradients for each task. The store grows with the number of tasks, and Algorithm 1 adds each finished task's gradients to S.
- **Data.** Permuted MNIST has 3 and 5 tasks. Rotated MNIST has 2 tasks, at angles of 10 to 40 degrees, and 5 tasks at 0 to 40 degrees in steps of 10. Split MNIST has 2 tasks of 5 labels, and 5 tasks of 2 labels. Split MNIST uses a separate output head for each task.
- **Network and training.** An MLP with two hidden layers of 100 ReLUs and 10 logits. Plain SGD, learning rate 10⁻³, batch size 10, 5 epochs for each task.
- **Baselines.** EWC, and A-GEM, a loss-gradient constraint on 200 stored examples. Plain SGD is the lower bound. Multi-task training on all data seen so far (MTL) is the upper bound.
- **Statistics.** Mean and standard deviation of test accuracy over 10 random runs for 2- and 3-task runs, and 5 runs for 5-task runs.

**Findings, as the source reports them.** Each number is the accuracy (%) on a task's test set after the last task, mean ± standard deviation.

| Setting | Result |
| --- | --- |
| Permuted MNIST, 5 tasks, tasks 1 / 2 / 3 / 4 / 5 (Table 1) | MTL 93.2 / 91.5 / 91.3 / 91.3 / 88.4. OGD 79.5 / 88.9 / 89.6 / 91.8 / 92.4. A-GEM 85.5 / 87.0 / 89.6 / 91.2 / 93.9. EWC 64.5 / 77.1 / 80.4 / 87.9 / 93.0. SGD 60.6 / 77.6 / 79.9 / 87.7 / 92.4 |
| Rotated MNIST, 5 tasks at 0° to 40° (Table 2) | MTL 92.1 / 94.3 / 95.2 / 93.4 / 90.5. OGD 75.6 / 86.6 / 91.7 / 94.3 / 93.4. A-GEM 72.6 / 84.4 / 91.0 / 93.9 / 94.6. EWC 61.9 / 78.1 / 89.0 / 94.4 / 93.9. SGD 62.9 / 76.5 / 88.6 / 95.1 / 94.1 |
| Split MNIST, 5 tasks of 2 labels (Table 3) | MTL 99.6 / 99.8 / 98.8 / 98.2 / 99.1. OGD 98.6 / 99.5 / 98.0 / 98.8 / 99.2. A-GEM 92.9 / 96.3 / 86.5 / 92.3 / 99.3. EWC 90.2 / 98.9 / 91.1 / 94.4 / 99.3. SGD 88.2 / 98.4 / 90.3 / 95.2 / 99.4 |
| Split MNIST, three other label orders (Tables 5 to 7) | OGD's task-1 accuracy is 99.6, 99.0 and 98.1. A-GEM's is 99.2, 94.1 and 92.1. The authors say the order of methods is the same in all three |
| Variants and store size, task 1 after task 2, rotated MNIST, 20 / 200 / 2,000 gradients (Table 4) | ALL 75.7 / 79.9 / 86.6. AVE 75.3 / 75.5 / 77.7. GTL 76.4 / 82.9 / 87.1 |
| The same, permuted MNIST (Table 4) | ALL 87.3 / 89.7 / 90.5. AVE 86.8 / 86.9 / 89.4. GTL 86.5 / 89.4 / 91.4 |
| More epochs, 2-task permuted MNIST, 20 to 120 epochs (Appendix A.2) | In words only: A-GEM and OGD stay competitive, and SGD and EWC first rise and then drop |

The authors read Table 1 as OGD "significantly better than EWC and SGD while being on par with A-GEM". They read Table 2 as OGD ahead of the other methods at 10, 20 and 30 degrees.

## Where the claims go beyond the evidence

- **"On par with A-GEM" hides a gap on the oldest task.** In Table 1, A-GEM keeps 85.5 on task 1 and OGD 79.5, with spreads of 1.7 and 2.3. That is the task with the most forgetting, which is what the benchmark measures.
- **The rotated-MNIST claim does not match Table 2.** At 30 degrees (task 4), SGD has 95.1 and EWC 94.4, both above OGD's 94.3. At 40 degrees, A-GEM's 94.6 is above OGD's 93.4. OGD is ahead at 0, 10 and 20 degrees.
- **"Significantly" with no test.** The paper reports means and standard deviations over 5 or 10 runs. It gives no statistical test, no interval and no adjustment for many comparisons.
- **One small network on MNIST.** Every result is a two-layer MLP of 200 hidden units on MNIST variants, with plain SGD. The paper's claim to lead the field rests on these alone.
- **Hyper-parameters set on the test results.** The paper names no validation set. The learning rate is chosen as small enough "that decreasing it further did not improve the performance". The number of epochs is chosen to saturate the first task. The text does not say on which data.
- **The store is not compared at equal cost.** OGD and A-GEM each store 200 items. An OGD item is a gradient with one entry for each parameter, and an A-GEM item is one MNIST image. The paper does not compare the memory in bytes.
- **Small slips in the text.** The text refers to "Table 4.1", which is Table 1. Algorithm 1, line 8, loops over T_t where the task index is k. The table captions say "class" where the rows are tasks.

The authors state three limits themselves. OGD stores only part of each task's gradients, so some forgetting remains. All methods fail on dissimilar tasks, such as rotations above 90 degrees. OGD is sensitive to the learning rate, because the stored gradients hold only near w*_A.

## Use in this repo

- **Nothing in the repo uses this method today.** A search of the code, docs, papers and skills finds no gradient projection and no orthogonal update. The repo has never trained a model. The sister reviews of [GEM](gradient-episodic-memory.md), [EWC](elastic-weight-consolidation.md) and [synaptic intelligence](synaptic-intelligence.md) report the same.
- **OGD keeps outputs, not explanations.** The constraint holds the stored logits fixed to first order. Many directions orthogonal to those logit gradients still change hidden units. [`papers/neuron-development/`](../../papers/neuron-development/README.md) shows how far units can move while outputs barely move. The loss of Pythia-70m is within 1% of its final value by step 16,000. Yet the median correlation of its VERIFIED poles with their final form is 0.62 there. Adjacent checkpoints after that step correlate at a median of 0.78 to 0.82. So an OGD-style step that keeps outputs is no evidence that a pole's explanation survives. This is our inference. The paper never looks at a hidden unit.
- **The same projection could hold a pole fixed.** OGD protects any function whose gradient is in S. The gradient of a pole's activation on its recorded probe texts is such a function. Add those gradients to S, and an update leaves the pole's activations unchanged to first order. Then a re-run of the recorded probes is the check, as in the neuron-development study. This is our suggestion, and the paper does not test it.
- **Model gradients matter for the " days" error.** For pythia-70m's " days" error (#4781, #5015), [`fixing-models.md`](fixing-models.md) names the shortlist prompts where copying is right as the hard negatives. Those prompts are only partly fitted: " legs" has p 0.5494 and " wheels" 0.4149. The paper's argument is about examples whose loss gradient is near zero. Here the loss gradients are not near zero, so the gap between model and loss gradients may be small. This is our inference from the shortlist's numbers.
- **Shared units are where the projection costs most.** A new-task gradient loses the part that lies in S. If old and new tasks share neurons, as in superposition (`openinterpretability/explain_groups.py`), that part is large. So the projected step learns less. The paper reports no measure of how much of g the projection removes. Our attribution patching already computes gradients at the activations (`openinterpretability/core/wiring.py`). A per-neuron share of that removed part would show which units carry the conflict. This is our suggestion.
- **A question for the maintainer.** Should a pole-preserving projection, or the overlap between a fix's gradient and the VERIFIED poles' activation gradients, be recorded as evidence for #4781-style fixes? It needs gradients only, not training. We record it as a question and open no issue, because the fix campaign is the maintainer's.
- **No single-neuron or explanation content.** The paper's units are whole networks and their test accuracy. Nothing in it bears on our review protocol, probes or verdicts.

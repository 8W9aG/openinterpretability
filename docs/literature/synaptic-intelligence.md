# Continual Learning Through Synaptic Intelligence

**Citation.** Friedemann Zenke, Ben Poole and Surya Ganguli (all Stanford University, with equal contributions from Zenke and Poole). "Continual Learning Through Synaptic Intelligence". *Proceedings of the 34th International Conference on Machine Learning*, Sydney, PMLR 70, 2017. [arXiv:1703.04200](https://arxiv.org/abs/1703.04200), version 3 of 12 June 2017 (version 1: 13 March 2017). The arXiv comment says "ICML 2017", and the venue agrees with the dates. The paper gives no code or data link. Funding: the Swiss National Science Foundation and the Wellcome Trust (Zenke). Also Stanford fellowships (Poole), and the Burroughs Wellcome, McKnight, Simons and James S. McDonnell foundations and the Office of Naval Research (Ganguli).

**Status.** Planned. #4782 (catastrophic interference, weight-space programming and LoRA) would use it. The maintainer designs any campaign to fix a model, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we downloaded the arXiv PDF (version 3) and read its text, extracted with `pdftotext`. We read all of it in the source's own text: the abstract and sections 1 to 6. Also appendices A (the CIFAR network), B (split CIFAR-10) and C (the path integral against the Fisher information). We did not see the figures as images. Most results are curves in Figures 3, 4, 6 and 7, so this page gives their direction only. No model read the source for us.

## Summary

**Claim.** Each parameter, a "synapse", can keep a running estimate of how much it helped lower the loss on past tasks. A quadratic penalty on changes to the important parameters then lets a network learn new tasks with little forgetting. The cost is small, and each parameter updates its estimate online during training.

**Method.**

- **The importance measure.** During training on task µ, each parameter k sums the product of its gradient and its update. The sum ω is that parameter's share of the drop in loss (Equation 3). At the end of a task, Ω sums ω over past tasks (Equation 5). Each term divides by the squared distance the parameter moved, plus a damping term ξ.
- **The penalty.** The loss for a new task adds c · Σ Ω (θ̃ − θ)² (Equation 4). Here θ̃ is the parameter at the end of the previous task. The strength c trades old memories against new ones. The authors motivate this surrogate exactly only for two tasks.
- **Theory.** For a quadratic loss under gradient descent, the summed importances, averaged over random starts, give the Hessian up to a scale. Without averaging, the same holds when the Hessian is diagonal or of rank 1.
- **Split MNIST.** Five tasks, each of two consecutive digits. The network is an MLP with two hidden layers of 256 ReLUs and a separate readout head for each task. Training uses ξ = 0.001, Adam with learning rate 0.001, batch 64, 10 epochs. Consolidation on (c = 1) against off (c = 0), 10 runs.
- **Permuted MNIST.** Ten tasks, each with its own random permutation of the pixels. The network is an MLP with two hidden layers of 2,000 ReLUs and one softmax output. Training uses ξ = 0.1, batch 256 and 20 epochs. A coarse grid search on a held-out validation set chose c = 0.1. The baselines are SGD, SGD with dropout, and EWC "extracted and replotted from Kirkpatrick et al. (2017)".
- **Split CIFAR-10/100.** The Keras CIFAR-10 network: 4 convolutional layers and 2 dense layers with dropout. It trains 60 epochs on CIFAR-10, then on 5 tasks of 10 CIFAR-100 classes each. The search for c covered 0.001 to 0.1. Controls: c = 0 on all tasks in sequence, and a network trained from scratch on each task. 5 runs each.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Split MNIST, Task 2 learned | Both networks reach accuracy "close to 1" on Task 2. Without consolidation, Task 1 shows "substantial loss in accuracy". With it, Task 1 has "only minor impairment" |
| Split MNIST, all five tasks learned | Without consolidation, the first two tasks drop "back to chance levels". With it, they show "only minor degradation" |
| Permuted MNIST, 10 tasks | Without consolidation the network "rapidly forgets" earlier tasks. With c = 0.1 it keeps "high classification accuracy" on Task 1 |
| Permuted MNIST, against joint training | Consolidation is "only slightly worse" than one network trained on all the data at once, and "comparable" to EWC |
| Permuted MNIST, importances across tasks (Figure 5) | First-layer importances show no correlation between tasks in both cases. Without consolidation, second-layer importances "become more correlated" as tasks accumulate. With it, they stay "largely uncorrelated" |
| Split CIFAR-10/100 | With consolidation, accuracy is similar across tasks. Without it, older tasks decline. Consolidation is better "except on the last task" |
| Split CIFAR-10/100, against training from scratch | Consolidation "either significantly outperformed" the network trained from scratch or matched it, except on Task 1 (CIFAR-10) |
| Split CIFAR-10, Task B (Appendix B) | Consolidation is better by a "small but significant difference (≈ 4.5%)" in validation accuracy, for c from 0.1 to 10, 10 runs |
| Path integral against the Fisher (Appendix C) | At the minimum of a quadratic, the empirical Fisher is 0, but the path integral is proportional to the Hessian's diagonal |

## Where the claims go beyond the evidence

- **Small networks and image tasks only.** The largest network is the Keras CIFAR-10 network. All tasks are image classification. The closing claim that "we may need to add intelligence to our synapses" is wider than these experiments.
- **The split benchmarks use a separate head for each task.** Each task's loss uses only its own outputs. The authors say this avoids "crosstalk between digits at the readout layer". It also makes forgetting easier to prevent than with one shared output.
- **The theory covers a quadratic loss only.** The authors state this limit. For general losses they "cannot expect any simple mathematical correspondence" between importance and the Hessian. The surrogate loss is exact only for two tasks, and the case of more tasks rests on experiment.
- **Few numbers, and tests that are not named.** The text gives one number for a result, the ≈ 4.5% of Appendix B. All other results are curves. The paper says "significantly" several times but names no statistical test. Split MNIST reports the standard error of 10 runs. Split CIFAR reports the standard deviation of 5 runs. Permuted MNIST gives no count of runs.
- **The EWC baseline was not re-run.** The authors took the EWC curve from Kirkpatrick et al. (2017), so its settings may differ from this paper's. The claim that the two methods are "comparable" rests on that copy.
- **The choice of c may use the reported data.** For permuted MNIST, a held-out validation set chose c. For split CIFAR, the text says it searched for "the best c" but does not name the set. The reported results are validation accuracies.
- **Small inconsistencies.** The text calls the network without consolidation the "blue line" in Figure 4, but the figure's caption makes "Our approach" blue. The captions of Figures 6 and 7 say "Validation error", but the axes plot validation accuracy. The split MNIST digits run "from 0 to 10", but MNIST has 0 to 9. The CIFAR search range for c is "1×10⁻³ < c < 0.1", but Figure 6 uses c = 0.1.

The authors credit EWC and say both methods may work because their importance measures correlate. They also make "no claim that biological synapses behave like the intelligent synapses" of their model.

## Use in this repo

- **Nothing in the repo uses this method today.** A search of the code, docs, papers and skills finds no use of synaptic intelligence, EWC or a per-parameter importance penalty. The sister review of EWC is #4812, in the same batch.
- **The importance measure is a gradient times a change, summed over training.** Our attribution patching is a gradient times an activation change at one point (`openinterpretability/core/wiring.py`, `openinterpretability/cli_pole_effects.py`). The paper's ω does the same for weights, along the whole training path. Appendix C argues why that matters: at a minimum the gradient is near 0, but the path integral is not. A one-point estimate at the final weights can miss what a unit did on the way there.
- **Late change in Pythia-70m's poles suggests a low importance does not mean a still unit.** [`papers/neuron-development/`](../../papers/neuron-development/README.md) found that the 86 VERIFIED poles keep changing after the loss settles. At step 16,000 their median correlation with their final form is 0.62. Adjacent checkpoints correlate at a median of 0.78 to 0.82. In the paper's terms, such weights move but lower the loss little, so ω would rate them unimportant. A penalty built on ω would leave them free to move further. So consolidation of this kind would not, by itself, keep an explanation true. This is our inference. The paper does not test explanations.
- **Shared units are where tasks interfere.** Figure 5 is the paper's one look inside the network. Without consolidation, the second layer's importances become correlated across tasks, and the authors name this as the likely cause of forgetting. That fits the case of a VERIFIED pole that serves several features, as in the superposition records (`openinterpretability/explain_groups.py`). Such a pole is where a later fine-tune would most likely change an explanation. The paper measures importance per weight, not per neuron, and on MNIST only.
- **Evidence for a fix of #4781, not a design.** The paper gives one way to protect what a model knows while it learns a new fact. A fix of pythia-70m's " days" error would need to keep the prompts where copying is right (see [`fixing-models.md`](fixing-models.md)). Like the interleaving of [`complementary-learning-systems.md`](complementary-learning-systems.md), consolidation addresses that need. It does so without access to the old data. Whether a campaign uses either is the maintainer's question.
- **A question for the maintainer.** Pythia publishes its training checkpoints. A coarse ω for each neuron could be computed from the change in weights between checkpoints and the gradient of a held-out loss. That would test whether the poles that change late have low importance. It would be research evidence, not part of a fix campaign. We record it as a question and propose no issue.
- **No single-neuron or explanation content.** The paper's units are weights and biases, and it never explains a neuron. Nothing in it bears on our review protocol or our verdicts.

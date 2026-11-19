# Overcoming Catastrophic Forgetting in Neural Networks

**Citation.** James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Dharshan Kumaran and Raia Hadsell (DeepMind, London). Claudia Clopath (Bioengineering department, Imperial College London). "Overcoming catastrophic forgetting in neural networks". [arXiv:1612.00796](https://arxiv.org/abs/1612.00796), v1 of 2 December 2016, v2 of 25 January 2017. Published in *Proceedings of the National Academy of Sciences*, 2017, volume 114, number 13, [doi:10.1073/pnas.1611835114](https://doi.org/10.1073/pnas.1611835114). The arXiv text gives no code or data links, and no funding statement. Its acknowledgements thank five people. Most authors are at DeepMind, which also built the Atari agent that the paper extends.

**Status.** Planned. #4782 (catastrophic interference and weight-space programming) would use it, and this page proposes one follow-up that measures evidence. The maintainer designs the campaign to fix models, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we downloaded the arXiv v2 PDF and extracted its text with `pdftotext`. The reader was this page's author, a model (Claude Opus 5.5). We read all of the extracted text in the source's own words. That is the abstract, sections 1 to 3, the references and the appendix (4.1 MNIST, 4.2 Atari, 4.3 Fisher overlap), with Tables 1 and 2. We did not see the figures as images. Most results are curves in Figures 2, 3 and 4, so we have their captions and the text about them, but not their values. We did not read the PNAS version or its supporting information, which can differ from arXiv v2.

## Summary

**Claim.** A network can learn tasks in sequence without forgetting the old ones. It does this when the weights that matter to an old task learn more slowly. The authors call the method elastic weight consolidation (EWC). They show it on permuted MNIST and on ten Atari games learned in sequence.

**Method.**

- **The penalty.** After task A, the loss on task B gains a quadratic term for each parameter: λ/2 · Fᵢ (θᵢ − θ*_A,i)². Here θ*_A is the solution for task A, and Fᵢ is the diagonal of the Fisher information matrix. The term comes from a Laplace approximation of the posterior after task A, as a Gaussian with diagonal precision. λ sets how much the old task counts. A third task adds a second penalty, or one merged quadratic penalty.
- **Why it should work.** The authors argue that over-parameterisation makes many weight settings equally good. So a solution for task B is likely to lie near the solution for task A. The Fisher is cheap: it uses first derivatives only, and it is positive semi-definite.
- **Permuted MNIST.** Each task is MNIST with its own fixed random permutation of the pixels. The networks are fully connected, with ReLU units. The authors compare plain SGD, a uniform L2 pull to the old weights, SGD with dropout and early stopping, and EWC. A third experiment measures the overlap of two tasks' Fisher matrices, layer by layer. In it, the permutation covers either a central 8x8 square of pixels or a 26x26 square. The overlap is one minus the Fréchet distance of the two Fishers, each scaled to unit trace.
- **Atari.** The agent is a Double DQN with about four times the standard network's parameters. It has task-specific biases and gains in each layer, and a task-recognition model. That model is a hidden Markov model over the game context, after the "forget me not" process. Each run uses ten games, drawn from a pool of 19 that a single DQN plays at human level within 50 × 10⁶ frames. The games come in random order and come back several times. The score is the sum of the ten human-normalised scores, each clipped to 1, so the maximum is 10. A control gives the agent the true task label. A last test adds Gaussian noise to a one-game agent's weights. The noise is uniform, shaped by the inverse Fisher, or within the Fisher's nullspace.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Permuted MNIST, plain SGD (Figure 2A) | Catastrophic forgetting of task A, which "compounds further with more training time" and with more tasks |
| Permuted MNIST, uniform L2 penalty | Task A degrades "much less severely", but task B "cannot be learned properly", because the penalty protects all weights equally |
| Permuted MNIST, EWC | Learns task B "well without forgetting task A" |
| Earlier methods on permuted MNIST | "Reasonable results on up to two random permutations" (Srivastava et al., 2013, and Goodfellow et al., 2014) |
| Dropout against EWC (Figure 2B), 50 random hyperparameter settings, best one reported | Dropout "does not scale to more tasks". EWC learns many tasks "with only modest growth in the error rates" |
| Fisher overlap by depth (Figure 2C) | Similar tasks (8x8 permuted) use similar weights in the whole network. Dissimilar tasks (26x26) use separate weights in early layers, but the layers near the output are "reused for both tasks" |
| Atari, plain gradient descent | "Never learns to play more than one game". The total score "remains below one" (maximum 10) |
| Atari, EWC (10 sets of 10 games, 4 seeds each) | The agents "do indeed learn to play multiple games", but below the score of ten separate DQNs |
| Atari, true task labels against inferred ones | "The improvement here was only modest" |
| Weight noise on a one-game agent (Figure 3C) | The agent loses less under noise shaped by the inverse Fisher than under uniform noise. Noise in the Fisher's nullspace hurts as much as inverse-Fisher noise, though the approximation predicts no effect |
| Atari settings (Table 2) | Fisher multiplier λ = 400. The Fisher uses 100 minibatches from the replay buffer. The penalty starts after 20 million frames of a game |

The text gives no MNIST accuracies or Atari scores as numbers. The results above are the text's description of curves that we did not see.

## Where the claims go beyond the evidence

- **No numbers, intervals or tests in the text.** Every comparison is a curve. The Atari runs have 4 seeds for each of 10 game sets, but the text gives no spread. The MNIST runs give no seeds at all.
- **The MNIST λ and Fisher sample are not given.** Table 1 lists learning rates, widths and epochs, but not λ for MNIST or how many samples its Fisher uses. A reader can't rerun Figure 2A from the text.
- **The Atari agent changes more than EWC.** The agent also has task-specific gains and biases, a task-recognition model and a buffer for each task. The text doesn't say whether the plain-SGD agent has these too. So the gap between the two curves isn't shown to come from the penalty alone.
- **The dropout baseline had the best of 50 settings, and the reader can't tell what EWC had.** Table 1's 3B column gives ranges for dropout's runs. The text doesn't say whether EWC had the same search.
- **Permuted MNIST shares its output.** Every task has the same ten labels and permutes only the input. The Fisher overlap result itself shows that the tasks share the output layers. That makes the tasks easier to keep apart than tasks that need different outputs.
- **Inconsistencies within the source.** Table 1's header and the appendix text name "Figure 3" for the MNIST graphs, which are Figure 2. Table 2's caption says MNIST, but it lists the Atari settings. The discussion points to "Fig 4C" for the nullspace result, which is Figure 3C. Appendix 4.3 calls the overlap plot Figure 3C, but it is Figure 2C. The text gives a replay buffer of 5 × 10⁵ experiences, and Table 2 gives a memory of 50,000. Section 2.1 says that at the switch "performance for task B falls rapidly, while for task A it climbs steeply". Forgetting would make task A fall and task B rise.
- **Credit: the authors state the main limit themselves.** The nullspace test shows the diagonal Fisher is "over-confident about certain parameters being unimportant". The discussion calls the point estimate "a significant weakness".

## Use in this repo

- **The repo already measures what EWC protects, with no penalty.** [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) tests VERIFIED explanations again after instruction tuning. 38 of 43 keep on Qwen2.5-1.5B (88%), with a median activation correlation of 0.983. 252 of 311 keep on Qwen2.5-0.5B (81%), at a median of 0.974. So the fine-tuning that made these instruct models moved most poles little, without EWC. The failures sit near a gate on the base model, which reads as a probe or two moving, not a forgotten unit.
- **EWC's premise fits poles that move while the loss stays flat.** EWC assumes many weight settings give the same loss. [`papers/neuron-development/`](../../papers/neuron-development/README.md) found Pythia-70m's VERIFIED poles still changing after the loss settles. At step 16,000 their median correlation with the final form is 0.62, and adjacent checkpoints correlate at a median of 0.78 to 0.82. Movement along directions that the loss barely feels is what that premise allows. The paper doesn't test a language model, so this is consistent with our result, not evidence for it.
- **A follow-up can test whether weight importance predicts which explanations keep.** The diagonal Fisher of a pole's input and output weights is a cheap importance score. On Qwen2.5-0.5B, 59 of 311 poles fail after instruction tuning. A test could read each pole's Fisher on the base model, and see whether low-Fisher poles move more and fail more often. This page proposes that issue. It measures evidence only and changes no weights.
- **The nullspace result is a caution for first-order importance.** The repo's answer attributions are first order, (a − ā) · ∂logit/∂a (`openinterpretability/explain.py`). The diagonal Fisher is also built from first derivatives, and the paper found it overconfident about which weights don't matter. A unit with a small first-order score may still matter. The `activation-patching-practice` skill already warns of attribution patching's failure modes, so this adds a second source, not a new rule.
- **Fisher overlap is the weight-level form of shared units.** The paper's similar tasks share weights in every layer. Its dissimilar tasks share only the layers near the output. In the repo, superposition records (`openinterpretability/explain_groups.py`) mark units that several features share, which is where a change could interfere. The [complementary-learning-systems](complementary-learning-systems.md) page makes the same point for focused learning.
- **EWC is a baseline in two editing papers we reviewed.** The [GRACE](aging-with-grace.md) and [WISE](wise-lifelong-editing.md) pages list fine-tuning with EWC among their baselines. The synthesis in [`fixing-models.md`](fixing-models.md) notes that the repo has no fine-tuning or weight-edit code. So the repo can't run EWC today.
- **A question for the maintainer.** Should a weight fix for #4781 carry a penalty like EWC's, anchored on the shortlist prompts where copying is right? That is a choice in the fix campaign, so this page records it as a question and opens no issue.
- **No single-unit content.** The paper studies weights and whole-task scores. It says nothing about single neurons, explanations, probes or random baselines for them.

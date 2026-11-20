# Editing Models with Task Arithmetic

**Citation.** Gabriel Ilharco, Marco Tulio Ribeiro, Mitchell Wortsman, Suchin Gururangan, Ludwig Schmidt, Hannaneh Hajishirzi and Ali Farhadi. The affiliations are the University of Washington, Microsoft Research and the Allen Institute for AI. The text we read doesn't map them to authors. "Editing Models with Task Arithmetic". [arXiv:2212.04089](https://arxiv.org/abs/2212.04089), version 3 of 31 March 2023. Version 1 is of 8 December 2022, and version 2 of 29 March 2023. Venue: ICLR 2023, as the arXiv comment field states. The paper's own header shows no venue line. The code is at [github.com/mlfoundations/task_vectors](https://github.com/mlfoundations/task_vectors). Funding: the acknowledgements name NSF IFML, Open Philanthropy, NSF, ONR and DARPA grants, a Sloan Fellowship and gifts from AI2. The paper names its tools: Detoxify, the Perspective API and Hugging Face's libraries and checkpoints.

**Status.** Planned. #5298 holds the task-vector merge rules for the patch format. #5404 would test the interpolation half of the paper on the registry's own pair of models. No code, skill or protocol here depends on it. The fix campaign (#5220) is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML full text (version 3) for this page's author, a model (Claude Opus 5.5). We have in the source's own text the abstract, Sections 1 to 8, the acknowledgements, Appendices A to D, and Tables 1 to 10. We have the opening of Appendix E and part of E.1. The rest of E.1, and E.2 and E.3, came back only as a summary, so this page gives none of their numbers. We didn't see any figure as an image, nor Tables 11 and 12. So the cosine similarities of Figure 5 appear here only as the text describes them.

## Summary

**Claim.** The change that fine-tuning makes to a model's weights, a *task vector*, can be edited with arithmetic. Negating it removes a behaviour, adding several builds a multi-task model, and an analogy of three improves a task with no data for it.

**Method.**

- **Task vector.** For a task $t$, $\tau_t = \theta^t_{\mathrm{ft}} - \theta_{\mathrm{pre}}$: the fine-tuned weights minus the pre-trained ones. A new model is $\theta_{\mathrm{new}} = \theta + \lambda \tau$. "Adding a single task vector to a pre-trained model with λ = 1 results in the model fine-tuned on that task."
- **Operations.** Negation uses $-\tau$. Addition uses $\sum_i \tau_i$. An analogy "A is to B as C is to D" uses $\tau_C + (\tau_B - \tau_A)$.
- **The scale.** $\lambda$ is chosen "using held-out validation sets", on a grid. For addition, one $\lambda$ for the sum, from 0 to 1 in steps of 0.05. For negation on images, the largest $\lambda$ that keeps at least 95% of the pre-trained model's accuracy on the control task. For negation on text, the largest that keeps perplexity within 0.5 points.
- **Models.** CLIP ViT-B/32, ViT-B/16 and ViT-L/14. GPT-2 Small, Medium and Large. T5-small, T5-base and T5-large, and T5-base checkpoints from the Hugging Face Hub.
- **Tasks.** Eight image classification sets (Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC45, SUN397, SVHN), with ImageNet as the control. Toxic generation, with WikiText-103 perplexity as the control. Four GLUE tasks (MRPC, RTE, CoLA, SST-2). Yelp and Amazon sentiment for analogies.
- **Baselines.** Gradient ascent, a random vector scaled per layer to the task vector's norm, fine-tuning on non-toxic text, and multi-task fine-tuning.
- **Metrics.** Accuracy, and normalised accuracy: each task's accuracy over its fine-tuned model's. For toxicity, the share of 1,000 generations that Detoxify calls toxic, and their mean toxicity.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Negation, CLIP, mean of 8 targets, target accuracy and ImageNet accuracy (Table 1) | ViT-L/14: pre-trained 64.8 and 75.5, negative task vector 19.0 and 72.9, random vector 60.9 and 72.9, gradient ascent 3.93 and 16.3 |
| Same, ViT-B/32 (Table 1) | Pre-trained 48.3 and 63.4, negative task vector 24.0 and 60.9 |
| Negation, GPT-2 Large, toxic share, mean toxicity, perplexity (Table 2) | Pre-trained 4.8%, 0.06, 16.4. Negative task vector 0.8%, 0.01, 16.9. Random vector 4.8%, 0.06, 16.4. Fine-tuned on non-toxic text 1.8%, 0.03, 17.2 |
| Negation, GPT-2 Medium and Small (Tables 9, 10) | 4.3% to 1.8% toxic, perplexity 18.5 to 18.9. 3.7% to 2.5%, perplexity 25.2 to 25.3 |
| Addition, CLIP, pairs of tasks (Section 4.1) | 98.9% normalised accuracy on average |
| Addition, CLIP, all 8 tasks (Section 4.1, D.2) | The best sum reaches 91.2% normalised accuracy. Multi-task fine-tuning reaches 99.4% |
| Addition, T5-base, mean of four GLUE tasks (Table 3) | Zero-shot 57.1, fine-tuned 78.1, fine-tuned plus a Hub task vector 78.6 |
| Analogies, T5, sentiment, auxiliary fine-tune against analogy against target fine-tune (Table 4) | Yelp, T5-small 88.6, 89.9, 91.1. Amazon, T5-base 90.8, 92.7, 93.2 |
| Two task vectors at $\lambda = 0.5$ against an ensemble (Appendix A) | Equal to a uniform average of the two fine-tunes. Accuracy correlates with the ensemble's at Pearson 0.99 |
| Scale (D.3) | "0.3 to 0.5 produce close to optimal results in many cases". "The variance in the optimal scaling coefficients can be large" |

Section 6 reports that task vectors from different tasks "are typically close to orthogonal". The authors "speculate that this enables the combination of task vectors via addition with minimal interference". The largest similarities are among MNIST, SVHN and GTSRB, and between EuroSAT and RESISC45. Intermediate task vectors "converge rapidly to the direction of the final task vector". A higher fine-tuning learning rate hurts both task vectors and the fine-tuned models.

## Where the claims go beyond the evidence

- **"Minimal interference" is a speculation, not a measure.** The paper shows that task vectors are near-orthogonal by cosine and that sums work. It doesn't test whether orthogonality causes the result. It doesn't measure interference except as the accuracy gap of a sum, which grows with the number of tasks (D.4).
- **Each use tunes the scale on held-out data.** Every operation picks $\lambda$ on a validation set, and negation picks it to protect the control score. So "keeps the control accuracy" is partly a selection rule, not only a result. Appendix D.3 calls the best scale's variance large.
- **One control per negation.** ImageNet guards the image negations, and WikiText-103 perplexity guards the text ones. A control can stay level while other abilities change. The text negation keeps perplexity within 0.5 points by construction.
- **Mostly single runs.** No table has an interval or error bar. Only the GLUE results average three seeds, and D.1's seed study covers two tasks.
- **The toxicity measure is one classifier on one prompt.** 1,000 generations from the prefix "I don't care if this is controversial", scored by Detoxify. Appendix C.3 adds RealToxicityPrompts, which we read only in part.
- **No look inside.** The paper works at the level of whole weight vectors. It doesn't ask which units a task vector changes, or whether a negation removes a function or masks it.

The authors state limits themselves. Task vectors need "models with the same architecture", fine-tuned "from the same pre-trained initialization". Removing a behaviour the model already shows strongly is harder (B.7). And a high learning rate calls for "more caution".

## Use in this repo

- **A patch is a task vector.** A patch's weight ops hold a change to the base model's weights, applied through hooks ([`docs/patches.md`](../patches.md)). The registry's one built merge rule, `additive` (`RESOLUTIONS` in `openinterpretability/patches/history.py`), gives $\theta_0 + \Delta_1 + \Delta_2$. That is the paper's addition with $\lambda = 1$ for each vector. The paper never uses $\lambda = 1$ for a sum. Its best sums sit at 0.3 to 0.5 per vector. So the paper's evidence is about a scaled sum, which `additive` isn't.
- **What #5298 would need, from this paper.** Three rules: a scaled sum with one $\lambda$ for all parents, negation as $-\lambda\tau$, and an analogy $\tau_C + (\tau_B - \tau_A)$. Each additive op already takes an output `scale` (`openinterpretability/patches/apply.py`). A `set` history op can change an earlier op's parameters. So a scaled sum and a negation can be written today, as `dense_add` ops with `scale` set to $\lambda$ or $-\lambda$. The rule would only choose $\lambda$. The paper chooses $\lambda$ on held-out data, and a merge in the patch format must bring its own evidence (`docs/patches.md`, *History and branches*). So the evidence run is where $\lambda$ would be chosen. `patch revert` drops ops. It is not a negation, because it removes a patch's change, where negation subtracts it from a model that never had it.
- **The control is the interference report.** The paper keeps one control score level. The interference report in [`docs/analyses/interference.md`](../analyses/interference.md) asks the same question for every VERIFIED pole and answer route. It compares each with a self floor and a random floor of the same size. The paper's random-vector baseline, scaled per layer to the task vector's norm, is the same idea as the random floor. In Table 2 the random vector leaves GPT-2 Large's toxicity and perplexity exactly where they were.
- **The registry has one task vector to test.** Qwen2.5-1.5B-Instruct minus Qwen2.5-1.5B is a task vector in the paper's sense. The interference report found it flips 8 of 59 VERIFIED poles above both floors, and its footprint holds 250,820 of 250,880 neurons. #5404 asks what happens at $\lambda$ = 0.25, 0.5 and 0.75. The paper suggests two more points: $\lambda < 0$ (negation, subtracting instruction tuning from the base model) and $\lambda > 1$ (extrapolation). We added them to #5404 as a comment, not as a new issue.
- **Near-orthogonal is a weight statistic.** The paper reads low cosine similarity between task vectors as a sign of little interference. The registry's evidence warns against reading function from weight statistics alone. Instruction tuning moves almost every neuron's activation, yet 50 of 59 VERIFIED poles keep their explanations (`interference.md`). The interference report would test the orthogonality claim at the unit level, on two patches whose weight deltas are near-orthogonal.
- **Intermediate task vectors converge early.** The direction of a task vector settles after a few hundred steps (Figure 7, by the text). [`papers/neuron-development/`](../../papers/neuron-development/README.md) finds a related pattern for units: Pythia-70m's VERIFIED poles reach their final form well before training ends. The two are different measures, one of a whole weight change and one of single units, so this page draws no conclusion from the match.
- **Related pages.** [`model-soups.md`](model-soups.md) is the same group's averaging paper, and Appendix A here shows the link: two task vectors at $\lambda = 0.5$ give a uniform soup. [`lorahub.md`](lorahub.md) composes adapters with learned weights. #4873 (tangent space), #4874 (TIES), #4877 and #4878 review the methods that build on this one, and #3403 decomposes task vectors into components.
- **Question for the maintainer.** Should a scaled merge in the patch format choose its $\lambda$ inside the merge's evidence run, as the paper chooses it on held-out data? Or should the patch author set it? The answer decides whether #5298's rule needs a search. This page records the question and opens no issue.

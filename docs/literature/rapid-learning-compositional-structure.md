# Linking Rapid Learning to Compositional Structure in Neural Networks

**Citation.** Abigail L. Tenenbaum and R. Thomas McCoy, of the Department of Linguistics and the Wu Tsai Institute, Yale University. The text doesn't say which affiliation belongs to which author. "Linking Rapid Learning to Compositional Structure in Neural Networks: Evidence from Additive Analogies in Meta-Learned Parameter Spaces". [arXiv:2611.17314](https://arxiv.org/abs/2611.17314), version 1 of 12 November 2026, the only version. Preprint, no venue given. The paper gives no code or data link, no funding and no disclosure of tool or LLM use.

**Mismatch.** The abstract in the HTML has one more sentence than the abstract page. That sentence says the analogies "emerge across neural architectures and function domains." The abstract page leaves it out.

**Status.** Planned. It is evidence for #4782's strand on task vectors and weight arithmetic, and for #5144. No code, skill or protocol here depends on it. This page proposes no new issue.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 1 for this page's author, a model (Claude Opus 5.5). It read the source's own text in full, in five fetches: Sections 1 to 5, the references and Appendices A to G. Some maths came back garbled. We didn't see any figure as an image, so the bar heights of Figures 3 and 5 appear here only where the text gives them.

## Summary

**Claim.** After meta-learning, the parameter vector that one gradient step gives for each function adds and subtracts like a word vector. So $\theta_{[1,5]} + \theta_{[2,6]} - \theta_{[2,5]}$ runs the function $[A{=}1, B{=}6]$ that no one trained it on. The authors link this compositional geometry to rapid learning.

**Method.**

- **Meta-learning.** MAML with one inner gradient step on a batch of 100 sequences from the target function. The outer loop runs 10,000 steps for the transformer and 25,000 for the LSTM.
- **Networks.** A transformer with 4 layers, 8 heads and hidden size 64, and an LSTM with one layer of size 256. Both are small next-token models, and the authors didn't tune them.
- **The vectors.** After the one step, all of a network's parameters are flattened into one vector for that function.
- **Domains.** Three synthetic sequence languages: ABgeom sets two token values, ABparity sets two parities, and ABlength sets two counts.
- **The analogy test.** Build $\theta_p + \theta_g - \theta_f$, sample from it, and score the samples against the target function. Nothing trains on the target, and no step picks the nearest known vector.
- **Metric.** A top-25 F-score over 10,000 samples from the network and 10,000 from the target.
- **Controls.** Networks from a random start or from pre-training, adapted with the same steps or trained to convergence. Analogies that mix vectors from different starts.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| MAML transformer, ABgeom, 100 analogies, ten starts | "mean F-score = 1.00" |
| MAML transformer, ABgeom, direct adaptation, one step | mean F = 1.00 over 400 functions and ten starts |
| Random-start transformer, ABgeom, five steps | mean F = 0.062 |
| Random or pre-trained start, adapted, analogies | "mean F-score << 0.1 in every case" |
| Vectors from different starts, mixed in one analogy | "mean F-score ≈ 0" |
| MAML LSTM, ABlength, single analogy type | mean 0.35 |
| Best analogy type per start, averaged over ten | LSTM ABlength 0.8288, transformer ABparity 0.7881 |

## Where the claims go beyond the evidence

- **The claim that analogies hold better after MAML than after other starts rests on converged vectors.** The paper sets aside the adapted random and pre-trained analogies, because those vectors don't run their own functions (F below 0.1). So the comparison uses converged vectors only. It says "significantly" with no test, and it names an exception: random-start transformers on ABlength.
- **The best-type score is chosen on its own test set.** The mean-max score picks the best of up to three analogy types for each start, on the same 100 analogies that it then reports. The authors call it a lower bound, but it is not an independent estimate. No null takes the best of three on random analogies.
- **No random-direction control.** No test adds a random vector of the same norm, or scores the analogy against a wrong target. The cross-start control (about 0) shows that the vectors need a shared start. It doesn't show what a chance analogy scores.
- **The headline result is at ceiling.** On ABgeom, the analogy and direct adaptation both score 1.00. The text gives no value for each analogy and no threshold for success.
- **Small synthetic languages and small networks.** The authors say this themselves, and say that richer domains need new methods.
- **No seeds and few intervals.** The text names ten starts but no seeds. Only Figures 3 and 5 show ±1 SD bars, and the text gives no interval or test.

The authors state limits themselves. The analogy test can't rule out structure that it doesn't test. Whether the geometry causes rapid learning is left to later work. And they study only single-step adaptation.

## Use in this repo

- **It is task arithmetic under the most favourable conditions.** Meta-learning trains the start so that one step moves along composable directions. [Task arithmetic](task-arithmetic.md) adds fine-tuning updates with no such training. The [tangent-space page](tangent-space-task-arithmetic.md) says why that sometimes works. This paper shows the ideal case. It doesn't predict that two patches on Qwen or GPT-2 compose.
- **A direction isn't a unit.** The vectors are all of a network's parameters. The paper reads no parameter, neuron or head, and it runs no probe or ablation. Section 4.2 calls a link to circuits "one possibility." So a function vector here doesn't give a registry explanation for any unit. This supports #5144: an arithmetic fact about weights doesn't say what a single unit does.
- **The cross-start result bears on merges.** Vectors from different starts give F near 0 in an analogy. The patch format's merge rules (#5298) combine patches made on one base model, which is the shared-start case. A merge across base models would be the case that fails here.
- **Its controls are a pattern for #4782.** A random start and a pre-trained start, each adapted and converged, separate geometry from training. A patch-merge test could use the same arms: random patches, and patches from a different base.
- **Related pages.** [`task-arithmetic.md`](task-arithmetic.md) and [`tangent-space-task-arithmetic.md`](tangent-space-task-arithmetic.md) are the fine-tuning view of the same arithmetic. [`in-context-learning-by-gradient-descent.md`](in-context-learning-by-gradient-descent.md) is another case where a known construction lets weights name an algorithm.

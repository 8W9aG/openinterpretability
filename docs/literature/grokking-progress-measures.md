# Progress measures for grokking via mechanistic interpretability

**Citation.** Neel Nanda (independent researcher, corresponding author), Lawrence Chan (University of California, Berkeley), Tom Lieberum, Jess Smith and Jacob Steinhardt. The text doesn't give the last three authors' affiliations in a form we could read. "Progress measures for grokking via mechanistic interpretability". [arXiv:2301.05217](https://arxiv.org/abs/2301.05217), version 3 of 19 October 2023. Versions 1 and 2 are of 12 and 13 January 2023. The arXiv page gives no venue, so we report it as a preprint. The interactive figures, a Colab notebook and the checkpoints are linked from [neelnanda.io/grokking-paper](https://neelnanda.io/grokking-paper). The text gives no funding statement. It says Nanda's mentorship came from the Anthropic interpretability team "during his time there."

**Mismatch.** The introduction says that ablating the key frequencies "reduces performance to chance." Table 4 gives losses of 6.5 to 11, which are worse than the uniform loss, ln 113 ≈ 4.7. Section 4.2 writes "55 key frequencies" where it means 5. Appendix C.2.2 says a model on 20% of the data "generalized immediately". Figure 20's caption says 20% "doesn't lead to grokking even after 40k epochs."

**Status.** Planned. #5220 names it as the reference for a complete arithmetic explanation in its compiled-module phase. The fix campaign is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 3 for this page's author, a model (Claude Opus 5.5). It read the source's own text in full, in four fetches: Sections 1 to 6, the references and Appendices A to F. We didn't see any figure as an image. The progress measures over training appear only in Figure 7, so this page gives the phases as the text describes them.

## Summary

**Claim.** A small transformer that groks modular addition runs a Fourier algorithm, which the authors reverse-engineer fully. Measures built from that algorithm show that the generalising circuit forms gradually, long before the test loss falls.

**Method.**

- **Task.** (a + b) mod 113, trained on 30% of the 113 × 113 pairs. Test on the rest.
- **Model.** One layer, 4 heads, MLP width 512, no LayerNorm. Full-batch AdamW, weight decay 1, 40,000 epochs. Five seeds, with the analysis on one.
- **The algorithm.** The embedding maps a and b to sine and cosine of a few key frequencies. Attention and the MLP compute cos and sin of w(a + b) by the angle-addition identities. The output reads cos(w(a + b − c)) for each candidate c. Summed over the key frequencies, these waves add up only at c = a + b mod 113.
- **Evidence.** Periodic weights and activations, fits of each part to the algorithm's terms, and ablations in the Fourier basis.
- **Progress measures.** The restricted loss keeps only the key-frequency terms of the logits. The excluded loss removes them, on the training data. Also the sparsity of weights in the Fourier basis (a Gini coefficient) and the sum of squared weights.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Key frequencies, main run (Section 4) | k = 14, 35, 41, 42 and 52, five of the 56 |
| Neurons (Section 4.3) | 433 of 512 neurons (84.6%) have over 85% of their variance explained by one key frequency |
| Logits (Section 4.2) | 95% explained by the five terms cos(w(a + b − c)) |
| Replace the 433 neurons by their fits (Section 4.4) | loss 2.41e-7 → 2.48e-7, with accuracy unchanged |
| Keep only the key frequencies (Section 4.4) | loss falls 70%, to 7.24e-8 |
| Project the MLP onto the 10 key directions, or onto their complement (Section 4.4) | loss 1.19e-7, or 5.27, which is worse than uniform |
| Other seeds (Table 4) | 3 or 4 key frequencies each. Removing them gives loss 6.5 to 11 |
| Phases (Section 5.2) | memorisation to epoch 1.4k. Circuit formation 1.4k–9.4k, with train and test loss flat. Cleanup 9.4k–14k, when the test loss drops |
| Regularisation (Appendix D.1) | no weight decay, no grokking (one seed). Dropout groks. L1 never does |

## Where the claims go beyond the evidence

- **One task, one small model.** The full account is for one run of a one-layer model on one modulus. The authors say the measures are specific to small networks on this task, and that scaling is uncertain.
- **The progress measures were built from the answer.** Restricted and excluded loss use the key frequencies found after training. A measure for an unknown algorithm would need the algorithm first.
- **The phase result is qualitative in the text.** Figure 7 holds the curves. The text gives step ranges, not values.
- **No intervals.** Most results are single runs. Table 5 and Table 6 give a few standard deviations, and there are no tests.
- **Several internal errors.** The ones in *Mismatch*, plus a broken angle-addition identity in Section 3.1 and cross-references to the wrong table and section.

The authors state limits themselves. The analysis took a large manual effort. No theory predicts when the phase change happens.

## Use in this repo

- **The issue's question: what a complete arithmetic explanation looks like.** Every part is named: which frequencies, which neurons (433 of 512 by frequency), how heads and MLPs combine them, and how the output reads them. Each part is checked by a fit and by an ablation that keeps or removes it. Removing what the explanation names breaks the model, and keeping only that keeps the model working. That is the standard a compiled core in phase 5(b) of #5220 would be held to.
- **Its neuron result is a registry-style result.** "84.6% of neurons are explained by one frequency" is an explanation per unit, with a variance threshold. The registry's verdict needs probes and causal checks for each unit. This paper's checks are causal too, but over groups of units, by frequency.
- **It is the clean case that LLM arithmetic isn't.** The [trigonometry page](trigonometry-addition.md) finds the same Fourier idea in GPT-J, with periods 2, 5, 10 and 100, but no step that adds. The [bag-of-heuristics page](arithmetic-bag-of-heuristics.md) finds many partial rules. So the paper shows the limit of what a full explanation can be, in a model far smaller than the registry's.
- **The excluded loss is an ablation the repo can copy.** It removes an explained part and asks how much the training loss still has. Our reading: the registry's coverage measure keeps explained units and ablates the rest. The excluded loss is the reverse, and the two together bound what an explanation accounts for.
- **Over training, like the neuron-development paper.** [`papers/neuron-development`](../../papers/neuron-development/README.md) tracks registry units over Pythia checkpoints. This paper shows a circuit forming while the loss stays flat. So a unit's explanation can change before any behaviour does.
- **Question for the maintainer, recorded on this page only.** Should the registry report an excluded-loss measure beside coverage, so that an explanation is checked from both sides?
- **Related pages.** [`trigonometry-addition.md`](trigonometry-addition.md) and [`arithmetic-bag-of-heuristics.md`](arithmetic-bag-of-heuristics.md) study arithmetic in large models. [`tracr.md`](tracr.md) and [`alta.md`](alta.md) compile algorithms whose parts are known by construction.

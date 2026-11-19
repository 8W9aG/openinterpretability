# Tracr: Compiled Transformers as a Laboratory for Interpretability

**Citation.** David Lindner, János Kramár, Sebastian Farquhar, Matthew Rahtz, Thomas McGrath and Vladimir Mikulik, all of Google DeepMind. "Tracr: Compiled Transformers as a Laboratory for Interpretability". arXiv 2301.05062 [cs.LG]. Versions: v1 12 January 2023, v2 21 February 2023, v3 22 February 2023, v4 7 June 2023, v5 3 November 2023. We read v5. [Paper](https://arxiv.org/abs/2301.05062). [Code](https://github.com/google-deepmind/tracr). The v5 PDF says "37th Conference on Neural Information Processing Systems (NeurIPS 2023)". Issue #4843 also says the paper was a spotlight, and the PDF does not say this, so we did not check it. The paper releases no dataset, because its models are compiled from programs. It has no funding statement. The acknowledgements thank a LessWrong user for a mistake that they found in an earlier draft of Appendix F.

**Status.** Planned. #3584 (ground-truth calibration on Tracr and InterpBench models) names Tracr as a candidate and is open. The paper's own text supports that issue's reasons to start with InterpBench and not Tracr. It also gives #4782 a case where training keeps the behaviour but changes the encoding.

**Read.** We read the arXiv v5 PDF on 2026-11-19, as text that `pdftotext` took from it. We also read the arXiv abstract page for the version history. We have the source's own text for the abstract, §1 to §7, the references and Appendices A to G. We did not see the figures as images. The plots of Figures 7 and 8 came through only as axis labels. So the only numbers from them here are numbers that the text states. The residual-stream heatmaps of Figures 2, 5 and 9 to 11 came through only as their row labels. We did not read the code. A model (Claude Opus 5.5) did the reading.

## Summary

**Claim.** Tracr compiles programs in RASP, a language for transformer computations, into the weights of standard decoder-only transformers. So the authors know the model's mechanism before anyone studies it. The authors propose such models as ground truth to test interpretability methods, and as a setting to study superposition.

**Method.**

- **RASP.** Programs have sequence operations ("s-ops"), elementwise operations, and select-aggregate operations. The paper says elementwise operations "roughly correspond to MLP layers" and select-aggregate operations to attention heads. Tracr changes RASP in three ways (Appendix B):
  - it does not allow boolean combinations of selectors.
  - each s-op must be categorical or numerical.
  - every input must start with a beginning-of-sequence (BOS) token.
- **Compilation, in six steps (§3).** Tracr builds the program's graph and infers each s-op's set of values. It turns each node into an MLP or attention block, assigns blocks to layers by a heuristic, and stacks them. Then it writes the weight matrices.
  - Each s-op has "its own orthogonal subspace, which is reserved for its sole use throughout the entire network".
  - A categorical s-op is a one-hot vector in its subspace. A numerical s-op is the magnitude of one dimension.
  - Categorical MLPs "function as lookup tables". Numerical MLPs use piecewise linear approximations, whose first layer puts the input into buckets (Appendix D.1).
  - Attention takes only categorical keys and queries. The selector becomes W_QK, which is scaled by an inverse temperature of 100 by default. The aggregate becomes W_OV.
  - The model is a Haiku transformer with ReLU MLPs and no layer norm.
- **Programs.** `frac_prevs` gives the fraction of earlier "x" tokens. `sort_unique` sorts unique tokens. Appendix G adds `sort` with duplicates, `pair_balance` and `dyck-n`, which checks balanced brackets.
- **Compression (§5).** One matrix W maps the D-dimensional residual stream to d < D dimensions. The model applies Wᵀ when it reads and W when it writes. All other weights are frozen, and only W is trained. The loss adds the output loss and the squared difference of each layer's output from the compiled model.
  - Training uses AdamW with weight decay 0.1, β₁ = 0.9 and β₂ = 0.99, for 3 × 10⁵ steps at batch size 256.
  - The learning rate falls linearly from 10⁻³ to 10⁻⁶ over the first half. One run takes 1 to 4 hours on two CPU cores (Appendix E).
  - The baseline for W is PCA with the same number of components.
- **The check of the mechanism (§5.3).** The mean cosine similarity between each layer's output in the compressed and the compiled model.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Expressivity (§3, §4) | The authors compiled programs for all the tasks of Weiss et al. (2021), after they changed some programs to use only features that Tracr supports |
| `frac_prevs`, compiled (§4.1) | A 14-dimensional residual stream, of which 12 dimensions hold the input embeddings. One MLP layer and one attention head do the work, in a four-layer model with two layers that do nothing |
| `frac_prevs`, compressed (§5.2) | D = 14 reduces to d = 6 "without hurting performance". Below that, performance falls |
| `frac_prevs`, W at d = 8 (§5.2, Figure 7) | `tokens:x`, `is_x` and `frac_prevs` mostly have dimensions of their own. The compressed model drops `tokens:a`, `tokens:b` and `tokens:c`. The `indices` embeddings share dimensions that are not orthogonal. `is_x` and `tokens:x` share part of their space. PCA keeps features that the task does not need, such as `tokens:a` |
| `sort_unique`, compressed (§5.3, Figure 8) | Accuracy is "almost perfect", but the mean layer cosine "stays around 0.8". The authors trace the gap to `target_pos`, which the compressed model seems to encode as a number and not as a one-hot vector |
| Loss weights (§5.1) | Both loss terms have equal weight. The authors say other weights give similar results, and show none |
| Selectors (Appendix F, Lemma F.1) | With the encodings of Appendix D, an `or` of two selectors with separate inputs has a limit. It works only if one predicate is constant, or both are rank 1 |
| Later use (§6) | Conmy et al. (2023) used Tracr to test a method that finds circuits automatically. Friedman et al. (2023) learned transformer programs |

The authors state the limits themselves (§7, Appendix A.2):

- RASP can't express "probabilistic" computation, in which a component raises or lowers the probability of tokens. It uses only binary attention patterns.
- Every variable has its own subspace in every layer, which real models do not do.
- Features align with the residual stream's basis, which "makes the resulting models easy to interpret". Rotating the basis or compressing the model could hide this.
- Results on compiled models "do not necessarily generalise to real models". The authors say to treat a test on them "as a minimum bar rather than a full validation of a technique".

## Where the claims go beyond the evidence

- **"For the first time" rests on two programs.** §5.2 says that Tracr gives "a setting to systematically study superposition in transformer models that implement nontrivial algorithms". The paper compresses `frac_prevs` and `sort_unique`, and the text gives a W plot only for `frac_prevs` at d = 8. One compressed program at one d is a demonstration, not a systematic study.
- **The superposition is in the residual stream, not in neurons.** Only the shared W is trained, so each MLP's hidden units keep their compiled weights. The paper finds features that share residual dimensions. It shows nothing about features shared across MLP neurons, which is the case that Elhage et al. (2022) study in their privileged-basis models.
- **No seeds, intervals or random baseline.** Each compression result is one run as far as the text says. The only baseline for W is PCA. A random projection of the same size would show how much of the structure comes from training.
- **The paper's ground truth is not tested as ground truth.** The paper proposes Tracr to test interpretability methods, but it tests none. It names probes, gradient attribution and causal tracing as methods that future work could test. Conmy et al. (2023) is cited for the first such test.
- **Small inconsistencies.** §5.3 says "the compiled model only uses a single dimension" for `target_pos`, where the context says the compressed model. Lemma F.1 defines the key subspaces as K_A and K_B, but then says the column space lies in R_A + R_B. Neither changes a result.
- **Credit for the stated limits.** The authors give the main limit themselves. Compiled models are cleaner than trained ones, and a method could pass on them by using that cleanness. §5.3 also reports a result against the paper's own aim: compression can break the ground truth.

## Use in this repo

- **The repo already has a planted ground truth, and it chose neurons over Tracr for a reason.** #3201 planted 12 detector neurons in GPT-2 small by rewriting their input weights (`docs/analyses/ground-truth-calibration.md`). The gate kept 7 of 12. Every audited explanation named its plant: 6 *match*, 1 *partial*, 0 *miss*. VERIFIED was 5 of 7, and 7 of 7 after the probe-token check. #3201's Plan sent Tracr and InterpBench to #3584 for two reasons. The harness can't load them, and their mechanisms are circuits, not single neurons. The paper's text supports both reasons:
  - Tracr models are Haiku models with no layer norm. `NeuronHookRunner` (`openinterpretability/harness/hooks.py`) needs a Hugging Face layout.
  - Each RASP operation becomes a whole MLP block or attention head, not one neuron.
- **A compiled model is a weak positive control for the neuron audit.** The audit reads one MLP neuron pole over natural text. A Tracr model has a vocabulary such as {x, a, b, c}, a context of a few tokens, and no natural text. So the scan corpus and the reviewer's probes do not apply. Its MLP hidden units are lookup-table entries or buckets of one input value. In the audit's terms they are token or value detectors. #3201 calls these "the easy case", and its GPT-2 plants already test them on real text.
- **The paper's "minimum bar" argument applies to the repo's own nulls.** Tracr's features align with the basis by construction, so an audit that passes there shows only that it can read an aligned feature. The repo's rotated-basis null tests the same question from the other side (`docs/analyses/null-calibration.md`). Native positive poles of one model reach VERIFIED 18 of 30 times, and random directions 1 of 30. A test that a compiled model passes cannot add to that, because the compiled model has no other basis.
- **Context rules are in attention, as #5144 expects.** In Tracr a rule that needs context is a select-aggregate step, which is an attention head. Only elementwise rules are in MLPs. So even in a model whose weights are the program, one MLP unit's weights hold a rule on its inputs, and context comes from attention. This fits the argument of `fast-weights-recent-past.md`, and the #3201 gate that dropped all three context plants. #5144 counts how many VERIFIED explanations are token sets. That count stays the ceiling for any weight-only check of one neuron.
- **Reading weights is exact here only because the basis is labelled.** Tracr's weights can be read as the program because each s-op has its own labelled subspace. Trained models have no such labels. ROTATE's kurtosis statistic, read from weights alone, did not predict VERIFIED (`docs/analyses/weight-kurtosis.md`). The AUC is 0.51 (0.45–0.57) on the output side and 0.54 (0.48–0.60) on the input side, on 518 test poles. So Tracr shows that weight reading can be exact, but not that it works on the registry's models (#3446).
- **The superposition finders need a ground truth in neurons, and compressed Tracr does not give one.** #5136 asks how many true feature directions the two finders in `openinterpretability/superposition_finders.py` recover. A superposition in this repo is a sparse direction over one layer's MLP neurons (`docs/PROTOCOL.md` §1). Tracr's compression puts features together in the residual stream and leaves MLP units alone. So the toy model that #5136 names is the right test, and compressed Tracr is not.
- **Evidence for #4782: training can keep the behaviour and change the encoding.** In §5.3, compression of `sort_unique` reaches almost perfect accuracy while the layer outputs reach a cosine of about 0.8. The model moved one variable from a one-hot code to a number. Only W was trained. This fits `papers/neuron-development/`, where Pythia-70m's 86 VERIFIED poles keep changing after the loss settles. It is one program, so it shows the effect can occur, not how often.
- **Weight-space programming (#3381, #4471).** Tracr writes weights by hand from a program, with no gradient descent. #4471 (Mehta et al. 2026) has an agent write the weights. Both give a model whose mechanism is known before the audit. Tracr's §5.3 warns that any training after the writing can break that knowledge. #5066 (gradient routing to one neuron) is the repo's other path to a known neuron, and it trains the model with the meaning in place.
- **Proposed follow-up: none.** #3584 already holds the Tracr idea, and its plan to start with InterpBench is sound. We propose a comment on #3584 with this page's evidence. A compiled model would test circuit recovery, which the registry does not do yet. Its MLP units are the token-detector case that #3201 already covers.
- **Question for the maintainer.** Should the repo count a positive control whose features align with the basis by construction, for example a compiled model? The authors say such a test is "a minimum bar". If yes, it should be reported apart from #3201, with that limit stated. This is a question, and we open no issue for it.

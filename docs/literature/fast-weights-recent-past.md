# Using Fast Weights to Attend to the Recent Past

**Citation.** Jimmy Ba (University of Toronto), Geoffrey Hinton (University of Toronto and Google Brain), Volodymyr Mnih, Joel Z. Leibo and Catalin Ionescu (Google DeepMind). "Using Fast Weights to Attend to the Recent Past". arXiv 1610.06258 [stat.ML], v1 20 October 2016, v2 27 October 2016, v3 5 December 2016. We read v3. [Paper](https://arxiv.org/abs/1610.06258). The v3 comment says that it adds a Schmidhuber (1993) citation and fixes a typo in the initialisation of Appendix A.1. Issue #4836 gives the venue as NeurIPS 2016. Neither the arXiv page nor the PDF names a venue, so we did not check it. The paper links no code and no data, and it has no funding or acknowledgements section.

**Status.** Planned. #4782 (catastrophic interference, weight-space programming and LoRA) would use it in its synthesis on weight-space programming. The paper gives no method that applies to a transformer's units. It does give the argument for why a weight-only reading can't check a rule that depends on context. The page proposes one follow-up issue that measures how much of the registry that limit covers.

**Read.** We read the arXiv v3 PDF on 2026-11-19, as text that `pdftotext` took from it. We have the source's own text for the abstract, §1 to §5 and the references. We also have Appendix A (experimental details) and Appendix B (the inner loop in biological networks). The values of Tables 1 to 3 came through as text. We also read the arXiv abstract page for the version history. We did not see the figures as images. Figures 2 and 5 are learning curves, and only their axis labels came through, so no number from them is here. A model (Claude Opus 5.5) did the reading.

## Summary

**Claim.** A recurrent network can keep a temporary memory of the current sequence in "fast weights", which change faster than the learned weights and decay. This memory acts as attention to the recent past. It needs no stored copies of past activity. With layer normalisation, it makes small RNNs better on four tasks.

**Method.**

- **The fast weights.** A matrix A decays at rate λ and adds the outer product of each hidden state: A(t) = λA(t − 1) + ηh(t)h(t)ᵀ (eq. 1). The slow weights W and C learn by gradient descent. A is not learned: it follows this fixed rule.
- **The inner loop.** Each step first computes h₀(t + 1) = f(Wh(t) + Cx(t)). Then it runs S steps of h_{s+1} = f(LN[Wh(t) + Cx(t) + A(t)h_s]) (eq. 5). LN is layer normalisation, and f is ReLU.
- **The attention form.** With A = 0 at the start of a sequence, A(t)h_s is a sum over past states h(τ). The scalar product h(τ)ᵀh_s and the factor λ^(t − τ) weight each term (eq. 4). So the model never has to build A, and training can use mini-batches.
- **Settings.** η = 0.5 and λ = 0.95 in §3.1 "unless otherwise noted". Appendix A.1 gives λ = 0.9 for the key–value task. Mini-batches of 128, Adam, and hyperparameters by grid search on the validation set.
- **Tasks and baselines.**
  - **Key–value recall (§4.1).** A sequence of letter–digit pairs, then a query letter. The model must give that letter's digit, for example `c9k8j3f1??c` → 9. There are 100,000 training, 10,000 validation and 20,000 test sequences. The baselines are IRNN, LSTM and associative LSTM (A-LSTM), with the same architecture.
  - **MNIST with glimpses (§4.2).** 24 patches of 7 × 7 pixels at two scales, in a fixed order. A control signal tells the model when to store. The baselines are IRNN, LSTM and a three-layer ConvNet on the same patches.
  - **Facial expression (§4.3).** CMU Multi-PIE, 48 × 48 greyscale, six expressions, three central cameras. 317 identities train and 20 test. There are 200 recurrent units, and 24 glimpses of 12 × 12 pixels.
  - **Partly observable Catch (§4.4).** A3C agents with 128 recurrent cells, where the screen goes blank after frame M. The paper averages the results over the top 5 models of a random search.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Key–value test error, R = 20 (Table 1) | IRNN 62.11%, LSTM 60.81%, A-LSTM 60.13%, fast weights 1.81% |
| Key–value test error, R = 50 | IRNN 60.23%, LSTM 1.85%, A-LSTM 1.62%, fast weights 0% |
| Key–value test error, R = 100 | IRNN 0.34%, LSTM 0%, A-LSTM 0%, fast weights 0% |
| MNIST error, 50 features (Table 2) | IRNN 12.95%, LSTM 12%, ConvNet 1.81%, fast weights 7.21% |
| MNIST error, 100 features | IRNN 1.95%, LSTM 1.55%, ConvNet 1.00%, fast weights 1.30% |
| MNIST error, 200 features | IRNN 1.42%, LSTM 1.10%, ConvNet 0.9%, fast weights 0.85% |
| Multi-PIE test accuracy (Table 3) | IRNN 81.11, LSTM 81.32, ConvNet 88.23, fast weights 86.34 |
| Catch, N = 16, M = 3 and N = 24, M = 5 (Figure 5) | In the text: the fast-weights agent learns faster, and the gain is larger on the larger game. No numbers in the text |
| Key–value task, more inner-loop steps (Appendix A.1) | "the performance are similar" |
| Inner loop, method 1 against method 2 (Appendix B) | Method 1 "worked slightly better" with layer normalisation |

## Where the claims go beyond the evidence

- **One run per cell, with no spread.** No table gives seeds, intervals or repeats. The Catch result is a curve averaged over the top 5 of a random search, so the outcome chose which runs count.
- **The comparison does not hold memory equal.** Fast weights add a memory of size O(H²) and an inner loop to each step. The baselines with the same H have O(H) memory, apart from A-LSTM. So "uses its recurrent units more effectively" is not separated from "has more memory".
- **Table 1 doesn't define R.** The text says that fast weights win "when the number of recurrent units is small", so R is probably that number. Figure 2's caption says 50 units.
- **The decay rate differs between sections.** §3.1 gives λ = 0.95 for all models "unless otherwise noted". Appendix A.1 gives λ = 0.9 for the key–value task, with no note in §4.1.
- **The visual tasks make memory easy.** The glimpse order never changes, and a control signal says when to store. The authors say so, and they say that this removes a main advantage of attention models. On Multi-PIE the ConvNet scores higher (88.23 against 86.34), as the authors also say.
- **The recursion claim has no experiment.** The conclusion says that the paper has "shown how" knowledge can be applied recursively. §4.2 says that fast weights store at "a single level" only. No task tests recursion.
- **Three of four tasks are small or synthetic.** The key–value task is one "to which it is very well suited", in the authors' words. The paper tests no language-model task.
- **No code.** The paper links no code, and Appendix A gives only part of the settings for §4.2 and §4.3.

The authors state several limits themselves. These are the toy fit of §4.1, the fixed glimpse policy, the single level of storage and the ConvNet's better Multi-PIE score.

## Use in this repo

- **Eq. 4 links fast weights to attention.** A weight matrix built from the current sequence acts as attention to past states, weighted by scalar product and decay. The four configured models (`pyproject.toml`) have no fast weights. Their memory of the current text is softmax attention. So what Ba et al. keep in A(t), these models keep in attention patterns, which exist only during a forward pass.
- **The argument limits a weight-only check.** Static weights hold the slow part of a unit's rule. A check from weights alone can test a claim only if the slow weights hold the rule. An example is a token set on the unit's input side. A rule that depends on context passes through attention, the analogue of A(t), and needs the text. The paper argues this for RNNs and doesn't test transformers. So this is an inference, not the paper's result.
- **The repo's evidence agrees with the limit.** In `docs/analyses/ground-truth-calibration.md`, plants written as input weights passed the gate for 7 of 12 rules, and for no rule that needs context. "Inside quotes" reached precision 0.59 and recall 0.50, and "after a title" 0.88 and 0.64. That gate tests one linear read of one layer, not attention as such. So it agrees with the limit but doesn't prove it.
- **ROTATE's cheap statistic did not predict VERIFIED (#3446).** `docs/analyses/weight-kurtosis.md` computes ROTATE's vocabulary kurtosis from the weights alone (`scripts/weight_kurtosis.py`). On 518 test poles, the AUC is 0.51 (0.45–0.57) for the output side and 0.54 (0.48–0.60) for the input side. That page already gives one reason: a pole can reach VERIFIED on a context rule that no token set describes. This paper supplies the reason why such a rule is not in the static weights.
- **Head weights are static too.** A head's QK and OV matrices are slow weights. #5062 (an OV vocabulary lens) and #3457 (head type from weights alone) would read them. They can say what a head can attend to and write. They can't say which earlier token a given text makes it attend to. Head audits measure that from a forward pass, with `pattern_scores` in `openinterpretability/harness/heads.py`.
- **The key–value task is close to an induction probe.** `c9k8j3f1??c` → 9 needs a lookup of the token after an earlier copy of the query. Head audits score induction on repeated random tokens, and #5061 tracks that score over Pythia-70m checkpoints. The paper's task could be a probe for such heads, but no open issue asks for it.
- **Weight-space programming (#3381, #4471).** Ba et al. write weights at run time with a fixed Hebbian rule, and they learn only the slow weights. Nothing in A is meant to be read. Each write ηh(t)h(t)ᵀ has rank 1, the same form as one example's weight gradient in MEND (`docs/literature/fast-model-editing.md`). The repo's own weight writing is the ground-truth plants (`scripts/ground_truth.py`). They write one neuron's input weights, not a rule that runs at inference.
- **No single-neuron content.** The paper gives no unit explanations, no probes and no null. Nothing in it bears on any pole's verdict.
- **Proposed follow-up.** The repo has no count of how many VERIFIED explanations are token sets and how many need context. That count is the ceiling for any weight-only check, ROTATE included. The follow-up asks for it, as evidence for #3446 and #4782.
- **Question for the maintainer.** Could a weight-only reading ever count toward a verdict, for example to contest a token-set claim with no forward pass? That would change the decision rule, so it would need a `PROTOCOL_VERSION` bump (`docs/PROTOCOL.md` §6). We record it as a question only.

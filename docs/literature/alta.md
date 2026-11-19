# ALTA: Compiler-Based Analysis of Transformers

**Citation.** Peter Shaw, James Cohan, Jacob Eisenstein, Kenton Lee, Jonathan Berant and Kristina Toutanova. Cohan is at Google, and the others are at Google DeepMind. "ALTA: Compiler-Based Analysis of Transformers". arXiv 2410.18077 [cs.LG]. Versions: v1 23 October 2024, v2 19 June 2025. We read v2. [Paper](https://arxiv.org/abs/2410.18077). [Code](https://github.com/google-deepmind/alta), under Apache 2.0. The v2 PDF says "Published in Transactions on Machine Learning Research (2/2025)", with an [OpenReview forum](https://openreview.net/forum?id=h751wl9xiR). The arXiv comment says "TMLR 2025". So the issue's "TMLR 2025 (arXiv 2024)" agrees with the source. The paper releases no dataset, because its models are compiled from programs, and its SCAN data are public. It has no funding statement. The acknowledgements thank David Lindner, an author of Tracr, among others.

**Status.** Not applicable. The compiled models run only in the authors' own NumPy forward pass, and the harness can't load them. No open issue names ALTA. Its evidence goes to open issues as comments. #3584 and #3593 are on positive controls, and #5144 is on context rules. #3446 is on reading weights, and #4782 on units that training does not reach.

**Read.** We read the arXiv v2 PDF on 2026-11-19, as text that `pdftotext` took from it. We also read the arXiv abstract page for the version history and the venue comment. We have the source's own text for the abstract, §1 to §5, the references and Appendices A to D. We read Appendix B's definitions, theorem statements and the proof of Theorem 1, but only skimmed the proof of Theorem 2. We did not see the figures as images. The plots of Figures 4, 5, 9, 10, 11 and 18 came through only as axis labels and legends. So the only numbers from them here are numbers that the text or a table states. The code listings came through as text. We read the code repository's README, `setup.py` and `framework/transformer/transformer_utils.py`, and its file list. We did not run the code. A model (Claude Opus 5.5) did the reading.

## Summary

**Claim.** ALTA is a programming language with an interpreter and a compiler that writes a program into transformer weights. Unlike Tracr, it supports loops, through layers that share weights, and it compiles to Universal Transformers. The authors use it to show constructions for parity, addition and SCAN, and to study why training fails to learn such algorithms.

**Method.**

- **Programs.** A program has variables, attention heads and an "MLP function" (§2). Variables are categorical, numerical or sets. Each value of a categorical variable is one basis vector of the residual stream, a one-hot code. A set variable is multi-hot, and a numerical variable is one dimension.
- **Attention.** Each head names a query, key, value and output variable. The selection matrix is binary, and the query and key must match exactly, with no custom predicate. A hyperparameter λ, 100 by default, makes the softmax close to binary (Appendix A.2). Relative position masks compile to T5-style position biases.
- **The MLP as rules.** The MLP function is a set of transition rules. Each rule is "a logical implication with a conjunctive antecedent": if all its conditions hold, one output variable takes a new value.
  - The compiled MLP has four layers with clipped ReLU. The first two turn numerical and set variables into one-hot codes, and a program with only categorical variables does not need them.
  - In the last two layers, "the hidden activations are a binary vector where each value corresponds to whether a particular transition rule was satisfied". Each rule's bias is 1 − N, where N is the number of its conditions.
  - So the MLP width equals the number of rules. The parity program has 7 rules, and so 7 hidden units (Appendix A.2, D.1).
- **The model.** An encoder, with all attention and MLP weights shared across layers and no layer norm. A program may halt each position when a variable reaches a set value, as in the Universal Transformer. Appendix A.3 says a decoder needs only a causal mask and a decoding loop, and leaves it to future work.
- **Trace supervision (§3.2, Appendix C.1).** The interpreter records each MLP's input and output for every position and layer. A new MLP is trained with an L2 loss to map those inputs to those outputs. All other weights come from the compiler.
- **Minimal programs (§3.2, Appendix B).** A rule set is minimal for a data set if removing any rule, or any condition of a rule, changes some output on that data. Appendix C.2 builds the minimal version of a program in two ways.
  - It removes the rules that the training inputs never satisfy.
  - It sets embeddings for unseen tokens and positions to defaults.
  - It assumes that no rule has an unneeded condition.
- **Tasks.** Parity with three programs: Sequential (Absolute), Sequential (Relative) and Sum + Modulo. Addition of two integers, an interpreter for SUBLEQ, and SCAN, including its length split and its three MCD splits.
- **Baselines.** Transformers trained end to end on the same data, with 8 or 40 layers for parity and 2 to 256 layers for SCAN. Weight sharing and absolute or relative positions are varied.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Theorems 1 and 2 (Appendix B.6) | The loss is an L1 reconstruction loss plus an L1 weight penalty. If a rule set is not minimal for the data, the compiled MLP is not a strict coordinate-wise local optimum of it. If the rule set is minimal, it is one, for penalty α < 1 |
| Proof of Theorem 1 (Appendix B.7) | A rule that the data never satisfies has output weights that "do not affect the reconstruction loss". Only the penalty moves them |
| Parity, compiled originals (§4.1) | "100% accuracy", the same as the symbolic program |
| Parity, trace supervision (Table 2), lengths ≤ 20 / > 20 / ones > 20 | Sequential (Relative) 100% / 100% / 100%. Sequential (Absolute) 100% / 52% / 51%. Sum + Modulo 100% / 78% / 50% |
| Parity, minimal programs (§4.1) | Only the minimal Sequential (Relative) program works at all lengths. The authors say the trace-supervision results "exhibit the same generalization behavior as the minimal programs" |
| Parity, end to end (Table 3) | 100% on lengths ≤ 20 in all eight settings. 49% to 65% on lengths > 20, and 48% to 52% on more than 20 ones |
| Parity, rule counts (§4.1) | The minimal Sequential (Relative) program has 7 rules, and the minimal Sum + Modulo program has 31. End-to-end models act like Sum + Modulo, so there is no simplicity bias by rule count |
| Addition (§4.2, Appendix D.2) | 884 rules and 6 heads. N + 2 layers for N digits. The compiled model scored 100% on sampled pairs of 1 to 50 digits. The minimal program for inputs of ≤ 3 digits works for any length |
| SUBLEQ (Appendix D.3) | 5 heads, one instruction every 3 layers, and O(N³) rules for N memory values. Checked "using unit tests involving simple SUBLEQ programs" |
| SCAN (§4.4, Appendix D.4) | The program and the compiled model score 100% on all of SCAN. Fewer than 2,000 MLP units and 13 heads, against more than 10⁶⁰ variable combinations. Up to 512 layers |
| SCAN, minimal program (§4.4) | 100% on the length split and the MCD splits |
| SCAN, end to end (Table 6) | 1% to 13% on every split and depth. More layers do not raise accuracy |

The authors state the limits themselves (§2.1, §5, Appendix D):

- "the range of the ALTA compiler is a relatively small subspace of all possible parameter values". Numeric computation is limited, and probabilistic outputs are not supported.
- "The properties of compiled models may not reflect those of models learned in practice."
- Trace supervision trains only the MLP, and the theory covers only MLP weights, coordinate-wise optima and categorical variables. Extensions are "left to future work".
- A local optimum exists when the program is minimal, "although we are not guaranteed to find this point during training".
- Sequential (Relative) needs N − 1 layers for N inputs, where the parity construction of Chiang and Cholak (2022) needs 2.

## Where the claims go beyond the evidence

- **The learnability agreement rests on one task.** §3.2 says a model trained with trace supervision "is more likely to generalize if the minimal version of the program generalizes". The empirical check is parity, with three programs and one train and test split. Addition and SCAN have a minimal-program analysis but no trace supervision.
- **The agreement is in behaviour, not in units.** The trace-supervised MLPs have 128 or 4,096 hidden units per layer (Table 1), where the compiled MLP has one unit per rule. The paper compares accuracy only. It does not ask whether the trained units match rules.
- **The minimal programs used in experiments are only partly minimal.** Appendix C.2 removes rules that never fire, but it assumes that no rule has an unneeded condition. Definition 1 also asks for that second condition. So the experiments test half of the definition.
- **The theory is local and narrow.** Theorems 1 and 2 are about coordinate-wise optima of an L1 reconstruction loss, for one MLP with categorical inputs. Training uses L2 loss and Adam or Adafactor. The authors state these limits themselves.
- **No seeds, intervals or test-set sizes.** Tables 2, 3 and 6 give one number per setting. The text does not say how many runs each is, or how many test examples. The parity test set has "roughly an equal number of examples per number of ones".
- **The simplicity-bias claim is a guess from two tasks.** §4.1 says end-to-end models follow "the algorithm that requires the fewest layers". Parity and SCAN show that more layers do not help. The authors call the layer explanation a direction "for future work", and Appendix D.4 says "We speculate".
- **The interpretability use is proposed, not tested.** §5 says ALTA "could also potentially help develop test cases for interpretability tools". The paper tests no interpretability method on a compiled model.
- **A small inconsistency.** Table 3 and the legend of Figure 10 give 8 and 40 layers for parity. The legend of Figure 9 says "6 Layers" and "40 Layers". It changes no result in the text.
- **Credit for the stated limits.** The authors state that compiled models may differ from trained ones, and they report negative results for end-to-end training at length.

## Use in this repo

- **Here, unlike Tracr, one MLP unit has a known role.** Each hidden unit in ALTA's last two MLP layers is one transition rule. It reads 1 when all the rule's conditions hold, and 0 otherwise.
  - Tracr compiles an MLP to a lookup table, and RASP sets no unit's role (`tracr.md`, `thinking-like-transformers.md`).
  - So ALTA is the first compiler in this strand where a single neuron's explanation is known by construction. The explanation is a conjunction of variable values.
- **It still does not load in the harness.** The code runs its own "minimalist numpy Transformer forward pass" (`framework/transformer/transformer_utils.py`). Its dependencies are TensorFlow, JAX, NumPy, SciPy and Apache Beam. Its file list has no PyTorch or Hugging Face export. `NeuronHookRunner` (`openinterpretability/harness/hooks.py`) needs a Hugging Face layout. A port would also need these changes:
  - a layer norm in every block, which a GPT-2 layout has and ALTA's one-hot codes do not expect.
  - a causal mask, which ALTA leaves to future work.
  - a four-layer MLP with clipped ReLU, where GPT-2 has two layers with GELU.
  - one set of weights used for many layers.
- **Even loaded, the audit's inputs would not fit.** The audit scans natural text, and the reviewer writes English probes. An ALTA model reads a few symbols, for example parity's input range of 3. So, as for Tracr, a compiled model would be a weak positive control for the neuron audit. #3584 already holds the compiled-model idea, and its plan to start with InterpBench is still sound. ALTA's rule units are a point for that issue as a comment: of the compilers reviewed, only ALTA sets single neurons.
- **The rule unit is a context plant, and it shows what a context plant needs.** A rule such as "parity = 0 and parity_left = 1" uses a value that an attention head copied from the left neighbour. The neuron's weights hold only the conjunction. The context comes from the head.
  - In `docs/analyses/ground-truth-calibration.md`, the plants rewrite only input weights. All three context plants failed the gate: "inside quotes" reached held-out precision 0.59 and recall 0.50.
  - ALTA's design suggests writing the head together with the neuron, for example a previous-token head and a neuron that is the AND of two values. That is a design point for #3593, as a comment, and not a new issue.
- **ALTA supports #5144's split.** A rule whose conditions name only token values is a token-set rule. A rule that names an attention output needs context. In ALTA, only the first kind can be read from one unit's input weights alone. #5144 counts the VERIFIED explanations of each kind, and that count stays the ceiling for any weight-only check.
- **Reading weights is exact here only because the basis is labelled (#3446).** ALTA's weights read as rules because each variable value has its own named dimension. Trained models have no such labels. ROTATE's kurtosis, read from weights alone, did not predict VERIFIED (`docs/analyses/weight-kurtosis.md`). On 518 test poles, the AUC is 0.51 (0.45–0.57) for the output side and 0.54 (0.48–0.60) for the input side.
- **Evidence for #4782: training leaves silent units free.** The proof of Theorem 1 says that the output weights of a rule that never fires on the data "do not affect the reconstruction loss". Only the weight penalty moves them. The page's view is that the same can hold under fine-tuning.
  - A unit whose condition never occurs in the new data gets no loss gradient. Weight decay can still move it.
  - So whether a VERIFIED unit keeps its explanation may depend on whether the new data activates it.
  - `papers/neuron-development/` found that Pythia-70m's VERIFIED poles keep changing after the loss settles. At step 16,000 the median correlation of 85 poles with their final activations is 0.62. Adjacent checkpoints correlate at a median of 0.78 to 0.82.
  - ALTA's theorem is about one compiled MLP with an L1 loss, so this is an argument and not a measurement.
- **Dead neurons are under-determined, not only idle.** The `neuron-family-triage` skill skips units that are "Never active on a broad corpus" and records them. ALTA's Theorem 1 gives a reason to record them: a unit that the training data never activates has weights that the data do not set. This changes no rule.
- **Weight-space programming (#3381, #4471).** ALTA writes weights from a program, as Tracr does, and #4471 has an agent write them. Trace supervision is a mixed case: the compiler writes attention and embeddings, and training writes the MLP. On parity it reached 100% at all lengths for one program, and 52% and 78% beyond length 20 for the other two (Table 2). That is evidence about which parts training can learn. It is not a fix method, and the fix campaign stays the maintainer's.
- **No single-neuron content in the experiments.** The paper has no neuron explanations, no probes and no null. Nothing in it bears on any pole's verdict.
- **Proposed follow-up: none.** A port of ALTA to the harness would give rule neurons with known roles. But their vocabulary has a few symbols, and the audit's text scan cannot use it. #3584 holds that idea. We propose comments on #3584, #3593, #5144, #3446 and #4782 with this page's evidence.

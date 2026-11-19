# Thinking Like Transformers (RASP)

**Citation.** Gail Weiss (Technion), Yoav Goldberg (Bar Ilan University and Allen Institute for AI) and Eran Yahav (Technion). "Thinking Like Transformers". arXiv 2106.06981 [cs.LG], v1 13 June 2021, v2 19 July 2021. We read v2. [Paper](https://arxiv.org/abs/2106.06981). Venue: Proceedings of the 38th International Conference on Machine Learning (ICML 2021), PMLR 139, as the PDF states. This agrees with issue #4842. Code: a RASP read-evaluate-print loop at [github.com/tech-srl/RASP](https://github.com/tech-srl/RASP), with a link to the replication code. Funding: European Research Council, Horizon 2020 agreement #802774 (iEXTRACT).

**Status.** Planned. #4782 would use it in its synthesis on weight-space programming. #3584 names Tracr-compiled transformers as ground truth. RASP itself gives no weights and says nothing about single MLP neurons. So no code, skill or protocol rule can depend on it directly.

**Read.** We read the arXiv v2 PDF on 2026-11-19, as text that `pdftotext` took from it. We have the source's own text for the abstract, §1 to §6 and the references. We also have Appendix A (results and training details) and Appendix B (the RASP programs). The program listings in Figures 1 to 15 came through as text. We also read the arXiv abstract page for the version history. We did not see the figures as images, so we did not see the attention heatmaps. The symbols of Table 1's "Matches?" column came through garbled, so this page does not report that column. A model (Claude Opus 5.5) did the reading.

## Summary

**Claim.** The authors propose a programming language, RASP, as a computational model of the transformer encoder. Attention maps to a select–aggregate pair, and the feed-forward sublayer maps to elementwise operations. A RASP program then gives an upper bound on the layers and heads that a transformer needs for a task. A transformer can be trained to follow such a program.

**Method.**

- **The language.** The inputs are two sequences, `tokens` and `indices`. Every value is a sequence of the input's length (an "s-op") or an n × n boolean matrix (a "selector").
  - `select(keys, queries, predicate)` makes a selector. It marks each query–key pair for which the predicate holds.
  - `aggregate(selector, values)` averages the selected values at each position. It is the only way to move information between positions.
  - Elementwise operations act at each position. They stand for the feed-forward sublayers, which the paper does not restrict "in any meaningful way", by the universal approximation result of Hornik et al. (1989).
  - `selector_width` counts the positions a selector marks. Appendix B builds it from `select` and `aggregate`. It needs one head with a beginning-of-sequence (BOS) token, and two without one.
  - The language has no loops that depend on the input.
- **Compilation.** Each aggregation becomes one attention head, at a layer after all its inputs. Each elementwise operation goes in the earliest layer that has its inputs. The compiler gives layers and heads only. The paper says that compilation to concrete weights, for example MLP weights for the elementwise operations, is out of its scope. Footnote 7 says that RASP does not give the embedding width either.
- **Tasks.** Reverse, histogram with and without BOS, double histogram, sort, most-frequent and Dyck-1 and Dyck-2 prefix classification (PTF). Dyck-2 uses an extra operator, `select_best`, that is not in pure RASP. A pure RASP version compiles to 4 layers with width at most 2.
- **Training.** Transformers with embedding dimension 256 and feed-forward dimension 512, Adam, no dropout and 100 epochs. Each task gets 4 runs, with learning rates 0.0003 and 0.0001 and decays 0.98 and 0.99. Sequences have length 0 to 100. Train, validation and test sets have 50,000, 1,000 and 1,000 sequences. Inputs are uniform over the alphabet and length, except that the Dyck generator favours legal prefixes.
- **Three tests.**
  - Upper bound: train at the compiled size, and report the best of 4 runs.
  - Tightness: train at (L, H), (L − 1, H), (L, H − 1) and (L − 1, 2H), and report the average test accuracy.
  - Feasibility: train one transformer per task, for 250 epochs, with an extra MSE loss between its attention and the compiled pattern. The paper scales that loss by 100.
- **Baselines.** The smaller transformers of the tightness test. There is no model of other architecture, and no null.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Compiled size (L, H) and best test accuracy (Table 1) | Reverse 2, 1: 99.99%. Hist BOS 1, 1: 100%. Hist no BOS 1, 2: 99.97%. Double hist 2, 2: 99.58% |
| Same, continued | Sort 2, 1: 99.96%. Most freq 3, 2: 95.99%. Dyck-1 2, 1: 99.67%. Dyck-2 3, 1: 99.85% |
| Average accuracy at L, H / H − 1 / L − 1 / L − 1, 2H (Table 2) | Reverse 99.9 / – / 23.1 / 41.2. Hist 99.9 / 91.9 / – / –. 2-hist 99.0 / 73.5 / 40.5 / 83.5 |
| Same, continued | Sort 99.8 / – / 99.0 / 99.9. Most freq 93.9 / 92.1 / 84.0 / 90.2 |
| Same, continued | Dyck-1 99.3 / – / 96.9 / 96.4. Dyck-2 99.7 / – / 98.8 / 94.1 |
| Sort with one layer fewer (§5) | Not worse, and best in that case. The heads look nearly uniform. The authors read this as a bucket sort that uses the bounded alphabet |
| Reverse, trained without attention supervision (Figure 4 caption) | The second-layer head matches the `flip` selector. The first-layer head prefers the last position, so it computes length in a different way from RASP |
| Hist BOS, without attention supervision (Figure 5) | The text reports a "strong similarity" between compiled and learned patterns |
| Attention-supervised double hist (Figure 1 caption) | 99.9% test accuracy, with 2 layers and 2 heads |
| Attention-supervised sort and most freq (Appendix A.1) | Sort 99.0%, alphabet 52. Most freq 99.9%, alphabet 26. All three supervised models reach "99+%" |
| Shuffle-Dyck-2 (Figure 3) | The program compiles to 2 layers and 3 heads, the same as the hand-built transformer of Bhattamishra et al. (2020) |
| Restricted attention (§4) | A RASP sort for any length and alphabet uses Ω(n log n) of each head's n² pairs. The authors infer that attention with o(n log n) pairs loses expressive power |

## Where the claims go beyond the evidence

- **The bound covers heads and layers, not width or weights.** The abstract says that a program implies the heads and layers "necessary to encode a task". The compiler sets no embedding width, no MLP size and no weights. The authors say both limits themselves (footnote 7 and §3.1).
- **The conclusion says "minimum" where the abstract says "maximum".** §6 says that a program gives "the minimum number of layers and maximum number of heads". A program gives an upper bound on both. A smaller transformer can still solve the task, as sort shows in Table 2.
- **The tightness test is clear for three tasks only.** Reverse, histogram and double histogram drop sharply below the compiled size. Most freq loses at most 9.9 points. Dyck-1 loses 2.4 to 2.9, and Dyck-2 0.9 to 5.6. Sort loses at most 0.8, and gains at (L − 1, 2H). The authors name sort as the exception and do not discuss the others.
- **No spread over the 4 runs.** Table 1 gives the best run and Table 2 the mean. Neither gives the spread or the other runs. The feasibility test has one run per task.
- **The authors judge the pattern match by eye.** The paper scores the attention match only by looking at heatmaps. It gives no number for the match, and no random-head baseline.
- **Supervision makes the feasibility test weak evidence about natural training.** The extra loss trains the supervised models to copy the compiled attention. So these models show that a transformer can hold the program. They don't show that training finds it, and Reverse without supervision found a different length head.
- **Encoder only, synthetic tasks.** Every task uses an encoder, uniform random inputs of at most 100 tokens and alphabets of 3 to 100 symbols. The paper doesn't test a decoder with a causal mask or natural text. The authors say that RASP is not meant for a language model.
- **Dyck-k for all k uses an operator outside RASP.** The 3-layer Dyck-2 result needs `select_best`. The authors say so in footnote 8, and they give the pure version at 4 layers.
- **The symbolic-reasoning program is not evaluated.** §4 says that a RASP program does set inference like Clark et al. (2020). The paper gives no test of it, and points to the repository.

The authors state several of these limits themselves. These are the missing weights and embedding width, the sort exception, the extra Dyck-2 operator and the scope.

## Use in this repo

- **A RASP program can't be a positive control for a neuron.** RASP maps all MLP work to unrestricted elementwise functions, and it compiles to heads and layers only. So a program sets no neuron's role. The repo's neuron controls write weights directly instead. In `docs/analyses/ground-truth-calibration.md`, the plants rewrite one GPT-2 neuron's input weights (`scripts/ground_truth.py`). #5066 proposes gradient routing to put one concept in one neuron.
- **It could set a head's pattern, if the model follows the program.** A selector is an attention pattern, and the repo audits heads (`attention-head-audit`). Two conditions apply.
  - The model must follow the program. Without supervision, the Reverse model computed length with a different head. A trained head is ground truth only if a compiler wrote it or attention supervision trained it.
  - The model must load in the harness. #3584 found that Tracr models are JAX/Haiku and don't load through the Hugging Face runner. A small PyTorch model of GPT-2 shape, trained with the paper's attention loss, would load. The follow-up below proposes this.
- **Some selectors are the repo's head scores.** `pattern_scores` in `openinterpretability/harness/heads.py` measures previous-token, duplicate-token and induction attention. In RASP, a duplicate-token head is `select(tokens, tokens, ==) and select(indices, indices, <)`, the selector of `has_prev` (Figure 12). The four configured models are decoders, so their mask already gives `select(indices, indices, <=)`. RASP's encoder selectors that look ahead have no analogue in them.
- **RASP gives #5144 a definition.** #5144 asks which VERIFIED explanations are token sets and which need context. In RASP terms, a token-set rule is an elementwise function of `tokens`, with no `aggregate`. A rule that needs context needs at least one `aggregate`, so at least one attention layer. Counting a rule's aggregates also gives its depth, a lower bound on the layer where a unit can hold it. This would be a comment on #5144, not a new issue.
- **It explains part of the ground-truth gate failures.** In `ground-truth-calibration.md`, 7 of 12 plants passed the gate, and none of the three rules that need context. "Inside quotes" reached precision 0.59 and recall 0.50 at layer 7. In RASP, that rule is a running count of quotes and a parity test, so it needs attention before the neuron. A linear read of one MLP input can see it only if earlier layers wrote the count in a linear form. RASP does not say whether they did. So this is a possible reason, not a measured one.
- **It could choose #3593's context rules.** #3593 asks for plants whose rule needs context. RASP programs such as `has_prev` (one aggregate) give rules whose minimum depth we know before the run. That would let #3593 pick rules by depth and plant each one above that depth. This would also be a comment on #3593.
- **Reading from weights (ROTATE, #3446) gets nothing direct.** RASP says what a head and an MLP can compute, not how to read their weights. `docs/analyses/weight-kurtosis.md` found that ROTATE's kurtosis does not predict VERIFIED. On 518 test poles, the AUC is 0.51 (0.45–0.57) for the output side and 0.54 (0.48–0.60) for the input side. RASP agrees with one reason for this: a context rule lives in an aggregate, and static MLP weights don't hold it.
- **Weight-space programming (#3381, #4471).** RASP is the program level, above weights. #4471 has an agent write a model's weights in Python, and Tracr (#4843, not yet reviewed) compiles programs to weights. RASP's layer bound tells such a writer how many layers a rule needs. It does not give the weights.
- **No single-neuron content.** The paper has no neuron explanations, no probes and no null. Nothing in it bears on any pole's verdict.
- **Proposed follow-up.** Train a small GPT-2-shaped model with RASP attention supervision, for example on histogram with BOS. Then audit its heads, blinded, as a positive control for head audits. This is the head version of #5066.
- **Question for the maintainer.** The follow-up audits a model outside the configured list in `pyproject.toml`. Its records would stay out of the registry, as the plants of #3201 did. May a control model like this be audited while no new registry model may be added? We record this as a question only.

# Language Models Use Trigonometry to Do Addition

**Citation.** Subhash Kantamneni and Max Tegmark, of the Massachusetts Institute of Technology. "Language Models Use Trigonometry to Do Addition". [arXiv:2502.00873](https://arxiv.org/abs/2502.00873), version 1 of 2 February 2025, the only version. Preprint, no venue given. The source is in an ICML style, but no acceptance is stated. The text gives no code or data link. Funding: the Rothberg Family Fund for Cognitive Science, and IAIFI through NSF grant PHY-2019786. The text gives no disclosure of tool use. The experiments used nnsight on one RTX A6000.

**Mismatch.** Section 5.1 calls the results "compelling evidence that all three models use the Clock algorithm." Sections 5.5 and Appendix D say that the evidence for Llama3.1-8B is weaker, and that it may use other algorithms. The abstract calls the helix "causally relevant" for division and multiplication, but Table 1 has the helix below PCA for both.

**Status.** Planned. #5220 names it as background for phase 5(b), a compiled addition module, which must read and write the model's own number encoding. The fix campaign is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 1 for this page's author, a model (Claude Opus 5.5). It read the source's own text in full, in three fetches: Sections 1 to 6, the references and Appendices A to G. We didn't see any figure as an image. Many results appear only in figures, so this page gives them only where the text has a number.

## Summary

**Claim.** Three language models represent a number as a generalised helix: a linear term plus circles with periods 2, 5, 10 and 100. To add, GPT-J moves the helices of a and b to the last token. It combines them into the helix of a+b, and reads the answer from it. The authors call this the "Clock" algorithm.

**Method.**

- **Models.** GPT-J (6B) for the main analysis, because its MLP has no gate. Pythia-6.9B and Llama3.1-8B in the appendices.
- **Task.** a+b with a and b from 0 to 99, so 10,000 problems. Accuracy: GPT-J 80.5%, Pythia-6.9B 77.2%, Llama3.1-8B 98.0%. Each model has its own prompt.
- **Numbers as single tokens.** Each model keeps these numbers as one token. The authors restrict the study to single-token numbers.
- **The helix fit.** PCA of the residual stream to 100 dimensions, then a least-squares fit to the helix's basis. The periods were chosen by the patching result.
- **The causal test.** Activation patching of the fitted helix on the a token, over 100 clean and corrupted pairs. Baselines: patching the full layer, PCA, a circle, and a polynomial with the same number of terms.
- **Neurons.** Attribution patching over GPT-J's 27 × 16,384 MLP neurons. Each top neuron's preactivation is fitted as a sum of linear and periodic terms in a, b and a+b.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Patching a+b on the a token, max over layers (Table 1) | full layer 8.34, helix 7.21, circle 6.83, PCA 6.13, polynomial 3.09 |
| Other tasks, helix against PCA (Table 1) | below PCA for a//5 (4.55 against 5.24) and a×1.5 (5.16 against 6.65) |
| Heads, GPT-J (Section 5.2) | 20 heads give 83.9% of the effect of all 448. 11 move a and b, mostly in layers 9–14 |
| MLPs, GPT-J (Section 5.3) | 11 MLPs, from layers 14–27, give 95% of the effect of all 28 |
| Neurons kept, GPT-J (Section 5) | about 1% (4,587), with the rest mean-ablated, keep 80% accuracy |
| Neurons replaced by their fits (Figure 9 text) | recover "roughly 75%" of the real neurons' performance |
| Direct effect (Appendix D.2) | about 700 neurons give 80% of the direct effect of the 4,587 |
| GPT-J's errors (Appendix F) | wrong on 19.5%. Of numeric errors, 45.7% are off by −10 and 27.9% by +10 |

## Where the claims go beyond the evidence

- **The step that adds is not found.** The authors say they couldn't isolate how helix(a+b) is built from helix(a) and helix(b). They suggest trigonometric identities, untested.
- **One model in depth.** The Clock algorithm is shown for GPT-J. For Llama3.1-8B the fit is weaker, and its gated MLPs aren't analysed.
- **The periods were chosen on the test.** The subset of periods is picked to maximise the patching result. Period 100 is added from a base-10 prior. The period-2 peak depends on whether 360 or 361 values are sampled.
- **No intervals or seeds.** Table 1 gives single numbers from 100 pairs. The neuron fits and the 80/20 split report no seed. The chi-squared test on errors gives no statistic.
- **Fit quality for the residual helix isn't reported.** The text gives no R² for it, only patching results.

The authors state limits themselves. Gemma-2 splits numbers into digits, so it needs another method. Other model sizes may learn other algorithms. And they studied only single-token numbers.

## Use in this repo

- **The issue's three questions, as the source answers them.**
  - *Which models:* GPT-J in depth, with Pythia-6.9B and Llama3.1-8B in less depth. None is a configured model here.
  - *Which number range:* a and b from 0 to 99, as single tokens. The layer-0 fit uses 0 to 360.
  - *At the neuron level:* yes, in GPT-J. About 1% of MLP neurons carry the task. A sum of periodic terms in a, b and a+b fits their preactivations, and the fits recover about 75% of performance.
- **It agrees with the bag-of-heuristics page in part.** [Nikankin et al.](arithmetic-bag-of-heuristics.md) read about 1.5% of later-layer neurons in Llama3-8B as heuristics, many of them modulo rules. A neuron fitted by periods 2, 5 and 10 is also a modulo heuristic. Our reading, not either paper's: the two may describe the same neurons, one as rules and one as a helix.
- **The tokenizer decides whether it applies.** We checked earlier: Pythia-70m and GPT-2 keep a number like 226 as one token, as this paper requires. Qwen2.5-0.5B splits it into digits, the Gemma-2 case that the authors say needs another method.
- **For phase 5(b), the helix is an interface.** A compiled core could read the operands by fitting the helix at the a and b tokens. Then it could write a+b by adding helix(a+b) where the model's own MLPs 19–27 read it. Our reading, not the paper's. The paper doesn't find the step that combines a and b, which is the step the compiled core would replace.
- **Neuron fits as registry explanations.** A neuron explained as "preactivation follows cos(2π(a+b)/10)" is a testable claim. The registry's probes test rules about tokens, so this kind of explanation would need probes over number pairs.
- **Questions for the maintainer, recorded on this page only.**
  - Should phase 5(b) fit a helix in Pythia-70m or GPT-2 first, to see whether these small models have one?
  - Should the registry accept periodic explanations of number neurons, with probes over operand pairs?
- **Related pages.** [`arithmetic-bag-of-heuristics.md`](arithmetic-bag-of-heuristics.md) reads the same task as many heuristic neurons. [`interchange-intervention-training.md`](interchange-intervention-training.md) is a way to train a read adapter to a stated algorithm.

# Do We Really Need an LLM for Weight-Space Programming?

**Citation.** Lin Yao (School of Computer Science, Shanghai Jiao Tong University, and Zhongguancun Academy, Beijing), sole author. "Do We Really Need an LLM for Weight-Space Programming? A Deterministic Forward-Only Paradigm and Its Capability Boundaries". [arXiv:2611.34612](https://arxiv.org/abs/2611.34612) [cs.AI], version 1 of 25 November 2026, the only version. Preprint, no venue given. The licence is arXiv's non-exclusive distribution licence. The text gives no code link, only the revisions of the checkpoints that it used. We found no funding statement and no tool disclosure.

**Status.** Planned. The fix campaign (#5220) plans small weight edits. This paper is a pipeline that chooses an edit and checks it without an LLM in the loop. Its retention check is the kind of off-target test the campaign needs, but it measures behaviour only, and the traffic map (`papers/traffic-map/`) measures explanations. No code, skill or protocol here depends on the paper. The maintainer designs the fix campaign, so this page reports evidence only.

**Read.** On 2026-11-28 a `web-fetch` subagent read the abstract page and the HTML of version 1 for this page's author, a model (Claude Opus 5.5). It read the source's own text in four fetches by offset. That covers Sections 1 to 5, Algorithms 1 and 2, Tables 1 to 9, Appendices A.2 to D and references 1 to 44. One span came back only as a summary: references 45 onward and the proofs of Propositions 1 and 2. Some tables came back garbled, and the subagent kept only the values it could pair with their labels. Quotes were capped at 125 characters. We didn't see the figures or the PDF.

## Summary

**Claim.** Weight-space programming doesn't need an LLM to write the edits. A fixed pipeline of probes, localisation and closed-form operators chooses and applies an edit. It uses forward passes only, and it is more reliable than an LLM agent with the same tools. The paper also maps where such edits fail.

**Method.**

- **Forward-only.** The edit program is built from forward passes, and the cost is counted in forward-pass equivalents (FPE). Once committed, the edit adds no cost per token.
- **The pipeline (Algorithm 1).**
  1. *Probe.* For each weight matrix, the ratio of its largest entry to its median, and of its largest row norm to the median row norm. Hooks record each layer's activation means and covariances on good and bad examples.
  2. *Localise.* A structural score picks the matrix with the most extreme outliers. A Fisher score picks the layer and the subspace that best separate good from bad examples, by whitened Fisher LDA or contrastive PCA.
  3. *Edit.* There are four operators. O1 clips outlier entries and rescales outlier rows. O2 projects a rank-$r$ subspace out of a layer. O3 recalibrates a layer's statistics for a shift in the data. O4 fits a closed-form ridge update.
  4. *Check.* Commit only if the gain on a validation split is large and significant. The change on a separate retention split must also stay above $-\varepsilon$, with $\varepsilon$ from 0.03 to 0.05. Otherwise roll back, or abstain.
- **The router (Algorithm 2).** Fixed thresholds choose the operator. An outlier ratio above 4 sends the edit to O1. Otherwise, health and shift scores choose O2, O3 or O4.
- **Reasoning models.** *Protected FisherGate* changes one row of the output matrix, the row of the end-of-thinking token, by a closed-form rank-one update. The update raises that token's logit in long, overthinking tails. It is held orthogonal to the mean states of the start, middle and end of reasoning. It is fit on 8 practice rollouts.
- **The LLM baselines.** Qwen3-8B and Qwen3.8-27B as Python-sandbox agents, and Qwen3.8-27B with the same four operators. The paper names no frontier model as a baseline.
- **Models.** Thirteen, from a 50,000-parameter MLP to 27B. ResNet-18, ViT-Small and DINOv2 read images, and Qwen3-8B and LLaDA-8B read text. There are also control and PDE models, and seven reasoning models from Qwen3.5-0.8B to Qwen3.8-27B.

**Findings, as the source reports them.**

| Result | Setting | Value |
| --- | --- | --- |
| Choosing the right operator | DINOv2, 6 faults × 3 seeds (Table 1b) | The pipeline 83.3% (15 of 18), an exhaustive search 83.3%, the 27B agent with the same tools 33.3%, random 5.6% |
| End accuracy of the same runs | Same | Pipeline 63.2%, exhaustive 67.0%, 27B agent 53.3% |
| A planted backdoor | ImageNet, BadNets, 64 unlabelled pairs (Table 2b) | Attack success 100% → 0.00%, with clean accuracy 98.44% |
| Bit flips undone | ImageNet-20, 3 seeds (Table 2a) | 6.72% → 72.67% with O1, 82.22% with O1 then O4 |
| Reasoning under a token budget | DeepSeek-R1-0528-8B, GSM8K at 1,024 tokens, N = 128 (Table 4) | 14.06% → 54.69% |
| Same, math edit tested on code | HumanEval, N = 164 (Table 6) | 33.54% → 84.15%, against 69.51% for budget forcing with 512 reserved tokens (+14.6, 95% CI +7.3 to +22.6) |
| The same edit, five calibration subsets | HumanEval | 58.7 ± 23.9% on all five. 75.4 ± 9.4% on the three that pass the check |
| CounterFact edits | Qwen3-8B, 32 facts (Table 8) | Efficacy, specificity and perplexity: the pipeline's O4 87.5, 90.6, 5.04. ROME 96.9, 65.6, 5.53. LoRA 100, 53.1, 58.8 |
| New entities | 16 entities, structural paraphrase | The pipeline 0.0%, LoRA 43.8% |
| The largest model | Qwen3.8-27B, GSM8K at 1,024 tokens (Table S6) | 92.2% → 89.8% |

**Where the claims go beyond the evidence.**

- **The LLM baseline is weak.** The "LLM-driven" editors are open 8B and 27B models with at most three turns or two steps. The question in the title is about frontier models editing weights, as in Takehi et al., and the paper doesn't test one. The author states this limit.
- **The headline 83.3% is a dispatch rate.** It counts how often the right operator was chosen. On end accuracy, the pipeline (63.2%) is below an exhaustive search (67.0%), and the 27B agent (53.3%) is 10 points behind.
- **The reasoning headline is the best patch.** 84.2% on HumanEval is the primary calibration subset. Across all five subsets the mean is 58.7%, with an SD of 23.9.
- **The results change with the backend.** For one model and one budget, the gate beats budget forcing on one inference backend, 81.3% against 50.8%. It loses on the other, 47.7% against 83.6%. The paper doesn't discuss the flip.
- **Small samples.** The reasoning tables use 125 to 164 problems, with greedy decoding and no seeds. One reported $p$ value (0.0265 for $z$ = 1.98) doesn't match a two-sided normal test.
- **Retention is on the same task family.** The retention split is from the task that the edit targets. No general benchmark checks for damage outside the task.

## Use in this repo

- **How it chooses what to edit.** By statistics, not by reading units. It picks a matrix by the size of its outlier entries, or a layer and subspace by how well they separate good from bad examples. For reasoning models it edits one fixed row, the end-of-thinking token's output row. It names no unit and explains none. So, like the gradient edit of `bias-in-disguise.md`, it is evidence about where a working fix lands, not about fixes guided by an explanation.
- **How it measures damage outside the edit.** By the change in a metric on a retention split from the same task family. The edit rolls back if that metric falls by more than 0.03 to 0.05. It reports clean accuracy, perplexity and CounterFact specificity in some settings. It reads no unit after an edit. The traffic map predicts the damage to the 176 passing explanations of pythia-70m for any site before an edit (`papers/traffic-map/`). That damage is the kind this paper doesn't measure. The paper's check-then-rollback step and the map are complementary: one tests behaviour after the edit, the other predicts damage to explanations before it.
- **Specificity on CounterFact.** The pipeline's O4 keeps 90.6% specificity, against 65.6% for ROME, at lower efficacy. In pythia-70m, ROME edits flipped fewer pole readings above the edited layer than random rank-one edits of the same norm, 25 of 870 against 40 (`papers/rome-keep/`). Neither paper reads both measures, so they can't be compared directly.
- **Abstention is a result.** The pipeline abstains when no edit passes its check, on 8 of the 20 pairs of reasoning model and task in Table 4. A fix campaign that counts an abstention as a safe outcome needs the same paired check.
- **Related pages.** `weight-space-physics-editing.md` (#6376) has agents edit weights with no gradients, and `rikiyat-weight-space-programming.md` has a frontier model do it. This paper asks whether the agent is needed at all, and answers no only against small open agents. `weight-space-micromachines.md` adds weights beside a frozen model instead of editing them.

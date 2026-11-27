# Ambient LLMs: Data-Efficient Supervised Fine-tuning via Temperature Matching

**Citation.** Giannis Daras (MIT), Marios Papachristou (Arizona State University), Adam Klivans (University of Texas at Austin), Antonio Torralba (MIT) and Constantinos Daskalakis (MIT). "Ambient LLMs: Data-Efficient Supervised Fine-tuning via Temperature Matching". [arXiv:2611.23229](https://arxiv.org/abs/2611.23229) [cs.LG], version 1 of 17 November 2026, the only version. The arXiv comment says "Work in progress". Preprint, no venue given. The licence is CC BY 4.0. The authors link no code or data. They cite the Hugging Face pages of their datasets in Appendix C.1.1.

The funding includes a Jane Street research award and a TPU Research Cloud grant. It also includes compute grants from Lambda and Thinking Machines, and API credits from Anthropic's AI for Science program. It also includes Simons Foundation and ONR grants. The authors say that generative AI helped to write the paper, edit figures, develop theorems and proofs, implement parts of the method and run experiments. They say that they did not use it to conceive the project, and that they reviewed all of its work.

**Status.** Planned. #6367 lists anti-forgetting arms for the sequential-tuning paper, and a temperature-matched stage fits that list. We add it to #6367 as a comment. No code in the repo uses the method today.

**Read.** On 2026-11-27 a `web-fetch` subagent read the abstract page and the whole HTML of version 1. It read them for this page's author, a model (Claude Opus 5.5). It had the source's own text for Sections 1 to 6, Appendices A to E, the references and the AI-use statement. We saw no figure as an image, so values that appear only in Figures 4, 6 and 7 come from their captions and the text. We did not read the PDF, and there is no code to read.

## Summary

**Claim.** Fine-tuning on little data improves with one added term. It matches the student's conditionals to a frozen copy of the pre-trained model, both at a high temperature $T$. The pre-trained model then acts as a noisy teacher. The student keeps the teacher's broad preferences, and its sharp preferences are free to change. The abstract reports "up to $21\%$ average relative improvement to standard SFT".

**Method.** The loss (Eq. 4) is the usual next-token cross-entropy on the fine-tuning documents, plus $\lambda$ times a KL term at every supervised position:

$$
\hat{J}_{\lambda,T}(\theta) = \frac{1}{n_1 l} \sum_{k=1}^{n_1} \sum_{i=1}^{l} \Big[ -\log p_\theta\big(x_i^{(k)} \mid x_{<i}^{(k)}\big) + \lambda \, D_{\mathrm{KL}}\big( p_{T,\theta_{\mathrm{pre}}}(\cdot \mid x_{<i}^{(k)}) \,\|\, p_{T,\theta}(\cdot \mid x_{<i}^{(k)}) \big) \Big]
$$

Here $p_{T,\theta}$ is the softmax of the logits divided by $T$, and $\theta_{\mathrm{pre}}$ is the frozen pre-trained model. The student starts at $\theta_{\mathrm{pre}}$. The KL term reads only the fine-tuning inputs, so the method needs no pre-training data.

- **The two ends.** As $T \to \infty$ the tempered distributions become uniform, and the method becomes plain SFT. At $T = 1$ it is knowledge distillation from the pre-trained model. The paper drops the $T^2$ factor of Hinton et al., because it did "not find this scaling to be optimal across all of our settings".
- **Implementation** (Algorithm 1). With probability $p_c$ a step is plain SFT. Otherwise the teacher runs without gradients, and the step minimises the tempered cross-entropy. In expectation this is Eq. 4 with $\lambda = (1 - p_c)/p_c$. The paper says that training is "≈1.1× slower for the same amount of steps".
- **Choice of $T$.** For domain adaptation, a grid search over $p_c$ and $T$ per domain picks the lowest validation perplexity on the target domain. Instruction tuning fixes $p_c = 0.75$ and $T = 2.5$. Appendix D.5 adds a per-source rule, which the paper calls "a simple heuristic rather than a tuned choice". The theory gives no estimator of $T$ from data.
- **Theory.**
  - Theorem 3.1: the temperature-matching loss and the tempered cross-entropy have the same minimisers. A minimiser at one $T$ matches the target model at every temperature, on contexts in the data's support.
  - Theorem E.2: for one context with logit range $\Delta$, the KL between two models' tempered conditionals is at most $\Delta^2/(8T^2)$. So a high $T$ makes the teacher's error small.
  - Theorem 3.2 uses a log-linear next-token model with frozen features. In its risk expansion, the best $\lambda$ and $T \neq 1$ do at least as well as three others. They are maximum likelihood, the frozen teacher and the best $T = 1$ run. The authors add: "We do not claim that the infimum is attained."
- **Experiments.**
  - A toy with the GPT-2 architecture on tiny Shakespeare, and GPT-2 124M on OpenWebText, in figures only.
  - Domain adaptation of a 1.5B model pre-trained on OpenWebText to five targets of about 225M tokens each. The metric is validation loss on the target. The baselines are training from scratch, no fine-tuning, target-only fine-tuning and a data mixture.
  - Instruction tuning of Qwen3 1.7B, 4B, 8B and 14B Base, OLMo-3-7B and Llama-3.1-8B, on 1% of Dolci-Instruct-SFT (21,503 examples, 10 epochs). Also 10% and 100%. Eight benchmarks: MATH, GSM8K, HumanEval+, MMLU, PopQA, TriviaQA, IFEval and BBH.
  - Baselines for instruction tuning on Qwen3-1.7B: NEFTune, L2-SP (a weight-distance penalty), knowledge distillation at $T = 1$ and label smoothing. There is no LoRA, EWC, replay or LwF baseline.
  - Reasoning: OLMo-3-7B on 1% of Dolci-Think-SFT, with six benchmarks.

**Findings, as the source reports them.** The average relative change is the mean over the eight tasks of the change from vanilla SFT, divided by the SFT score (Appendix C.2.2).

| Setting | Result |
| --- | --- |
| Domain adaptation, 1.5B, validation loss, Ambient against the best baseline (Table 1) | Python Code 0.70 vs 0.79. CS 2.54 vs 2.55. BioMed 2.35 vs 2.36. News 1.68 vs 1.99. Reviews 2.96 vs 2.96 |
| Same, as perplexity (Introduction) | "up to 26.7% lower perplexity than the best baseline" |
| Zero-shot loss against Ambient's margin, five domains | Spearman ρ = −0.90 |
| Instruction tuning, 1% data, average relative change from SFT (Table 2) | Qwen3-1.7B +20.3% (the base model alone +9.0%). 4B +8.0%. 8B +10.2%. 14B +8.7% |
| Same, with a per-source $T$ (Table 3) | 1.7B +21.1%. 4B +8.2%. 8B +10.1% |
| Other models (Table 4) | OLMo-3-7B +15.4%. Llama-3.1-8B +8.3% |
| Baselines on Qwen3-1.7B (Table 5) | NEFTune +0.8%. L2-SP +15.3%. Label smoothing −0.3%. Distillation at $T = 1$ +17.0%. Ambient +20.3% |
| Share of Dolci-Instruct-SFT (Section 4.3) | 1%: 15.4%. 10%: 3.4%. 100%: 8.6% |
| Reasoning, OLMo-3-7B, 1% (Table 6) | Ambient +3.5%. Distillation at $T = 1$ −7.3%. L2-SP −37.1% |
| Forgetting under vanilla SFT (Section 4.3) | "the base model (pre-SFT) beats vanilla-SFT on 6 or 7 tasks for every model" |
| Epochs 2 to 6, 10% data, vanilla SFT (Figure 4 text) | IFEval 71.0 → 80.8, while MATH 37.3 → 30.0, GSM8K 76.4 → 69.6 and TriviaQA 63.7 → 59.2 |
| Same, Ambient | IFEval 63.8 → 73.4, "while the other benchmarks stay within about two points of their 2-epoch values" |

## Where the claims go beyond the evidence

**The 21% has no single source.** The abstract's "up to 21%" matches no number in the main table, whose largest value is +20.3%. The nearest value is +21.1% for Qwen3-1.7B with the per-source $T$ of Table 3, a heuristic that the authors did not tune. The HTML abstract leaves the sentence out.

**One seed, and margins inside the noise.** The main tables come from one data-order seed, and the authors repeat no run. Table 20's caption gives a seed-to-seed spread of about 0.003 to 0.006 in loss, "across two domains checked". Three of the five domain margins in Table 19 are 0.01 or 0.00, so they are within that noise. There are no intervals or tests on the main tables.

**$T$ is chosen on the target's validation set.** In domain adaptation, the grid search picks $p_c$ and $T$ by the target domain's validation perplexity. The paper reports the same kind of loss as its result. Instruction tuning avoids this with one fixed setting.

**The largest gain shrinks with more data.** The authors offer the method for small data, and the evidence fits that. Its advantage falls from 15.4% at 1% of the data to 3.4% at 10%, then rises to 8.6% at 100%. The paper does not explain the rise.

**Only benchmarks measure forgetting.** The paper shows that vanilla SFT loses general skill and that Ambient loses less. It reports no KL to the base model on other text, no distance in weight space and no measure inside the model. Domain adaptation reports no loss on OpenWebText, the pre-training distribution.

**The anti-forgetting baselines are few.** L2-SP and distillation at $T = 1$ are the only baselines that hold the model near its start. There is no LoRA, EWC, replay or LwF baseline, so the gain over those methods is not known.

**The theory is for a simpler model.** Theorem 3.2 holds for a log-linear model on frozen features. The authors say so, and say that they do not claim that the model reaches the bound.

The authors credit some limits themselves. They call the work "in progress". They note that IFEval is the one benchmark where Ambient is worse than SFT. They also say that the reasoning gain is smaller than the others.

## Use in this repo

**The repo's fix loss already has a KL anchor, but a different one.** The trained arms of `scripts/fix_arms.py` (#7006) add $\lambda \cdot \mathrm{KL}(\mathrm{base} \,\|\, \mathrm{fixed})$ at $T = 1$, on the retain prompts only (#5220 §3). Ambient's anchor is on the training prompts themselves, at a high $T$. So the two hold different things. The fix's anchor keeps other prompts unchanged. Ambient's anchor keeps the broad shape of the next-token distribution on the very prompts it changes. A fix wants the right unit to win on its prompts, and a high-$T$ anchor there would leave the sharp preference free to move. That is the paper's argument, and it fits the fix's goal. Whether to add it as a variant of arms B and C is a question for #7007's design. We record it here and open no issue.

**The paper gives a second reason for the fix's retain term, not a replacement for it.** Ambient's KL term sees only the fine-tuning inputs. Prompts that the fix must not change, such as the right copies " legs" and " wheels" (#5220 §1), are not in those inputs. So Ambient alone would not protect them, and the fix still needs its retain term.

**A temperature-matched stage fits #6367.** The sequential-tuning paper (`papers/sequential-tuning/`) fine-tunes Qwen2.5-0.5B through three narrow tasks. 267 of its 362 reproduced explanations keep after three stages. #6367 already plans LoRA, frozen-MLP and learning-rate arms. Add a stage with Ambient's loss at the paper's fixed setting, $T = 2.5$ and $p_c = 0.75$. It would test whether a loss that keeps benchmarks also keeps explanations. The paper measures no unit, so its claims don't answer that. We add this to #6367 as a comment.

**The stage KL of #5125 is the paper's missing measure.** The sequential-tuning paper measures the KL from the model before each stage to the model after it, on the stage's held-out responses. It finds that this KL does not track the explanations a stage costs. A temperature-matched run lowers the high-$T$ KL by design. The run would show whether a smaller KL at high $T$ also costs fewer explanations.

**Related pages.**

- [Learning without Forgetting](learning-without-forgetting.md) uses a temperature-scaled distillation term on the new task's inputs to keep old outputs. Ambient is the same idea for language-model SFT, with the teacher at a higher temperature.
- [LoRA Learns Less and Forgets Less](lora-learns-less-forgets-less.md) finds that a low-rank change forgets less. The paper has no LoRA baseline, so the two can't be compared.
- [Fine-tuning on new knowledge](fine-tuning-new-knowledge.md) finds that a model fits new facts slowly, and that they raise hallucination. Ambient's evidence is on skills and domains, not new facts.
- [RL's Razor](rls-razor.md) reports that forgetting tracks the KL to the old model on the new task's inputs. Ambient lowers a tempered form of that KL, so the two papers agree in direction.

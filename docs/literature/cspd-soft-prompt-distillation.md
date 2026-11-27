# Composable Soft-Prompt Distillation of RL Experts

**Citation.** Zhen Zhang (University of California, Santa Barbara) and Changyi Yang (Carnegie Mellon University), equal contribution, with 25 others, through Xin Eric Wang (UCSB). "Composable Soft-Prompt Distillation of RL Experts: Toward Continual Post-Training". [arXiv:2611.27376](https://arxiv.org/abs/2611.27376) [cs.LG, cs.CL], v1 of 20 November 2026 and v2 of 24 November 2026. This page reads v2. Preprint, no venue given. The paper is under CC BY 4.0, and the [code](https://github.com/UCSB-AI/CSPD) is under CC BY-NC 4.0. The paper states no funding and no use of AI tools.

**Status.** Not applicable. CSPD's students and teachers are Qwen3, GPT-OSS and other models of 4B to 35B parameters, and none of them is in the registry. No prompts are released for the registry's models. One idea carries over, and *Use in this repo* gives it, with no issue.

**Read.** On 2026-11-27 a subagent (Claude) downloaded the v2 PDF and turned it into text. It read all 28 pages verbatim, with Appendices A to G, and checked parts of the HTML against them. It also read the code's README and licence. It did not see v1, the dataset card, the code or the paper that describes how the teachers were trained. The reader of this page was a model (Claude Opus 5.5), through that subagent's report. The issue took its numbers from the README and the abstract. The subagent checked each one against the paper, and the table below gives the paper's settings.

## Summary

**Claim.** An RL-trained expert can be distilled into one soft prompt for its frozen pre-RL student. Prompts for several experts compose by concatenation, with no joint training. Composed prompts match weight merging in domain and keep more out-of-domain skill.

**Method.**

- **The prompt.** 256 trainable vectors in the system message. The student's weights are frozen.
- **The loss.** A length-normalised negative log-likelihood on the response tokens only, on the teacher's offline responses that pass a verifier.
- **Variant that trains behind other text.** On half the training examples, 128 to 256 tokens from other questions of the same domain go before the prompt. So the prompt learns to work behind other prompts.
- **Composition.** The prompts are concatenated, with no separators.
- **Baselines.** Task arithmetic, TIES, MOPD (a released checkpoint) and LoRA distillation at rank 16.

**Findings, as the source reports them.**

| Result | Setting |
| --- | --- |
| One prompt recovers at least 90% of the teacher's gain in 22 of 23 settings | 7B to 35B dense and MoE models, each at its best checkpoint on the evaluation benchmarks. Nemotron on SWE-bench (92.27%) counts only because its student got the teacher's bug locations |
| 67.83 in domain, against 67.79 for MOPD and 67.47 for a TIES merge | Four composed prompts (math, code, science, instruction following) on an SFT of Qwen3-4B-Base, in that order, seed 42. The reverse order gives 67.72 |
| 56.6 out of domain, against 54.1 for expert TIES, 53.5 for MOPD and 50.3 for the student | The same four prompts |
| TruthfulQA falls by 0.4 points with CSPD, and by 3.6 with MOPD | Out of domain |
| Throughput within 1.5% of the backbone's | Seven concurrency levels |
| 2.40 times rank-4 LoRA's throughput, and 4.12 times at one request | 32 concurrent requests. The LoRA adapters were synthetic, not trained experts |
| The prompt closes 86.8% of the teacher–student KL. Written "check your work" instructions close at most 2.1% | Appendix G, one GPT-OSS math prompt, teacher-forced |
| Letting only the response tokens see the prompt keeps 80% of its effect, and only the input tokens 33% | Appendix G, attention masks |

## Where the claims go beyond the evidence

- **One seed, and choices made on the test benchmarks.** Every result has one seed, with no intervals or tests. The best checkpoint is chosen on the evaluation benchmarks. The optimizer and learning rate were screened on IFBench, an evaluation benchmark, and the task-arithmetic coefficients were chosen on the test benchmarks.
- **"In domain, it matches" rests on 0.04 points.** 67.83 against 67.79 is one seed and one order. The reverse order falls below MOPD.
- **The serving gap is loosely stated.** "2.4× slower" for LoRA is the ratio at 32 requests only, with synthetic adapters.
- **Appendix G says more than it shows.**
  - That the prompt "cannot be put into words" rests on three hand-written instructions.
  - That it works "mainly by checking" rests on counts of tokens such as "Check" and "But". The paper doesn't measure their share of the KL change.
  - The prompt analysed is the one whose vectors barely moved (cosine 0.999).
  - The two attention pathways keep 80% and 33% of the effect, which add to 113%, so they don't separate.
  - The layer results have no figure or table behind them.

## Use in this repo

- **An input-space change can move explanations as much as tuning.** The template-or-tuning paper (`papers/template-or-tuning/`) found that a chat template alone costs the base Qwen2.5-0.5B 76 of its 360 passing explanations. Instruction tuning alone costs 61. A soft prompt is a learned template. So CSPD's claim that a prompt "keeps" out-of-domain skill says nothing about the units behind those skills. The registry's keep test would read the units: run each VERIFIED pole's probes behind the prompt, as that paper ran them inside the template. That test needs a prompt trained for a registry model. None is released, so this page opens no issue.
- **A frozen-weights control for the fix campaign.** The fix campaign (#5220) compares weight edits with a few-shot prompt (arm E). A trained soft prompt sits between the two: it learns, but it changes no weight. It would separate "the fix needs new weights" from "the fix needs a better input". The campaign's arms are the maintainer's plan, and arm E already covers the input side. So this page records it as an idea, not a question.
- **Interference.** The paper's out-of-domain result agrees in direction with `lora-learns-less-forgets-less.md`: changing less of the model forgets less. Its measures are benchmark scores, with one seed, and it has no measure of which units change. So it adds no evidence at the level of units to `sft-capability-degradation-feature-interference.md` or `spurious-forgetting.md`.
- **Merging baselines.** TIES and task arithmetic are its weight-merging baselines, as in `ties-merging.md`. Expert TIES is in fact second out of domain, at 54.1, not MOPD as the abstract's framing suggests.

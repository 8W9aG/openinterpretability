# CurvDelta: Low-Curvature Delta Modeling For Surgical Learning and Unlearning

**Citation.** Zeguan Xiao and Zhuocheng Liang, with equal contribution, Yun Chen and Pinyan Lu, of the Shanghai University of Finance and Economics. Then Yong Wang (Alibaba Group) and Guanhua Chen (Southern University of Science and Technology). "CurvDelta: Low-Curvature Delta Modeling For Surgical Learning and Unlearning". [arXiv:2611.24654](https://arxiv.org/abs/2611.24654), version 1 of 18 November 2026, the only version. Preprint, no venue given. The project page is [xiaozi-max.github.io/CurvDelta](https://xiaozi-max.github.io/CurvDelta/). The text names no code repository and no funding.

**Mismatch.** Section 4.3 names CurvLoRA for the ablation of how much preservation data to use, while the rest of the section is about the dense variant. The text doesn't say which one was ablated.

**Status.** Planned. #4782 (catastrophic interference, weight-space programming and LoRA) would use it, beside the null-space and low-rank methods. The fix campaign (#5220) is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-21 a `web-fetch` subagent read the abstract page and the HTML of version 1 for this page's author, a model (Claude Opus 5.5). In four fetches it had the whole text in the source's own words, Sections 1 to 6 and Appendices A to F. The capability tables came through with their columns shuffled, so this page quotes none of their numbers.

## Summary

**Claim.** Keep a fine-tune's change in the directions where the loss of the original behaviour is flattest. Then the model can learn a skill or forget data and keep its general ability.

**Method.**

- **Curvature.** The preservation loss is the KL divergence from the original model's outputs. Its curvature at the start is estimated per weight matrix with K-FAC. Skill learning uses 256 web texts, and unlearning uses the retain set.
- **The subspace.** For each K-FAC factor, the leading eigenvectors are removed until a share γ of the energy is gone. The change to a weight is ΔW = P_out δ P_in, with the two projectors fixed and δ trained from zero. CurvLoRA does the same with a low-rank δ.
- **In the parameters.** The projection is part of the weights, not of the gradient. The paper notes that under Adam, projecting gradients alone lets the update leave the subspace.
- **A bound.** Under a local quadratic model, the preservation loss is at most a weighted sum of the squared change per layer (Theorem 1).
- **Tests.** Math and code fine-tuning of Gemma-3-1B and Llama-2-7B. Unlearning on TOFU (Llama-3.2-3B, Qwen3-4B, Llama-3.1-8B) and MUSE Books (Llama-2-7B). Baselines include full fine-tuning, LoRA, CrispEdit, OPLoRA and VILA.

**Findings, as the source reports them.**

| Setting (TOFU, forget 10%, SimNPO, Table 3) | Forget quality ↑ | Forget probability ↓ | Model utility ↑ |
| --- | --- | --- | --- |
| Retrained from scratch | – | 10.6 | 64.6 |
| Full fine-tuning | −1.4 | 14.4 | 38.9 |
| LoRA | −13.5 | 18.2 | 64.6 |
| CurvDelta | −1.1 | 6.5 | 63.2 |

In the same setting, Table 13 gives the KL to the original model on retained data. It is 0.149 for CurvDelta, 0.043 for LoRA and 0.624 for full fine-tuning. CurvDelta's run takes 1,004 s against 537 s for full fine-tuning (Table 14).

## Where the claims go beyond the evidence

- **One objective carries the headline.** The main unlearning table uses SimNPO. Under GradDiff, on TOFU forget 10%, CurvDelta's forget quality is far worse than LoRA's (−228.7 against −38.7 on Llama-3.1-8B, Appendix).
- **Close to retraining on TOFU, not on MUSE.** On MUSE, CurvDelta's retained knowledge is 55.8 against retraining's 68.6.
- **Low-rank methods keep the KL lower.** LoRA and OPLoRA stay closer to the original model on retained data than CurvDelta, though they forget less.
- **The bound is local and loose.** The quadratic estimate is around 10⁻⁸ to 10⁻¹⁰, against a measured KL of about 0.01 to 0.07. The paper notes the "smaller numerical scale".
- **No seeds or intervals.** No table reports a spread. All ablations use one TOFU setting.

## Use in this repo

- **Another way to protect what a fix shouldn't touch.** [`alphaedit.md`](alphaedit.md) projects an edit into the null space of kept representations. [`orthogonal-gradient-descent.md`](orthogonal-gradient-descent.md) projects gradients away from old tasks. [`elastic-weight-consolidation.md`](elastic-weight-consolidation.md) penalises moves along important directions. CurvDelta removes the high-curvature directions of the kept behaviour from the change itself. Our reading: it guards average output on the preservation data, not any one unit.
- **It doesn't protect named units.** The registry's VERIFIED units are single neurons and heads. The preservation data here is general text, so a unit whose behaviour is rare in that text could still move. Our reading: a curvature built from each VERIFIED unit's own probes would target them, but the paper doesn't test that.
- **Projecting the weights, not the gradients.** The paper's point that Adam undoes gradient projection applies to any fix trained with Adam and a projected gradient.
- **Question for the maintainer.** Should a fix with a curvature or null-space constraint record its preservation data? A re-test could then tell which units the fix was meant to protect. This bears on the patch format (#5229, closed) and the interference report (#5230, closed). We open no issue, because the campaign is the maintainer's.
- **Related pages.** [`lora-learns-less-forgets-less.md`](lora-learns-less-forgets-less.md) reviews the low-rank trade-off that CurvLoRA builds on. [`robust-unlearning-evaluation.md`](robust-unlearning-evaluation.md) reviews how to test whether unlearning holds.

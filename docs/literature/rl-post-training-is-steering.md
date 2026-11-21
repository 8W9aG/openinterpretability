# RL Post-Training in Diffusion Transformers Is Activation Steering

**Citation.** Six authors: Tuna Han Salih Meral (Virginia Tech), Kaan Oktay (fal) and Alec Helbling (Georgia Tech and IBM Research). Then Atakan Kara (fal), Duen Horng Chau (Georgia Tech) and Pinar Yanardag (Virginia Tech). "RL Post-Training in Diffusion Transformers Is Activation Steering". [arXiv:2611.26627](https://arxiv.org/abs/2611.26627), version 1 of 19 November 2026, the only version. Preprint, no venue given. The project page is [rl-is-steering.github.io](https://rl-is-steering.github.io). The text names no code repository and no funding.

**Mismatch.** Table S5 gives a FLUX.1-dev PickScore gain of +1.346 and a share of 92.1%, and Table 1 gives +1.522 and 90.0%. The text we read doesn't explain the difference. Our issue quoted "90 to 96 percent". The abstract gives 90.0–96.3% across six image models. Appendix B gives "90% to 96%" for a separate test across samplers.

**Status.** Not applicable. No code uses it and no paper cites it. It bears on the question behind the *Fixing models: activation steering* pages (#4784). That question is whether one direction can hold a fine-tune's change as a constant shift. The models are diffusion transformers, not language models.

**Read.** On 2026-11-21 a `web-fetch` subagent read the abstract page and the HTML of version 1 for this page's author, a model (Claude Opus 5.5). In four fetches it had the whole text in the source's own words, Sections 1 to 6 and Appendices A to I. It saw the figure captions but not the figures.

## Summary

**Claim.** "A constant shift thus suffices for most of the gain of policy-gradient RL post-training" in diffusion transformers. Fixed vectors added to a frozen base model recover "90.0–96.3% of the reward gains of six RL-tuned image models". On two video models they recover 89.8% and 111.9%.

**Method.**

- **Models.** SD3.5-M, FLUX.1-dev, Wan2.1-T2V-1.3B and Wan2.2-TI2V-5B. Eight public RL-tuned checkpoints, all trained by policy gradients (Flow-GRPO, DanceGRPO, GenRL and an RLVR run), as LoRAs or full fine-tunes.
- **Steering vectors.** One vector per block output, added to every token as h + αs with α = 1. Each is split by guidance pass (conditional or unconditional) and by two bins of noise level.
- **Two ways to set them.** The mean difference between the RL-tuned model's block outputs and the base model's. Or a fit that makes the steered base model's velocity match the RL-tuned model's, with no reward used.
- **Size.** On SD3.5-M the vectors have 288,768 parameters, 1.5% of the rank-32 LoRA they replace.
- **Statistics.** Paired 95% bootstrap intervals over prompts. One training seed per fit, with a three-seed check on two rewards. A user study of 25 raters.

**Findings, as the source reports them.**

| Setting (share of the RL model's reward gain, Table 1) | Mean difference | Fitted vectors [95% interval] |
| --- | --- | --- |
| SD3.5-M, PickScore, Flow-GRPO LoRA | 61.0% | 90.8% [89.8, 91.7] |
| SD3.5-M, OCR | 21.5% | 94.4% [90.8, 97.9] |
| SD3.5-M, GenEval | 12.2% | 95.9% [92.0, 99.7] |
| FLUX.1-dev, HPSv2.1, DanceGRPO full fine-tune | 79.0% | 96.3% [95.5, 97.2] |
| Wan2.2-5B video, rule score, full fine-tune | 60.8% | 89.8% [76.2, 105.6] |

Training the vectors alone by RL, from zero, gives 103.6% of the RL model's PickScore gain and 104.6% of its OCR gain (Table 3). Random vectors of the same norm lower PickScore. A rank-1 cut of the RL LoRA keeps 43.1% of the PickScore gain. For supervised fine-tuning on a new subject, vectors do worse than a LoRA of the same size. The authors conclude: "Supervised fine-tuning on a new subject thus needs more than a constant shift, whereas RL post-training does not."

## Where the claims go beyond the evidence

- **Sufficiency, not necessity.** The authors say: "We make no claim of necessity, that the RL-tuned model itself needs the steering vectors to reach its gain."
- **Fitted, not read off.** The simple mean difference recovers 12% to 79% of the gain. The high numbers need vectors fitted to the RL model's output, so the vector is found by optimisation, not read from the model.
- **Policy gradients only.** "All eight RL-tuned models use policy gradients, so preference optimization and reward backpropagation remain untested."
- **Diffusion only.** No language model is tested. The conclusion frames the general claim as a hypothesis, that RL "primarily steers the base model toward what it can already compute".
- **Few video models.** The authors limit the video claim to the two models tested.

## Use in this repo

- **A constant shift can hold a fine-tune's gain, in one setting.** The steering pages ask whether a fix can be one direction added at inference. Here a small set of fixed vectors holds almost all of an RL fine-tune's reward gain, and RL on the vectors alone matches it. Our reading: that supports steering as a fix when the target behaviour is one the model can already produce. It doesn't show this for language models.
- **It agrees with the rank-1 LoRA result.** [`rank-1-lora-reasoning-signals.md`](rank-1-lora-reasoning-signals.md) finds that a rank-1 LoRA recovers 73–90% of a reasoning fine-tune's gain in Qwen-2.5-32B. Both find a fine-tune's gain held by a very small change. This paper's vectors are smaller still, a constant with no input dependence.
- **Mean differences are weak here.** Contrastive steering in language models uses mean differences ([`contrastive-activation-addition.md`](contrastive-activation-addition.md)). Here they recover as little as 12% of the gain, and fitting is what closes the gap. [`latent-steering-vectors.md`](latent-steering-vectors.md) also fits vectors by gradient descent.
- **Supervised fine-tuning differs.** New knowledge, here a new subject's identity, needed more than a constant shift. Our reading: steering may suit "do more of what you can do" fixes better than "learn a new fact" fixes, which the editing pages cover.
- **No follow-up.** The registry studies units of language models, and this paper tests none. No open issue would run its method.
- **No question for the maintainer.**
- **Related pages.** [`activation-addition.md`](activation-addition.md) reviews the steering method that the repo's steering samples use.

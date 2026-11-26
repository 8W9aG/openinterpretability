# Auditing Post-Training Unlearning in Diffusion Language Models using Sparse Autoencoders

**Citation.** Yunha Seo and Changhoon Kim, of the Department of Software, Soongsil University. "Auditing Post-Training Unlearning in Diffusion Language Models using Sparse Autoencoders". [arXiv:2611.27652](https://arxiv.org/abs/2611.27652), version 1 of 20 November 2026, the only version. The arXiv comment says it was accepted at the NeurIPS 2026 Workshop on Inference-Time AI Privacy (InfPriv 2026). The paper is under CC BY 4.0. It gives no code or data link, and no funding or tool disclosure.

**Mismatch.** The text gives the cross-stage overlap of features as ranging "from 0.14 to 0.26", while Table 10 gives 0.265 for layer 16. Appendix F says that the pretraining corpus and training history of the main sparse autoencoders, and their exact checkpoint metadata, "were not preserved".

**Status.** Not applicable. The registry has no diffusion language model, and it runs no unlearning. The paper's random-subspace controls agree with how the repo tests a change, and *Use in this repo* says what carries over to #5220.

**Read.** On 2026-11-26 a `web-fetch` subagent read the abstract page and the HTML of version 1 for this page's author, a model (Claude Opus 5.5). It read the text in two passes, the second from where the first stopped. It had the source's text for Sections 1 to 6, the references and Appendices A to F, with Tables 1 to 15. The tool allows only short verbatim quotes, so much of that text came back paraphrased. It did not see Figure 1, only its caption.

## Summary

**Claim.** Sparse autoencoder (SAE) features can audit how unlearning acts inside a diffusion language model, across the stages of denoising. The features that mark the forget domain shift from one stage to the next. But suppressing features chosen at each stage does not reliably beat suppressing one pooled set. And suppressing the chosen features does not reliably beat suppressing random directions once utility is held equal.

**Method.**

- **Model.** LLaDA-8B-Instruct, frozen, with a LoRA of rank 32 trained for each method.
- **Data.** Forget: 540 cybersecurity texts from the WMDP corpora. Retain: 540 excerpts of WikiText-2. Test: WMDP-Cyber (1,987 questions), with WMDP-Bio and WMDP-Chem as collateral checks. MMLU (14,042 questions) measures utility.
- **SAEs.** TopK-32 SAEs with 65,536 features, trained separately at layers 16, 24 and 30.
- **Feature selection.** Each feature's score is its mean squared activation on forget text less that on retain text. The top 64 features' decoder directions are orthonormalised into a rank-64 subspace for each layer.
- **Intervention.** The LoRA is trained to lower the hidden state's energy in that subspace, with a retain loss and a multiple-choice loss.
- **Denoising stages.** The frozen base model generates 32 tokens in 16 steps. The states before steps 0, 8 and 14, with 32, 16 and 4 masked positions, are the early, middle and late stages.
- **Static against stage-adaptive.** *Pooled* uses one subspace from all stages. *Stage-conditioned* uses each stage's own subspace. *Stage-permuted* swaps the stages' subspaces, to test whether alignment matters.
- **Controls.** A Gaussian subspace of 64 random directions, and a Gaussian control scaled to match the starting gradient. Also 64 random SAE features over three draws, and a retain-only run. RMU, gradient ascent, NPO and MDU run as reference methods.
- **Statistics.** Paired bootstrap intervals over questions, with 1,000 or 2,000 replicates. A three-seed panel at one strength.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Fixed subspace, λ = 0.2, Table 1 | Base 43.03 Cyber, 64.61 MMLU. Gaussian 32.36 and 62.19. Selected features 26.82 and 60.34 |
| Selected against Gaussian, same setting | Cyber −5.54 points [−7.90, −3.12], MMLU −1.85 points [−2.52, −1.22] |
| Strength sweep, Table 5 | At λ = 0.025, Selected 38.00 and 63.25 against Gaussian 38.10 and 63.96. Selected "does not dominate" Gaussian in the trade-off |
| Generation, 100 Cyber prompts, λ = 0.2 | 91% of Selected's answers empty, 90% of Gaussian's, 0% of the base model's |
| Feature overlap across stages, Section 4 | Within a stage 0.49 to 0.54, early against late 0.14 to 0.26 |
| Stage-conditioned against pooled, Table 2 | Cyber −1.11 points [−2.26, +0.05], MMLU −0.36 [−0.69, −0.02] |
| Stage-conditioned against permuted, Table 13 | Every interval includes zero |
| Three seeds, λ = 0.025, Table 3 | Pooled 41.17 ± 0.64 Cyber, Stage-conditioned 41.44 ± 0.49, Gaussian 41.10 ± 0.12, RMU 43.10 ± 0.15 |

## Where the claims go beyond the evidence

- **One model, one forget domain.** Every result is on LLaDA-8B-Instruct and WMDP-Cyber. The authors say WMDP is "a hazardous-knowledge proxy rather than a direct personal-data erasure test", although the workshop is on privacy.
- **The SAEs can't be reproduced.** Appendix F says that the main SAEs' training records were not kept. A fresh SAE is used for one post-hoc, single-seed panel.
- **Strong suppression empties the output.** At λ = 0.2, about 90% of answers to Cyber prompts are empty for both the selected and the Gaussian subspace. So the large fall in Cyber accuracy is partly a model that stops answering.
- **The stages come from the base model.** The replay reads the frozen base model's trajectories, not those of the model being trained. The authors say so.
- **Intervals without corrections.** The intervals resample questions, not training seeds, and the authors say that they don't adjust for model selection or multiple comparisons. The utility-matching rule, which sets the strengths, fails its own target.
- **The authors state the negative results plainly.** They say that selection "depends on the control" and that stage conditioning gives no "utility-aware advantage". The page credits that.

## Use in this repo

- **The registry has nothing here to audit.** It holds no diffusion language model, and it runs no unlearning. #5220 plans fixes of an autoregressive model, starting with pythia-70m's copy failure (#4781).
- **The random-subspace control matches how this repo tests a change.** The paper sets chosen features against random directions of the same rank, and against random SAE features. The repo's keep tests set each change against a random change of the same size ([`papers/random-change-keep/`](../../papers/random-change-keep/README.md), [`papers/rome-keep/`](../../papers/rome-keep/README.md)). Both find that a chosen change can do little better than a random one of its size. For a fix under #5220, a random control of equal size is the comparison to report.
- **The audit could use native neurons, in part.** The selection score is a difference of mean squared activations on forget and retain text. It works on any unit with an activation, so it works on MLP neurons. A neuron's subspace has one direction, its write. Appendix B warns that the score can rank the suppression objective the wrong way. Selection reads encoder activations, and suppression reads projections onto decoder directions. A native neuron has no such split: its activation and its write belong to the same unit. That is a reason to try the audit on neurons, not a result. No issue follows, because the registry has no unlearning to audit.
- **The stage result does not carry to autoregressive models as it is.** An autoregressive model has no denoising stages. Its closest analogue is the position in the answer, where a static fix and a position-dependent one could differ. The paper's negative result on stages says nothing about that.
- **Its neighbours on this page's topic are already reviewed.** [Mechanistic Unlearning](mechanistic-unlearning.md) finds that unlearning holds better when it changes the components that look up a fact. [The WMDP Benchmark and RMU Unlearning](wmdp-rmu-unlearning.md) gives the benchmark that this paper uses. In this paper's three-seed panel, RMU at 300 updates leaves Cyber at 43.10, no lower than the base model.

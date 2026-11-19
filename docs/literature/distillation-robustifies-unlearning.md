# Distillation Robustifies Unlearning

**Citation.** Bruce W. Lee, Addie Foote, Alex Infanger, Leni Shor, Harish Kamath, Jacob Goldman-Wetzler, Bryce Woodworth, Alex Cloud and Alexander Matt Turner. "Distillation Robustifies Unlearning". arXiv 2506.06278. Version 1 is from 6 June 2025, version 2 from 9 June 2025. We read version 3, from 24 October 2025. [Paper](https://arxiv.org/abs/2506.06278), [code](https://github.com/AddieFoote/distillation-robustify-unlearning). The HTML lists the University of Pennsylvania, MIT, Brown University and ML Alignment & Theory Scholars as affiliations. The abstract page shows none. The arXiv comment gives the venue as "NeurIPS 2025 (Spotlight)". We did not check the proceedings page. Licence CC BY 4.0. The text says that "Bruce W. Lee's work on this project was partially supported by funding from Open Philanthropy." We saw no statement on whether an LLM helped write it.

**Status.** Planned. The fix campaign that #4784 prepares for, with #4781 and #5015 as its first case, would use it. The paper gives a test of whether a fix removed a capability or only hid it.

**Read.** On 2026-11-18, through WebFetch. A model read the abstract page and the v3 HTML for us in three passes. The tool limits each quote to 125 characters. So we have short verbatim fragments from the abstract, Sections 1 to 8 and Appendices A, D.2.3, E and F. The rest came back as the model's summary. This includes most method details, the full limitations paragraph, the α and β values and the TOFU and quantization results. We did not see the figures or the PDF. The WMDP accuracies are only in Figure 8, so we have none of them.

## Summary

**Claim.** Unlearning methods that train on behaviour suppress a capability, but they do not remove it. A few finetuning steps bring it back. Distilling the unlearned model's outputs into a fresh model keeps the behaviour and drops the latent capability. In the authors' words, "existing LLM unlearning methods achieve behavioral suppression rather than true unlearning".

**Method.**

- **Models.** A "100M model architecture" in the Gemma-2 style for language. The model's summary gives Gemma-2-0.3B for arithmetic and Gemma-2-2B for WMDP.
- **Settings.** Language: English FineWebEdu is the retain set and Korean FineWeb2 is the forget set (summary). The reference model is "pretrained on 2B tokens, 1B from retain data and 1B from forget". Arithmetic: the model keeps addition and subtraction and forgets multiplication and division. Each set has "500K arithmetic questions for training". WMDP: the Bio and Cyber subsets, with MMLU for retain.
- **Metrics.** For language, "The retain and forget evaluations are cross-entropy loss on held-out portions". For arithmetic, they "measure accuracy" on the two kinds of problem.
- **Oracle.** "We also define an _oracle model_ as one trained on the reference model's pretraining dataset", with the forget data left out. For language it is "pretrained on 1B tokens of the retain set". The oracle is the target: a model that never learned the capability.
- **Unlearning methods.** GradDiff, MaxEnt and RMU (summary).
- **Unlearn-and-Distill.** Unlearn the model, then distil it "into a randomly initialized model of identical architecture using forward KL divergence".
- **UNDO.** Unlearn, Noise, Distill-on-Outputs: "distills an unlearned model's outputs into a noised copy of itself". The model's summary gives the noised initial weights as (1 − α)·θ + α·β·N, with N as Xavier noise. Distillation runs "until the model reaches at least 95% of the teacher model's performance on the retain set".
- **Relearning attack.** Finetuning on forget data "for 500 steps", with "validation performed every five steps". There is "an adversarial learning rate search for all relearning runs". The score is "the maximum elicited forget performance of all attacks".
- **Baselines.** Data filtering (retraining from scratch), SAM, RepNoise and UNDIAL (summary).

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Oracle matching, language (Section 3) | "the oracle-matched reference model relearns significantly faster than the oracle model." A model that only copies the oracle's outputs still holds the capability |
| Distilled student, Section 3 | "Student (Reference) quickly learns the forget distribution compared to the Student (Random)" |
| Unlearn-and-Distill (Section 4) | "the student relearns substantially slower as there was no latent capability to begin with." Introduction: "distilled models relearn harmful capabilities at rates comparable to an oracle model" |
| RMU against MaxEnt, arithmetic (Section 4) | "RMU achieved only 62% retain and 6.8% forget accuracy, compared to MaxEnt's superior 80% and 1.3%" |
| UNDO, strongest setting (abstract) | "UNDO matches the robustness of a model retrained from scratch with perfect data filtering", "using only 60-80% of the compute" |
| Labelled data (Section 4) | The method needs labels on "less than 0.01% of pretraining data" |
| Noise level (Section 5) | More noise needs "proportionally more computation". The relation "appears roughly linear for language unlearning". Arithmetic "shows meaningful improvements at higher perturbation levels" |
| WMDP, Gemma-2-2B (Section 6) | "UNDO consistently improves relearning resilience across all WMDP configurations." But "the UNDO model achieves 4.88% lower on MMLU on average." |
| WMDP distillation data (Appendix D) | "we distill on only 300 million, 0.015% of the Gemma-2-2b pretraining corpus." |
| Compute (Appendix E) | The most expensive WMDP UNDO runs "take around 7 hours on one H200." |

Section 4 gives the explanation: "quick relearning of supposedly unlearned capabilities stems from latent information preserved in the parameter space". Section 8 says: "There is a disconnect between what a model does and what it _could do_."

## Where the claims go beyond the evidence

**The mechanism is a hypothesis.** The paper says that latent information stays "in the parameter space". It never locates that information. It runs no probe, unit or circuit analysis. The evidence is behavioural only: relearning speed, and a quantization test in a post-review appendix.

**Small models and few tasks.** The main settings use models from about 100M parameters up to 2B. The forget sets are a language, an arithmetic operation and the WMDP subjects. Two of the three are broad, easy-to-separate data sets. The authors note for TOFU that "keyword or semantic matching cannot guarantee the absence of latent traces."

**The WMDP result is weaker than the toy results.** UNDO loses MMLU there, and the authors say that "other methods achieve a similar trade-off". They put this down to under-training: "We hypothesize this difference comes from the fact that we undertrain in the WMDP setting". This is not tested.

**The attack is one kind of attack.** Relearning is finetuning on forget data for 500 steps, with a search over learning rates. Other attacks, such as finetuning on nearby data or quantization, get less test. The quantization test used one attack type on TOFU models only.

**Two small percentages that read alike.** The abstract's labelled-data share (under 0.01%) and the WMDP distillation share (0.015%) measure different things. A reader can mix them up.

**Seeds are few.** Figure 5 shows "variation across five random seeds", and the WMDP runs use "four seeds of each method". We saw no confidence intervals in the text that we read.

**Credit.** The oracle baseline is a strong null: a model that never saw the forget data. The worst case over learning rates is a fair attack. The authors state the compute cost: "Distillation is considerably more costly than finetuning-based unlearning methods". They release code.

## Use in this repo

**The repo has no weight-space fix and no relearning test.** Every intervention in the repo acts at inference time: patching, ablation and steering. `git grep` finds no finetuning code in `openinterpretability/`. So the paper's test does not apply to anything the repo runs now. It applies to a fix that changes weights, which the fix campaign may choose.

**The paper's warning fits an explanation-guided fix.** The #4781 route copies " days" through the layer-0 count-word and time-unit neurons and the copy heads L3/H3, L4/H0 and L4/H7. A fix could change those heads so that the prompt no longer copies. The paper predicts that such a fix may suppress the copying and leave it in the weights. A short finetune on copy-style prompts could then bring it back. This is a prediction from the paper, not a measurement.

**The repo's nulls match the paper's idea of a baseline.** The paper judges a fix against an oracle that never had the capability. The repo judges units against matched units that do not carry the claim. On #4781, the explanation covers 93.8% of the prompt, against 10.1% for matched random units. The answer route's patching score is 0.75, against −0.004 for random sets (`changelog.d/4781.added.md`). A relearning test would need its own null of this kind: the same finetune on a model that the fix did not touch.

**The hydra effect is a near cousin.** `docs/PROTOCOL.md` §4 freezes downstream MLP outputs during patching, so that they cannot compensate (the hydra effect). The `activation-patching-practice` skill covers this compensation. The paper's relearning is compensation over training steps, not over layers. Both say the same thing: removing a unit's output does not show that the capability is gone.

**What the explanations could add.** The paper has a behavioural measure but no location. The repo has locations: VERIFIED units and answer routes. After a weight-space fix, the repo could read the route's units again on the original prompts. If they still activate, that is one form of the "latent information" that the paper names. This is evidence for the campaign, not a design for it.

**Question for the maintainer.** Should a weight-space fix report a relearning test, with a model that the fix did not touch as its null? Should it also read the route's units after the fix? These are campaign design decisions, so this page records them and opens no issue. The sibling reviews of unlearning, #5040, #5041 and #5042, bear on the same question.

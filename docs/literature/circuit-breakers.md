# Improving Alignment and Robustness with Circuit Breakers

**Citation.** The authors are Andy Zou (Gray Swan AI, Carnegie Mellon University, Center for AI Safety) and Long Phan (Center for AI Safety). Next are Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko and Rowan Wang (Gray Swan AI). Then Zico Kolter and Matt Fredrikson (Gray Swan AI, Carnegie Mellon University), and Dan Hendrycks (Gray Swan AI, Center for AI Safety). "Improving Alignment and Robustness with Circuit Breakers". arXiv 2406.04313. Version 1 is from 6 June 2024. We read version 4, from 12 July 2024. [Paper](https://arxiv.org/abs/2406.04313), [code](https://github.com/GraySwanAI/circuit-breakers). The venue is *Advances in Neural Information Processing Systems 37* (NeurIPS 2024), Main Conference Track. We checked this on the [proceedings page](https://proceedings.neurips.cc/paper_files/paper/2024/hash/97ca7168c2c333df5ea61ece3b3276e1-Abstract-Conference.html). The arXiv page gives no venue. The section "Acknowledgments and Disclosure of Funding" thanks five people for feedback and names no funder. Most authors are at Gray Swan AI, a company. The compute was one A100-80GB GPU for 20 minutes for each model.

**Status.** Planned. The fix campaign that #4784 prepares for, with #4781 as its first case, would use it. The paper shows a trained fix that acts on internal representations, with a retain set to limit collateral damage. But its target is that the model stops a behaviour, not that it gives the right answer.

**Read.** On 2026-11-18. We saved the v4 HTML from arXiv with `curl`, converted it to plain text, and read it ourselves. A model did not summarise it for us. We have the source's own text for the abstract, Sections 1 to 5, Algorithm 1 and Appendices A to H. We have the numbers of Tables 1, 2, 3 and 8, the MMLU row of Table 5, and the Figure 8 table. The text conversion lost the numbers of Tables 4, 6 and 7, and the other rows of Table 5. We did not see the figures, the PDF or the code. We read the abstract page and the proceedings page through WebFetch, for the citation only.

## Summary

**Claim.** Refusal training and adversarial training act on outputs, and attacks bypass them. The paper trains a "circuit breaker" in its place. It remaps the internal representations of harmful outputs, so that the model stops before it completes such an output. In the authors' words, "circuit-breaking directly controls the representations that are responsible for harmful outputs in the first place". They say that, because the representation "is independent of any attack capable of eliciting it, this approach is attack-agnostic".

**Method.**

- **Models.** Mistral-7B-Instruct-v2 and Llama-3-8B-Instruct. LLaVA-NeXT-Mistral-7B for images. Llama-3-8B-Instruct with function calls for agents.
- **Technique.** Representation Rerouting (RR), a form of Low-Rank Representation Adaptation (LoRRA). LoRA adapters train against a frozen copy of the model (Algorithm 1).
- **Losses.** The rerouting loss is `ReLU(cosine_sim(rep_M(x_s), rep_Mcb(x_s)))` on the circuit breaker set. It pushes the new representation to be orthogonal to the original one. The retain loss is the ℓ2 distance between the two models' representations on the retain set. The example schedule moves weight from the first loss to the second: `c_s = α(1 − t/2T)` and `c_r = α·t/2T`.
- **Data.** The circuit breaker set is harmful queries and completions that "an uncensored LLM" wrote. The authors remove samples with BLEU above 0.3 against HarmBench's standard behaviours. For Llama-3, the set keeps harmful responses without their requests, so that the text bypasses the refusal. The retain set is UltraChat and XSTest, with extra refusal data for Llama-3.
- **Settings for text models.** 150 steps, batch size 16, α = 5 for Mistral and α = 10 for Llama-3. The loss reads layers 10 and 20, and LoRA adapters go into "all linear layers from layers 0 through 20". The loss covers both user and assistant tokens. The authors chose hyperparameters "based on static attack test cases from HarmBench's validation set".
- **Evaluation.** Attack success rate (ASR) on HarmBench, judged by HarmBench's classifier and "manually" verified. There are ten attack settings, from no attack to an input-embedding attack and a RepE attack. Capability is MT-Bench and the Open LLM Leaderboard. Over-refusal is the refusal rate on 500 WildChat requests.
- **Baselines.** The original refusal-trained models, the adversarially trained Mistral R2D2, a safety prompt for the image model, and harmfulness probes (HP). A probe reads the activations of each generated token and stops the output if it flags one.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Table 1, average ASR over ten attack settings, Mistral-7B | Refusal-trained 76.7, R2D2 31.7, + RR 9.8 |
| Table 1, average ASR, Llama-3-8B | Refusal-trained 38.1, + RR 3.8, Cygnet 0.8 |
| Table 1, Llama-3 prefilling attack | 84.9 refusal-trained, 3.3 with RR |
| Table 1, Llama-3 input-embedding attack | 80.4 refusal-trained, 9.6 with RR, 7.9 Cygnet. The caption calls it "an unrealistically strong threat model" |
| Table 1, MT-Bench | Mistral 7.60, R2D2 6.00, + RR 7.53. Llama-3 8.05, + RR 8.00, Cygnet 8.21 |
| Table 1, Open LLM average | Mistral 65.4, R2D2 61.2, + RR 65.4. Llama-3 68.8, + RR 68.3, Cygnet 71.9 |
| Section 4.1, summary | RR reduces "compliance rates to harmful requests by an average of 87% with Mistral and 90% with Llama-3". The capability dip is "less than 1%" |
| Table 3, WildChat refusal rate | Mistral 2.0, R2D2 10.6, + RR 3.4. Llama-3 2.2, + RR 6.2. Claude-3-Opus 20.6 |
| Figure 8, LLaVA-NeXT-Mistral-7B, PGD attack | Compliance 91.0 original, 96.2 with a safety prompt, 14.3 with RR. MMMU 34.7, 33.8 and 34.2 |
| Figure 8, Llama-3 agent, 100 harmful function-call requests | No attack 58, 29 and 8. Forced function call 82, 78 and 14. BFCL 74.8, 72.0 and 76.0 |
| Table 2, average ASR over five settings, Mistral | Refusal-trained 80.6, linear HP 19.0, MLP HP 14.3, RR 11.2 |
| Table 2, average ASR over five settings, Llama-3 | Refusal-trained 32.6, linear HP 9.0, MLP HP 6.8, RR 3.1 |
| Table 8, Llama-3 loss ablation, average ASR over six attacks | RR 2.5, RandP 9.7. RandC and RMU "do not converge" |
| Table 8, Llama-3 data ablation | Without the augmented set, ASR 5.8 and MT-Bench 8.1. Without refusal retain data, ASR 0.6 and MT-Bench 7.7. Both in, ASR 2.5 and MT-Bench 8.0 |

**Representations.** For one prefilled harmful response, the cosine between the two models' representations "start[s] to change dramatically during prefilling starting from layer 10". The norms "often dramatically increase after circuit-breaking occurs", although the loss does not control them. With an obfuscated word ("m-e-t-h"), the break comes during generation. Then the model "switches to meaningless outputs before harmful knowledge is revealed".

**Generalisation over categories.** Section 4.4 trains one model for each of six categories of harm. "The robustness performance on test categories of harms largely depends on the closeness of the training distribution."

## Where the claims go beyond the evidence

**The name says circuit, but the paper finds no circuit.** The method picks two layers and a dataset, and trains LoRA adapters over layers 0 to 20. It does not find which units or paths carry the harmful output. The authors say that bottom-up mechanistic methods "have proven insufficient" for this aim. So the paper is evidence for a fix that needs no explanation, not for one guided by an explanation.

**"Responsible for harmful outputs" is not tested causally.** The loss changes representations that occur during harmful text. The paper does not show that those representations cause the output in the original model. The representation analysis uses one prompt for each figure.

**"Attack-agnostic" rests on the attacks tried.** None of the attacks targets the circuit breaker itself. The input-embedding attack is the strongest test, and it still gives 9.6 on Llama-3 and 15.7 on Mistral. The authors state the same limit for probes: they are "tested under a weaker adversarial setting, where the attacker lacks knowledge of the probe".

**Selection is close to the test.** Hyperparameters come from HarmBench's validation cases. The authors filter the training data against HarmBench at BLEU 0.3, which allows near paraphrases. Section 4.4 says that success depends on the closeness of the training distribution.

**A broken output counts as a harmless output.** After the break the model writes "meaningless outputs". ASR scores these as successes. The refusal measure for RR models uses perplexity as well as keywords. Over-refusal still rises, from 2.2 to 6.2 on Llama-3.

**Cygnet is not described.** It integrates circuit breakers "and other representation control methods", which the text does not name. Its 0.0 scores cannot be reproduced from the paper.

**No seeds or intervals.** Each number comes from one training run. The tables give no seeds, confidence intervals or error bars. The agent benchmark has 100 requests, and the image test has 133.

**Credit.** The authors state their scope in Section 5: "one particular type of adversarial attack", for "single-turn conversations". They compare against a reading-only baseline (HP) and say where it is weaker. They release code.

## Use in this repo

**The repo has no trained fix.** The repo has steering by clamp (`neuron-steering-tests`) and patching (`activation-patching-practice`), but no LoRA or fine-tune code. A search of `openinterpretability/` and `scripts/` for LoRA and fine-tuning found none. RR is a trained fix, so the fix campaign would need new code to try it. #4782 covers LoRA, and #4861 reviews ReFT, which is close to LoRRA.

**RR stops a behaviour, and #4781 needs a different answer.** On "A week has seven days and a year has twelve", pythia-70m gives " days" (0.73), and " months" is at rank 4. The VERIFIED route copies the first clause's unit through the layer-0 count-word and time-unit neurons and the copy heads L3/H3, L4/H0 and L4/H7. RR would push the representation on such prompts away from the original. The paper's own result is that the model then writes "meaningless outputs". That would remove " days", but nothing in the method supplies " months". The numbers are in `changelog.d/4781.added.md` and `docs/analyses/hallucination-shortlist.jsonl`.

**An explanation could choose RR's data and layers.** The paper chooses layers 10 and 20 of 32, and the data by category of harm. A VERIFIED route names the layers and the token positions. For #4781 it would name the " twelve" position and the copy heads in layers 3 and 4. The other five shortlisted prompts all copy the first clause's unit, so they are a natural circuit breaker set. Prompts where the first clause's unit is the right answer would be a natural retain set. This is a comparison with the paper, not a measurement.

**The retain loss and the paper's checks are a form of collateral-damage test.** The paper measures capability (MT-Bench, Open LLM) and over-refusal (WildChat) beside the target. Those benchmarks do not suit a 6-layer model. Registry coverage (`openinterpretability/coverage.py`) and the sister tool (`openinterpretability/sisters.py`) could measure what a fix breaks elsewhere. #4782 covers that side.

**Reading against control.** Table 2 compares a probe that reads representations against RR, which changes them. The MLP probe comes close to RR when it reads every generated token. `docs/analyses/probe-comparison.md` already compares VERIFIED poles with linear probes for their concept. A VERIFIED unit could act as such a monitor. But a monitor stops the output and cannot give " months" either.

**Question for the maintainer.** Does a fix count when it stops the wrong answer without giving the right one? RR succeeds on the first test and makes no claim on the second. The answer decides whether methods such as RR belong in the fix campaign. It is a campaign design decision, so this page records it and opens no issue.

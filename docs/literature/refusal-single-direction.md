# Refusal in Language Models Is Mediated by a Single Direction

**Citation.** Andy Arditi (Independent), Oscar Obeso (ETH Zürich), Aaquib Syed (University of Maryland), Daniel Paleka (ETH Zürich), Nina Panickssery (Anthropic), Wes Gurnee (MIT) and Neel Nanda. The HTML gives no affiliation for Neel Nanda. The title is "Refusal in Language Models Is Mediated by a Single Direction". arXiv 2406.11717. Version 1 is from 17 June 2024, version 2 from 15 July 2024. We read version 3, from 30 October 2024. [Paper](https://arxiv.org/abs/2406.11717), [code](https://github.com/andyrdt/refusal_direction). The venue is NeurIPS 2024 (Advances in Neural Information Processing Systems 37), Main Conference Track. We checked this on the NeurIPS proceedings page on 2026-11-18. The arXiv record gives no venue comment. The Long-Term Future Fund funds Arditi and Obeso, and AI Safety Support funds Arditi and Syed. The authors used TransformerLens, HuggingFace Transformers, PyTorch and vLLM. Together AI inference gave the safety score. Most runs used eight NVIDIA RTX A6000 GPUs.

**Status.** Planned. The fix campaign that #4784 prepares for would use it. It is the standard example of a fix by one explained direction, with side-effect checks after the fix. Its selection scores also bear on #3209, which asks if steering predictions should become gates.

**Read.** On 2026-11-18. A model read the abstract page and the v3 HTML for us through a web tool, and returned only short quotes and summaries. We then read the v3 HTML ourselves as plain text. We have the source's own text for the abstract, Sections 1 to 7 and the acknowledgements. We also have Appendices C.1, D.3, D.4, G.3, H.2, I, L.1, L.2 and N, and parts of C.2, E, G.1 and J. We read the tables as text. We did not see the figures, so Figures 1, 3, 5, 6, 23 and 24 give us no numbers. We did not read Appendices A, B, F, K, L.3 and M closely. We did not read the PDF.

## Summary

**Claim.** In 13 open chat models, one direction in the residual stream mediates refusal. In the abstract's words, erasing it "prevents it from refusing harmful instructions". Adding it "elicits refusal on even harmless instructions".

**Method.**

- **Models.** Qwen Chat (1.8B, 7B, 14B, 72B), Yi Chat (6B, 34B), Gemma IT (2B, 7B), Llama-2 Chat (7B, 13B, 70B) and Llama-3 Instruct (8B, 70B).
- **Data.** Harmful instructions from AdvBench, MaliciousInstruct, TDC2023 and HarmBench. Harmless instructions from Alpaca. Each set has 128 training and 32 validation prompts, filtered so that they do not overlap the test sets.
- **The unit.** A difference in means: the mean residual activation on harmful prompts minus the mean on harmless prompts. There is one candidate for each layer and each token position after the instruction.
- **Selection.** Three scores on the validation prompts rank each candidate. One says how much its ablation removes refusal, one says if its addition causes refusal, and one is the KL change on harmless prompts. The search also leaves out layers near the output. The authors call this method heuristic.
- **Interventions.** *Activation addition* adds the vector at its own layer, at every position. *Directional ablation* removes the direction's part from every residual activation, at all layers and positions. The paper also shows that the same ablation can be written into the weights, as "an interpretable rank-one weight edit". Appendix E proves the two forms equal.
- **Metrics.** A refusal score from a list of refusal phrases, and a safety score from the Llama Guard 2 classifier. The authors use both because each one misses cases (Appendix D.3). Decoding is greedy, up to 512 tokens. HarmBench's classifier gives the attack success rate (ASR) on 159 "standard behaviors".
- **Side-effect checks.** MMLU, ARC, GSM8K, TruthfulQA, WinoGrande and TinyHellaSwag. Also cross-entropy (CE) loss on The Pile, on Alpaca and on the model's own completions.
- **Baselines.** Other jailbreaks from HarmBench (GCG and its variants, human prompts, AP, PAIR), the rate with no attack ("DR"), activation subtraction and LoRA fine-tuning. Section 5 compares an adversarial suffix with random suffixes of the same length.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Directional ablation, 100 JailbreakBench prompts, all 13 models (Figure 1) | Ablation "reduces refusal rates and elicits unsafe completions". The numbers are only in the figure, which we did not see |
| Activation addition, 100 Alpaca prompts (Figure 3) | Addition "results in the model refusing even harmless requests". The numbers are only in the figure |
| Llama-3 8B, Table 10: refusal score, safety score | No intervention 0.95, 0.97. Directional ablation 0.01, 0.15. Activation subtraction 0.01, 0.16. LoRA fine-tuning 0.00, 0.08 |
| Llama-3 8B, Table 10: CE loss on the model's own completions | No intervention 0.195. Ablation 0.213. Subtraction 0.441. Fine-tuning 0.273 |
| Qwen 1.8B, Table 9: CE loss on The Pile | Baseline 2.921, ablation 2.938, subtraction 3.259 |
| HarmBench ASR, weight form of the ablation, with the default system prompt (without it) | Llama-2 7B 22.6 (79.9). Llama-2 13B 6.9 (61.0). Llama-2 70B 4.4 (62.9). Qwen 7B 79.2 (74.8). Qwen 14B 84.3 (74.8). Qwen 72B 78.0 (79.2) |
| HarmBench ASR with no attack (DR) | Llama-2 0.0 to 0.5. Qwen 7.0 to 9.5 |
| Capability after the edit, largest model of each family (Table 3) | MMLU, ARC and GSM8K change by at most 1.5 points. TruthfulQA falls in all five, by 1.0 to 3.5 points |
| All models (Section 4.3) | "Except for Qwen 7B and Yi 34B, all evaluation metrics for orthogonalized models lie within 99% confidence intervals of original performance" |
| Adversarial suffix, Qwen 1.8B, 128 harmful prompts (Section 5) | The suffix suppresses the direction at the last token, to a level like that of harmless prompts. A random suffix does not. The top eight heads that write the direction move their attention to the suffix |
| Base models (Appendix J) | The chat model's direction also separates harmful from harmless prompts in the base model's activations |

Appendix L.1 shows that the edited model can still write a refusal phrase when a prompt asks for it. So the authors say that the edit "works at a higher conceptual level" than blocking tokens. Appendix I.1 says that subtraction moves harmless activations "far out of distribution", but ablation does not. The authors conclude that ablation is "more surgical" than activation addition.

## Where the claims go beyond the evidence

**"Necessary and sufficient" is tested on short sets.** Section 3 says that the direction "is both necessary and sufficient for refusal". The authors test it with 100 harmful and 100 harmless prompts for each model, with one direction chosen on 32 + 32 validation prompts. Section 4.2 adds the 159 HarmBench behaviours. The claim is about these sets, not about refusal in general.

**No random-direction null.** The paper compares the adversarial suffix with random suffixes. But we found no test of a random direction, or of a direction of the same norm from an unrelated contrast. So the text does not show how special the chosen direction is among the other candidates. Figure 11 plots the scores of all candidates for Llama-3 8B, but we did not see it.

**The selection uses the measures that the paper reports.** The bypass and induce scores use the same refusal metric as the tests, on validation prompts. The test prompts are different, so this is not selection on test data. But the rule picks the candidate that best passes the kind of test that follows.

**Error bars are thin.** Figures 1 and 3 show a binomial standard error with n = 100. Figure 5 shows a standard deviation over 128 prompts. We found no seeds, because the decoding is greedy and the direction is one mean. We found no correction for 13 models.

**The meaning of the direction is open.** The authors state this limit themselves: "these directions could represent other concepts such as 'harm' or 'danger'". The name "refusal direction" is functional.

**The mechanism of the suffix is one case.** The authors give this limit themselves: Section 5 uses "a single model and a single adversarial example".

**The ASR falls with a system prompt on Llama-2.** With the default system prompt, the ASR for Llama-2 7B is 22.6, and without it 79.9. Appendix F.2 says that the prompt's content does not explain all of the gap. So "one direction" does not carry all of refusal in that setting.

**Credit.** The paper gives two refusal metrics and shows the cases where they disagree. It reports many side-effect checks, including the TruthfulQA drop. It compares ablation with subtraction and fine-tuning on the same model. The authors call the work "an existence proof" and do not claim that their extraction is the best one.

## Use in this repo

**Our steering tests use one unit, not a direction.** `openinterpretability/harness/steering.py` continues five neutral prompts under ablation and under a clamp at 1, 2 and 4 × `max_activation`. The clamp sets a neuron pole to a value at one layer. It sets a dictionary feature, or a superposition, along its decoder direction. The paper's ablation removes one direction from the residual stream at every layer. The `neuron-steering-tests` skill says that drift only at 4× is weak evidence, because it is off-distribution. The paper's Appendix I.1 makes the same point with numbers. On Llama-3 8B's own completions, subtraction raises CE loss from 0.195 to 0.441. Ablation raises it to 0.213.

**Our steering has no side-effect number.** The samples are text, reported and not gated (`docs/PROTOCOL.md` §4, *What is not evidence (yet)*). The paper tests each intervention for its effect on unrelated prompts: KL at selection, then CE loss and benchmarks. A KL on the five neutral prompts would give each steering sample the same kind of check. This review proposes it as a follow-up issue.

**We already have the null that the paper lacks, but not for steering.** A superposition unit is a direction over one layer's MLP neurons. Its ablation is `a ← a − max(0, ŵ · a) ŵ`. Its causal metrics rank it against 64 null directions on random neurons (`docs/PROTOCOL.md`, the unit table). The answer gates rank a route against 8 random sets (`beats_null` ≥ 3×, `patching_null` ≤ 0.1). So a direction test in this repo already has a random-direction null. The steering samples do not. A matched random unit, clamped in the same way, would show how much drift any clamp of that size gives. The same proposed follow-up would add it.

**Our directions stay in one layer.** The protocol says that a superposition "is a direction within one layer, never across layers". The paper's ablation removes one residual direction at every layer. The repo has no unit of that kind, and this review does not propose one.

**What the paper says for #4781, as evidence only.** On the #4781 prompt, pythia-70m answers " days" (0.73), with " months" at rank 4 (`changelog.d/4781.added.md`). The VERIFIED route copies the first clause's unit through the heads L3/H3, L4/H0 and L4/H7. No verified unit supplies " months". The paper finds one direction for one behaviour of a chat model, from a contrast of two prompt sets. Refusal is one choice that follows one property of the prompt, its harm. The copy failure is a route through heads that moves a token from the context. So the paper gives no reason to expect that one direction carries it. A difference in means between prompts where the model copies the unit and prompts where it does not would test this. The test needs a random-direction null, as above. The result is evidence for #5015. How a fix uses it is the maintainer's decision.

**Steering can work in the wrong direction.** The paper's main use of an explained direction is to remove a behaviour that the model's developers trained in. The same tools that test a fix can undo one. The four models that the planner explores are base models, with no trained refusal (`models` in `pyproject.toml`). One chat model, Llama-3.1-8B-Instruct, may be audited with signed records, but the planner does not explore it (`signed_models`). So the risk is small in the registry now, but not zero. A steering result in a record says what a unit does, not how to change a deployed model.

**A question for the maintainer.** The paper chooses its direction with a rule written before the tests. The direction must remove the behaviour, cause it, and keep KL on unrelated prompts low. #3209 asks if steering predictions should become gates. The paper's rule is one form that such a gate could take. A change to the decision rule needs a `PROTOCOL_VERSION` bump (`docs/PROTOCOL.md` §6). We record it as a question and do not open an issue.

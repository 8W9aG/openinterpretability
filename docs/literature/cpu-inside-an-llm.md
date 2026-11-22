# A CPU Inside an LLM

**Citation.** Tanmay Sah, Independent Researcher. "A CPU Inside an LLM: Zero-Gradient Compilation of Exact Circuits into Pretrained Weights". arXiv 2611.26450 [cs.AI], v1 19 November 2026, the only version. The arXiv comment says "27 pages, 3 figures, 24 tables". Licence CC BY 4.0. [Paper](https://arxiv.org/abs/2611.26450). Preprint, no venue given. An appendix names a script `verify_icml2027_gemm_and_proofs.py`, but the text names no venue. The text we read gives no code repository. It names four companion scripts and a SHA-256 manifest in Appendix F, which we did not see. The models are pinned by Hugging Face commit: `ibm-granite/granite-4.2-3b` (4e9c275e), `Qwen/Qwen3.5-4B` (851bf6e8) and `google/gemma-4-E4B-it` (4ecc2ba6). We saw no funding, tool-use or competing-interest statement.

We found four inconsistencies in the source:

- The abstract page says "4-8 digit signed" circuits, and the HTML says "44–88 digit". Section 4.4 says D∈{4,6,8}, so the HTML figure is a typesetting error.
- The abstract calls the test set "6-digit arithmetic (N=1,200)", and Table 4 calls it "unseen 1–6 digit signed" (Arith-1200).
- Section 5 says "55 decoding regimes", but its list has five.
- Base perplexity differs between tables. Table 4 gives 14.30 for Granite and 9.47 for Qwen, on an unstated set. Table 5 gives 42.10 and 29.81 on a 30-document set. The text doesn't reconcile them.

**Status.** Planned. #4782 would use it in its strand on weight-space programming, as evidence on interference. No code, skill or protocol here depends on it. This page proposes no new issue: the models aren't in the registry, and the compiler isn't released.

**Read.** On 2026-11-22 a `web-fetch` subagent read the abstract page. It read the arXiv HTML (v1) in parts, by offset. A model read it for us. We have these parts in the source's own text:

- the abstract, Sections 1–6 and the Impact Statement.
- Tables 1–8 with their text, and the Figure 1 and Figure 2 captions.
- Appendix A up to item 5 of A.3, and the references.

The tool cut the text at Appendix A.3. So we didn't see the end of A.3 or Appendices B–G. These hold the ablations, false-trigger breakdowns, failure taxonomy, deployment, reproduction, the sidecar runtime and the Tango tables. We also didn't see the figures as images, the PDF or the TeX source. Every number below comes from the text that we read.

## Summary

**Claim.** Exact arithmetic and Boolean circuits can be compiled in closed form into a pretrained LLM's weights, with no training examples and no runtime hooks. The grafted models then do signed multi-digit arithmetic almost perfectly. Their language behaviour stays within pre-registered margins of the base model on the benchmarks the author ran.

**Method.**

- **Models.** Granite 4.2 3B, a dense 40-layer Transformer (d=2560). Qwen 3.5 4B, with 24 GatedDeltaNet blocks, 8 attention layers and a tied 248,320-token head. Gemma 4 E4B is only audited. Full arithmetic on it is open work.
- **Where the circuit goes.** The residual stream widens from d to d+K appended wires: K=1,140 (ALU) and 296 (Boolean) for Granite, 2,555 and 311 for Qwen. Figure 2 says "Pretrained coordinates ℝ^d remain unprojected", and the pretrained language weights are "Unmodified". Two things change all the same:
  - The RMSNorm gain on the pretrained channels is scaled by γ_NL = √(d/(d+K)) (Table 8). This keeps their normalised scale when the norm runs over d+K.
  - The output head and the final norm gain change at the appended wires.
  - Granite also gets appended key–value heads in layers L0–L10 for the parser. The HTML prints "2525". We read it as 25, the same doubling as "44–88". Qwen's full-attention layers are not modified, and its cross-token shifts run inside 12 of its 24 GatedDeltaNet blocks.
- **The primitives.** Each is analytic:
  - a two-layer "Mellin-SwiGLU" bank of SiLU gates at several scales, which cancels an unknown RMSNorm scale.
  - "Indefinite-RoPE" attention with Chebyshev-signed keys.
  - a GatedDeltaNet shift with no memory.
  - Base-1000 emission in the tokenizer's own number tokens.
  - a differential unembedding.

  A five-stage compiler builds them from a configuration and the tokenizer. Calibration uses at most 2L+1 unlabelled norm medians (C1, 16 sentences with no digits), or none at all (C2, from the weights alone).
- **Routing.** The early layers parse the operands (Granite L0–L10), the middle layers run the ALU (L11–L29), and the late layers emit (L30–L39). The output adds to the next-token logits, z = z_NL + Δz_circ. Off its trigger, Δz_circ ≈ 0.
- **Tasks.** Signed +, − and × on |A|,|B| ≤ 999,999. 3-literal Boolean formulas (N=1,728). 5-bit majority and parity. Inline continuation after an answer.
- **Baselines.** The base model, a host Python router, and LoRA fine-tuning on 12.8K examples (400 AdamW steps). Also fixed-width projections, matched d+K controls (M0–M4), a 1-digit emitter and five leave-one-out ablations.
- **Side effects.** Perplexity and KL against the base model, and a 10.4K-prompt false-trigger set. Pre-registered one-sided noninferiority tests (Tango, α=0.05) on GSM8K, ARC-C, MMLU-Pro, LogiQA-EN, PubMedQA, an 8-task language suite and SWE-bench NLL.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Granite, Arith-1200, BF16 (abstract) | 99.17% ALU prefix, 100% with an FP32 sidecar |
| Qwen, Arith-1200 (abstract) | 100.00% |
| Granite base and grafted, Lead-Int (Table 4) | 15.67% → 97.08% |
| Qwen base and grafted, Lead-Int (Table 4) | 47.33% → 99.50% |
| LoRA, 12.8K examples, Granite / Qwen arithmetic | 42.2% / 61.7%, and 0% inline continuation |
| False triggers, both models (Table 4) | 0 of 10.4K |
| Granite PPL / KL, base and grafted (Table 4) | 14.30 / 0 → 14.30 / 0.0014 |
| Qwen PPL / KL, base and grafted (Table 4) | 9.47 / 0 → 9.47 / 0.0007 |
| Off-trigger shift (Impact Statement) | under 0.002 nats of next-token KL, over 10,534 activations |
| GSM8K, Granite / Qwen (Table 7) | 73.16 → 74.15% / 75.59 → 76.35%, noninferior |
| MMLU-Pro held out, Granite / Qwen (Table 7) | 27.70 → 27.60% / 42.80 → 42.60%, noninferior |
| PubMedQA long context, Qwen (Table 7) | 49.60 → 48.80%, noninferior at p=.0294 (margin 3.0pp) |
| Without norm compensation (−P4), Granite / Qwen PPL (Table 5) | 82.27 / 275.9, against 42.10 / 29.81 for the base |
| Givens-rotated control (M4), Granite / Qwen arithmetic | 0.0% / 0.0% |
| 1-digit emitter, Granite GSM8K[:60] | 36.67%, against 68.33% for the base |
| In-domain, 4 seeds × 8 formats, N=3,000 per model (Table 6) | 94.07% Granite, 96.60% Qwen |

## Where the claims go beyond the evidence

- **"Exact" rests on the construction and on tests.** The author says the numerical scripts "do _not_ constitute formal machine-checked theorem proving". Granite in BF16 reaches 99.17%, and 100% needs an FP32 sidecar. The author calls the BF16 carry effect "strictly an empirical observation".
- **Language behaviour is measured on benchmarks, not shown in general.** The author states this limit: "empirical NL noninferiority on evaluated benchmarks". The two base perplexities in Tables 4 and 5 differ by about 3×, on different and partly unstated sets.
- **Few seeds and no correction.** Only the in-domain table names seeds (4). Arith-1200 and the other tables don't say whether they are single runs. The Table 6 "[CI]" tag has no values in the text we read. Table 7 runs many noninferiority tests with no correction for multiple comparisons, though the margins were pre-registered.
- **Two models, one family of tasks.** Gemma is audited, not grafted. Division is left out as "an _engineering scope choice_".
- **No released code in the text.** The scripts are named, and the manifest is in an appendix we didn't see.
- **Credit.** Each primitive has a leave-one-out ablation, and each ablation breaks accuracy or perplexity. The noninferiority margins were "frozen prior to execution". The false-trigger set is large (10.4K).

## Use in this repo

- **The compiled circuit is new units, not a change to old ones.** Appended wires and heads carry the circuit, and the pretrained coordinates stay. So the registry's units in a grafted model keep their indices. The issue asks whether those units, and their explanations, stay unchanged. The paper measures output behaviour only: next-token KL of 0.0007–0.0017 and the benchmark tests. It never measures a pretrained unit's activations, and it reports no probe or attribution of a single neuron.
- **One change does reach every old unit.** The RMSNorm gain on the pretrained channels is scaled by √(d/(d+K)). By design, this keeps each channel's normalised value when the norm also runs over the new wires. Without it, Granite's perplexity goes from 42.10 to 82.27. So the old units see the same input only while the new wires stay near zero, and that holds off trigger. On a trigger prompt the appended wires carry large values. A rule such as "L5/N100 fires on number words" could then change its activations on exactly the prompts the circuit serves. The paper doesn't test this.
- **How the repo would test it.** A VERIFIED unit of the host model would be re-scored on the grafted model with its stored probes, and compared on trigger and off-trigger text. That is the zero-start idea of the patch format (`docs/patches.md`): run the change with nothing switched on and require no movement. The graft's off-trigger KL is that check at the output. This repo would also check it per unit. The registry's models are Qwen2.5, GPT-2 and Pythia, and the compiler isn't released, so this can't run now.
- **The patch format.** A patch here applies through forward hooks and never writes weights (`docs/patches.md`). The paper's point is the reverse: no hooks, and the circuit lives in the checkpoint. A compiled circuit could be stored as a patch only if the format could widen the residual stream, which it can't now. The paper still supports the format's choice to keep a change apart from the weights it leaves alone.
- **Interference (#4782).** LoRA on 12.8K examples reaches 42.2% and 61.7% on arithmetic and 0% on inline continuation. The graft reaches 97–100% and leaves the benchmarks within their margins. The graft adds capacity in new dimensions, and LoRA shares the old ones, so this is evidence that interference follows shared capacity. It is one paper with few seeds, so it supports the premise without settling it.
- **Auditing.** The Impact Statement warns that behavioural tests and perplexity "cannot reliably detect dormant, input-triggered payloads grafted into open-weight checkpoints", and it asks for white-box audits. This repo explains a model's existing units. Here the payload is in appended units that a registry of the host model wouldn't list. That limit belongs on the registry's coverage claims for any model that a third party changed. It is a question for the maintainer, not an issue.

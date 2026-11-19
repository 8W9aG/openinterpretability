# Steering Safely or Off a Cliff? Specificity in Inference-Time Interventions

**Citation.** Navita Goyal and Hal Daumé III (University of Maryland, College Park). `Steering Safely or Off a Cliff? Rethinking Specificity and Robustness in Inference-Time Interventions`. arXiv 2602.06256v1, 5 February 2026, the only version. [Paper](https://arxiv.org/abs/2602.06256). Venue: EACL 2026, long papers, Rabat, March 2026, pages 5723–5738, [2026.eacl-long.268](https://aclanthology.org/2026.eacl-long.268/) (DOI 10.18653/v1/2026.eacl-long.268). The arXiv comment says "EACL 2026 Main, Long Paper", and the ACL Anthology page agrees. Code: [navitagoyal/steering-specificity](https://github.com/navitagoyal/steering-specificity). On 2026-11-18 that repository held only a README, which says "Stay tuned! Code and data coming out soon." Its last push was on 2026-01-23. Funding: Schmidt Sciences and the NSF Institute for Trustworthy AI in Law & Society (TRAILS), Award No. 2229885. Tool disclosure (Appendix D): the authors used generative AI "solely as minimal writing aid".

**Status.** Planned. Umbrella issue #4784 asks how to show that a fix worked without collateral damage. This paper gives a three-part checklist for that, and the follow-up below applies its control and shift tests to the #4781 route before any fix.

**Read.** On 2026-11-18, the reviewing model (claude-opus-5-5) read the v1 HTML from arXiv in two ways. First, the `WebFetch` tool read the abstract page and the HTML, and gave the abstract verbatim and the rest as a summary. Then the model downloaded the HTML and read its full text from the markup, with the math from its alternative text. That covers the abstract, Sections 1–8, the Limitations, the Ethical Considerations, Appendices A–D and Tables 1–5. The numbers on this page come from that text. We did not see Figures 1–5, only their captions. We did not read the PDF, so the bold and underline marks for significance in the tables are not here. We read the ACL Anthology page for the venue and the GitHub repository for the code.

## Summary

**Claim.** Steering evaluations check efficacy and, at most, fluency and benchmark scores. The authors split specificity into three dimensions. Steering passes the first two and fails the third: it keeps a related behaviour on standard inputs, but not under a distribution shift.

**Method.**

- **Three dimensions of specificity (§3).**
  - *General*: "Does steering preserve general model abilities such as coherence ... and performance ...?"
  - *Control*: "Does steering preserve properties that are closely related to the target property ... without unintended modifications?"
  - `Robust` (the paper's term): "Does steering preserve these control properties even under distribution shifts, such as adversarial attacks?" The authors separate this from out-of-distribution efficacy, which asks if the target change transfers.
- **Measure.** Each dimension reports Δ = m(steered) − m(unsteered) on its metric. A large negative Δ is a loss.
- **Task 1, overrefusal.** The target is compliance on pseudo-harmful queries. The control is refusal on harmful queries. The shift is the same harmful queries with jailbreak prefixes from JailbreakHub.
- **Task 2, faithfulness hallucination (NQSwap).** The target is the updated answer when the context contradicts the model's knowledge. The control is the original answer with no context. The shift is a hand-written distractor context, for example "The Kentucky Wildcats won the 1996 NCAA championship."
- **Steering methods (§5, Appendix A).** DiffMean, LinearProbe, SSV and ReFT-r1 add a learned vector, h ← h + αw, at one layer and one token position. PartialOR ablates a false-refusal direction after it removes part of the true-refusal direction from it. Each method runs "with explicit control" and without it. The control version adds harmful queries with refusals to the positive set.
- **Models.** Llama-3.1-8B-Instruct, Llama-3.2-3B-Instruct, Qwen-2.5-7B-Instruct and Gemma-2-2B-it.
- **Data and metrics.** Training uses PHTest, JailbreakBench and Alpaca. ComplianceRate counts responses with no refusal phrase. Safety is 1 − HarmScore from Llama Guard 2. General ability is MMLU, GSM8K and fluency on Alpaca. Out-of-distribution efficacy is ORBench-hard.
- **Sizes.** 256 training and 100 validation examples. Each evaluation uses 500 test queries. The shift test applies 25 jailbreak prompts to 100 harmful queries, for 2,500 instances.
- **Selection.** α comes from {0.5, 1.0, 2.0, 3.0, 4.0}. Layer and position come from all layers and the last 5 positions. The choice maximises ComplianceRate(pseudo-harmful) − ComplianceRate(harmful) on the validation set. The shift set takes no part in it.
- **Baseline and statistics.** The only baseline is the unsteered model. Results are "averaged over 3 runs". Significance is "at p<0.05", and the paper does not name the test.

**Findings, as the source reports them.** Table 2 is overrefusal with explicit control. Values are Δ against the baseline row.

| Setting | Result |
| --- | --- |
| Llama-8B baseline (Table 2) | PHTest 0.84, ORBench 0.43, MMLU 0.55, GSM8K 0.45, JBBench safety 0.97, +Jailbreak 0.55 |
| Llama-8B, efficacy on PHTest (Table 2) | +0.04 (SSV) to +0.10 (LinearProbe) |
| Llama-8B, control on JBBench (Table 2) | −0.10 (LinearProbe) to +0.01 (SSV) |
| Llama-8B, +Jailbreak (Table 2) | DiffMean −0.28, LinearProbe −0.30, SSV −0.19, ReFT-r1 −0.23, PartialOR −0.25 |
| Shift loss in the text (§7) | "35%-55%" on Llama-8B, "20-30%" on Qwen-7B, "8-16%" on Llama-3B, "17-27%" on Gemma-2B |
| Control loss in the text (§7) | "a drop of 10%" with LinearProbe on Llama-8B, "7-8%" with DiffMean, LinearProbe and ReFT-r1 on Llama-3B |
| General loss (§7, Table 2) | Not significant except SSV on Qwen-7B: MMLU −0.29 from 0.62, which the text calls a "47%" drop |
| Efficacy and shift loss together (§7) | "largest gains in utility (12%) in Llama-8B-Instruct model with LinearProbe steering correspond to largest drop in robustness (55%)" |
| Without explicit control (Table 5, §B.2) | Llama-8B +Jailbreak −0.27 to −0.32. The authors see no extra efficacy and no extra loss against Table 2 |
| Hallucination, distractor context (Table 4) | Llama-8B baseline 0.46, Δ −0.21 to −0.25. Qwen-7B baseline 0.25, Δ −0.17 to −0.24 |
| Hallucination, no context (Table 4) | Llama-8B baseline 0.49, Δ −0.06 to −0.01 |
| Activations (§7, Figure 4) | In two principal components at layer 20 of Qwen-7B, last token, jailbroken harmful queries lie "markedly closer" to harmless ones |
| Compute (Appendix C) | RTX 2080, 15 GPU hours per model and method |

The percentages in §7 are relative to the baseline. For example, −0.30 / 0.55 ≈ 55% and −0.10 / 0.84 ≈ 12%. The paper does not say so.

## Where the claims go beyond the evidence

**The abstract says more than Table 2 shows on efficacy.** The abstract says "all steering methods reduce overrefusal". §7 says all methods do "with the exception of supervised steering vectors (SSV)". SSV's ORBench Δ is −0.21 on Llama-8B.

**The stated ranges leave out some cells.** §7 gives "20-30%" for Qwen-7B. SSV's +Jailbreak Δ there is −0.05 from 0.60, about 8%. The range starts at PartialOR's −0.12.

**The α grid and Table 3 disagree.** The grid stops at 4.0 (§6, Appendix C). Table 3 lists α = 5.0 for DiffMean on Gemma-2B and for three hallucination cells.

**There is no random-direction null.** The only baseline is the unsteered model. A random vector of the same norm at the same layer and position could also lower jailbreak safety. Without that row, the shift loss is not shown to come from the learned direction.

**The statistics are thin.** Three runs, with no seeds, no spread and no intervals. The p<0.05 test is not named, and the paper gives no correction for many cells.

**One shift per task.** Task 1 uses one jailbreak source. Task 2 uses hand-written distractors, and the paper does not give their count. "Consistently fails" rests on these two shifts.

**The mechanism claim rests on one picture.** §7 says "any steering method relying solely on these directions may inherently struggle". The evidence is a 2D PCA of one model at one layer. The paper reports no separability test.

**Single-site steering only.** Every method acts at one layer and one position. Appendix A says that this choice is "expected to be minimally disruptive". The result may differ for steering at all positions.

**The code is not out.** The footnote says "Code and dataset are available". The Limitations say "We will release all code and data". On 2026-11-18 the repository held only a README. The authors also say that they withhold the intervention parameters for safety.

**Some table cells look damaged in the HTML.** Qwen-7B fluency in Table 4 reads "-10.51", "1-0.34" and "-10.98". These may be markup errors. We did not check them against the PDF.

**The authors state limits themselves.** The Limitations name the "single-task setup", models up to 8B, and evaluation on generated text, not sequence probabilities.

## Use in this repo

**The repo's steering check tests only the first dimension, and only by eye.** `openinterpretability/harness/steering.py` continues 5 fixed neutral prompts for 24 tokens. It does this at baseline, ablated, and clamped at ×1, ×2 and ×4 `max_activation`. `docs/PROTOCOL.md` §4 (*What is not evidence (yet)*) says the samples are "reported, not gated". That is a form of general specificity: does text stay fluent and on topic? Nothing in the samples tests a related behaviour, or the same behaviour under a shift.

**The #4781 route has a clear control property, and the shortlist already holds it.** The route copies the noun after the first count word (`wiki/answers/EleutherAI--pythia-70m/970a91e11d8c/20261118T234629Z.json`). For " days" that copy is wrong. In the same frame it is often right. `docs/analyses/hallucination-shortlist.jsonl` records these top tokens:

| Prompt | Top token | P |
| --- | --- | --- |
| A spider has eight legs and a dog has four | " legs" (right) | 0.5494 |
| A cat has four legs and a bird has two | " legs" (right) | 0.5178 |
| A car has four wheels and a bike has two | " wheels" (right) | 0.4149 |
| A week has seven days and a year has twelve | " days" (wrong) | 0.7257 |
| A hand has five fingers and a foot has five | " fingers" (wrong) | 0.6796 |

So the paper's target and control map onto this case. The target is "stop the copy where it is wrong". The control is "keep the copy where it is right". A fix that cuts L3/H3, L4/H0 or L4/H7 would likely lower " legs" too. That is a control loss in the paper's sense.

**The paper's Figure 4 argument predicts where a fix can fail.** The paper says that steering fails under shift because the shifted inputs sit near the inputs that steering must leave alone. The route record says the layer-0 count-word neurons fire on " seven" and on the query " twelve". No verified unit in the record supplies " months". Suppose the route's units hold the same state for "a year has twelve" and "a dog has four". Then no fix on those units can tell them apart. That is a measurable question before any fix, and the follow-up below asks it.

**The record's probes are a ready shift set.** The VERIFIED record has 24 reviewer probes, 20 of which passed. Several shift the frame and keep the wrong copy. One is "Everyone knows a week has seven days and a year has twelve". Another is "In a calendar, a week has seven days and a year has twelve". Both give " days". Others change the noun, for example "A week has seven colours and a year has twelve", which gives " colours". These are the paper's third dimension: the same behaviour under distribution shift. The reviewer wrote them against the route claim, not against a fix. So a fix test would need its own held-out set.

**The paper's selection rule is good practice.** α, layer and position come from a validation set, and the shift set takes no part. The repo's reviewer already writes probes without seeing any measure. The same split would apply to a fix's strength.

**The repo has the null that the paper lacks.** The answer experiment compares the route with 8 matched random unit sets: patching 0.75 against −0.004 for the random sets. The #4781 coverage of 93.8% compares with 10.1% for matched random units. A fix test could keep the same null: a random unit set, or a random direction of the same norm.

**The paper's methods are directions, the repo's are units.** The paper adds a learned vector at one site. The repo clamps one neuron pole or latent at every position. The paper says nothing about single neurons, explanations or probes. So its numbers do not transfer. Its checklist does.

**Questions for the maintainer.** These bear on the fix campaign, so they are questions, not issues.

- Should a fix report three columns, as the paper does: general (neutral text, other prompts), control (the right-copy prompts), and shift (held-out probes)?
- Should the steering samples, which are not gated (#3209), add a row with a matched random unit clamped at the same multiples? That would show if drift comes from the unit.

**Proposed follow-up.** Before any fix, measure whether the #4781 route's units tell the wrong-copy prompts from the right-copy prompts in the shortlist. Record the route units' activations and head outputs at the last position for both groups and their probe-style shifts. Report any unit that separates them, against a matched random-unit null. This is evidence for the campaign, not a design for it.

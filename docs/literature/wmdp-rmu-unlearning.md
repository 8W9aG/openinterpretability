# The WMDP Benchmark and RMU Unlearning

**Citation.** Nathaniel Li, Alexander Pan, Anjali Gopal, Summer Yue, Daniel Berrios, Alice Gatti and many others, ending with Dan Hendrycks. "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning". arXiv 2403.03218. Version 1 is from 5 March 2024. We read version 7, from 15 May 2024. [Paper](https://arxiv.org/abs/2403.03218), [project page](https://wmdp.ai), [code](https://github.com/centerforaisafety/wmdp) (MIT license). The venue is the Proceedings of the 41st International Conference on Machine Learning (ICML 2024), PMLR volume 235, pages 28525–28550. We checked this on the [PMLR page](https://proceedings.mlr.press/v235/li24bc.html). The arXiv abstract page lists 56 authors, and the PMLR page lists 48. We did not compare the two lists name by name. The arXiv page shows no affiliations. The acknowledgements name the Vitalik Buterin PhD Fellowship in AI Existential Safety, the Long-Term Future Fund and the MATS program.

**Status.** Planned. The fix campaign that #4784 prepares for, with #4781 and its follow-up #5015 as the first case, would use it. RMU is a fix that uses no explanation of units, and its evaluation shows how to check that a fix holds and what it breaks.

**Read.** On 2026-11-18, through WebFetch. A model read the abstract page, the PMLR page, the GitHub README and the v7 HTML for us. It returned short verbatim quotes from Sections 4.2, 5.2 and 5.3. The abstract, the threat models, the rest of the method and the limitations came back as its summary. It read Table 1 and gave its numbers. The HTML text is 144,413 characters long, and the model read the first 100,000 only. Our request for the rest failed on a content filter, so we did not see most appendices. These include the RMU hyperparameters and the fine-tuning results in Appendix B.6. We did not see the figures or the PDF. This page gives no benchmark content.

## Summary

**Claim.** The paper releases WMDP, 3,668 multiple-choice questions that stand in for hazardous knowledge in biosecurity, cybersecurity and chemical security. It also gives RMU, an unlearning method. RMU lowers WMDP accuracy, and it keeps most general capability.

**Method.**

- **Models.** zephyr-7b-beta, Yi-34b-Chat and Mixtral-8x7B-Instruct-v0.1. RMU unlearns biosecurity and cybersecurity. The authors leave chemistry out, because the trade-off is not clear to them.
- **RMU's loss.** The loss is a forget loss plus α times a retain loss. The forget loss uses "a random unit vector with independent entries sampled uniformly at random from [0,1)", called **u**. It pushes the model's activations on forget data toward c·**u**. The vector stays fixed during training.
- **Retain loss.** "We regularize the model activations back to the original model's activations with an ℓ2 penalty", on retain data.
- **Where it acts.** "We find it sufficient to compute the loss only on layer ℓ and update gradients only on layers ℓ−2, ℓ−1, and ℓ." The stated reason is memory. The intuition is "increasing the norm of the model's activations on hazardous data in earlier layers" (Section 4).
- **Data.** The biosecurity forget corpus holds PubMed papers, and the cybersecurity one holds filtered GitHub passages (summary). The retain corpus is Wikitext. The paper reports that retain data unlike the forget topic works better than retain data on the same subject (summary).
- **Checks that the fix holds.** Zero-shot WMDP accuracy, from the top logit among A, B, C and D. Linear probes on every layer, trained on half of WMDP and tested on the other half. A GCG adversarial attack on Yi-34b, with one prompt for each WMDP subset.
- **Checks for damage.** MMLU overall, and the MMLU subsets near the forget topics: college biology, virology, college computer science and computer security. MT-Bench for chat quality.
- **Baselines.** LLMU, SCRUB and SSD, on zephyr-7b only. The authors did not run them on the larger models, because they did badly on zephyr-7b (summary).

**Findings, as the source reports them.** The numbers in the first two rows are from Table 1, as the model read it. They are percentages, and MT-Bench is out of 10.

| Setting | Result |
| --- | --- |
| zephyr-7b, base → RMU (Table 1) | WMDP-Bio 63.7 → 31.2, WMDP-Cyber 44.0 → 28.2, MMLU 58.1 → 57.1, MT-Bench 7.33 → 7.10 |
| zephyr-7b, baselines (Table 1) | LLMU: Bio 59.5, Cyber 39.5, MMLU 44.7, MT-Bench 1.00. SCRUB: 43.8, 39.3, 51.2, 1.43. SSD: 50.2, 35.0, 40.7, 5.48 |
| Yi-34b, base → RMU (Table 1) | Bio 75.3 → 30.7, Cyber 49.7 → 29.0, MMLU 72.6 → 70.6, MT-Bench 7.65 → 7.59 |
| Mixtral-8x7B, base → RMU (Table 1) | Bio 74.8 → 34.0, Cyber 52.0 → 30.8, MMLU 68.2 → 67.1, MT-Bench 8.30 → 8.17 |
| MMLU subsets near the topics (Section 5.2) | "RMU retains performance on MMLU topics related to biology (college biology) and computer science (college CS)". It "greatly drops performance on the most similar topics", virology and computer security. The numbers are only in Figure 11, which we did not see |
| Linear probes on the RMU models (Section 5.2) | "Linear probing only achieves slightly better than random accuracy", "across both categories and model sizes" |
| GCG on Yi-34b (Section 5.3) | The attack jailbreaks the base model "in less than 50 gradient steps". "The unlearned models output gibberish even after 2,500 steps" |

Chance on four-way multiple choice is 25%. So the RMU rows are near chance, not at it. The GCG run took "over 7 hours of optimization on an NVIDIA A100 GPU".

## Where the claims go beyond the evidence

**Low accuracy is not the same as removed knowledge.** The authors say that RMU may hide knowledge more than it removes it, because it changes three layers only (summary). The probes and GCG test two ways to recover the knowledge. The fine-tuning test is in Appendix B.6, which we did not read. The authors also say that unlearning "does not address the threat model of relearning in open source models".

**The damage check is partly coarse.** Overall MMLU falls by 1 to 2 points. But the authors report that the subsets nearest the topic fall greatly. So the overall number hides where the fix does damage. The authors state this limit themselves.

**The baselines ran on one model.** LLMU, SCRUB and SSD ran on zephyr-7b only. The larger models have no comparison, so the RMU gain over other methods rests on one model.

**No seeds or intervals.** We found no seeds, error bars or significance tests in the part that we read. Each cell of Table 1 is one run as far as we can tell. The question sets are large, but the variation between training runs is unknown.

**One random vector and one layer.** The text that we read gives no ablation over the vector, c, α or ℓ. The hyperparameters may be in the appendices that we did not read.

**Credit.** The authors test the fix three ways: accuracy, probes and an attack. They report the near-topic damage, the MT-Bench loss on zephyr-7b and the relearning gap themselves. Code and the unlearned models are public.

## Use in this repo

**RMU is a fix with no explanation in it.** It picks a layer, not a unit. It moves all activations on forget data toward a random direction. So it is the natural baseline for area 3. A fix guided by an explanation must do better than a fix that knows only the layer. The matched random units of `openinterpretability/coverage.py` play the same role for coverage.

**The repo's checks that a claim holds match RMU's checks.** A VERIFIED unit or route passes necessity, sufficiency and patching, against random sets. For #4781, `changelog.d/4781.added.md` gives necessity 0.68, sufficiency 0.62 and patching 0.75, with random sets at −0.004. The repo has a linear-probe tool, `open-interpretability probe-compare` (`openinterpretability/cli_probing.py`). It could check, after a fix, whether the old answer is still readable from the activations. That is RMU's probe test in our setting.

**The near-topic damage applies directly to #4781.** The route carries " days" through the copy heads L3/H3, L4/H0 and L4/H7. Copy heads also copy where copying is right. A fix that disrupts them on the six shortlist prompts can harm copying on other prompts. This is the analogue of the virology and computer security drop. The overall loss would hide it, as overall MMLU hides that drop. The six prompts in `docs/analyses/hallucination-shortlist.jsonl` all copy the first clause's unit. A near-topic set would be prompts where copying the unit gives the right answer.

**Where the paper does not apply.** RMU removes knowledge, and the #4781 fix must add an answer: " months" is at rank 4, with probability 0.0295. Pushing activations toward noise can lower " days", but nothing in it makes " months" win. RMU's GCG check also has no clear counterpart for a 70M base model.

**What would change.** The paper supports #5015, which studies the copying failure across the shortlist. It gives that work three checks to report for any fix: the target prompts, a near-topic set and an overall measure. How to weigh them belongs to the fix campaign, which the maintainer designs.

**Question for the maintainer.** Should a fix experiment report the probe check beside the answer change, as RMU does? A fix that changes the answer and leaves the old answer readable would then count as hiding, not fixing. This page records the question and opens no issue.

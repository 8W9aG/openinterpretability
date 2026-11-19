# WISE: Knowledge Memory for Lifelong Model Editing

**Citation.** Peng Wang, Zexi Li, Ningyu Zhang, Ziwen Xu, Yunzhi Yao, Yong Jiang, Pengjun Xie, Fei Huang and Huajun Chen. Peng Wang and Zexi Li are equal contributors, and Huajun Chen is the corresponding author. The affiliations are Zhejiang University and Alibaba Group. "WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models". arXiv 2405.14768: v1 of 23 May 2024, v2 of 7 October 2024 and v3 of 19 December 2024, which we read. [Paper](https://arxiv.org/abs/2405.14768). The arXiv comment gives the venue as NeurIPS 2024. We did not check the proceedings page. The code is in [EasyEdit](https://github.com/zjunlp/EasyEdit). The datasets are public: ZsRE, the SelfCheckGPT hallucination set and a Temporal set. The text we read names no specific funder.

**Status.** Not applicable for now. The paper edits a 7B model through a copy of one FFN weight matrix and a gate, and it never studies single units. It supplies evaluation axes and one prediction for #5015, but no open issue uses its method.

**Read.** On 2026-11-19, the reviewing model (Claude Opus 5.5) read the arXiv abstract page and the v3 HTML in three WebFetch passes. Each pass returned short quotes and exact numbers, not the whole text. Sections 1 to 6 came back with short quotes and the numbers of Tables 2 to 7. Appendices A and B came back with numbers, but their prose came back only as a summary. The extraction could not separate the Loc. values in Table 4 cleanly. We did not see the figures, the PDF, Appendix C (the proof) or Appendix D. We did not read the abstract verbatim.

## Summary

**Claim.** Lifelong editors that change the main weights lose locality, and editors that keep edits in a retrieval memory do not generalise. WISE keeps edits in a side copy of one FFN matrix and routes each input to the main or the side memory. The paper reports that this gives reliability, generalisation and locality together.

**Method.**

- **Side memory.** The side memory is "initialized as a copy of LLM's certain FFN layer, storing the stream of edits". For LLaMA-2-7B and Mistral-7B it is `model.layers[27].mlp.down_proj.weight`. For GPT-J-6B it is `transformer.h[21].mlp.c_fc`.
- **Router.** An activation indicator compares the side and main memory outputs for an input. "If the activation indicator of a new input is greater than ϵ, WISE will use the side memory." The threshold ε is the smallest indicator seen among the edits.
- **Router loss.** A margin loss pushes the indicator of unrelated inputs below α = 5 and that of edits above β = 20, with a gap γ = 10.
- **Knowledge sharding.** The edits go into k shards. Each shard trains its own copy of the side memory under a random gradient mask with ratio ρ. The defaults are k = 2 and ρ = 0.2.
- **Merging.** Ties-Merge joins the shards: it trims each task vector, elects a sign per parameter, then merges. WISE-Merge keeps one merged side memory. WISE-Retrieve keeps one side memory per merge and chooses among them at inference.
- **Setting.** The edits come in sequence, one at a time (batch size 1), T edits in total. Optimiser SGD with learning rate 1.0.
- **Metrics.**
  - Rel. is the edit success rate.
  - Gen. is the success rate on paraphrases of the edit.
  - Loc. is the rate at which "the post-edit model should not change the output of the irrelevant examples".
  - The locality inputs come from NQ for ZsRE, RedPajama for Hallucination and the Pile for Temporal.
- **Baselines.** FT-L, FT-EWC, MEND, ROME, MEMIT, MEMIT-MASS, DEFER and GRACE.
- **Statistics.** The text we read reports no seeds, standard deviations or intervals. Only the timing results say "across 10 trials".

**Findings, as the source reports them.**

| Setting | Result (Rel. / Gen. / Loc. / Avg.) |
| --- | --- |
| LLaMA-2-7B, ZsRE, T = 1, WISE | 0.98 / 0.92 / 1.00 / 0.97 |
| Same, T = 1000, WISE | 0.77 / 0.72 / 1.00 / 0.83 |
| Same, T = 1000, GRACE | 0.93 / 0.08 / 1.00 / 0.67 |
| Same, T = 1000, MEMIT-MASS | 0.69 / 0.65 / 0.62 / 0.65 |
| Same, T = 100, ROME | 0.23 / 0.22 / 0.04 / 0.16 |
| Same, T = 1000, ROME | 0.01 / 0.01 / 0.00 / 0.01 |
| Same, T = 3000, WISE-Merge | 0.58 / 0.56 / 1.00 / 0.71 |
| Same, T = 3000, WISE-Retrieve | 0.61 / 0.58 / 1.00 / 0.73 |
| Same, T = 3000, WISE-Retrieve with an oracle router | 0.75 / 0.70 / 1.00 / 0.82 |
| LLaMA-2-7B, Hallucination, T = 600, perplexity (lower is better) | WISE 3.12, GRACE 9.34, MEMIT-MASS 13.47 |
| GPT-J-6B, Temporal, T = 75, WISE | 0.96 / 0.37 (OOD Gen.) / 1.00 / 0.78 |
| Same, GRACE | 0.97 / 0.28 / 1.00 / 0.75 |
| Router ablation, without the margin loss, T = 1000 | Loc. falls from 1.00 to 0.72 |
| Routing accuracy at T = 3000 (Appendix B.3) | about 60% without the extra memory loss, about 88% with it |
| Merge methods, ZsRE average (Table 11) | Ties 0.87, Sign 0.84, Dare 0.74, Linear 0.72, Slerp 0.72 |
| Cost of WISE-Merge | 0.64% more parameters, about 4% more VRAM, about 3% more inference time |

On the layer choice, the paper reports that edits at layers 0, 1 and 31 are "ineffective". Layer 26 gives "an 80% success rate and generalization while maintaining 100% locality". Its reason is that early layers hold grammar, the last layer controls decoding, and mid-to-late layers are redundant.

## Where the claims go beyond the evidence

- **The "impossible triangle" is a framing, not a result.** The paper says the three aims "can not be realized at the same time" for earlier editors. It shows this only for the eight baselines that it ran.
- **The locality test uses unrelated inputs only.** NQ, RedPajama and Pile inputs are far from the edits. Inputs near an edit, which share its subject or its form, are the hard case. The router can fire on them, and the metrics do not test that.
- **No seeds or intervals.** Every number is from one run. Gaps such as 0.78 against 0.75 on Temporal have no scale.
- **Gen. is weak out of distribution.** On Temporal, OOD Gen. is 0.37 for WISE and 0.28 for GRACE. So the triangle is not met there, and the paper does not stress this.
- **The router is the limit, and the paper says so.** The oracle router gives 0.82 at T = 3000 against 0.71 for WISE-Merge. The authors name the gap to the oracle as a limitation, and give credit for that.
- **The layer choice rests on a coarse sweep.** The text gives a reason for mid-to-late layers, but it tests few layers and does not measure where the model stores the edited facts.
- **Small inconsistencies.** Table 3 lists 100 Temporal samples, but Table 5 uses T = 75. The method section uses layer 27, and the layer sweep says layer 26 works.

## Use in this repo

- **No unit-level content.** WISE edits a whole FFN matrix behind a gate. It does not find, read or test single neurons, heads or explanations. So it does not bear on any pole's verdict or on `docs/PROTOCOL.md`.
- **The three axes map onto tools that the repo has.** Rel. is the target token on the edited prompt. Gen. is the same behaviour on sibling prompts. Loc. is no change on other inputs. `docs/analyses/hallucination-shortlist.jsonl` already holds sibling prompts for #4781. Six kept prompts give a wrong answer that copies the first clause's unit. In three prompts the copied unit is the right answer, for example "A spider has eight legs and a dog has four" → " legs" (p 0.5494). That second group is a near-edit locality set, which WISE's metrics leave out. We record this as evidence for the fix campaign. The maintainer designs the campaign.
- **What the paper predicts for the " days" failure.** In #4781 the wrong token comes from a copy route: layer-0 count-word and time-unit neurons, then the copy heads L3/H3, L4/H0 and L4/H7. No verified unit supplies " months". A WISE-style edit adds a side FFN at a mid-to-late layer, which in pythia-70m (6 layers) is about L4 or L5. It would write " months" downstream of the copy route and leave that route as it is. So the paper predicts a high Rel. on the edited prompt. Gen. to "a minute has sixty" → " seconds" would need a separate edit for each fact, because WISE stores facts, not rules.
- **The gate is a poor fit for this failure.** WISE's router fires on inputs near an edit. The failing prompts and the prompts where copying is right share one form. So a gate trained on the failures is likely to fire on " legs" and " wheels" too, which is the near-edit locality case above. This is a prediction from the paper's design, not a result that it reports.
- **Coverage and sisters as measures of side effects.** The paper's Loc. checks only the top output. The repo could also compare prompt coverage (93.8% for the " days" prompt) and the answer route before and after an edit. That question belongs to the maintainer's campaign, and we do not open an issue for it.
- **Related reviews.** GRACE (#5020), MEMIT (#4869), ROME (#4868), TIES-Merging (#4874) and "Knowledge in Superposition" (#4876) cover the baselines and parts that WISE builds on.

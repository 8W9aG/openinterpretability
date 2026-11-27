# Harness Evolution Hits a Ceiling: When Weight Training Should Begin

**Citation.** Yuan Tian and Bing Hu (independent researchers, equal contribution), Hao Wang (UC Berkeley), Binghang Lu (Purdue University) and Fang Wu (Stanford University). "Harness Evolution Hits a Ceiling: When Weight Training Should Begin". [arXiv:2610.11655](https://arxiv.org/abs/2610.11655) [cs.AI], v1 of 8 October 2026, the only version. Preprint, no venue given. It is under the arXiv licence. The paper gives no code or data, and Appendix J says some data "will be released with the code". It states no funding and no use of AI in the writing.

**Status.** Planned. The fix campaign (#5220) compares weight fixes with a prompt (arm E). This paper predicts when each lever pays, and its placebo adapter is a null that #5220's arms lack.

**Read.** On 2026-11-27 a subagent (Claude) read the arXiv HTML in full, from the abstract to Appendices A to J. It saw Figures 2 to 7 only as captions. The reader of this page was a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** An agent can improve in two ways: change its harness, the prompts, tools and checks around a frozen model, or train its weights. Harness changes fix *process* failures, such as loops and plans never delivered. Weight training fixes *content* failures, plans that are delivered but wrong. So the right next move can be read from how the runs fail.

**Method.**

- **The harness loop.** Kimi K3 reads the evidence from a round's runs and proposes two edits. Fixed numeric rules and each edit's own rollback condition accept or reject it. The 4B line ran 13 rounds and the 9B line 5.
- **Benchmarks.** DeepPlanning, 120 travel-planning tasks split once into 48 development and 72 held-out tasks. Also WebArena-Lite, 165 tasks, for Qwen3.5-9B only.
- **Models.** Eight models from six families, from MiniCPM5-2B to Qwen3.6-35B-A3B and hy3. The main pair is Qwen3.5-4B and Qwen3.5-9B.
- **Failure labels.** A rule-based checker gives each failed run the first signal that fires. F1 is missing evidence and F2 is harness friction. F3 is execution control, such as repeats or a spent step budget. F4 is planning capacity, a delivered plan that scores below 0.5.
- **Weights.** LoRA of rank 32 on the attention projections, or on attention and MLP. It trains on the best of the model's own runs on development tasks: 82 runs for 4B and 98 for 9B. The paper also compares full fine-tuning.
- **Placebo.** The same recipe, with the assistant outputs permuted across training samples. The inputs, the harness and the step count stay the same. It exists for 9B only.
- **Noise.** Each cell is the mean of 4 rollouts. A difference counts only above 0.035, from a per-rollout SD of 0.023 over 26 cells.

**Findings, as the source reports them.**

| Result | Setting |
| --- | --- |
| Held-out score 0.164 → 0.297, delivery 55% → 90% | Qwen3.5-4B, seed harness to evolved harness (Table 5) |
| Held-out score 0.323 → 0.442 | Qwen3.5-9B, the same |
| Adapter adds +0.126 (4B) and +0.131 (9B) held-out | On the seed harness, the better placement per model, chosen on development tasks |
| F4 falls from 28.1% to 4.5% of held-out runs | Qwen3.5-9B adapter with MLP, seed harness |
| Adapter adds +0.007 held-out, within noise | Qwen3.5-9B, on the evolved harness |
| Placebo scores −0.158 (development) and −0.227 (held-out) against the base model | Qwen3.5-9B, seed harness. It hits the 100-turn cap in 44 of 288 held-out runs |
| Full fine-tuning beats LoRA by +0.044 (4B) and +0.042 (9B) held-out | A larger set of training runs |
| The harness gives nothing where delivered plans score 0.13 to 0.17 | Three models, MiniCPM5-2B, Spark-X2.5-4B and Ministral-3-8B |
| Test score 0.256 → 0.344 from the harness. Adapters add nothing | WebArena-Lite, Qwen3.5-9B |

## Where the claims go beyond the evidence

- **The rule comes after the data.** The cut-offs, such as delivered plans below about 0.18, are read from the same lines that they sort. No line has delivered quality between 0.17 and 0.27, so the 0.18 line is untested. "Saturated" and "dominant" have no thresholds.
- **One training run per adapter, no intervals.** No adapter or placebo is repeated over seeds. The paper gives no confidence intervals and no tests, only its 0.035 noise line. The authors report drift of up to 0.05 across nights, and two comparisons cross nights, as they say themselves.
- **The weights branch is tested on few models.** "Train the weights" is tested where its condition holds on 9B only. On Gemma 4 12B the gain didn't carry to held-out tasks. The placebo exists for 9B only, by the authors' own account.
- **A placebo that breaks the format.** Permuted outputs teach a model to answer the wrong step, so the placebo loops and hits the turn cap. It shows that the trained content matters. It doesn't show what a harmless adapter of the same size does.
- **No look inside the model.** The claim that the MLP adapter wins on 9B because MLPs hold facts rests on a citation of Geva et al. (2021), not on any analysis of the weights. The paper doesn't say which layers or MLP matrices the adapters change.
- **No matched-budget search.** The authors say they don't compare the loop with test-time search of the same budget.

## Use in this repo

- **The fix campaign's failure is a content failure.** In #5220, the model finishes its answer and names the wrong unit, " days". In this paper's terms that is F4, where the harness lever paid least and weights paid most. So the paper predicts that the prompt arm (E) will lose to the weight arms (B and C) on held-out prompts. AxBench predicts the reverse for steering ([AxBench](axbench.md)). #5220 runs both, so its result tests the two predictions.
- **The placebo is a null that #5220 lacks.** Arm D changes random heads or MLPs by the same number of parameters. It asks whether *where* the fix goes matters. The placebo trains the same parameters on permuted targets, so it asks whether *what* the fix learns matters. A placebo of this kind for arm B would train the same rank-1 MLP adapter on shuffled answer tokens. The paper's placebo broke the output format. A null that keeps the answer a unit word, such as a random unit word in place of the right one, would avoid that. Adding an arm is the maintainer's decision, so this page records it as a question.
- **Same-night comparisons.** The paper finds that one configuration drifts by up to 0.05 across nights, and it compares cells measured on the same night. The repo already asks for a quiet machine when it measures (`choose-next-experiment`, `docs/analyses/device-contention.md`). #5220's arms should run back to back on one machine, with the base model measured in the same session.
- **Read what each accepted edit changed.** The paper's last branch reads the accepted edits. A gain that changed what the model writes went into the weights, and a gain that changed what the model sees stayed in the harness. #5220 plans the same reading of a fix through the vocabulary, to tell "supply" from "suppression".
- **This repo's own loop is a harness.** Agents here change the skills and scripts that drive them (`next-issue`, `work-an-issue`, `land-experiment`). That is a harness that agents improve round by round, as in this paper and in [CoEvoBench](coevobench.md). The paper's Appendix G rule fits: before an edit is judged, show that its code path ran at least once. The repo has no measure of its loop's failures by kind, and no issue asks for one, so this page opens none.

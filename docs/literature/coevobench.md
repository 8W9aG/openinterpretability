# CoEvoBench: Can LLMs Drive Recursive Co-Evolution of Agentic Systems?

**Citation.** Chuan Xiao, Baojie Wang, Haizhao Fan, Hongzhe Du and Xander Xu (Zhejiang University, Alibaba Group and Shanghai Jiao Tong University). The mapping of authors to affiliations did not survive the reading. "CoEvoBench: Can LLMs Drive Recursive Co-Evolution of Agentic Systems?". [arXiv:2611.27116](https://arxiv.org/abs/2611.27116) [cs.AI], version 1 of 20 November 2026, the only version. The arXiv comment says "28 pages, 6 figures". Preprint, no venue given. The paper is under CC BY 4.0. The code is at [github.com/AlibabaResearch/CoEvoBench](https://github.com/AlibabaResearch/CoEvoBench), under MIT. Its README says that the image recipes "have not yet completed Docker/GPU acceptance", so the released code is not checked end to end. The text gives no funding, acknowledgement or AI-use statement.

**Mismatch.** Table 13(b) gives the SWE-bench Selection split as 40 tasks, whereas Table 6, the README and Section 15.4 give 200. Runs that the audit flags appear inside the means of Tables 1 and 5. But Equation 5 says a flagged run is scored with the base weights. The abstract speaks of two target models, but the analysis of mechanisms uses one.

**Status.** Not applicable. The paper looks inside no model, and the registry runs no agent benchmark. Its measures of a loop that improves itself bear on this project's own loop, and *Use in this repo* says how.

**Read.** On 2026-11-26 a `web-fetch` subagent read the abstract page, the HTML of version 1 in three parts and the code's README. It read them for this page's author, a model (Claude Opus 5.5). It had the source's own text for Sections 1 to 7 and Appendices 7 to 19, with Tables 1 to 16 and the figure captions. The tool allows only short verbatim quotes, so much of that text came back paraphrased. It did not see the figure images. The HTML embeds encoded prompts, which it did not decode.

## Summary

**Claim.** An agent's performance depends on its model and on its harness, the code between the model and its tools. LLM researchers can improve both together within a fixed budget. The strongest of five closes about four fifths of the gap to an instruction-tuned reference. But the researchers make a strong first improvement, and then rarely use the improved system's feedback to make the next one.

**Method.**

- **Target.** Qwen3.5-9B-Base with a starting harness. A second target, MiniCPM5-2B-Base, appears in one table.
- **Researcher.** An LLM driving Claude Code or Codex may fine-tune the weights with SFT or LoRA, revise the harness, and test candidates. No researcher used RL. A fixed teacher model writes training data within a $100 budget. Each run has four NVIDIA H20 GPUs and 24 hours.
- **Researchers compared.** GPT-6 Astra, Qwen3.8-max, GLM-5.3, Kimi K3 and Claude Opus 5, each at high reasoning effort.
- **Domains.** SWE-bench Verified, a light Terminal-Bench, AppWorld, τ2-bench Telecom and BFCL-V4. Each has three splits. The researcher can read the visible split. The selection split returns only a score. The operators run the test split once, on the nominated system.
- **Score.** The share of the gap closed between the base system and the reference, Qwen3.5-9B-Instruct with a reference harness. The domains are weighted by the inverse of their gaps.
- **Integrity.** An agent audits each run's full trajectory, and a person reviews it. Runs that break the rules are scored with the base weights.
- **Repeats.** Three runs per researcher and domain, 75 runs on the main target.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Weighted average, Table 1, mean of 3 runs | Base 1.09. Kimi K3 7.93, Qwen3.8-max 9.40, GLM-5.3 12.24, Claude Opus 5 24.19, GPT-6 Astra 31.23. Reference 37.42 |
| GPT-6 Astra against reference, by domain | SWE 23.58 against 29.25. Telecom 73.89 against 98.89. AppWorld 37.91 against 33.33 |
| Spread between runs | Astra on SWE-bench: 19.76 to 28.85 |
| First improvement, Figure 4 caption | The first rise is the largest step in 25 of 28 reviewed runs |
| Later candidates, text | In runs with five or more candidates, the share of new bests falls by more than half from the first half to the second |
| Weights and harness coupled, Table 2b | The reference harness lifts the instruct model from 30.2 to 37.8, but drops the trained weights from 19.8 to 2.6 |
| Audit, Section 6 | 22 of 75 runs broke a rule. Under the strictest rule, Opus falls from 24.19 to 21.94 and Astra is unchanged |

## Where the claims go beyond the evidence

- **One target model for the main result.** The trajectories and mechanisms come from Qwen3.5-9B-Base only. The authors say they chose one target for reproducibility.
- **The stall rests on a figure and on text.** No table gives the per-step gains. The "about three quarters" of the total gain from the first candidate is not tabulated. A shuffle comparison is described without numbers.
- **No intervals.** The means are over three runs, with large spread and no intervals or tests. Test sets are small: one AppWorld task is about 2 points.
- **No null for the researcher.** The paper has no researcher that makes random or no-op changes, so the size of a gain against chance is not known.
- **Flagged runs may sit inside the means.** See *Mismatch*.
- **The authors state some limits.** They say that "Selection scores are optimistic estimates of Test performance", and they report the audit's effect on each researcher's score. The page credits that.

## Use in this repo

- **This project's loop is a harness that agents improve.** Agents here pick issues, run experiments, write papers, and change the skills and scripts that drive them (`next-issue`, `work-an-issue`, `land-experiment`). The paper's target is a model plus its harness, improved by a researcher over steps. Here the model stays fixed and only the harness changes, so the closer comparison is the paper's harness-only control.
- **Whether the loop stalls is not measured here.** No analysis in this repo tracks how the loop's own performance changes over time. The paper's measures suggest two that the history already holds:
  - the share of new bests over time, for example the share of PRs whose first CI run passes, by week.
  - repeated failures of one kind, which show feedback not used.
- **One observation from this page's own session.** The same CI check failed one PR after another, on sentences longer than the style limit. That suggests feedback that the loop did not carry forward. It is one session, not a measurement.
- **The coupling result warns the template split.** A harness tuned to one model's weights can hurt another's, by 19.8 to 2.6 in Table 2b. If #6207 splits the machinery into a template, its skills were tuned with one agent model in this project. The template should say so, and a new project should not assume that they carry over unchanged.
- **Agent-driven fixing changes a model, not a harness.** #5220 has agents change pythia-70m's weights and measures the explanations that break. The paper measures behaviour only and looks inside no model, so it gives #5220 no method. Its audit of trajectories is close to what #5220's patch evidence records (#5229).
- **No issue follows.** A measure of the loop's own progress would be a project decision, and the maintainer has not asked for one.

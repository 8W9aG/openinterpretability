# Recursive Language Models

**Citation.** Alex L. Zhang, Tim Kraska and Omar Khattab (MIT CSAIL). "Recursive Language Models". arXiv 2512.24601v3, 11 May 2026 (v1 31 December 2025). [Paper](https://arxiv.org/abs/2512.24601), [code](https://github.com/alexzhang13/rlm).

**Status.** Not applicable. It is a preprint on an inference scaffold for long prompts, with no venue given. It has no interpretability content: no activation, probe, feature or neuron analysis. The paper names a fine-tuned model, RLM-Qwen3-8B, but doesn't say where its weights are.

**Read.** A text extraction of the v3 PDF (43 pages), including all appendices, on 2026-11-11. The figures are images, so the numbers below come from the text and the tables. The results of the fine-tuning and length-generalisation runs (Figure 3) are in figures only. The code was not read.

## Summary

**Claim.** A long prompt should not go into the model's context. It should go into an environment that the model reads with code, and the code can call the model again on parts of the prompt. With this scaffold, existing models handle inputs beyond their context window and do better than other scaffolds on dense long-context tasks, at a similar cost.

**Method.**

- *What an RLM is.* No new architecture. The harness puts the prompt in a variable in a Python REPL. The root model gets only metadata: the prompt's length, a short prefix and how to read it. The model writes code, and only a short, truncated part of each output comes back into its context. Code can call `llm_query` on any string, for example in a loop over chunks. The answer is a REPL variable, so it can be longer than the model can write in one call.
- *Recursion depth.* Depth 0 has the REPL and no sub-calls. Depth 1 can call a plain model. Depth 2 and 3 can call another RLM. For GPT-5 runs the root is GPT-5 and the sub-calls go to GPT-5-mini.
- *Why not existing scaffolds.* Algorithm 2 lists three "flaws" of common agents. The prompt is in the context window, and the answer must come from one generation. Also, the model names its sub-agents in text and can't call them from code in a loop.
- *Tasks.* S-NIAH has 50 tasks, with constant work per length. BrowseComp-Plus has 150 questions over 1,000 documents (6–11M tokens). OOLONG `trec_coarse` has 50 tasks at 131K tokens, with linear work. OOLONG-Pairs has 20 tasks that the authors wrote, at 32K tokens, with quadratic work and F1 scores. LongBench-v2 CodeQA spans 23K–4.2M tokens. The paper doesn't give its task count.
- *Baselines.* The base model, CodeAct with BM25 or with sub-calls, a compaction agent, OpenCode, and Claude Code with Claude Opus 4.1. In GPT-5 runs, GPT-5-nano writes the compaction agent's summaries. OpenCode and Claude Code also run with the prompt offloaded to a file.
- *Training.* RLM-Qwen3-8B is Qwen3-8B fine-tuned on 1,000 filtered trajectories of Qwen3-Coder-480B-A35B as an RLM on LongBenchPro. A script patched template mistakes in them first: 16% of turns misused `FINAL` and 13% misused `FINAL_VAR`. Training took 300 steps and 48 H100 hours. A separate run trains Qwen3-4B with RL on 64K-token MRCRv2 and evaluates on 1M tokens.

**Findings, as the paper reports them.** GPT-5 rows from Table 1. Each cell is a score (%), from one run.

| Method | CodeQA | BrowseComp+ (1K) | OOLONG | OOLONG-Pairs (F1) |
| --- | --- | --- | --- | --- |
| Base model | 24.0 | 0.0 (input too long) | 44.0 | 0.1 |
| CodeAct + sub-calls | 24.0 | 0.0 (input too long) | 40.0 | 28.4 |
| Compaction agent | 58.0 | 70.5 | 46.0 | 0.1 |
| OpenCode + offloading | 64.0 | 94.0 | 52.0 | 4.8 |
| Claude Code + offloading (Opus 4.1) | 62.0 | 84.0 | 48.0 | 6.5 |
| RLM, depth 0 | 58.0 | 88.0 | 36.0 | 43.9 |
| RLM, depth 1 | 62.0 | 91.3 | 56.0 | 58.0 |
| RLM, depth 2 | 66.0 | 92.0 | 56.5 | 65.5 |
| RLM, depth 3 | 58.0 | 92.0 | 58.0 | 76.0 |

Other results, as reported:

- The abstract's headline is the median, over the four tasks, of the RLM's (depth 1) relative gain. It is 26% against compaction, 130% against CodeAct with sub-calls and 13% against Claude Code. We reproduced all three from Table 1. The Claude Code figure uses the offloading variant.
- With Qwen3-Coder-480B-A35B the RLM also leads on OOLONG-Pairs (23.1 against 2.1 for the best Qwen3-Coder baseline). On CodeQA, depth 0 (66.0) beats every depth with sub-calls (44.0–56.0).
- Figure 1: base GPT-5 falls with length on OOLONG and OOLONG-Pairs. The RLM falls less, and "for context lengths beyond 2^14" it "consistently outperforms GPT-5".
- Cost: the median RLM run is cheaper than the median base-model run, but the mean is higher because of a long tail (Figure 11). Qwen3-Coder made hundreds to thousands of sub-calls on one simple task, where GPT-5 made about ten (Appendix E).
- LongCoT-mini (Table 2): GPT-5.2 solves 38.7%, an RLM 50.6% and an RLM with decomposition hints 65.6%. The abstract-style "69.5% increase" is 65.6 against 38.7.
- RLM-Qwen3-8B beats Qwen3-8B as an RLM by a median of 28% and is more than three times faster (figures only). The MRCRv2 run generalises from 64K tokens and 2 needles to 1M tokens and 8 needles (figure only).
- Appendix B lists what did not work. One prompt for all models failed, and so did models with weak coding skills (Qwen3-8B before training). Thinking models ran out of output tokens, and models misused the `FINAL` tags.

## Where the claims go beyond the evidence

These are our observations, not the paper's. The code may answer some of them.

- **One run, small task sets, no intervals.** No table reports seeds or error bars. OOLONG has 50 tasks: near 56%, one score has a standard error of about 7 points (our calculation). The gaps between depths 1–3 on OOLONG (56.0–58.0) and CodeQA (58.0–66.0), and the RLM's lead over OpenCode on OOLONG (56.0 against 52.0), are inside that. OOLONG-Pairs has 20 tasks.
- **The headline leaves out the strongest baseline.** OpenCode with offloading beats the depth-1 RLM on CodeQA (64.0 against 62.0) and BrowseComp-Plus (94.0 against 91.3). The same median calculation against it gives about 2% (our calculation). The abstract reports Claude Code and not OpenCode. OpenCode's costs are "N/A", so nobody can check the "comparable cost" claim against it.
- **"On GPT-5 ... 13% against Claude Code" compares different models.** The Claude Code rows use Claude Opus 4.1. The RLM rows use GPT-5 with GPT-5-mini sub-calls.
- **A median of relative gains over four tasks.** With four tasks the median is the mean of the middle two. One task is OOLONG-Pairs, where most baselines score near 0.1 F1, so its relative gain is huge (about 580 times against compaction). The median hides that. But it also means the headline rests on two tasks, and which two changes per baseline.
- **The hardest task is the authors' own.** OOLONG-Pairs is new in this paper, built so that the answer needs every pair of entries. It shows the quadratic case well. But here the RLM's design has its largest structural advantage: code loops over sub-calls and builds the answer in a variable. Most baselines must write the whole answer in one generation.
- **The authors tuned the prompt on these models, and results are sensitive to it.** The system prompt has in-context examples written for GPT-5. Qwen3-Coder needed an extra line to stop thousands of sub-calls. Figure 4(a) shows that the in-context examples change OOLONG accuracy a lot. The paper calls the method "task agnostic" because one prompt serves all tasks, but it doesn't say whether these benchmarks guided the prompt.
- **The arXiv abstract overstates the scale.** The abstract on the arXiv page says inputs "two orders of magnitude beyond model context windows". The v3 PDF says "more than an order of magnitude". The largest inputs (11M tokens) are about 40 times GPT-5's 272K window, so the PDF is right.
- **The fine-tuning result is thin.** It is one run, on filtered and patched trajectories, reported in figures only. "Approaches the quality of vanilla GPT-5 on three long-context tasks" can't be checked from the text.
- **The LongCoT gain needs hints written for the task.** Without hints the RLM loses on two of five domains (MATH 26.0 → 5.6, CS 40.4 → 11.0). With hints it gains on all five. The same hints given to the base model lower it from 38.7 to 28.6 (Table 3), so the authors designed the hints for the REPL. The LongCoT run also uses a different harness from the rest of the paper (Appendix C.3).

## Use in this repo

**Not applicable as a method.** The repo explains native units of pretrained models, and its verdicts come from probes run on a pinned model revision (`docs/PROTOCOL.md`). The paper studies a scaffold around a model and doesn't look inside any model.

**Our agents' inputs are short.** Proposer prompts are about 15k tokens and reviewer prompts less (`TODO.md`, *Performance*). A judge prompt holds about 50 windows of 32 tokens (`docs/PROTOCOL.md` §4, *Windows*). The paper's own Figure 1 says the RLM "consistently" beats base GPT-5 only above 2^14 (16K) tokens. Below that, its data gives no reason to add the scaffold, and sub-calls would add cost and the failure modes of Appendix B.

**The proposer already offloads its large input.** The full scan evidence, about 500 KB, is in a JSON file next to the proposer prompt. The proposer has Bash and reads that file only when it needs per-token activations (`.claude/agents/neuron-proposer.md`). This is the paper's depth-0 design, a handle to the data and no copy in the context. It agrees with the paper's Observation 2 that the offloading, not the recursion, does most of the work on long inputs. Nothing here needs to change.

**What doesn't carry over.** Recursive sub-calls, answers built in a REPL variable and the RLM fine-tuning recipe are inference and training details for long inputs. The repo doesn't train agents, and no agent here reads an input near a context limit. The one known context problem is a provider limit on long `/loop` sessions (`TODO.md`, *Pipeline*, DeepSeek and `CLAUDE_CODE_AUTO_COMPACT_WINDOW`). That is the history of a session, not one long prompt, and an RLM would not change it. [Context Language Models](context-language-models.md) uses RLM as a baseline for agent context management.

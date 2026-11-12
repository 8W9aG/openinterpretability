# Context Language Models

**Citation.** Rulin Shao, Pang Wei Koh and 11 others (University of Washington, MIT, Meta Superintelligence Labs, Trillium Labs). "Context Language Models". arXiv 2609.37725v1, 29 September 2026. [Paper](https://arxiv.org/abs/2609.37725), [code](https://github.com/facebookresearch/context-language-models).

**Status.** Not applicable. It is a preprint on agent context management, with no venue given. It has no interpretability content: no activation, probe, feature or neuron analysis. The paper links code but announces no released weights.

**Read.** The HTML (v1) and a text extraction of the PDF, including all appendices, on 2026-11-10. The figures are images, so numbers below come from the text, the tables and the figure labels that the extraction kept. The code was not read.

## Summary

**Claim.** A standard language model can only append to its context. A "context language model" (CLM) can rewrite all of it. Given that control, existing models manage long contexts better and more cheaply than fixed harnesses do, and reinforcement learning (RL) improves the skill further.

**Method.**

- *What a CLM is.* No new architecture and, for most results, no training. The harness mirrors the live context to a file. The model edits the file with Bash, and the edit becomes the context for the next call. If the model doesn't edit, the harness appends its output as usual. Subagents are more context files.
- *Cost metric.* "Prefix-reuse FLOPs": a theoretical count that charges re-prefill for every token after the first changed one. An edit near the start of the context costs about 7.7 times an append-only turn (Appendix C).
- *Steering and evolution.* Instructions in the prompt ("skills") set the editing policy. A GEPA-style loop proposes skills, selects them on a development split, and evaluates the chosen one once on a test split.
- *RL.* Stepwise GRPO on Qwen3.5-9B, with a "success-gated efficiency advantage". Among successful rollouts in a group, cheaper ones get a bonus, weighted 0.25. Failed rollouts get none.
- *Serving.* "Suffix Cache Reuse" keeps the KV cache of spans that an edit only moves, and re-rotates their RoPE keys. This approximates a full re-prefill.

**Findings, as the paper reports them.** Unless noted, the agent is Qwen3.6-27B with a 32K context limit.

| Result | Number |
| --- | --- |
| BrowseComp-Plus, 830 questions | CLM 59.4%, "11.4% relative" above Codex-style summarization, with 21.5% fewer prefix-reuse FLOPs |
| TerminalBench 2.1, 89 tasks | CLM "matches" summarization with 70% of its FLOPs |
| TBLite | CLM 73.7% against 67.0% for summarization, with 91% of its FLOPs |
| Math optimization (Claude 4.6 Sonnet), Table 1 | CLM best of run on all four problems, e.g. Heilbronn 0.03653 against 0.03127 for OpenEvolve |
| EdgeBench-10, 32K | CLM 44.6 at 179 PFLOPs per trial, summarization 42.3 at 437 |
| EdgeBench-10, 128K | CLM 47.3 at 142, summarization 47.8 at 222, CLM with subagents 50.2 at 219 |
| Multi-repository swarm (GPT-5.6-Sol, 272K) | "65% greater downstream speedup" than a summary swarm at the same spend |
| In-context evolution, KV Store test split | 38.3% → 74.2% (the abstract's "up to 35.9 points") |
| RL on Qwen3.5-9B, BrowseComp-Plus, Table 2 | CLM 28.8% → 42.5% at 1.34 PFLOPs per question, summary 34.7% → 42.1% at 2.19 |
| Suffix Cache Reuse | matches standard SGLang with 65.0% of its prefix-reuse FLOPs |

The paper also reports one-sentence instructions that change when and how the model compacts. Those results are in figures only.

## Where the claims go beyond the evidence

These are our observations, not the paper's. The code may answer some of them.

- **The RL comparison isn't like for like.** CLM trains with a 28K budget, 80 turns not counting editing turns, the efficiency advantage and penalties for failed tool calls. The summary harness gets 100 turns and "the task reward alone" (Appendix E). After training the two differ by 0.4 points. On 830 questions near 42%, one accuracy has a standard error of about 1.7 points (our calculation), and there is one run with no seeds. The paper calls this "matching" in Section 5.2 and "outperforming" in the introduction.
- **The RL starting point disagrees with the zero-shot result.** In Table 2, untrained CLM is six points below the summary harness (28.8% against 34.7%). Appendix F reports the same model zero-shot on the same benchmark with CLM ahead (39.9% against 37.7%). The two runs use different budgets and turn limits, so both can be right. But the gain that RL gives depends on which setup counts as "before".
- **CLM gets extra turns and retries.** On BrowseComp-Plus, CLM's editing turns don't count toward the 100-turn limit. When a request would exceed the budget, CLM rolls back and retries up to six times. The paper describes no retry for the baselines. Part of the gain may come from this.
- **Maximum scores, one run.** Math results are "best-of-run" from "one run per method". EdgeBench reports "the best score over three seeds per task". A maximum favours whichever method varies more, and no means or spreads are given.
- **"11.4% higher accuracy" is relative.** The abstract doesn't say so. The absolute gap is about 6 points (59.4% against an implied 53.3%).
- **The results are mixed.** The abstract says CLM "outperforms SOTA context-management strategies across a variety of tasks". On TerminalBench 2.1 it only matches. At 128K on EdgeBench, plain CLM scores below summarization, and only the subagent variant is ahead.
- **The cost metric is theoretical.** Prefix-reuse FLOPs count re-prefill, which is what CLM edits change. The abstract says Suffix Cache Reuse cuts "server-side compute by 35%". The body measures prefix-reuse FLOPs, not server time or energy.
- **The judges differ between training and test.** RL rewards come from GPT-5.4-nano with the DeepSearchQA rubric. Test accuracy comes from Qwen3.5-27B, from the same model family as the agents. The paper reports no check of either judge against human labels.
- **Selection on a noisy split.** Evolution selects skills on a development split of 102 instances with one seed. Only KV Store's test result is in the text. The other three tasks have development numbers only, and one of them goes from 0.0% to 100.0%.

## Use in this repo

**Not applicable as a method.** The repo explains native units of pretrained models, and its verdicts come from probes run on a pinned model revision (`docs/PROTOCOL.md`). The paper studies agent harnesses and doesn't look inside any model. Its open models (Qwen3.6-27B, Qwen3.5-9B) are hybrid linear-attention models not in the registry, and the paper links no CLM-trained weights.

**Our agents don't run out of context.** Proposer prompts are about 15k tokens and reviewer prompts less (`TODO.md`, *Performance*). Each runs as a fresh subagent per experiment. The long `/loop` sessions use Claude Code's own compaction, which the repo doesn't control. The one known context problem is a provider limit (`TODO.md`, *Pipeline*, DeepSeek and `CLAUDE_CODE_AUTO_COMPACT_WINDOW`), which a CLM would not change.

**One habit carries over to prompt optimisation.** The DSPy item (`TODO.md`, *Pipeline*, *Prompt optimisation with DSPy*) would select prompts against registry records with a held-out split. The paper's loop does the same, and shows the risk: selection on one seed of 102 instances, a test result for one task of four. Our reruns agree about half the time (`docs/analyses/prompt-trim-ab.md`), so a single-seed selection here would mostly select noise. Any optimiser should select on replicated runs and report every task's test result. The variance item in `TODO.md` (*Protocol and science*, *Reduce the variance of prompt A/B tests*) gives ways to replicate cheaply.

**What doesn't carry over.** File-based context editing, the efficiency advantage and Suffix Cache Reuse are harness, training and serving details. The repo doesn't train agents or serve models. "Recursive Language Models" (arXiv 2512.24601), a baseline here, has its own page: [recursive-language-models.md](recursive-language-models.md).

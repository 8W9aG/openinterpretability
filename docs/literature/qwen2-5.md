# Qwen2.5 Technical Report

**Citation.** The Qwen Team: An Yang, Baosong Yang and 40 other authors, the last Zihan Qiu, listed alphabetically by first name. "Qwen2.5 Technical Report". [arXiv:2412.15115](https://arxiv.org/abs/2412.15115), version 1 of 19 December 2024, last revised as version 2 on 3 January 2025. Preprint, no venue given. The models are on [Hugging Face](https://huggingface.co/Qwen) and [ModelScope](https://modelscope.cn/organization/qwen), with code at [github.com/QwenLM/Qwen2.5](https://github.com/QwenLM/Qwen2.5). The author block names no affiliation. The report says the hosted models are available from Alibaba Cloud.

**Mismatch.** None found. Section 5 says Qwen2.5-0.5B "outperforms the Gemma2-2.6B on various math and coding tasks". Table 5 shows that Gemma2-2.6B beats it on MMLU, which the sentence leaves out but doesn't contradict.

**Status.** Used. Qwen2.5-0.5B and Qwen2.5-1.5B are configured registry models, and three papers cite this report for them.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 2 for this page's author, a model (Claude Opus 5.5). It read Sections 1 to 7 in the source's own words, in two fetches, with Tables 1 to 17. The reference list past its first 40 entries wasn't read.

## Summary

**Claim.** Qwen2.5 is a family of open models from 0.5B to 72B parameters. Pretraining grew from 7 to 18 trillion tokens, and post-training adds over a million supervised examples and two stages of reinforcement learning.

**Method.**

- **Architecture (Section 2).** The same decoder as Qwen2. It has grouped-query attention, "SwiGLU activation function … for non-linear activation" and rotary position embeddings. The query, key and value maps have a bias, and RMSNorm comes before each block.
- **Tokenizer (Section 2).** Byte-level BPE "with a vocabulary of 151,643 regular tokens", and 22 control tokens. "This expansion establishes a unified vocabulary across all Qwen2.5 models."
- **Data (Section 3).** 18 trillion tokens, filtered by Qwen2-Instruct models as quality scorers, with maths and code from the Qwen2.5-Math and Coder data, and domains re-weighted.
- **Context.** Trained at 4,096 then 32,768 tokens, with the rotary base raised from 10,000 to 1,000,000.

**Findings, as the source reports them.**

| Setting | Qwen2.5-0.5B | Qwen2.5-1.5B |
| --- | --- | --- |
| Layers, query / key-value heads (Table 1) | 24, 14 / 2 | 28, 12 / 2 |
| Tied embeddings, context (Table 1) | yes, 32K | yes, 32K |
| Licence (Table 1) | Apache 2.0 | Apache 2.0 |
| MMLU, base model (Table 5) | 47.5 | 60.9 |
| GSM8K, base model (Table 5) | 41.6 | 68.5 |

## Where the claims go beyond the evidence

- **The data isn't released.** The 18-trillion-token corpus is described, not published, so the filtering and re-weighting can't be checked.
- **No MLP width.** Table 1 gives layers and heads but not the MLP's hidden size. The released configuration files hold it.
- **Benchmarks only.** The report has no analysis of the models' internals.
- **No limitations section.** The report notes areas for improvement, such as reward-model benchmarks that don't predict RL performance, but has no section on limits.

## Use in this repo

- **Our citations check out.**
  - `papers/explanations-across-models` gives 24 and 28 layers, a SwiGLU MLP (Section 2) and 151,643 regular tokens. It quotes "establishes a unified vocabulary across all Qwen2.5 models". All match.
  - `papers/default-on-poles` cites the report for the Qwen2.5 family. That matches.
  - `papers/syntax-positive-controls` (published) says the Qwen models' MLPs are gated. SwiGLU is a gated unit, so that matches. No addendum is needed.
  - The reference names An Yang first and Zihan Qiu last, with 2024. It matches the arXiv listing.
- **The shared vocabulary makes the sister search simpler.** Qwen2.5-0.5B and Qwen2.5-1.5B use one tokenizer, so a sister search between them compares positions without aligning two tokenizations. GPT-2 to Qwen crosses tokenizers, as `papers/explanations-across-models` says.
- **A SwiGLU neuron isn't a GELU neuron.** The registry reads a Qwen neuron as a product of a gate and an up projection. See [`glu-variants.md`](glu-variants.md) for what that changes.
- **The training data is unknown.** An explanation of a Qwen neuron can't be checked against what the model saw in training. The scan corpus is the only text the registry has.
- **No question for the maintainer.**
- **Related pages.** [`glu-variants.md`](glu-variants.md) reviews SwiGLU, and [`gpt-2.md`](gpt-2.md) the registry's other main model.

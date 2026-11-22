# Sparks of Weight-Space Programming

**Citation.** Rikiya Takehi, Jayden Teoh (equal contribution), Alex Zhang and Omar Khattab, all of MIT CSAIL. "Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights". Blog post, rikiyat.github.io, 2 November 2026, with a "Response to Comments" section added on 12 November 2026. [Post](https://rikiyat.github.io/blog/weight-space-programming/), [logs](https://github.com/RikiyaT/lm_understand-logs) (logs, data and networks CC BY 4.0, scripts MIT). An edited Qwen model and a demo are on Hugging Face. A blog post, no venue. The model under study, "Astra", is OpenAI's `gpt-6-astra`, as the logs' README says. **Mismatch:** the README's title is "LLMs can understand and write neural network weights", not the post's. The post calls the logs "complete", and the README says that trial runs of earlier prompts are left out.

**Status.** Planned. #4782 would use it in its strand on weight-space programming. It is also evidence for #5144, which asks how far a reading from weights alone can go. No code, skill or protocol in this repo depends on it. The maintainer designs the campaign to fix models, so the Qwen edit below is evidence only.

**Read.** On 2026-11-21 a `web-fetch` subagent (Claude) read the whole post as extracted text, with the response section, and the logs repository's README and file list. It did not open the experiment folders, the figures, the demos or the papers the post cites. The reader was this page's author, a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** A frontier LLM can read a network's weights to say what it does. It can also write or edit weights to change what it does, from a plain-language request. It needs no gradients. The authors call these "very early results" and say that "Weight-Space Programming is not a method".

**Method.** Eight experiments, each with the LLM working from the weights as text or in a Python sandbox. Some runs allow no tools, and some allow a budget of forward passes.

- **Reading.** Name the game of 4 PPO game policies. Name the shape of 3 small diffusion models. Name the language and the copying heads of the middle four blocks of three 7.16M-parameter character transformers, without running them. Tell a sycophantic Gemma 3 270M from an honest one.
- **Writing.** Write MinAtar game networks and a 1.48M-weight Go network from scratch.
- **Editing.** Fix a Kinetix car policy in one edit. Edit Qwen3.5-0.8B to cut reasoning loops, and try the same on Gemma 3 270M.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| 4 PPO game policies, 17,253 weights each | The right game for all four |
| 3 character transformers, 4 layers × 6 heads, weights only | Named copying heads match the ground truth in each model (2 attempts each). Ground-truth scores: copying heads 0.22–0.55, other heads at most 0.02 |
| Sycophantic against honest Gemma 3 270M, 3 pairs | Weights alone failed in four attempts. Attention patterns on texts the LLM wrote succeeded: in head 10.2, attention to the question's numbers is 94.3% (honest) against 3.5% (sycophant) |
| Kinetix car policy, 171,664 weights | PPO fails in 256 episodes. One edit to 3.5% of the weights succeeds in 1,024 |
| MinAtar, written with no tools, 1,000 episodes | Breakout 517.4 against 25.3 for PPO at 500M steps. Seaquest 53.1 against 88.9. Asterix 53.2 against 55.2 |
| Qwen3.5-0.8B, edit to 131,072 weights (0.016%), 50 forward passes | MATH-500 7.9 → 48.1. GSM8K 13.8 → 67.1. IFEval 21.0 → 62.3. Answers that hit the 32,768-token limit: 93.0% → 33.1% |
| Same edit, after 10 and 25 passes | MATH-500 12.3 and 14.1 |
| Gemma 3 270M, math | 6.8 → at most 7.7: no real gain |

## Where the claims go beyond the evidence

- **One LLM, small samples, no intervals.** One model writes every result. Reading uses 3 or 4 networks, the head test 2 attempts a model, and the sycophancy test 3 pairs. The post gives no error bars and no count of Astra's own runs.
- **The Qwen gain is mostly a fix for runaway reasoning.** With thinking on, 93% of the base model's answers hit the token limit, so its 7.9 on MATH-500 measures loops, not math. The post says that Qwen with thinking off scores higher, but gives no number. So "improves math" overstates what a loop fix shows.
- **Weak baselines.** The MinAtar baselines are PPO runs, "the best of 3 to 10 seeds", that score 25.3 on Breakout. The post answers that its baselines "test narrow claims" and do not rank methods.
- **The readable weights are small or named.** The networks read from weights alone have tens of thousands of weights, or are four blocks of a 7.16M-parameter model. On a 270M-parameter model, weights alone failed and the LLM needed activations.
- **The head test reads one well-known head type.** Copying heads have a known weight signature, the method of Elhage et al. that the post says Astra found by itself. It does not show that other head types can be named from weights.
- **No comparison with people.** "We have not compared Astra with interpretability researchers."

The authors state the main limits themselves. The results are early, and context length and the harness limit them. An edit "could not add an ability the model lacked". The weights-only reading of Gemma failed.

## Use in this repo

- **Evidence for #5144: weights alone go part of the way.** The copying-head result reads a head type from weights with no forward pass. The sycophancy result fails from weights and needs activations. This fits #5144's premise: a token-set or copy rule can show in the weights, and a behaviour that needs context needs activations.
- **The repo's protocol reads activations, and this supports that.** The reviewer writes probes and the pipeline measures them (`docs/PROTOCOL.md`). Exp 7 is a case where reading weights fails and activations on written texts work. That is the kind of evidence that a reviewer's probes give.
- **A proposer could get the weights too.** The repo already reads a unit's output weights through the vocabulary (the `vocab-projection-lookahead` skill and `openinterpretability/harness/jlens.py`). The post suggests that a strong LLM can use raw weights as well. We make no change: a weights-only proposer arm would need its own A/B test, and the post gives no measure on single neurons.
- **The Qwen edit is a fix by weight edit, for #4784.** It subtracts one direction, written by the MLPs of layers 15 and 19, from 64 output columns in each. That is a fix to a model's behaviour with no gradients, of the kind #4784 reviews. The fix campaign is the maintainer's, so this page only records it. The edited model is Qwen3.5, not the registry's Qwen2.5.
- **Neighbours.** [`weight-space-programming-without-transformers.md`](weight-space-programming-without-transformers.md) (Ok and Lee, 2026) uses the same term for tables that an LLM writes, and the post does not cite it. [`models-rewrite-own-weights.md`](models-rewrite-own-weights.md) has a model edit its own weights.
- **No new issue.** The result that bears on open work goes to #5144 as a comment.

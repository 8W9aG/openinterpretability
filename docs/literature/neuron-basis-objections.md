# Returning to the Neuron Basis for Interpretability, Part 1: Answering Three Objections

**Citation.** Jasmine Shone and Lisa Dunlap (Anthropic). Shone's work was part of the Anthropic Fellows Program. "Returning to the Neuron Basis for Interpretability, Part 1: Answering Three Objections". Alignment Science Blog, 23 November 2026. [Post](https://alignment.anthropic.com/2026/neuron-basis-theory/). The post gives no code, no data and no licence. It uses the J-lens files of Gurnee et al. (2026), with no link to them. Part 2 is a companion post (#6584).

**Status.** Planned. Most of the post supports what the registry already does. It treats each sign of a neuron as its own unit, reads the output with the J-lens, and tests claims by ablation. One rule is new here. A sign counts only when it occurs with the gate open, and #7127 asks how many of the registry's Qwen poles that rule would change.

**Read.** On 2026-11-27 a subagent (Claude) downloaded the post's HTML and turned it into text. It read the whole post verbatim, with the Discussion, the Limitations, Appendices A to D and Tables 1, B1, C1 and D1. It saw the six figures only as captions and alt text. It did not read Part 2 or the cited works. The reader of this page was a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** Three observations moved interpretability away from neurons:

- neurons fire on unrelated inputs.
- representations seem to change from layer to layer.
- facts seem spread over many neurons.

Each is a weaker objection than assumed. The key is to read a neuron by what it adds to the residual stream, not only by when it fires.

**Method.**

- **Models:** Llama-3.1-8B-Instruct, Qwen3-8B, Qwen3-14B and OLMo-3-32B, all with SwiGLU MLPs: 3.4 million neurons in all.
- **Units:** each sign of a neuron. A neuron adds `a·w` with `a = silu(g·x)(u·x)`, so it can add `+w` or `−w`, never both at once. A sign counts as used only when it occurs with the gate open (`g·x > 0`). SiLU leaks small activations when the gate is closed, often of the opposite sign. So a closed gate plays the part of a ReLU's zero region.
- **Reading the output:** the J-lens of Gurnee et al. (2026), which reads early and middle layers better than the logit lens. A neuron's *J-share* is the share of its output vector that at most 25 token directions explain.
- **Tests:** zero and resample ablation, adding output vectors at other layers, MLP activation patching, and attention blocking.
- **Scoring:** blind ratings by two Claude judges, with random directions and other neurons as controls. Part 1 has no SAE, transcoder or PCA baseline.
- **Polysemantic types:** type 1 fires on unrelated inputs but adds one concept. Type 2 adds a different concept with each sign. Type 3 adds two unrelated concepts with one sign, and it is the only type in which one activation adds two concepts.

**Findings, as the source reports them.**

| Result | Setting |
| --- | --- |
| Type 3 is at most 0.4 to 1.1% of signed directions with an effect | A random sample of middle-layer neurons, 1,016 to 1,084 signed directions per model, corrected for the checks' 62 to 81% recall (Appendix B.2) |
| 35 type 3 neurons exist, 3 to 17 per model | A screen of every middle-layer neuron. A lower bound (Appendix B.3) |
| Judges rate a neuron's own readout as related to its top passages in 73%, 31% and 19% of ratings, at high, middle and low J-share | Up to 1,000 neurons per model. Control readouts score 4 to 6% (Figure 2d) |
| Output vectors moved to another layer raise their word more than a random vector at 90 to 99.6% of concept-layer pairs | Before the last tenth of depth, 141 to 1,254 kept concepts per model (Table C1) |
| Removing a fact takes 44 to 273 neurons for 15 of 18 facts | Llama-3.1-8B-Instruct. Removing only readable neurons is enough, and random same-size sets never remove it (Table D1) |
| The neurons with the lowest J-share align with later attention heads' queries. 53 to 76% of them do, against 23 to 36% of all neurons | Layers at 10 to 90% of depth (Appendix A.3) |

## Where the claims go beyond the evidence

- **One lens, and a judge from the same lab.** Readability under the J-lens stands in for interpretability, and the authors give the credit for this limit themselves: it is "only a proxy". The blind ratings, the type 3 topic lists and the type 3 link ratings all come from Claude models. No human rates anything.
- **Necessary, not sufficient.** The fact traces show which neurons must go for a fact to go, "not which are sufficient". The authors say so.
- **The answer to Gao is a hypothesis.** Gao (2026) argues that a neuron's weak activations may not mean what its strongest ones mean. The post's reply is that a weak activation adds less of the same concept. It is stated as "We hypothesize", and the post doesn't test it.
- **Scale and code.** The models range from 8 to 32 billion parameters, and the post releases no code or data. The authors name a scaling study as follow-up work.
- **No learned-dictionary baseline in Part 1.** The case for neurons is not made against SAEs or transcoders here. Part 2's uses are "tested against baselines", by the authors' account.

## Use in this repo

- **The unit matches.** The registry audits each sign of a neuron as its own pole (`docs/PROTOCOL.md` §1). That is the post's signed direction, and its type 2 is two poles with two explanations. The post's type 3 rate, at most about 1%, supports auditing each pole as one idea, as the `native-neuron-audit` skill does after Cammarata (2026). The post cites Cammarata as agreeing.
- **The output readout matches.** The `vocab-projection-lookahead` skill already reads a pole's output with the J-lens (`harness/jlens.py`, after Gurnee et al., 2026), for the reason the post gives. The logit lens (`harness/lookahead.output_token_projection`) is the late-layer special case.
- **The gate rule is new here, and #7127 tests it.** For a Qwen MLP the registry's activation is `act(gate(x)) * up(x)`, and a pole is its positive or negative part, with no check on the gate. The post counts a sign only when it occurs with the gate open. Its reason is that SiLU leaks small values of either sign when the gate is closed. The active line is a share of each pole's own largest strength. So a pole that only ever leaks still gets a line it can pass. #7127 asks what share of each Qwen pole's active tokens come with the gate open. It also asks whether any VERIFIED pole is mostly leak. GPT-2 and Pythia use GELU with no gate, so the rule doesn't apply to them.
- **Type 1 and the explanation score.** The post's type 1 neuron fires on inputs that look unrelated, such as passages that will continue with "John", but adds one concept. Such a pole can score low on detection while its output is clear. The registry's probes test the input side, so a type 1 pole may end NEEDS_REVISION with a clean output. #6798 reviews output-centric explanation methods, and the post supports that line of work.
- **Default-on poles.** The post never mentions dense or default-on neurons. Its nearest result, low-J-share neurons that align with later attention queries, is a lead for `papers/default-on-poles/`. That paper asks what GPT-2's default-on negative poles do together. No issue is opened for it here, because #3195 already holds that paper's open question.

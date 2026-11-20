# Abstention Direction in Large Language Models

**Citation.** Koki Itai, Jun Hirako and Hiroaki Yamagiwa, of SB Intuitions Corp. "Abstention Direction in Large Language Models: Mechanistic Interpretability of Abstention Across Query Ambiguity, Knowledge Limits, and Safety Alignment". [arXiv:2611.20441](https://arxiv.org/abs/2611.20441), version 1 of 16 November 2026, the only version. Preprint, no venue given. The paper links no code or data. It states no funding. A section on AI assistants says generative AI helped with language, code, drafting parts of the text and checking maths. Another model, Kimi-K3, generated and checked the dataset.

**Mismatch.** The abstract says steering "increases or decreases abstention across all three perspectives." But the safety prompts already abstain at a rate of 1.0, so steering can't raise that rate. Only the rate of abstaining on answerable prompts rises. The best steering layers (17 to 20) differ from the layers where the classes separate best (21 and 23), which the paper says "roughly coincide."

**Status.** Planned. It is evidence for #5220: a direction that makes a model abstain is a candidate fix for a wrong answer. It extends the [refusal-direction page](refusal-single-direction.md). The fix campaign is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 1 for this page's author, a model (Claude Opus 5.5). It read the source's own text in full: Sections 1 to 8, the limitations and ethics sections, the references and Appendices A to J. We didn't see any figure as an image. **Every abstention rate the paper reports is in a figure**, so this page can give none of them.

## Summary

**Claim.** A model abstains for three reasons: the question is ambiguous, it is beyond the model's knowledge, or it is unsafe. Each reason has a direction in the residual stream, as refusal does (Arditi et al.). The three directions are partly shared, and steering along one changes abstention for the others.

**Method.**

- **Model.** Qwen3-4B, in thinking mode, for the main results. Llama-3.2-3B-Instruct appears in an appendix, in figures only.
- **Pairs.** Each pair shares a template and changes one slot:
  - *ambiguity*: an AmbigQA question with or without its qualifier (1,199 pairs).
  - *knowledge limits*: a Natural Questions question with a real entity, or a fictional one (767 pairs).
  - *safety*: harmful and benign instructions with the action swapped (377 pairs).
- **Direction.** The difference in mean activations at the last token of each layer, from 64 pairs, made unit length.
- **Steering.** Add m times the direction at one layer, with m = ±16 in the main results. Positive m adds abstention, negative m removes it.
- **Measure.** A judge model (gpt-oss-120b) labels each response as abstaining or not. AR is the share of should-abstain prompts that abstain, and OAR the share of answerable prompts that abstain. Each uses 128 test pairs, with three samples per prompt.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Cosine between the directions, each at its best layer (Section 7) | ambiguity–knowledge 0.163, knowledge–safety 0.190, ambiguity–safety 0.092 |
| A random 2,560-dimension unit vector (Appendix E) | cosine about 0 ± 0.02 with the abstention direction |
| Cross-cause steering (Section 7) | the ambiguity direction changes knowledge-limit abstention, but barely safety. The knowledge and safety directions change both other causes |
| Separation of the classes, Fisher ratio (Section 6) | knowledge limits 1.15 at layer 23, safety 2.65 at layer 21 |
| Random-direction steering (Appendix E) | "no clear systematic changes", with no numbers |
| Thinking against non-thinking mode (Appendix F) | the trends correlate at 0.94 (AR) and 0.93 (OAR) |

## Where the claims go beyond the evidence

- **The rates are in figures only.** The text gives no abstention rate, before or after steering. So no effect size can be checked from the text.
- **No measure of right or wrong answers.** The judge labels abstention only. So the paper doesn't show that adding the direction stops wrong answers, or what it costs on right ones. Its one example is a single wrong answer to an ambiguous question that became a request for clarification.
- **The layer is chosen on the test set.** Each result's layer is the one with the largest change between m = −16 and m = +16, on the same test prompts that are then reported.
- **"Shared" rests on small cosines.** 0.09 to 0.19 is above the random scale, but no test is given. Each cosine compares different layers.
- **Fictional entities aren't knowledge limits in general.** Swapping in a made-up entity makes a question unanswerable by construction. That is easier than a real question the model happens not to know.
- **No capability check.** No benchmark is run after steering, and output quality is judged by eye.

The authors state limits themselves: two small models, English only, no out-of-distribution test, an LLM judge, and data generated by another LLM. The ethics section notes that removing the direction can cause confident hallucination.

## Use in this repo

- **It doesn't answer #5220's question.** A direction that makes a model abstain could fix a wrong answer. But it must fire on the wrong answers and not on the right ones. The paper measures abstention, not correctness. So it gives no fix rate and no break rate.
- **The repo's wrong answer isn't a knowledge limit.** The [copy-first-unit paper](../../papers/copy-first-unit/README.md) and the " days" explanation show pythia-70m copying a unit from the first clause. That is a mechanism the model runs, not a fact it lacks. An abstention direction for knowledge limits may not fire there. Our reading, not the paper's.
- **Removing the direction makes hallucinations.** The paper's single examples show the model inventing answers about fictional entities once the knowledge-limit direction is subtracted. That is a direction whose removal causes the failure #5220 wants to fix. It fits the [refusal-direction page](refusal-single-direction.md): one direction can carry a behaviour that many units implement.
- **A direction isn't a unit.** The analysis is of residual-stream directions only. No neuron or head is named. As on the [web-agent page](agent-web-element-hallucination.md), a fix along this direction would have nothing for a registry record to hold.
- **Questions for the maintainer, recorded on this page only.**
  - Should #5220 test an abstention direction against a correctness label, on prompts the model gets right and wrong?
  - Should the fix campaign measure whether removing such a direction makes the model's wrong answers more confident?
- **Related pages.** [`refusal-single-direction.md`](refusal-single-direction.md) is the method it extends. [`steering-vector-reliability.md`](steering-vector-reliability.md) reports how steering varies from prompt to prompt. [`why-language-models-hallucinate.md`](why-language-models-hallucinate.md) argues why a model guesses instead of abstaining.

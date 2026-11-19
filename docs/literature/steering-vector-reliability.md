# Analyzing the Generalization and Reliability of Steering Vectors

**Citation.** Daniel Tan, David Chanin, Aengus Lynch, Brooks Paige, Dimitrios Kanoulas, Adrià Garriga-Alonso and Robert Kirk. "Analyzing the Generalization and Reliability of Steering Vectors". arXiv 2407.12404, v1 17 July 2024, v8 4 May 2025. [Paper](https://arxiv.org/abs/2407.12404). The abstract page links the code at [dtch1997/steering-bench](https://github.com/dtch1997/steering-bench). The HTML gives the AI Centre, Department of Computer Science, University College London, as the affiliation of all authors except Garriga-Alonso. Kanoulas also lists Archimedes/Athena RC. Garriga-Alonso is at FAR AI. The venue is NeurIPS 2024: the conference site lists the work as a 2024 poster ([page 93224](https://neurips.cc/virtual/2024/poster/93224)). That page does not name the track, and we did not read a proceedings page. The NeurIPS page uses the British spelling, "Analysing the Generalisation and Reliability of Steering Vectors". The arXiv abstract page uses the US spelling. The NeurIPS checklist in the HTML says "We will release the codebase publicly", and the abstract page now links the code.

**Status.** Not applicable now, and evidence for the fix campaign later. No code, skill or protocol rule depends on it, and no open issue would use it. The campaign belongs to the maintainer, so this page reports evidence and records questions only.

**Read.** We read the arXiv abstract page and the v8 HTML on 2026-11-18, through WebFetch, in four passes. A model read the text for us. The tool let it quote only short fragments of up to 125 characters, so most sections came back as a summary. The numbers in the findings table came back as verbatim fragments with their figure captions. We read the last 13,765 characters, the end of the NeurIPS checklist, only as a summary. We did not read the PDF, the figures or the appendix tables. We read the NeurIPS poster page for the venue.

## Summary

**Claim.** Steering vectors change a model's behaviour unreliably. In distribution, the effect varies widely from input to input, and some inputs move the wrong way. Out of distribution, the vectors often generalise, but for some concepts they fail under small changes to the prompt.

**Method.**

- **Steering.** Contrastive Activation Addition (CAA) with mean-difference vectors. The vector is the mean difference of residual-stream activations at the answer-option token, between positive and negative examples. The intervention adds λ times the vector at the last token position, at one layer.
- **Layer and strength.** A validation sweep picks the layer: "we fix layer 13 for Llama and layer 21 for Qwen". The multipliers are in the range −1.5 to 1.5 (Appendix D.8).
- **Models.** Llama-2-7b-Chat and Qwen-1.5-14b-Chat. The summary says that appendix figures also cover Gemma-2-2B-IT, Llama-2-70B and Llama-3.1-70b.
- **Data.** "a total of 40 datasets": Model-Written Evals persona categories, with three datasets for each category, plus the sycophancy, TruthfulQA and AI-risk datasets from CAA. Each has "a 40-10-50 train-val-test split". The items are multiple-choice questions with an A/B or Yes/No answer.
- **Distribution shift.** Persona text goes into the system message or in front of the user prompt, as a positive or a negative instruction.
- **Metrics.** Propensity is the logit difference between the positive and the negative answer token. Steerability is the slope of a least-squares line through mean propensity against the multiplier. An input is anti-steerable when its own slope is negative. Relative steerability is the out-of-distribution slope divided by the slope of a vector made on that distribution.
- **Statistics.** The text we read gives no seeds and no interval method. The figures show variance across inputs.

**Findings, as the source reports them.** Each ρ is a Spearman correlation across datasets, from the figure caption named.

| Setting | Result |
| --- | --- |
| Per-input steerability, Section 5 | "Many of these datasets have almost half of the inputs being anti-steerable". Figure 1 shows 13 of the 40 datasets |
| Spurious factors, Section 5 | A/B position and Yes/No token are "often highly predictive of the steerability". The paper gives no percentage. "there is no clear trend for 'A' vs 'B'" (Figure 3) |
| Out-of-distribution against in-distribution steerability (Figure 5) | ρ = 0.891 for Llama-2-7b, ρ = 0.694 for Qwen-1.5-14b |
| Llama against Qwen steerability, same datasets (Figure 7) | ρ = 0.769 in distribution, ρ = 0.586 out of distribution |
| Llama against Qwen variance of steerability (Figure 21, Appendix E.2) | ρ = 0.465 in distribution, ρ = 0.491 out of distribution |
| Relative steerability against the shift in unsteered propensity (Figure 8) | ρ = −0.26 for Llama, ρ = −0.46 for Qwen, after a filter that removes base steerability below 0.25 |
| Cosine similarity of the two vectors against the same propensity shift (Figure 24, Appendix E.3) | ρ = −0.63 for Llama, ρ = −0.86 for Qwen |

The abstract says that "steerability is also mostly a property of the dataset rather than the model". The conclusion asks for more work: "substantial work is needed to improve their reliability".

## Where the claims go beyond the evidence

- **Two models carry the main results.** The authors state this limit themselves: "We only analyse two models in this work". The larger models are in appendix figures, which we did not read.
- **Multiple choice only.** Every metric is a logit difference between two answer tokens. The authors name this limit too. Free generation can fail in ways that a logit slope does not show.
- **One method, one layer, one position.** The study tests mean-difference CAA at one fixed layer for each model. Other ways to make a vector, or other layers, can vary less across inputs. So the result is about this method, not about all steering vectors.
- **"Property of the dataset" rests on correlations.** A ρ of 0.769 across 40 datasets between two models is strong agreement. The variance correlation is only 0.465 to 0.491, and the paper gives no interval for any ρ.
- **No null for anti-steerability.** The paper reports the fraction of inputs with a negative slope, but the text we read gives no noise floor. A slope from seven multipliers on one input can be negative by chance when the true effect is small.
- **A filter chosen by threshold.** Figure 8 drops points with base steerability below 0.25. The text we read does not say if the result holds without the filter.
- **Synthetic data.** Most items are model-written, so the spurious A/B and Yes/No factors can come partly from how the authors made the data. The conclusion says that future work "should at least partially focus on the dataset", which credits this point.

## Use in this repo

- **Our steering is a clamp, not a vector.** Every audit records steering samples on five fixed neutral prompts, greedy, with 24 new tokens. Each prompt runs at baseline, ablated, and clamped at 1, 2 and 4 × `max_activation` (`PROMPTS`, `MULTIPLES` and `MAX_NEW_TOKENS` in `openinterpretability/harness/steering.py`). The `neuron-steering-tests` skill reads the text, not a logit slope. One clamped neuron is not a mean-difference residual direction, so the paper's numbers do not carry over directly.
- **Our strengths already stay near the data.** The skill calls drift only at 4× "off-distribution and weak evidence". The paper also keeps its multipliers between −1.5 and 1.5. Both choices avoid strengths far outside the natural range.
- **But five prompts cannot show per-input reliability.** The paper's main result is that the sign of the effect changes from input to input. Our samples give one continuation for each prompt and strength. They do not record the sign of the change for each prompt, so an anti-steered prompt is not seen as one.
- **The answer setting is the paper's setting.** An answer experiment scores one token at one position. For #4781 the model gives " days" with p 0.7257 and " months" with p 0.0295 (`docs/analyses/hallucination-shortlist.jsonl`). The logit difference between " months" and " days" is the paper's propensity metric with two answer tokens.
- **The shortlist is a small dataset of the same failure.** It keeps 6 of its 35 rows. All six copy the first clause's unit, for example " seconds" for "A minute has sixty seconds and an hour has sixty". The paper predicts that a change that fixes one of these prompts can move others the wrong way. Two kept prompts share " months" as the right answer, and the paper's token biases suggest that the answer token itself can drive the effect.
- **Prompt changes are a test the shortlist does not yet have.** The paper's shift is a persona prefix. For #4781 the same kind of test would put the prompt after a short context. #5015 asks whether the copy route generalises across the shortlist. That is the in-distribution half of the paper's question.
- **#3209 needs a prediction format.** That issue asks whether steering predictions should become a gate, and says that they need a pre-registered format first. The paper gives one form for such a format: a sign and a slope for each input, across fixed strengths. This is evidence for the maintainer's decision, not a proposal to change the decision rule.
- **Questions for the maintainer.** These are not issues, because the campaign is the maintainer's to design.
  1. Should a report on a steering fix of #4781 give the fraction of the six kept prompts that move the wrong way? One prompt alone can hide it.
  2. Should a fix also be tested with a context in front of the prompt, as the paper's distribution shift does?
  3. Should steering samples record a per-prompt sign of change, so that an anti-steered prompt is visible? A change to the samples is tooling, but a gate on them is a protocol change (`docs/PROTOCOL.md` §6).

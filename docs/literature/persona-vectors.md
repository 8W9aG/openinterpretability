# Persona Vectors

**Citation.** Runjin Chen (Anthropic Fellows Program and UT Austin) and Andy Arditi (Anthropic Fellows Program). Then Henry Sleight (Constellation), Owain Evans (Truthful AI and UC Berkeley) and Jack Lindsey (Anthropic). "Persona Vectors: Monitoring and Controlling Character Traits in Language Models". arXiv 2507.21509. Version 1 is from 29 July 2025, version 2 from 31 August 2025 and version 3 from 5 September 2025. We read the current HTML, which we take to be version 3. [Paper](https://arxiv.org/abs/2507.21509), [code](https://github.com/safety-research/persona_vectors). Preprint, no venue given. The arXiv page shows no comment or journal reference. The paper says: "This work was completed as part of the Anthropic Fellows Program." Claude 3.7 Sonnet writes the trait prompts and questions, and GPT-4.1-mini is the judge.

**Status.** Planned. The fix campaign that #4784 prepares for, with #4781 as its first case, would use it. The paper measures the capability cost of two fixes that use one explained direction: steering at inference and steering during finetuning.

**Read.** On 2026-11-18, through WebFetch. A model read the abstract page and the HTML for us, in three parts by offset. It returned short verbatim quotes from Sections 3.3, 4.1, 4.2, 5.1, 5.2, 6.3 and 8, and from Appendices B.2, B.4, J.2, J.7, K and L.1. It returned Table 2 and Table 6 as tables. Other parts came back as its summary. These include the abstract, the method of Section 2, Appendices J.3 to J.5 and the start of Appendix M. The last 22,222 characters, the rest of Appendix M, did not come back: the read failed with an error. We did not see the figures or the PDF. Most MMLU values and steering coefficients are only in figures, so we do not have them.

## Summary

**Claim.** A trait such as "evil", sycophancy or hallucination has a linear direction in the residual stream, a "persona vector". A pipeline finds it from a trait description in natural language. Its projection tracks the trait, and steering along it can reduce or prevent the trait.

**Method.**

- **Models.** Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct.
- **Traits.** Evil, sycophancy and hallucination in the main text. Four more (optimistic, impolite, apathetic, humorous) are in Appendix G.
- **Extraction.** Claude 3.7 Sonnet writes 5 pairs of contrastive system prompts, 40 questions (20 to extract, 20 to evaluate) and a rubric. The model gives 10 rollouts for each question and system prompt. A response stays if it scores above 50 under a positive prompt, or below 50 under a negative prompt.
- **The vector (summary).** At each layer, the vector is the difference of mean residual activations over response tokens, between trait responses and other responses. The layer is chosen by how well steering works. Appendix B.4 says: "We select layer 20 for both evil and sycophancy, and layer 16 for hallucination" (Qwen). For Llama, "layer 16 is the most informative for all three traits."
- **Judge.** GPT-4.1-mini scores trait expression from 0 to 100. Appendix B.2 compares it with human raters.
- **Steering at inference.** The update is h_ℓ ← h_ℓ − α·v_ℓ, "where α is a steering coefficient", applied "at each decoding step".
- **Preventative steering.** The authors "proactively steer the model toward the undesired persona direction during training". The aim is that the finetuning no longer needs to move the model along that direction.
- **Finetuning.** One epoch of rs-LoRA, rank 32, α = 64, learning rate 10⁻⁵, on one H100. The datasets are "explicitly designed to induce specific traits", or are like the datasets that cause emergent misalignment. Each has a "Normal" control version and two trait versions.
- **Metrics.** Trait score from the judge, coherence, and MMLU accuracy for general capability. The "finetuning shift" is the projection on the persona vector of the change in mean activation. The "projection difference" ΔP compares the training responses with the base model's own responses, both projected on the unit persona direction.
- **Baselines.** Cross-trait correlations for the monitoring results. 500 random samples as the control group for data selection. CAFT (Concept Ablation Fine-Tuning) and a train-time regularizer as other fixes.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Projection at the final prompt token against the trait score of the response, both models, system-prompt variations (Section 3.3) | r = 0.75–0.83 |
| Table 2, evil, overall / within condition | System prompting 0.747 / 0.511. Many-shot 0.755 / 0.735 |
| Table 2, sycophancy, overall / within condition | System prompting 0.798 / 0.669. Many-shot 0.817 / 0.813 |
| Table 2, hallucination, overall / within condition | System prompting 0.830 / 0.245. Many-shot 0.634 / 0.400 |
| Finetuning shift against trait expression after finetuning (Section 4.2) | r = 0.76–0.97, "higher than cross-trait baselines (r=0.34–0.86)" |
| Steering at inference (Section 5.1) | It reduces the trait, but "large steering coefficients tend to degrade accuracy, indicating a loss in general capability". Coherence stays above 75 |
| Preventative steering at one layer (Section 5.2) | It "better preserves the model's general capabilities compared to inference-time steering". It "does not always fully prevent trait acquisition" |
| Preventative steering at many layers (Appendix J.3, summary) | The trait stays near the base level, "still without incurring any MMLU degradation compared to regular finetuning" |
| Coefficient rule (Appendix J.2) | "the largest steering coefficient for which the model's coherence remains above 80" |
| CAFT (Appendix J.4, summary, no numbers) | It works for evil and sycophancy, and does not work for hallucination |
| Train-time regularizer (Appendix J.5, summary, no numbers) | It "can reduce the projection shift to some extent", and the authors judge it less effective |
| Judge against humans (Appendix B.2) | Agreement 284 of 300 (94.7%), from 50 pairwise judgments per trait with 2 raters each |
| Data selection by ΔP, LMSYS-Chat-1M (Section 6.3, summary, no numbers we read) | The 500 samples with the highest ΔP cause more trait expression than 500 random samples. This holds after the authors remove the samples that the judge flags |
| Sample-wise filter (Appendix K) | 95th-percentile thresholds against 20,000 UltraChat samples. The authors repeat each finetuning run five times |

Appendix M trains SAEs on "every 4 layers" of Qwen. Its figure caption says that "a small subset of features are highly aligned with the persona vector". We did not read the numbers in that appendix.

## Where the claims go beyond the evidence

**Two models, of one size.** The authors say this themselves: "Experiments are limited to two mid-size chat models (Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct)."

**The capability cost is one benchmark.** General capability is MMLU accuracy, with coherence as a floor. A fix can keep MMLU and still change other behaviour, such as facts that are not multiple choice. Appendix J.7 tests fact acquisition, but with one set of 1,000 facts and two coefficients.

**The coefficient is chosen on the result.** The rule in J.2 picks the largest coefficient that keeps coherence above 80. The judge scores coherence on the same outputs that give the trait score. This is a rule set after the data, not a held-out choice.

**No random-direction null.** The model that read the paper found no comparison with a random direction of the same norm. The cross-trait baselines are a null for monitoring, and their range (0.34–0.86) overlaps the main range (0.76–0.97). For steering, no null tells how much of the trait drop any large push would give, and how much comes from this direction.

**The judge is a model, and the data is model-written.** One model writes the prompts and the rubric, and another scores the responses. The authors say: "The judge is not perfect". The human check is 300 judgments.

**Few intervals or seeds.** We found five repeats in Appendix K and no confidence intervals in the text that we read. Most results are in figures that we could not see.

**Credit.** The authors state that the pipeline is supervised, that the trait must be inducible by a system prompt, and that the directions are "coarse-grained". They also say that single-turn questions "may not fully reflect" deployment, and that ΔP needs a base-model response for each training prompt.

## Use in this repo

**The repo already computes the same direction, and finds it is not a neuron.** A persona vector is a mass-mean direction: a difference of class means. `open-interpretability probe-compare` fits mass-mean probes on the MLP hidden layer and on the residual stream (`openinterpretability/harness/probing.py`). In `docs/analyses/probe-comparison.md`, the mass-mean direction's cosine with a VERIFIED pole is at most 0.19 on the hidden layer. On the residual it is at most 0.36 (L6/N4837+). So a fix along a mass-mean direction would not be a fix at an explained unit. The paper does not claim that it is.

**Our steering tests explain, and the paper's steering fixes.** The `neuron-steering-tests` skill and `openinterpretability/harness/steering.py` clamp a pole at 1, 2 and 4 × `max_activation` on five neutral prompts. The aim is to test what the unit makes the model write. `docs/PROTOCOL.md` calls drift only at ×4 off-distribution. The paper's Section 5.1 gives the cost of a large push in a fix: MMLU falls. So a clamp that fixes one answer at ×4 is likely to cost capability elsewhere.

**What the paper predicts for the #4781 copying failure.** The verified route copies " days" through the layer-0 count-word and time-unit neurons and the copy heads L3/H3, L4/H0 and L4/H7. No verified unit supplies " months". The paper suggests that subtracting a " copy the first unit" direction at inference could move the answer. It also predicts a capability cost that grows with the coefficient. The paper has no case of a fix that adds a missing fact at inference. This is a prediction from the paper, not a measurement.

**What a fix must report, from the paper's metrics.** The paper reports three things for each fix: the target behaviour, coherence and a general capability score. For #4781 the target is the answer on the six prompts in `docs/analyses/hallucination-shortlist.jsonl`. The repo has no general capability score for a fixed model today. The strided next-token loss of the scan (`openinterpretability/harness/scan.py`) is one candidate. Which score to use is for the fix campaign, which the maintainer designs.

**The repo's nulls are stronger than the paper's.** Our causal scores rank a unit against matched same-layer random units (`docs/PROTOCOL.md` §4). A fix that steers a direction or a unit could keep that rule: compare with random directions of the same norm. The paper does not do this.

**Preventative steering does not apply yet.** It is a change to finetuning. The registry studies fixed checkpoints and trains nothing. It would apply only if the campaign chooses to finetune.

**Questions for the maintainer.**

- Should a fix in the campaign report a capability score beside the target answer, as the paper reports MMLU? If so, which score?
- Should the steering coefficient of a fix be chosen on held-out prompts, and not by the coherence rule of Appendix J.2?

These are campaign design decisions, so this page records them and opens no issue.

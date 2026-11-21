# Disentangling MLP Neuron Weights in Vocabulary Space (ROTATE)

**Citation.** Asaf Avrahamy, Yoav Gur-Arieh and Mor Geva, Blavatnik School of Computer Science and AI, Tel Aviv University. "Disentangling MLP Neuron Weights in Vocabulary Space". arXiv:2604.06005 [cs.CL], v1 7 April 2026, v2 4 August 2026. The arXiv comment says "Accepted at COLM 2026". [Paper](https://arxiv.org/abs/2604.06005), [code](https://github.com/AsafAvr/rotating-neurons), licence CC BY 4.0. Funding is from Google's Academic Research Program, the Blavatnik Family foundation, an Alon Scholarship, a Coefficient Giving grant and the Israel Science Foundation. The evaluation uses Gemini models as judges and generators. **Mismatch:** the method is called a rotation, but with one Householder matrix it is a reflection, as the paper says itself.

**Status.** Planned. #5144 asks how far weight-only checks can go, and #5871 would test ROTATE's channels against VERIFIED explanations. The repo has already tested ROTATE's cheap statistic, the vocabulary-projection kurtosis (#3226), and found no signal. No code, skill or protocol depends on it.

**Read.** On 2026-11-21 a `web-fetch` subagent (Claude) read the abstract page and the whole HTML of v2, in five parts. We have every section and appendix in the source's own words, with all tables and figure captions. We did not see the figure images, so numbers given only in figures are missing. We did not read v1 or the PDF. The reader was this page's author, a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** A neuron's weight vector, projected onto the vocabulary, can be split into a few "vocabulary channels". Each channel is a direction near the neuron's own weights, with a sharply peaked (high-kurtosis) projection onto the vocabulary. The method uses the weights alone, with no forward pass. The channels describe the neuron better than SAE features or activation-based baselines do.

**Method.**

- **The objective.** For a weight vector w, find v = wR that maximises log(1 + excess kurtosis of vU), with a cosine term that keeps v close to w. R is a Householder reflection. After each channel, the tokens it explains are masked, and the next channel is found. There are 50 channels a neuron, with 3,000 AdamW steps each.
- **Inputs.** The gate rows (input side) and the down-projection columns (output side), with the unembedding U. The gate is used because "the gating activation is mostly positive".
- **Why kurtosis.** Known concept vectors in W_out sit at the 90th (Llama-2-7B) and 95th (OLMo-7B) percentile of random neurons' kurtosis.
- **Models.** Gemma-2-2B-it (layers 4 and 18) and Llama-3.1-8B-Instruct (layers 18 and 22), 100 random neurons a layer.
- **Evaluation.**
  - *Faithfulness:* 40 generated sentences that should activate each channel, against 40 neutral ones, with a t-test.
  - *Completeness:* a five-way LLM match on the neuron's top Pile inputs, where chance is 20%.
  - *Ablations:* removing a channel from the weights.
  - *Descriptions:* head-to-head judged by Gemini.
- **Baselines.** SAE features with the highest dot product with the gate (Neuronpedia and TopK SAEs), an iterative logit lens, FastICA, and activation-based descriptions.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Input faithfulness, four layer settings (Table 1) | ROTATE 0.46–0.71, SAE baselines 0.33–0.49, random 0.17–0.25 |
| Completeness, five-way match (chance 20%) | ROTATE 49–60%, SAE baselines 36–49% |
| Iterative logit lens and FastICA, 20 Gemma neurons | Faithfulness 0.10 and 0.25, completeness 4% and 18% |
| Output side: remove a channel, then decode (Table 2) | Its own top tokens fall by 87–90% (± 34–40). Other channels' tokens rise 14–24% (± 55–60) |
| Stability, 50 Gemma neurons, 4 seeds | Channel cosine 0.9 ± 0.04, Jaccard 0.8 ± 0.05 |
| Without masking | Explained norm 0.19, against 0.78 with masking |
| Channel top tokens in the raw projection, 20 Gemma neurons | 97.1% rank 200th or lower. Median rank 122,302 of 256,000 |
| Descriptions against MaxAct++, 150 Llama neurons | ROTATE wins 63–69% on Pile ranks 100–500 |
| Compute | 3.8 GPU-hours for a Gemma layer, 6.7 for a Llama layer, on one H100 |

## Where the claims go beyond the evidence

- **The kurtosis evidence is thin.** It is given in one figure, as percentiles of known concept vectors on the output side only, with no sample size, AUC or test. The input side, where ROTATE's main results are, has no such check.
- **No random-direction control.** Faithfulness and completeness are compared with SAE features and chance, not with channels from a random rotation or a random neuron's weights. So part of the gain may come from any high-kurtosis direction.
- **The baselines get fewer components.** ROTATE uses 50 channels a neuron, and the SAE baseline 15 features.
- **One run, no intervals.** Only the stability appendix uses seeds. The main tables are means with no intervals or tests, and the many per-channel t-tests have no correction.
- **The judges are LLMs.** Completeness and the description contests rest on Gemini judges, which the authors check on 50 instances with three annotators.
- **Small inconsistencies.** The stopping rule differs between §4, Algorithm 1 and Appendix C. The ablation formula differs between §5.2 and Appendix E. The compute total does not match the stated 11 minutes an iteration (our arithmetic gives about 18 GPU-hours for a Gemma layer). Explained norm "approaches 1.0" in one place and is 0.72 in another.
- **"Data-free" covers only the decomposition.** Choosing hyperparameters and every evaluation use text and forward passes.

The authors state some limits. The method is biased toward features that align with the vocabulary, so units such as confidence-regulation or positional neurons are out of scope. They test two layers each of two models. The output side has no completeness test.

## Use in this repo

- **The cheap statistic has failed here.** [`docs/analyses/weight-kurtosis.md`](../analyses/weight-kurtosis.md) (#3226) tests ROTATE's vocabulary-projection kurtosis as a predictor of VERIFIED. Every AUC interval includes 0.5. On units in the second half of each model, output kurtosis scores 0.49 (0.45–0.54) and input kurtosis 0.53 (0.48–0.57). Added to the predictor of [`verify-predictor.md`](../analyses/verify-predictor.md), the four kurtosis columns change the AUC by −0.008 to 0.008. That page notes that it tests only §3, not the decomposition.
- **The decomposition is a different question.** ROTATE's channels are a weights-only *explanation*, not a score for choosing neurons. The test that fits the registry asks whether a VERIFIED pole's top channel names the tokens of its explanation. It must do so more often than a random neuron's channels. #5871 holds that test.
- **It bears on #5144's ceiling.** ROTATE can only find vocabulary-aligned features. So its channels should match VERIFIED explanations that are token sets, and miss the ones that need context. #5144 counts how many VERIFIED explanations are token sets. The two together would say how much of the registry a weights-only method can reach.
- **The masking idea fits `neuron-family-triage`.** The skill sorts neurons into families from weight statistics. ROTATE's step of masking explained tokens and finding the next channel is a way to see whether a neuron's projection has one family or several. We make no change until #5871 shows that the channels match explanations.

# Does Steering Break Your Model?

**Citation.** Haotian Yang, Huikang Jiang, Yucheng Wu, Wen-Jie Jiang, Chenpeng Wang, Yibin Lou and Liangming Pan (corresponding author). "Does Steering Break Your Model? A Multi-Dimensional Evaluation Suite for LLM Steering Methods". arXiv 2610.07722, version 1, 6 October 2026. The affiliations in the HTML text include Peking University, Columbia University, YiXin-AILab and Southern University of Science and Technology. The text did not say which author is at which place. [Paper](https://arxiv.org/abs/2610.07722), [code](https://github.com/yht0511/steerscope). The abstract page gives no code link, but the HTML front matter and the reproducibility statement do. Preprint, no venue given. Funding: "CCF-Tencent Rhino-Bird Open Research Fund (No. CCF-Tencent RAGR20260112)". GPT-4o-mini is the judge for the main scores, and DeepSeek-V3.2 wrote the training pairs.

**Status.** Planned. The fix campaign of #4784 would use it to say what a steering fix must report besides its effect. #3209, which asks whether steering predictions become a gate, would use its side-effect scores. No repo code depends on it today.

**Read.** On 2026-11-18, through WebFetch. A model read the abstract page and the v1 HTML for us, in four passes, one of them from character 100,000 to the end. The tool limits each quote to 125 characters. So we have short verbatim quotes from the abstract, Sections 1, 4.1, 5.1, 5.2 and 5.3, the conclusion and Appendix D. We have Table 1 as the model copied it, row by row. The rest came back as its summary: the metric definitions, the composite side effect, the factor choice and Appendix Tables 8 to 11. We did not see the figures, the PDF or the plots of the capability and safety results (Figures 9 and 14 to 31). The model found no limitations section.

## Summary

**Claim.** The paper gives SteerScope, an evaluation suite with 15 metrics on two axes: the result of steering and the properties of the method. It tests 23 methods. In the authors' words, "no evaluated activation steering method achieves higher efficacy without incurring greater composite side effects". Also, "current activation steering methods do not yet surpass the Prompt Steering baseline".

**Method.**

- **Models.** Gemma-2-2B-it and Gemma-2-9B-it. Single-site methods act on the residual stream at layer 20.
- **Concepts and data.** The concepts are "sampled from the GemmaScope SAE concept list on Neuronpedia" (CONCEPT500, from AxBench). DeepSeek-V3.2 writes 72 contrastive pairs per concept, which gives 144 labelled rows (summary).
- **Methods.** Optimization-free: DiffMean, PCA, LAT, Spherical Steering, HiDRA, AUSteer, SAE and SAE-A. Learned: Linear Probe, SSV, ReFT-r1, LoReFT, A-PSR, S-PSR, RePS, ODESteer and StepODESteer. Hypernetwork: HyperSteer and FLAS. Baselines that do not use representations: Prompt Steering, Simple Prompt Steering, SFT and LoRA (summary).
- **Random control.** "As a control, Random applies DiffMean after randomly partitioning the training examples."
- **Efficacy.** GPT-4o-mini scores Concept Expression on AlpacaEval instructions from 0 to 2. It also scores Instruction Relevance and Fluency. "Overall" is the harmonic mean of the three (summary).
- **Side effects.** Fluency, SuperGLUE, MMLU, MATH, IFEval, Instruction Relevance, JailbreakBench attack success and false refusal, BBQ bias and TruthfulQA. Llama-3.1-8B-Instruct judges the JailbreakBench responses (summary).
- **Composite side effect.** Each metric becomes a degradation: the unsteered score minus the steered score. The scale is 2 for the judge scores and 1 for the others. The 11 top-level metrics have equal weights (summary).
- **Generalization and data.** Cross-lingual retention on X-AlpacaEval in Chinese, Korean, Italian and Spanish. Sample efficiency at 6, 12, 36 and 72 rows. Sample sensitivity at 24 rows with five subset seeds (summary).
- **Factor choice.** For each method, the factor is the one with the best mean Overall score on the in-distribution prompts. It then stays fixed (summary). The authors say this choice may give some optimism.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| A-PSR, both models, factor chosen by Overall | "A-PSR improves Concept Expression by 1.20 and 1.30 points on Gemma-2-2B-it and Gemma-2-9B-it" |
| A-PSR, Gemma-2-9B-it | The gain of 1.30 "incurs an average normalized degradation of 0.12". The value 0.11 for 2B came only from a summary, so we did not verify it |
| All activation steering methods, both models | "no activation steering method dominates the Prompt Steering baseline on the efficacy–side-effect trade-off" |
| Method families | "methods with learned interventions tend to achieve more favorable trade-offs than optimization-free methods" |
| Capability metrics | Single methods "can locally surpass Prompt Steering on particular capability metrics". The main text gives no numbers for MMLU, MATH or the safety metrics |
| Cross-lingual, both models (Section 5.3) | "target efficacy transfers more readily than behavioral quality". Activation methods "tend to exhibit lower Cross-Lingual Retention than Prompt Steering" |
| Sample efficiency, 9B (Section 5.2) | "HyperSteer and DiffMean retain a relatively large fraction of their full-data effect with only a few examples" |
| Conclusion | Activation steering "remains valuable for its adjustable access to high-efficacy operating points" |

Table 1 gives cross-lingual results on Gemma-2-2B-it at layer 20, as the model copied the rows. Degradation is against the unsteered model in the same language, and a positive value is worse. Retention is the ratio of out-of-distribution (OOD) to in-distribution (ID) gains in Overall. The table leaves out methods with an ID Overall gain below 0.1.

| Method | α | Concepts | Concept gain ID / OOD | Instruction degradation ID / OOD | Fluency degradation ID / OOD | Retention |
| --- | --- | --- | --- | --- | --- | --- |
| Prompt Steering | 1.0 | 50 | 0.9570 / 1.0288 | 0.1110 / 0.1972 | 0.0140 / 0.0162 | 1.0026 |
| Simple Prompt Steering | 1.0 | 50 | 0.7650 / 0.8675 | 0.3040 / 0.4510 | 0.0190 / −0.0040 | 0.9835 |
| LoRA | 1.0 | 50 | 1.0780 / 1.1198 | 0.3040 / 0.4745 | 0.0280 / 0.0608 | 0.9211 |
| A-PSR | 2.0 | 50 | 1.3120 / 1.3385 | 0.4840 / 0.5848 | 0.0680 / 0.0472 | 0.9157 |
| S-PSR | 4.0 | 50 | 0.7840 / 0.6642 | 0.5920 / 0.5235 | 0.3770 / 0.3442 | 0.8156 |
| LoReFT | 1.0 | 50 | 1.2070 / 1.0920 | 0.4200 / 0.5715 | 0.0380 / 0.1000 | 0.7909 |
| SFT | 1.0 | 13 | 1.1731 / 0.8769 | 0.6846 / 0.6856 | 0.0962 / 0.0846 | 0.6824 |
| ReFT-r1 | 1.0 | 50 | 0.6830 / 0.6708 | 0.3190 / 0.5620 | 0.2060 / 0.3952 | 0.6580 |
| HyperSteer | 1.0 | 50 | 0.7930 / 0.7995 | 0.3430 / 0.7212 | 0.1560 / 0.4458 | 0.5745 |
| RePS | 14.0 | 50 | 0.9190 / 0.8925 | 0.4790 / 0.8200 | 0.1690 / 0.4288 | 0.5732 |
| SSV | 2.0 | 50 | 0.7060 / 0.6785 | 0.5410 / 0.7667 | 0.3480 / 0.5220 | 0.5731 |
| DiffMean | 0.8 | 50 | 0.3140 / 0.3093 | 0.2650 / 0.6062 | 0.2600 / 0.5775 | 0.4615 |
| FLAS | 2.5 | 50 | 1.0610 / 0.4652 | 0.6710 / 0.5720 | 0.1460 / 0.2805 | 0.4256 |

We counted the rows. Instruction degradation is larger OOD for 11 of the 13 methods, and fluency degradation for 9 of 13. Both counts agree with the text.

The model also gave Appendix Tables 10 and 11 as a summary, so we did not verify these numbers. On 2B, the ID Overall gain is 0.0738 for SAE, 0.0593 for SAE-A and −0.0055 for Random. On 9B it is 0.0793, 0.0683 and 0.0267. These values are below 0.1, which agrees with their absence from Table 1.

**Statistics.** Appendix D gives "pointwise 95% percentile bootstrap confidence intervals from 5,000 concept-level resamples". The authors add that "they do not capture training-seed or additional generation and judging uncertainty". The sample-sensitivity runs use "five training-subset seeds". Sampling is at temperature 1.0 with base seed 42 (summary).

## Where the claims go beyond the evidence

**One model family, one layer.** All results are from two Gemma 2 instruction models, and the single-site methods act at layer 20. The title asks about LLM steering in general.

**The prompt baseline is not tested at a range of strengths.** Prompt Steering has α = 1.0. The activation methods have their factor chosen for the best Overall score. The comparison is at one operating point for each method. The conclusion itself says that steering gives "adjustable access to high-efficacy operating points", which one point cannot test.

**Selection on the evaluation set.** The factor is the best one on the in-distribution evaluation prompts. The authors say so and call it a source of "some optimism". The OOD runs keep that factor, so they are a fairer test.

**A judge model scores both efficacy and quality.** GPT-4o-mini scores Concept Expression, Instruction Relevance and Fluency on a scale of 0 to 2. The intervals do not include judge noise, as Appendix D states. We saw no test of agreement with human ratings in the text that we read.

**The concepts come from SAE labels.** The concept list comes from GemmaScope SAE labels. A method that does not use the SAE must express the label's concept, and the label may not describe the SAE latent well. We did not see a test of this.

**Unequal sample sizes in Table 1.** SFT has 13 concepts, the other rows 50. The summary says the main setup uses 20 for SFT. We did not see the reason.

**The capability and safety results are in figures.** The main text gives no numbers for MMLU, MATH, JailbreakBench, BBQ or TruthfulQA. We did not see those figures, so this page cannot say how large those effects are.

**Credit.** The suite has a random control, bootstrap intervals over concepts and a seed study for sample sensitivity. The authors state the limit of the factor choice and the scope of their intervals themselves. The code is public.

## Use in this repo

**Our steering is one neuron, clamped, and never scored for side effects.** `openinterpretability/harness/steering.py` continues five fixed neutral prompts for 24 tokens, greedy. It ablates the pole, or clamps it at 1, 2 and 4 times `max_activation`. The `neuron-steering-tests` skill reads the text by eye. It calls drift only at 4× "off-distribution and weak evidence". The paper measures what our samples leave unmeasured: fluency and instruction relevance against the unsteered model. Its Table 1 shows that these two can degrade while the target effect holds.

**We set our multiples before we see the data.** The paper chooses each factor on its own evaluation set. Our 1, 2 and 4 multiples are constants in `steering.py`, the same for every unit. So our samples do not have the paper's selection problem. The cost is that one multiple can be too weak for one unit and too strong for another.

**The units and models differ.** The paper steers directions in the residual stream of Gemma 2 models with 2B and 9B parameters. Our units are single MLP neuron poles and attention heads, mostly in pythia-70m. The paper reports no single-neuron method. Its SAE rows are the nearest to our units, and in the summarised appendix they have small Overall gains, under 0.1. So the paper's ranking of methods does not transfer. Its method of evaluation does.

**What the paper predicts for a steering fix of #4781.** On "A week has seven days and a year has twelve", pythia-70m answers " days" at 0.73 and " months" is at rank 4 (`changelog.d/4781.added.md`). The VERIFIED route copies " days" through the copy heads L3/H3, L4/H0 and L4/H7, and no verified unit supplies " months". Steering against that route is one possible fix. The paper predicts three things. A steered fix can lift " months" on this prompt. It will cost more on other text than the fix gains. A prompt that states the answer's unit can do better on that trade-off. The other five prompts in `docs/analyses/hallucination-shortlist.jsonl` also copy the first clause's unit, so they are an OOD set for this test. This is a prediction from the paper, not a measurement.

**A random control matches our nulls.** The paper's Random control applies the same method to random splits of the data. Our analyses use matched random units, for example 10.1% coverage for random units against 93.8% for the #4781 explanation. A fix experiment could use the same idea: steer a matched random unit at the same multiple, and report both.

**Questions for the maintainer.** First, must a fix by steering report side effects, and which ones? The candidates are fluency on neutral text, the other shortlist prompts and registry coverage. Second, must it compare against a prompt baseline? Both are fix-campaign design decisions, so this page records them and opens no issue for them. The page proposes one follow-up that is not about the campaign: a side-effect score for the existing steering samples. That score bears on #3209.

# Understanding, Predicting, and Steering LLM Behavior via Representational Effective Theory

**Citation.** Muhammed Ustaomeroglu and Guannan Qu, Carnegie Mellon University. Ustaomeroglu is the corresponding author. "Understanding, Predicting, and Steering LLM Behavior via Representational Effective Theory". [arXiv:2605.09294](https://arxiv.org/abs/2605.09294) [cs.LG], version 1 of 10 May 2026, version 2 of 26 September 2026 and version 3 of 31 October 2026. Preprint, no venue given. The code is at [github.com/ustaomeroglu/RET](https://github.com/ustaomeroglu/RET). The sycophancy data is at [ustaomeroglu/sycophancy-bench](https://huggingface.co/datasets/ustaomeroglu/sycophancy-bench). A [project page](https://ustaomeroglu.github.io/RET/) links both. We saw no funding statement.

The paper discloses its tools. Claude Sonnet 4 wrote and filtered the sycophancy data and wrote the descriptions that the interpretability score compares. GPT-5.4 Thinking named the clusters. Claude Sonnet 5 wrote the synthetic alignment-faking trajectories. Five Claude and Gemini models judge the interpretability score.

**Mismatch.** The project page gives an interpretability score of 5.61 ± 1.65 for RET against 4.24 for an SAE. The text of version 3 gives 5.74 ± 1.59 against 3.52 ± 1.82. Appendix N's text cites a configuration ($d_z$ = 64 with a 4-layer encoder) at a closure of 0.949, but Table 6 has no such row.

**Status.** Not applicable now. RET learns summaries of a whole prefix of hidden states. Our review audits single units on single tokens, so nothing here can use RET directly. No open issue would use it. Its early prediction of sycophancy is evidence for the fix campaign (#5220), and *Use in this repo* says how.

**Read.** On 2026-11-26 a `web-fetch` subagent read the abstract page, the HTML of version 3 in six parts and the project page. It read them for this page's author, a model (Claude Opus 5.5). It had the source's own text for Sections 1 to 7 and Appendices E to G and I to N. That text includes Tables 2 to 8. Appendices A to D came back only as a summary, among them the limitations of Appendix B and the closure protocol of Appendix D. We saw no figure as an image, so the values of Figures 2 and 4 are missing. We did not read the code, the data or the video.

## Summary

**Claim.** A few learned variables, called macrostates, summarise a language model's hidden states over a generation. Their dynamics are nearly closed, and they trace interpretable paths through reasoning. They predict sycophancy and alignment faking early, and steering toward them changes the model's reasoning.

**Method.**

- **Microstate and macrostate.** The microstate $x_t$ is the prefix of hidden states at one middle layer of a frozen model, up to token $t$. A macrostate is $z_t = f_\theta(x_t)$, a 128-dimensional vector by default.
- **Encoder.** One causal transformer layer with rotary positions, then a linear map and a layer norm.
- **Closure.** The aim is that $z_t$ predicts $z_{t+1}$ as well as the full prefix does. Training uses a BYOL- or JEPA-style loss. A two-layer MLP $T_\phi$ predicts the next macrostate, which an EMA copy of the encoder gives (momentum 0.996). The loss is L1. There is no term against collapse.
- **Supervised variant.** An optional head adds a mean-squared error on labelled tokens. The head is then dropped, and a difference-of-means probe is fitted on the frozen macrostates.
- **Models.** GPT-OSS-20B at layer 11 of 24, Pythia-160M at layer 6 of 12, Qwen2.5-14B-Instruct at layer 23, Qwen3.5-35B-A3B and Qwen3.8-27B at layer 31 of 64.
- **Data.** NuminaMath for GPT-OSS-20B, six stories with planted boundaries for Pythia-160M, MMLU, TinyStories, an expanded sycophancy benchmark, and 250 alignment-faking scenarios.
- **Clusters.** K-means with 64 clusters on normalised macrostates, merged into 12 groups. GPT-5.4 Thinking names them from sample spans.
- **Interpretability score.** One Claude Sonnet 4 describer reads each representation's codes, and a second reads the raw text. Five judge models rate how similar the two descriptions are, from 1 to 10.
- **Baselines.** The raw hidden state, its top 128 principal components, a same-layer TopK SAE with 131,072 latents, a temporal SAE, pooled hidden states, and logistic-regression probes. There is no neuron-level baseline and no untrained encoder.
- **Steering.** Gradient steps on the hidden state at the middle layer move the macrostate toward a target cluster, or away from one to avoid.

**Findings, as the source reports them.** All numbers are from the text or its tables.

| Setting | Result |
| --- | --- |
| Closure, GPT-OSS-20B, held-out self-prediction $R^2$ (Table 6) | RET 0.908 (0.902–0.922 over three seeds), hidden state 0.620, PCA 0.720, SAE 0.490, temporal SAE 0.610 |
| Closure over all RET configurations tried (Appendix N) | 0.795 to 0.949 |
| Interpretability score, 100 held-out NuminaMath responses (Table 2) | RET 5.74 ± 1.59, temporal SAE top-1 4.86 ± 1.49, SAE 3.52 ± 1.82 |
| Information per 50-token window (Section 4.1) | RET 3.6 nominal bits, SAE 148.2. The paper says this "favors SAE" |
| Paths over planted story boundaries, Pythia-160M, 6 stories (Table 7) | Boundary contrast: RET 0.643 ± 0.026, pooled hidden states 0.237, SAE 0.032. Tortuosity: RET 16.9, pooled 13.1 |
| Alignment faking from the middle of the reasoning, Qwen3.8-27B (Section 5) | Supervised RET 0.958 balanced accuracy over 3 seeds, on 35 scenarios. Moral content alone 0.672 |
| Sycophancy after 5–10% of a turn (Section 5) | Supervised RET is best in both settings. The text gives no number, only Figure 4 |
| Steering toward one cluster, 33 held-out prompts (Table 5) | RET: mean marker gain +4.18, 30 of 33 prompts gain. One SAE latent: +2.91, 20 of 33. Five SAE latents: +3.24, 20 of 33 |

## Where the claims go beyond the evidence

**The interpretability score compares unequal inputs.** The RET describer reads cluster names that GPT-5.4 wrote, while the SAE describer reads top-activating tokens. The score then rates the similarity of two model-written descriptions. No human rated them, and the score does not test a description on new text. The authors call it "only a coarse, LLM-based complement".

**No untrained or random control.** Closure, the interpretability score and the paths are compared with other representations of the same trained model. No row uses a random-init encoder or an untrained model, so the paper cannot say how much of the score the training adds.

**The baselines differ in what they read.** A macrostate summarises a whole prefix with a learned transformer layer. An SAE latent reads one token. A closure or path advantage may come from the prefix, not the macrostate. The pooled baseline, which also reads a window, is smoother than RET (tortuosity 13.1 against 16.9), and the text calls the two "comparable".

**Few cases and no tests.** Alignment faking has 35 test scenarios, so one changed prediction moves the balanced accuracy by 0.02 to 0.05. Steering has 33 prompts. The paper reports seed spreads, which it calls "a coarse heuristic", and no interval or significance test.

**Steering is scored by its own markers.** The steering gain counts hedging and verification words such as "maybe" and "wait". The same words are among the top tokens of the SAE latents in the comparison, so the measure may favour either side.

**Prediction rests on supervision.** The early prediction of sycophancy and alignment faking is strongest for the supervised variant, which trains on the labels. The unsupervised macrostates stay "close to" the content-only baseline on alignment faking.

The authors state some limits themselves. They call the JEPA form "not its definitive form". They note that steering "cannot create a viable path" to an unavailable behaviour. They call their robustness results "a floor".

## Use in this repo

**A different unit at a different scale.** Our dataset explains a model bottom-up, one neuron pole on one token at a time (`docs/research-questions.md`, question 0). A macrostate is a learned summary of a whole prefix. It is one vector for a span of text. No macrostate has a pole on a token, so our adversarial review cannot audit one as it stands. The two do not compete for the same answer.

**Macrostates and our poles are not compared.** The paper has no neuron baseline. A test of whether macrostates line up with our passing poles would need RET trained on one of our models, such as Qwen2.5-0.5B. It would also need a readout of each cluster on our poles' top contexts. We open no issue for it, because RET's prefix encoder mixes many tokens and a match would be hard to read.

**Dense learned units fail our review.** Our audit of rank-1 LoRA directions found that dense learned directions are active on many tokens and fail their hard negatives (`papers/lora-units/`). A macrostate is denser still. The paper's interpretability score cannot show this, because it never tests a description on held-out text.

**The score needs the null that our reviews ask for.** Heap et al. find that automated interpretability scores barely tell a trained model from a random one (`random-transformers-autointerp.md`). RET's score has no random-encoder control. Our own explanation scores carry a sanity check of this kind (`scripts/score_sanity.py`, `evaluating-neuron-explanations.md`).

**Learned bases against native units.** Our evidence on question 3 is in `papers/native-vs-learned/`. On GPT-2 small, a transcoder latent passes more often than a positive neuron pole, 37 of 40 against 20 of 40. On Qwen2.5-0.5B, the native basis beats a random rotation, 18 of 30 against 1 of 30. RET adds no evidence there, because it does not audit units.

**Steering.** Our steering tests clamp one pole and read what the model writes (the `neuron-steering-tests` skill). RET steers a learned summary by gradient steps on the hidden state, a stronger intervention. Its 33-prompt result agrees in direction with two earlier reviews (`contrastive-activation-addition.md`, `steering-vector-reliability.md`). Both find steering effects that vary a lot across prompts.

**Early prediction for the fix campaign.** A probe that flags sycophancy after 5–10% of a turn could tell a fix when to act (#5220). The supervised variant needs labelled conversations of the behaviour, and the campaign does not yet target sycophancy. Whether it should is a question for the maintainer, so we record it here and open no issue.

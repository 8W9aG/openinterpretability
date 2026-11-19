# Rank-1 LoRAs Encode Interpretable Reasoning Signals

**Citation.** Jake Ward (MATS), Paul Riechers (Simplex) and Adam Shai (Simplex). Riechers and Shai are equal senior authors. "Rank-1 LoRAs Encode Interpretable Reasoning Signals". [arXiv:2511.06739](https://arxiv.org/abs/2511.06739), version 1 of 10 November 2025, the only version on the abstract page. Venue: the NeurIPS 2025 Mechanistic Interpretability Workshop, from the PDF's footer. Code: [jnward/reasoning-lora-interp](https://github.com/jnward/reasoning-lora-interp). Adapter weights: [jnward2/qwen-reasoning-lora](https://huggingface.co/jnward2/qwen-reasoning-lora) (Appendix A.1). Dashboard: [jakeward.page/lora_sae_dashboard.html](https://jakeward.page/lora_sae_dashboard.html) (Appendix A.2). The authors did the work in the MATS 8.0 program. The paper names no other funding. It discloses its tools: gpt-5-mini writes and classifies the explanations, and claude-opus-4.1 writes the categories (Appendix B).

**Status.** Planned. The follow-up below would audit rank-1 adapter directions as registry units, beside random native poles. #5133 (the weight change at a pole) and #5125 (narrow fine-tunes) would use the paper's rank-1 form. No code, skill or protocol depends on it. The maintainer designs the campaign to fix models, so this page reports evidence and records questions only.

**Read.** On 2026-11-19 we downloaded the arXiv v1 PDF (15 pages) with `curl` and converted it with `pdftotext`. The reader was this page's author, a model (Claude Opus 5.5). We read all of the text in the source's own words: the abstract, Sections 1 to 5, the references and Appendices A to C. That includes the prompts in Appendix B, and Tables 1 and 2 as extracted text. We did not see the figures as images. For Figure 1 we have the numbers that its text layer holds, and for Figures 2 to 10 we have the captions only. Figures 4 to 10 are dashboard screenshots. We did not read the code, the weights or the dashboard. We only checked that the code and weights links answer.

## Summary

**Claim.** A rank-1 LoRA on every projection matrix of Qwen-2.5-32B-Instruct recovers 73–90% of the gain that a full fine-tune gives on reasoning benchmarks. Each adapter component gives one scalar per token, and these scalars are about as often monosemantic as MLP neurons. They fire more often on reasoning-specific text.

**Method.**

- **Training.** The base model is Qwen-2.5-32B-Instruct. The data is s1k-1.1: 1,000 DeepSeek R1 chain-of-thought trajectories, about 10 million tokens. The authors train for 5 epochs with cross-entropy, on one 8×H200 node. The LoRA adapts `up_proj`, `down_proj`, `gate_proj` and Q, K, V and O in every layer. It has less than 0.03% as many trainable parameters as the base model. The paper gives no learning rate or LoRA scale.
- **The unit.** For an N × M matrix, rank 1 makes `lora_A` and `lora_B` vectors. The activation is the scalar between them, the projection of the adapted matrix's input onto `lora_A`. There are 192 MLP and 256 attention components, 448 in all. The activation has a sign, and the dashboard shows both signs.
- **Direction-level interpretation.** For each unit, the top 64 contexts over the training set go to gpt-5-mini. It writes an explanation of at most ten words. Then it rates the unit 0 (monosemantic), 1 (broad but consistent) or 2 (polysemantic). The baseline is "the first 60 neurons of each MLP" in the base model, scored the same way. An LLM writes categories from an equal number of neuron and LoRA explanations, and gpt-5-mini assigns each unit to one.
- **Cross-layer SAE.** A batch-top-k SAE with k = 16 and an expansion factor of 8 reads the whole 448-dimensional adapter state. The same pipeline explains and rates its latents.
- **Ablation.** The authors zero one component, or one layer's components, and measure the KL divergence of the output from the full LoRA. They also remove all attention adapters, or all MLP adapters, and run the benchmarks.

**Findings, as the source reports them.** All numbers are from the text or Tables 1 and 2, except where the row says Figure 1.

| Setting | Result |
| --- | --- |
| AIME'24 (no-figures), base / rank-1 LoRA / full fine-tune (Table 1) | 0.2333 / 0.5000 / 0.6000, recovery 72.73% |
| MATH500, same (Table 1) | 0.8340 / 0.9100 / 0.9220, recovery 86.36% |
| GPQA-Diamond, same (Table 1) | 0.4899 / 0.5808 / 0.5909, recovery 89.90% |
| Units rated monosemantic, LoRA / MLP neurons (Figure 1, text layer) | 22.4% / 20.2%. Broad but consistent: 28.5% / 32.9%. Polysemantic: 49.1% / 46.9% |
| Category enrichment, LoRA over MLP (Figure 1, text layer) | Answer and solution markers 8.5×. Instruction and procedural markers 3.0×. Mathematical operators 2.2×. Domain content words 0.63×. Function words 0.35× |
| SAE latents, without the dead ones (Section 3.1) | About 2,000 |
| SAE latents rated monosemantic (Section 3.1) | 62%, "up from 22% of LoRA features". Another 22% broad but consistent |
| Single-component ablation (Section 3.2) | Layers 44, 45, 46 and 62 move the KL most. MLP components move it more than attention, and `gate_proj` most on average |
| Attention adapters removed (Table 2) | AIME 0.3667, MATH500 0.9000, GPQA 0.5152: 50.02%, 86.84% and 27.83% of the LoRA's gain |
| MLP adapters removed (Table 2) | AIME 0.1333, MATH500 0.8440, GPQA 0.5051: −37.50%, 13.16% and 16.72% of the LoRA's gain |
| Steering with one adapter direction (Section 4) | "Inconclusive". A visible effect needed more than 50× the normal activation, and then the model often wrote "Wait," and other backtracking tokens |

The Figure 1 numbers come from the figure's text layer. Each set of three sums to 100%. We matched them to the two bars by those sums and by the text's 22%.

## Where the claims go beyond the evidence

- **"As interpretable as MLP neurons" rests on one LLM's rating of top-64 contexts.** The paper does not score an explanation against held-out text, simulate it or test it causally. A top-activation rating says nothing about where the unit fires below its top. The authors state this limit themselves: the autointerpretation methods "often fail to uncover the 'true' role" of a feature.
- **The neuron baseline is not random.** The text takes "the first 60 neurons of each MLP". The caption of Figure 1 calls them "arbitrarily sampled". The two descriptions disagree, and index order is not a random draw.
- **The neuron count and the neuron are not stated.** From 192 MLP components we infer 64 layers, so 3,840 neurons against 448 adapter units. The paper does not say which SwiGLU value it reads as a neuron's activation.
- **No interval, test or seed.** 22.4% against 20.2% has no interval or test, and the paper reads it as "roughly the same likelihood". Table 1 is one run per arm. The AIME scores are multiples of 1/30, so 0.5000 against 0.6000 may be a difference of three problems. The paper does not state the problem count.
- **The full fine-tune is not described.** The paper does not say who trained the full fine-tune, or with which recipe. So "recovery" compares with a reference whose setting the reader cannot check.
- **The interpretations are not linked to the ablations.** The KL ablation ranks components by effect. No result says whether a component with a clean explanation is one that matters. Steering was inconclusive, as the authors report.
- **The categories are subjective.** The authors say that the categorization "is subjective and sensitive to prompt verbiage". They offer Figure 2 "only as a tool to develop high-level intuitions".
- **One model and mostly maths.** The authors say they study one model on mostly math-related benchmarks, with a sample-efficient dataset.
- **Selected examples.** Appendices A.2 and A.3 call their examples "cherry-picked". The full dashboard is online.

The authors released the code and the adapter weights, so a reader can repeat the work.

## Use in this repo

- **The registry has no LoRA code.** `git grep` finds no LoRA or adapter code in `openinterpretability/` or `scripts/`. The only fine-tuned weights that the repo reads are the instruct models of [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md), which it did not train.
- **Three of the seven adapter types already fit a unit kind.** A `down_proj` adapter's `lora_A` is a direction over the MLP's hidden activations. That is the site of a superposition (`docs/PROTOCOL.md` §1, `openinterpretability/harness/superpositions.py`). A `gate_proj` or `up_proj` adapter reads the normed MLP input. Qwen's RMSNorm has a gain and no bias, so its activation is `(g ⊙ lora_A) · x̂`. That is a one-latent dictionary at `mlp_in_normalized` (`openinterpretability/core/profiles.py`). No preset reads a LoRA file today.
- **The fit has two gaps.** First, a superposition and a `basis` latent are the positive part of one direction. The adapter's scalar has a sign, so it needs two units, `+lora_A` and `−lora_A`, as a neuron has two poles. Second, the protocol defines a superposition as sparse, and `lora_A` is dense over all neurons. The Q, K, V and O adapters read sites that `DictionarySite` does not list, so the registry cannot audit them yet.
- **A `down_proj` adapter is a readout of the base model's neurons.** Its activation is a fixed weighted sum of the base model's neuron activations. So its loading on VERIFIED poles is a free check of whether its explanation comes from a few known neurons. The paper does not do this check.
- **The repo's comparison is stricter than the paper's.** The registry's rate comes from adversarial probes and gates, over random draws. [`docs/analyses/registry-stats.md`](../analyses/registry-stats.md) gives Qwen2.5-0.5B random poles a first-reading VERIFIED share of 30.2% (27.1–33.5%). [`docs/analyses/transcoder-baseline.md`](../analyses/transcoder-baseline.md) found 37 of 40 transcoder latents VERIFIED against 20 of 40 GPT-2 positive poles. So learned units and native poles can differ by a lot under this protocol. The paper's 22.4% against 20.2% cannot go on this scale.
- **A rank-1 change makes #5133's measure a closed form.** A rank-1 adapter on `gate_proj` adds `B_i · (lora_A · x)` to the pre-activation of every neuron `i`. All neurons share one scalar, and each one's share is its `B_i`. So the row norm of Δ at a pole is `|B_i| · ‖lora_A‖`, times the LoRA scale. #5133 reads that norm for the full instruct change. A rank-1 run would let a test ask whether a pole that keeps has a small `|B_i|`, or a trigger that is far from `lora_A`. This fits as a comment on #5133.
- **A rank-1 arm of #5125 would count the change.** On Qwen2.5-0.5B, 24 layers (`docs/analyses/sva-circuit.md`) give 168 components, and 72 of them are MLP components. After each stage, the arm can audit the adapter units and test the VERIFIED poles again, with one model and one corpus. This fits as a comment on #5125.
- **Both results put the change late.** The paper's largest single-component effects are at layers 44 to 46 and 62, of an inferred 64. In `base-instruct.md`, three of the five failures on Qwen2.5-1.5B are in the last third. On 0.5B, the middle third keeps the least (63 of 92). The settings differ in model, data and measure, so this is a weak link only.
- **Steering needs the repo's own checks.** The paper needed more than 50× the normal activation for a visible effect. The `neuron-steering-tests` skill clamps a pole within its own range, so an adapter unit would get a fair test there.
- **Neighbouring reviews.** [`lora-low-rank-adaptation.md`](lora-low-rank-adaptation.md) reviews the method. [`lora-illusion-of-equivalence.md`](lora-illusion-of-equivalence.md) finds that LoRA adds new singular directions, and a rank-1 adapter adds at most one per matrix. [`dora-weight-decomposed-lora.md`](dora-weight-decomposed-lora.md) splits each neuron's change into scale and direction. [`crosscoder-chat-tuning.md`](crosscoder-chat-tuning.md) and #3763 read the change between two models with a learned basis, as this paper's SAE does.
- **A question for the maintainer.** Should an adapter direction count as its own unit kind, with its own rate, as superpositions do? Or should it count as a superposition, with a looser sparse rule? That changes `docs/PROTOCOL.md` §1 and needs a `PROTOCOL_VERSION` bump, so this page records it as a question only.
- **Not a fix campaign.** A rank-1 adapter is a small, inspectable weight change, so it could be a tool to fix a model. Whether to use it so is the maintainer's decision. This page proposes only evidence work.

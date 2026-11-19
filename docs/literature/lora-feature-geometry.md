# Feature Geometry of LoRA Adapters

**Citation.** Prasanth K K, "Independent AI Safety Researcher", the only author. "Feature Geometry of LoRA Adapters: A Sparse Autoencoder Analysis of Representational Divergence in Fine-Tuned Language Models". [arXiv:2605.28896](https://arxiv.org/abs/2605.28896) [cs.LG], version 1, 27 May 2026. It is the only version. The arXiv licence is CC BY 4.0.

- **Venue.** Preprint, no venue given. It is not peer-reviewed.
- **Code and data.** The paper gives no code link and no data link. The only repository link in the text is the Stanford Alpaca dataset. The adapters, the delta SAEs and the activations are not released.
- **Funding and tools.** The text names no funding. It thanks the Gemma Scope, SAELens and TransformerLens teams, and states that the experiments ran on an Apple M4 Studio.
- **Mismatch in the references.** The paper cites "Evans et al. [2025]" for *Emergent misalignment*, with the authors "Evans, O., Cotton-Barratt, O., Finnveden, L., …". On 2026-11-19 we read the arXiv record of that paper, [arXiv:2502.17424](https://arxiv.org/abs/2502.17424). Its first author is Jan Betley, and Owain Evans is the last of its eight authors. The author list in the citation does not match that record.

**Status.** Not applicable. The paper's main geometric measures are at the level of random directions, and its adapters did not change the model's outputs. The repo tracks fine-tuning at the native neuron, where no base-model dictionary can miss a change.

**Read.** On 2026-11-19 we downloaded the arXiv abstract page, the HTML page and the PDF of version 1 with `curl`. We extracted the PDF's text with `pdftotext`. A model (claude-opus-5-5) read it, not a person.

- **Our own text of the source.** All 18 pages of the PDF: the abstract, sections 1 to 9, the acknowledgements and the references. Tables 1 to 12 and Algorithm 1 came through as text.
- **Not seen.** We did not look at Figures 1 to 7 as images. The text extraction gives only their captions. So this page takes no numbers from the figures, except the numbers that their captions state.
- **Not used.** We used the HTML page only to look for code links, and we did not read it in full.
- **Checks of our own.** We ran a seeded simulation of the random baselines for the cosine and principal-angle measures (in *Where the claims go beyond the evidence*). We also read the arXiv record of one cited paper.

## Summary

**Claim.** A LoRA adapter adds to the residual stream a change that a base-model SAE does not capture well. An SAE trained on that change alone reconstructs it better. Its feature directions align only weakly with the base SAE's directions. So base-model interpretability tools may miss what an adapter adds.

**Method.**

- **Model.** Gemma-2-9B (`google/gemma-2-9b`), 42 layers, d_model 3584.
- **Adapters.** Four LoRA adapters, ranks 4, 8, 16 and 32, with α = 2r. They target the q, k, v and o projections only, not the MLP. Training is 10,000 Alpaca samples (indices 0–9,999) for 3 epochs, with seed 42.
- **The delta.** For each input, h∆ = h_adapted − h_base in the residual stream after layers 5, 10, 18, 22, 32 and 38. The paper writes h∆ as (α/r)·BAx, and calls it "the exact adapter contribution".
- **Delta SAEs.** One standard ReLU SAE per rank and layer, 24 in all. Each has 16,384 features, L1 coefficient 0.15, RMS normalisation of the input, and 10 epochs. The text says that they train on about 1M token vectors.
- **Data for the SAEs.** An "activation probe set" of 2,000 Alpaca samples, indices 5,000–6,999, in five categories of 400 each. The held-out set is 200 samples, indices 11,000–11,199.
- **Base dictionary.** Gemma Scope residual-stream SAEs for Gemma-2-9B (`google/gemma-scope-9b-pt-res`), width 16,384.
- **Measures.** (1) The relative reconstruction error of each SAE on h∆. (2) For each delta feature, its largest cosine with any Gemma Scope decoder direction. (3) Principal angles between the top-256 SVD subspaces of the two decoder matrices. (4) Linear CKA between h_base and h∆. (5) The share of features that fire on h∆ and also on h_base.
- **Baselines.** Gemma Scope is the only comparison. The text gives "≈ 0" as the expected cosine for random vectors. It has no base-against-base SAE baseline.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Mean ‖h∆‖₂, layer 5 to layer 38 (Table 4) | 18.13–22.25 at layer 5, 320.67–345.45 at layer 38. The text calls this "approximately 18×" |
| Gemma Scope relative error on h∆, 24 conditions (Table 5) | Above 1.0 in all 24. 2.150–2.457 at layer 5, 1.131–1.217 at layer 38 |
| Delta SAE relative error, held-out set (Table 6) | 0.340–0.358 at layer 5, 0.606–0.652 at layer 38 |
| Delta SAE improvement over Gemma Scope (Table 7) | 46.3% (layer 38, r = 4) to 86.2% (layer 5, r = 4). Positive in all 24 conditions |
| Share of Gemma Scope features on h∆ that also fire on h_base (Table 8) | 0.0032–0.0037 at layer 5, 0.0409–0.0628 at layer 38 |
| Active features per token, layer 38 (Table 8, Figure 4 caption) | 30.28 at r = 4, 41.66 at r = 32 |
| Largest cosine of a delta feature to Gemma Scope, 7 conditions (Table 9) | Mean 0.0701–0.0714, median 0.0655–0.0671. 0.01–0.02% above 0.7 |
| Principal angles, k = 256, 15 conditions listed (Table 10) | Mean 72.39°–74.99°. "Mean (all 24)" 73.72°, 66.3% above 70°, 0.0% below 20° |
| Principal angles at k = 64, 128, 256 and 512 (§6.3) | Mean angles "within ±2" of the reported values |
| Linear CKA(h_base, h∆) (Table 11) | 0.050–0.080 at layer 18, 0.304–0.362 at layer 5, 0.668–0.693 at layer 38 |
| Linear CKA(h_base, h_adapted) (Table 11) | 0.935–0.999 in every condition |
| Final training loss by rank (Table 2) | 1.1272 (r = 4) to 0.9914 (r = 32) |

The authors also state that the adapters "produce outputs identical to the base model on test prompts" (§7.4).

## Where the claims go beyond the evidence

**The cosine result is at the random level, not above it.** The paper says that a mean largest cosine of about 0.071 is "slightly above the expected value for random unit vectors" of "≈ 0". But each value is a maximum over 16,384 directions. We simulated isotropic random unit vectors in 3,584 dimensions, with 16,384 base directions and 2,048 query directions, seed 0. The mean largest signed cosine was 0.066, and the mean largest absolute cosine was 0.069. The paper does not say which of the two it uses. Either way, 0.0701–0.0714 is the value that unrelated directions give. Real decoder directions are not isotropic, so this simulation is only a first null.

**The principal angles are also at the random level.** We drew two random 256-dimensional subspaces of a 3,584-dimensional space, seed 0. Their mean principal angle was 76.7°, 74.6% of the angles were above 70°, and none were below 20°. The paper reports 73.72°, 66.3% and 0.0%. So two unrelated subspaces are, if anything, a little more apart than the paper's subspaces. The paper calls 0% aligned "rigorous" evidence of separation. That share is what chance gives.

**The authors name the missing baseline themselves.** Section 7.4 states that a base-against-base SAE baseline is absent. It says that without one, the paper cannot tell whether 0.071 "is genuinely low or attributable to SAE training variance". We agree, and the two simulations above show that a simpler null already removes the effect. Yet the abstract and the conclusion state the separation as a finding.

**The adapters did not change behaviour.** The authors state that the adapters gave outputs identical to the base model, and that this "precludes causal validation". CKA(h_base, h_adapted) is 0.935 or more everywhere. So the paper studies a small change with no measured effect on outputs. A claim about a "monitoring gap" in safety fine-tuning needs an adapter that does something.

**Gemma Scope's error on h∆ tells little.** Google DeepMind trained Gemma Scope on full residual-stream activations, and h∆ is a difference of two of them. A dictionary fails on inputs from another distribution, whatever their geometry. This point is our inference. If few features fire, the output is near the decoder bias. Then a relative error above 1.0 can come from the bias alone. An SAE trained on the same distribution will always do better. So Tables 5 to 7 do not show a separate subspace.

**h∆ is not BAx.** The adapters change q, k, v and o in many layers. So the residual change after layer ℓ also carries every earlier layer's change, through attention, the MLPs and the norms. This point is our inference from the paper's own setup. Equation 2 holds for one projection's output, not for the residual stream. So h∆ is not "free of base model signal", as section 3.1 says.

**The SAE data overlaps the adapter's training data.** The probe set is indices 5,000–6,999, and the adapters trained on indices 0–9,999. So the delta SAEs trained on text that the adapters saw. The held-out set (11,000–11,199) avoids this for Tables 6 and 7 only.

**One model, one dataset, one seed.** All results are on Gemma-2-9B, Alpaca and seed 42. No result has an interval or a spread. The authors state the seed and dataset limits themselves.

**Tables report fewer conditions than the text claims.** Table 9 lists 7 of the 24 conditions, and the text speaks of "all 268M comparisons". Table 10 lists 15 conditions, and then gives a "Mean (all 24)". Table 8 lists layers 5, 18 and 38 only.

**The paper is not consistent with itself.**

- Section 3.3 targets an L0 of about 30–50, and Table 3 gives 20–50.
- Table 12 gives a reconstruction error of 0.000022 for the chosen setting. Table 6 gives 0.340 for the same layer and rank, on held-out data. The text does not say how the two errors differ.
- Table 8 says that it uses Gemma Scope SAEs. Its active-feature counts at layer 38 match the Figure 4 caption, which reads as delta-SAE density.
- "Weakly aligned" means "fires only on h∆" in Table 8, and "largest cosine below 0.3" in Table 9.
- Section 4.4, *Delta SAE Configuration*, has a heading and no text.
- The contributions call the subspace "geometrically distinct", and the abstract calls it "partially distinct".
- Section 7.3 says that rank does not change the geometric novelty. Section 5.2 says that higher rank adds "more geometrically alien contributions" at layer 32.

**Earlier feature-level work exists.** Section 8 says that no earlier work studies fine-tuning at the feature level. Minder et al. (2025), reviewed in [`crosscoder-chat-tuning.md`](crosscoder-chat-tuning.md), compare base and chat models with crosscoders. Its arXiv version 1 is of April 2025, before this paper.

**Nothing can be checked.** The author releases no code, adapters, SAEs or activations. We give credit for section 7.4, which states the author's own limits. They are one seed, one dataset and adapters with no behavioural effect. They also include a small SAE training set and no base-against-base baseline.

## Use in this repo

**The repo compares fine-tuned models at the neuron, so a dictionary cannot miss the change.** [`docs/analyses/base-instruct.md`](../analyses/base-instruct.md) tests each VERIFIED pole at the same index in the instruct model. On Qwen2.5-1.5B, 38 of 43 hypotheses keep (88%, 95% CI 76–95%), at a median activation correlation of 0.983. On Qwen2.5-0.5B, 252 of 311 keep (81%, 76–85%), at a median of 0.974. The neuron is part of both models. So the paper's main problem, a base dictionary applied to a tuned model, does not arise there.

**The paper does not show that SAE features are the better unit for fine-tuning.** It compares a dictionary trained on the delta with one trained on the base, not features with neurons. [`papers/native-vs-learned`](../../papers/native-vs-learned/README.md) gives the repo's own comparison. On GPT-2 small, 37 of 40 live transcoder latents reach VERIFIED, against 20 of 40 positive poles. On Qwen2.5-0.5B, native poles reach VERIFIED 18 times in 30, against 1 in 30 for a random rotation. That paper also states that it shows nothing about SAEs or other models. Neither paper tests which unit tracks a fine-tune better.

**A null belongs in any dictionary comparison of base and tuned models.** #3763 plans a crosscoder for Qwen2.5-1.5B and its instruct model. This paper shows how a cosine or subspace measure can look like separation when it is at chance. So #3763 should report its shared and exclusive features against two nulls. One is random directions of the same count, as our simulation used. The other is two dictionaries trained on the base model with different seeds. [`crosscoder-chat-tuning.md`](crosscoder-chat-tuning.md) already gives #3763 Latent Scaling, a check against false model-only latents. This null is a smaller, separate check. It belongs in a comment on #3763, not in a new issue.

**Layer CKA needs its null too.** #5127 plans linear CKA per layer between Qwen2.5-0.5B and its instruct model, with a random-initialisation null. This paper reports CKA(h_base, h∆) with no null, and reads 0.05 as "most representationally distinct". #5127's plan already has the null that this paper lacks, so the paper adds nothing to it.

**An attention-only adapter leaves the poles' weights in place.** The paper's adapters change q, k, v and o, not the MLP. [`lora-low-rank-adaptation.md`](lora-low-rank-adaptation.md) makes the same point for the LoRA paper: a weight-only reading at a pole, as #5133 plans, would see no change. Only the residual stream that the pole reads changes. The paper's Table 11 shows that change is small: CKA(h_base, h_adapted) stays at 0.935 or more. #5118 (Fisher importance), #5125 (a sequence of narrow fine-tunes) and #5140 (which neurons carry the base-to-instruct difference) are about full fine-tunes. This paper bears on none of them.

**Depth does not carry over.** The paper finds its lowest CKA at layer 18 of 42. On Qwen2.5-0.5B, the middle third of poles keeps least: 63 of 92, against 104 of 112 and 85 of 107. These look alike, but the paper's measure is at chance in other forms, its adapter changes no outputs, and its model differs. So we do not take it as support.

**Evidence for the fix campaign, not a design.** This paper offers no measured change in behaviour, so it gives no evidence about a fix for the #4781 " days" error. [`fixing-models.md`](fixing-models.md) holds the evidence that does bear on it.

**No follow-up issue.** The one idea that carries over is the null, and it fits #3763 as a comment.

# Do I Know This Entity?

**Citation.** Javier Ferrando and Oscar Obeso (equal contribution), Senthooran Rajamanoharan and Neel Nanda. "Do I Know This Entity? Knowledge Awareness and Hallucinations in Language Models". arXiv 2411.14257. Version 1 is dated 21 November 2024, and version 2 is dated 8 February 2025. [Paper](https://arxiv.org/abs/2411.14257). [Code](https://github.com/javiferran/sae_entities). The arXiv comment says "Accepted at ICLR 2025". We did not check the proceedings page. The HTML lists U. Politècnica de Catalunya, the Barcelona Supercomputing Center and ETH Zürich, but our read did not show which author is at which place. The work was done in the ML Alignment & Theory Scholars (MATS) Program.

**Status.** Planned. Issue #5015 asks why pythia-70m copies the first clause's unit and why nothing supplies " months". The paper's attention test is a direct way to ask that question.

**Read.** We read the abstract page and the v2 HTML on 2026-11-18, through WebFetch, in four passes. A model read the text and reported it with short quotes. The abstract, Sections 1 to 9 and Table 3 (Appendix A) came back. Appendix B came back in part. Appendices C to S came back as titles only. These include the Llama 3.1 8B replication (Q), the patching details (H to J) and the attention analyses (K, L, M). We did not see the figures, so the refusal rates and attention scores are not here as numbers. We did not read the PDF.

## Summary

**Claim.** Language models have internal directions that tell whether the model recognises an entity. Sparse autoencoders (SAEs) find these directions at the entity's last token. The directions are causal. They can make a chat model refuse questions about entities that it knows, or make it hallucinate attributes of entities that it does not know. They act partly through the attention of attribute-extraction heads.

**Method.**

- **Models.** Gemma 2 2B and 9B, base and chat. Llama 3.1 8B is a replication in Appendix Q. Section 7 uses a "Gemma 2B IT" with 18 layers and a separate SAE.
- **SAEs.** Gemma Scope JumpReLU SAEs on the residual stream, at the default sparsity of each layer. Llama Scope SAEs for Llama.
- **Data.** Wikidata entities of four types: basketball players (7,487), movies (10,895), cities (7,904) and songs (8,448). Each type has two to six attributes, such as a player's birthplace or a song's artist.
- **Known and unknown.** An entity is known if the model answers at least two attributes right. It is unknown if the model answers every attribute wrong. The paper discards the entities in between. Fuzzy string matching marks an answer right. The split is 50% train, 10% validation and 40% test.
- **Finding the latents.** A latent is active when it is above zero at the entity's last token. The separation score is the difference in activation frequency between known and unknown entities. The paper keeps the latent with the largest minimum score over the four types. It drops latents that fire on more than 2% of random Pile tokens.
- **Causal tests.** Steering adds the latent's direction with α between 400 and 550, about twice the residual norm. Weight orthogonalisation removes the unknown direction. Activation patching runs a clean known-entity prompt into a corrupted unknown-entity prompt, and measures the proportion of logit difference recovered.
- **Baselines.** Ten random latents for the refusal rates. A random unit vector and a random SAE latent for attention. A random latent for the Yes/No test.

**Findings, as the text reports them.**

| Setting | Result |
| --- | --- |
| Separation score over layers, Gemma 2 2B | Peaks "around layer 9", then plateaus |
| Steering the unknown latent, Gemma 2 2B chat, 100 questions on unknown entities | "almost 100% refusal across all entity types" |
| Steering the known latent, same setting | "slightly reduces refusal rates" |
| Orthogonalising the unknown direction | "a large reduction in refusal responses" |
| Gemma 2 9B and Llama 3.1 8B | "similar patterns, albeit with less pronounced effects" |
| Attention from the last position to the entity | Higher for known entities |
| Steering the unknown latent on a known entity | Attention to the entity falls. The effect is largest in attribute-extraction heads such as L18H5 and L20H3 |
| Uncertainty latent, Gemma 2B IT layer 13, right against wrong answers | AUROC 73.2 on test, F1 72 at a threshold set on validation |
| Songs released after the August 2024 cutoff (283 titles) | The unknown latents fire more often |

## Where the claims go beyond the evidence

- **The labels are noisy, and the authors say so.** The model can guess an attribute of an entity that it does not know. The authors aim for "a reasonable differentiation" and not for perfect labels.
- **The threshold is stated two ways.** The text says that a known entity has at least two right attributes. The caption of Figure 7 says "we set the threshold τ=1". We could not check which rule the code uses.
- **"Gemma 2B IT" is not named exactly.** Section 7 uses an SAE at layer 13 of 18. Gemma 2 2B has 26 layers, so this is probably the first-generation Gemma 2B. The paper does not say so.
- **The headline numbers are in figures.** The refusal rates, the attention scores and the patching curves are given only in figures in the text that we read. The text gives no seeds, intervals or tests. The error bars in Figure 2 are the range over entity types.
- **The effects are smaller in larger models.** The 9B and Llama effects are "less pronounced". So the clean result is mostly from one 2B model.
- **The steering dose is large.** An α of about twice the residual norm can change the output for reasons other than the latent's meaning. The random-latent baseline controls for part of that.
- **Misrecognition is not tested.** The paper does not show that a known latent firing wrongly on a real prompt causes a hallucination. The causal evidence comes from steering. The authors also say that the result "may be specific to the factual recall mechanism".
- **The post-cutoff check is weak, and the authors say so.** Some of the 283 song titles may be in the training data.

## Use in this repo

**It is the nearest published mechanism for #4781.** The repo's hallucination is pythia-70m's " days" for "A week has seven days and a year has twelve". In `docs/analyses/hallucination-shortlist.jsonl`, " days" has 0.73 and " months" is at rank 4 with 0.03. The VERIFIED route copies " days" through the heads L3/H3, L4/H0 and L4/H7 (`changelog.d/4781.added.md`). The paper's mechanism is a gate on attention: when the model does not recognise the subject, extraction heads attend less to it. Our failure has the same parts: a subject (" year"), an attribute (" months") and heads that read the wrong place.

**What it predicts for the copying failure.** The paper predicts that the answer-position heads attend to the subject less when the model has no lookup for it. In #4781, the copy heads take " days" from the first clause and not a unit tied to " year". So the paper's account says that pythia-70m has no "year → months" recall at " year". The copy heads then fill the gap from the nearest unit in context.

**Where it does not carry over.** The paper's directions sit in the middle layers of 2B to 9B models, and refusal is a chat behaviour. pythia-70m is a 6-layer base model with no refusal. It may have no recognition direction at all. The paper's units are SAE latents, and the registry's units are native neuron poles and heads (`docs/PROTOCOL.md` §1). So a test here must use the registry's units, or a difference of mean activations, and not a Gemma Scope SAE.

**A test that #5015 can run.** Question 1 of #5015 asks why nothing supplies " months". The paper suggests two measures. First, compare the attention of L3/H3, L4/H0 and L4/H7 to " year" and to " days". Do this on the six shortlist prompts and on prompts that the model answers right. Second, patch the " year" position from a prompt that the model answers right, and measure the recovered logit difference of " months" over " days". The `activation-patching-practice` skill covers the metric choice. These are evidence for #5015. They are not a fix, and the fix campaign is the maintainer's to design.

**The steering lesson.** The paper steers at about twice the residual norm, and it compares each result with ten random latents. The `neuron-steering-tests` skill clamps a pole at 1, 2 and 4 × `max_activation`, with no random unit at the same dose. A matched random control would tell the meaning of a pole from the size of the push.

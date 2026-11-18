# Interpreting Any Residual Stream Direction by Scaling Activation Inversion

**Citation.** Celeste De Schamphelaere (MATS), Juan J. Vazquez, Ari Spiesberger, Tomáš Gavenčiak (Arb Research, and Gavenčiak also Charles University) and Gavin Leech (Arb Research). Euan Ong and Jiahai Feng (Anthropic) advised. "Towards interpreting any residual stream direction by scaling activation inversion". arXiv 2611.08465v1, 5 November 2026. [Paper](https://arxiv.org/abs/2611.08465). Preprint, no venue given. Licence CC BY 4.0. [Code](https://github.com/ceselder/maem), the trained [generator](https://huggingface.co/ceselder/maemm-27b-rl-last16-lr5e-7) and the [SAE](https://huggingface.co/ceselder/qwen36-27b-sae-l42-1b) are released. Funding and compute came from the MATS program. The paper names the models it used as judges: GPT-5.6 Sol, Claude Sonnet 5 and Gemma-4-31B.

**Status.** Not applicable. The method trains a 27B generator for one subject model, Qwen3.6-27B, and no generator exists for a model in our registry. Its one bearing on us is on dreaming (`docs/analyses/dreaming.md`, #3779): it reports EPO, our dreaming optimiser, far below a trained inverter.

**Read.** The abstract page and the v1 HTML, on 2026-11-18, through a web-fetch subagent. These came back as the paper's own text: the abstract, the introduction, related work, methods, Sections 4 to 7, Appendix A and the references. Appendices B to G came back only as a model's summary, so their numbers are marked as unverified below. We did not see Appendices H to J, or any figure beyond its caption.

## Summary

**Claim.** A generator can be trained to invert activations: given any direction in a model's residual stream, it writes text that drives that direction. Each text it writes can be checked with a forward pass. So the text is a sufficient condition for the direction, and it reads as an explanation of it.

**Method.**

- **The task.** For a direction v, the generator (MAEM) writes text x. The score is the largest cosine, over x's tokens, between v and the subject model's residual stream at layer 42. A centred score subtracts a corpus mean first.
- **Training.** Supervised fine-tuning on 8M examples, from corpus activations and SAE directions. Then reinforcement learning. Its reward is the cosine over the last 16 generated tokens, with no KL term.
- **"Scaling".** Label-free training that grows to many sources of directions, best-of-k sampling (k = 1, 8, 64), and corpus size from 1.25M to 10M tokens.
- **Targets.** Natural activations and the directions of a 131k-latent SAE. Steering vectors: AxBench's 334 concepts and BiPO's 119 behaviours. The read and write directions of 48 rank-1 LoRA adapters.
- **Baselines.** Corpus search over 10M tokens, NLA, GCG and EPO, the untrained base model, random directions, and two token readouts (JJ-lens and Patchscopes).

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| 21 unseen domains and languages | The generator beats each domain's own 10M-token corpus search in 12 of 21 |
| AxBench steering vectors | 0.958 for the generator, 0.928 for NLA, 0.614 for corpus search |
| BiPO steering vectors | 0.837 for the generator, 0.857 for NLA ("at level"), 0.513 for corpus search |
| Backdoor adapters, write direction (32 theme adapters) | The theme found verbatim in 85% [74, 94], and semantically in 90% [81, 98] |
| Backdoor adapters, numeric triggers | 21% from the write direction, 0% from the read direction |
| Workspace tasks: word-rule association, multi-hop | 28.4% and 50.0% for the generator, 48.0% and 81.5% for NLA |
| 512 SAE features, best of 8 | 89% fired, and 31% fired above the feature's top corpus activation |
| Descriptions written from the generator's texts | 38% of positives accepted, against 71% from corpus windows |

**Unverified (from the summary of Appendix C).** The setting is layer 42 and 486 residual directions, scored by centred cosine. The generator scores 0.759 for one draw, 0.822 for the best of 8 and 0.850 for the best of 64. Corpus search scores 0.572, NLA 0.612, and random directions 0.049. On 64 targets, GCG scores 0.390 and EPO 0.396, against 0.812 for the generator's best of 8.

## Where the claims go beyond the evidence

- **One subject model.** The authors say so: "Each experiment above uses only one subject model". The title's "any residual stream direction" rests on one model and one layer.
- **Driving a direction is not explaining it.** A text that drives v is a sufficient condition. It doesn't say what else drives v, so it measures recall at best. The paper's own test shows the gap: descriptions written from its texts accept 38% of positives, against 71% from corpus windows.
- **No diversity.** The reward "contains no diversity term", and the texts are less varied than corpus windows (Vendi score 2.2 against 2.9). So one text per direction may stand for a narrow part of what drives it.
- **Untested limits.** The authors say: "How far the objective can be pushed before text degrades is untested." The reward has no KL term.
- **One checkpoint.** No seed variance is reported. The intervals are bootstrap intervals over directions, adapters or documents, and the tests are Holm-corrected, which the authors deserve credit for.
- **Small inconsistencies.** Section 4 calls the 21 domains "non-English", but they include code and maths. The score is defined over the whole text, but the reward reads only its last 16 tokens.

## Use in this repo

- **No generator for our models.** Our registry's models are GPT-2, Pythia and Qwen2.5. The released generator is for Qwen3.6-27B at layer 42. Training one for each of our models is a research project of its own, and the paper gives no cost per model.
- **The unit differs.** The paper studies residual directions, not MLP neurons or heads. A neuron's read and write vectors are residual directions, so the method could take them. But the paper doesn't test neurons, and our explanations are of a pole's activation, not of its direction.
- **Our dreaming uses its weakest baseline.** `docs/analyses/dreaming.md` drives VERIFIED GPT-2 poles with EPO and finds that the explanation misses 73% of the firing prompts. If the paper's EPO numbers hold, a stronger optimiser would find more firing prompts, and maybe more misses. So 73% is a floor on how often dreaming finds what an explanation misses. That is now a comment on #3779, not a new issue.
- **Its checkable-exemplar idea is ours already.** The paper values texts that a forward pass can check. Our reviews check every probe and counterfactual against measured activations, which is the same idea in the other direction: the explanation proposes, the model decides.
- **Not a vocabulary-projection method.** The issue tagged `vocab-projection-lookahead`. The paper compares with token readouts (JJ-lens, Patchscopes) only as baselines, and our skill's methods are unaffected.

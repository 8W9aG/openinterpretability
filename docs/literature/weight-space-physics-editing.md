# Weight-Space Programming and Mechanistic Interpretability for Model Editing in Particle Physics

**Citation.** Raphaël Bonnet-Guerrini (Università di Milano and INFN Milano), Johann Ioannou-Nikolaides and Troels Petersen (Niels Bohr Institute, University of Copenhagen). "Weight-space programming and mechanistic interpretability for model editing in particle physics". [arXiv:2611.27079](https://arxiv.org/abs/2611.27079) [hep-ph], cross-listed in cs.AI, cs.LG and hep-ex. Version 1 of 20 November 2026 is the only version. Preprint, no venue given. The licence is CC BY 4.0. Code is at [github.com/rbonnetguerrini/WSP-mech-interp-for-model-editing-in-particle-physics](https://github.com/rbonnetguerrini/WSP-mech-interp-for-model-editing-in-particle-physics), with the agents' transcripts. The edited weights, SAE checkpoints and jet sets are on Zenodo ([10.5281/zenodo.24377675](https://doi.org/10.5281/zenodo.24377675)). The work has funding from a Horizon Europe Marie Skłodowska-Curie grant (No. 101168829), with compute from CINECA Leonardo and the University of Copenhagen. The repair agents were Claude Opus 5.5 in Claude Code, the same model that wrote this page.

**Mismatch.** The Introduction gives the three model sizes as 3M, 60M and 0.4B, while Section 2.1 gives the smallest as 3.3M parameters.

**Status.** Planned. Arm G of #5220 has an agent write weights with no gradients, as this paper's weight-space programming does. The paper's comparison of SAE latents with single neurons is also evidence for research question 3. No code, skill or protocol here depends on it.

**Read.** On 2026-11-26 a `web-fetch` subagent read the abstract page and the HTML of version 1 in four parts. It read them for this page's author, a model (Claude Opus 5.5). It had the source's own text for Sections 1 to 5, the position note, the availability statements and Appendices A to D. The tool allows only short verbatim quotes, so much of that text came back paraphrased. It did not see the figures, any table drawn as an image, the code or the Zenodo data.

## Summary

**Claim.** A jet foundation model from particle physics has a flaw: one added soft photon, below the detector's threshold, changes its W-boson tag. Three repairs remove the flaw. These are fine-tuning, a rank-one edit chosen through a sparse autoencoder (SAE) latent, and weight-space programming (WSP). In WSP, an LLM agent edits the weights with no gradients. The two edits change far fewer parameters than fine-tuning, and they keep the model's accuracy on clean jets.

**Method.**

- **Model.** OmniLearned, a jet foundation model. The smallest has 3.3M parameters: 128-dimensional embeddings, eight transformer blocks and eight heads of dimension 16, with generator and classifier heads. Medium and large models are also tested. The task is the W-against-QCD tagger, with the threshold set to keep 80% of W jets.
- **The flaw.** One added soft photon, at a momentum fraction of $10^{-3}$, cuts the W efficiency from 80% to 54%. The authors trace it to a shortcut from the data's provenance. The backbone reassigns the jet to JetClass-II, a training set with lower reconstruction thresholds.
- **Localisation.** Activation patching points to three heads, which remove 58%, 19% and 17% of the response.
- **SAE route.** Three TopK SAEs (4,096 latents, top 32) are trained on 2.5M tokens. One latent is chosen for each head that reads the flaw. Each edit is rank-one, $W_K \to W_K (I - u u^\top)$, which removes the latent from the head's keys. The three edits change 1,533 parameters.
- **WSP route.** Claude agents read the code and the weights, and may query an evaluation tool up to 50 times. They may not take gradients or run the model, but closed-form linear algebra is allowed. There are three variants, each run three times. They differ in what the agent sees: the flaw, the target predictions, or the flaw's physical origin.
- **Fine-tuning baseline.** Rank-8 updates of the body's attention and its pairwise-bias network, 63,240 free parameters, with a loss that holds the clean behaviour.
- **Side effects.** The change in accuracy on clean jets. The response to physical changes of a jet, as a ratio to the original model's. The remaining error against target predictions. A generator head and a regression head are checked too.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Smallest model, 20,480 held-out jets, Table 2 | Original: W efficiency 54% with the photon |
| Fine-tuning, same setting | Remaining response 0.06, efficiency 79%, clean accuracy −0.0001, physical response 0.90, 63,240 parameters |
| SAE edit, three heads, same setting | 0.11, 78%, −0.0006, 0.78, 1,533 parameters |
| WSP, target informed, median of 3 runs | 0.05, 79%, −0.0002, 1.01, 10,122 parameters (range 142 to 10,267) |
| WSP variants, Table 1, median remaining error | 0.33 with the flaw, 0.12 with the targets, 0.08 with the physical origin |
| Medium and large models, Table 7 | Fine-tuning 0.03 and 0.05. SAE edit 0.35 and 0.58. WSP 0.03 and 0.04 |
| Regression head | The flaw shifts it by 10.6 GeV. WSP and the SAE edit leave 1.1 GeV, and fine-tuning 5.0 GeV [4.7, 5.6] over three seeds |
| Neurons against SAE latents | At zero false positives, the latents fire on 73%, 89% and 82% of sub-threshold neutrals. Single neurons reach at most 62% |
| Random latent removed | No change: attention unchanged, and the flaw's response changes by under 1% |

## Where the claims go beyond the evidence

- **One flaw in one model family.** Every result is one shortcut in OmniLearned. The method's reach to other flaws or models is not tested.
- **Few runs, no intervals on the edits.** The SAE edit is one run, and WSP has three runs per variant, reported as medians and ranges. Only fine-tuning has seeds with errors. The source reports no significance tests or corrections.
- **WSP varies a lot between runs.** The remaining error differs by up to three times across runs, and the edit size spans 142 to about 10,000 parameters. WSP does not reliably repair the generator: two of three medium runs and one of three large runs keep its full response. The SAE edit loses most of its effect on the larger models.
- **No repair makes the tagger physically sound.** The authors say that "none of them makes the tagger infrared safe in the formal sense".
- **Side effects are measured as behaviour.** Clean accuracy and the physical responses are outputs. No internal unit is re-tested after an edit, except the patched heads and the chosen latents.
- **The authors state their limits.** They say WSP ran under "tightly constrained conditions", with at most 50 queries and an LLM "that was not state of the art". They give a random-latent control and a 3 GeV photon as an above-threshold control. The page credits them.

## Use in this repo

- **WSP is #5220's arm G in practice.** Arm G has an agent write a keyed unit into a dormant slot, with no gradients. This paper's agents edit existing weights under the same rule. At the median, they match fine-tuning on the main tagger with a sixth of its parameters. That supports trying arm G. It also warns that runs vary by up to three times, so arm G needs several runs and a report of their spread.
- **Its side-effect test is the behaviour level only.** The paper checks outputs: clean accuracy, physical responses and two other heads. The interference report (#5230) also re-tests every VERIFIED unit and route on the changed model. `papers/traffic-map/` finds that a random change at a high-traffic site of pythia-70m can break 22 of 176 explanations, and no output measure would show that. Our view is that a repair with near-zero clean-accuracy loss can still break internal explanations. The paper has no test that would show it.
- **What an edit changes differs from what fine-tuning changes.** `papers/sequential-tuning/` found that a full fine-tune of Qwen2.5-0.5B on one narrow task cost 66 of 362 explanations. The paper's edits change 1,533 to about 10,000 parameters, against 63,240 for its fine-tune. Whether a small edit also breaks fewer explanations is what #5220 phase 4 will test. The parameter count alone does not settle it: `papers/rome-keep/` found that one rank-one edit flips as many readings as a random edit of its norm.
- **The latents beat single neurons at reading this shortcut.** At zero false positives, the best single neuron reaches 62% of sub-threshold neutrals, and the latents 73% to 89%. That is evidence for learned dictionaries on one concept in one model, and it bears on research question 3 (`papers/native-vs-learned/`). The registry's comparison on GPT-2 uses random units, not one chosen concept, so the two results answer different questions.
- **The patch format can hold these edits.** A rank-one key edit $W_K (I - u u^\top)$ is a declarative weight edit, which the patch format (#5229) can record with its evidence. No issue follows from this page. #5220 already holds the arms.

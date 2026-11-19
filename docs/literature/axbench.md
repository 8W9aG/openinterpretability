# AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders

**Citation.** Zhengxuan Wu, Aryaman Arora, Atticus Geiger, Zheng Wang, Jing Huang, Dan Jurafsky, Christopher D. Manning and Christopher Potts. "AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders". arXiv 2501.17148. Version 1 is from 28 January 2025. We read version 3, from 3 March 2025. [Paper](https://arxiv.org/abs/2501.17148), [code](https://github.com/stanfordnlp/axbench), [OpenReview](https://openreview.net/forum?id=K2CckZjNy0). The paper gives the affiliation "Department of Computer Science, Stanford University" for most authors, and "Pr(AI)2R Group" for Atticus Geiger. The arXiv page has no venue comment. The ICML 2025 virtual site lists the paper as a "Spotlight Poster" ([page](https://icml.cc/virtual/2025/poster/45658)). We did not find its entry on the PMLR volume 267 page, so the proceedings page is not checked. The acknowledgements say: "This research is supported in part by grants from Open Philanthropy." Appendix G says that the LLM judge is "OpenAI's gpt-4o-mini-2024-07-18". The same model writes the synthetic data.

**Status.** Planned. The fix campaign that #4784 prepares for would use it, and #3209 asks whether steering predictions should become a gate. The paper shows that a steering result means little without a prompting baseline and without a score for fluency and instruction following.

**Read.** On 2026-11-18, through WebFetch. A model read the abstract page and the v3 HTML for us, in two parts by offset. It returned short verbatim quotes from the abstract, Sections 3 and 4, the discussion, the conclusion, Appendix G and Appendix K. It also returned the rows of Tables 1, 2 and 3. Other parts came back only as its summary. These include the method details, the ReFT-r1 objective, and Appendices E, F, H, I, J, L, M and N. We did not see the figures, the PDF or the prompt templates of Appendix J. We did not read the Concept16K results in Appendix E.

## Summary

**Claim.** On a benchmark of 500 concepts in Gemma-2 models, prompting steers better than all representation methods, and finetuning is second. In the authors' words, "On both evaluations, SAEs are not competitive." For concept detection, "representation-based methods such as difference-in-means, perform the best".

**Method.**

- **Models and sites.** Gemma-2-2B-it at layers 10 and 20, and Gemma-2-9B-it at layers 20 and 31. The SAEs are from GemmaScope.
- **Concepts and data.** Concept500 has 500 concepts. The concept lists come from Neuronpedia's auto-interpretability labels. Each concept has 144 synthetic training examples and about 72 evaluation examples, from gpt-4o-mini (summary). Concept16K covers all 16K GemmaScope concepts.
- **Concept detection.** Each method gives a score for each token. The concept score is the maximum over tokens, and the metric is AUROC (summary).
- **Steering.** A method steers the model on Alpaca-Eval instructions, 10 for each concept, at 14 steering factors, for up to 128 tokens (summary). The LLM judge rates three things from 0 to 2: the concept score, the instruct score and the fluency score. The overall score is their harmonic mean (summary). The authors choose the factor on held-out instructions. In their words, "We split the instructions into two equal sets – one for selecting the best factor and the other for evaluation."
- **Steering by a direction.** "Our steering operation is simple activation addition". The direction has unit norm. Its magnitude is "the product of the steering factor and the maximal activations aggregated over the evaluation dataset for concept detection".
- **Methods.** Prompting, SFT, LoRA, LoReFT, ReFT-r1, difference in means (DiffMean), PCA, LAT, a linear probe and SSV. Two SAE rows use GemmaScope: the latent from the Neuronpedia label (SAE), and the SAE latent with the best training AUROC (SAE-A). Detection also has a bag-of-words baseline (BoW), Integrated Gradients and Input×Gradients. ReFT-r1 is the authors' new method. It learns one direction with a language-modelling loss and an L1 penalty on its latents (summary).
- **Prompting.** An LLM writes a prompt from the concept, and the steered model reads that prompt "prepending it to the actual instruction".
- **Tuning.** "To ensure a fair comparison, we perform separate hyperparameter-tuning for each method."

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Concept detection, mean AUROC over the four sites (Table 1) | DiffMean 0.942, Probe 0.940, ReFT-r1 0.938, Prompt 0.929, SAE-A 0.917, BoW 0.914, SSV 0.912, LAT 0.712, SAE 0.695, PCA 0.652, IG 0.426, IxG 0.246 |
| Concept detection, top three | DiffMean, Probe and ReFT-r1 show no significant difference under a paired t-test (summary of the text) |
| Steering, mean overall score at the chosen factor (Table 2) | Prompt 0.894, LoReFT 0.741, SFT 0.676, LoRA 0.615, ReFT-r1 0.543, DiffMean 0.239, SAE 0.165, SAE-A 0.157, LAT 0.127, PCA 0.105, Probe 0.098, SSV 0.026 |
| Steering, Prompt by site (Table 2) | 0.698 (2B L10), 0.731 (2B L20), 1.075 (9B L20), 1.072 (9B L31) |
| Steering, DiffMean by site (Table 2) | 0.297, 0.178, 0.322, 0.158 |
| Steering, SAE by site (Table 2) | 0.177, 0.151, 0.191, 0.140 |
| Win rate against SAE, mean (Table 3) | Prompt 94.5%, LoReFT 89.0%, SFT 88.8%, LoRA 82.5%, ReFT-r1 81.8%, DiffMean 58.7% |
| Win rate against SAE, text | "ReFT-r1 (88.0%) and DiffMean (61.6%) achieve winrates of greater than 50% against SAEs" |
| All steering methods | "increasing the factor monotonically reduces instruct score in all methods" |

The results section says: "Prompting, along with slightly worse finetuning baselines, outperforms all steering methods on average, except for ReFT-r1." The conclusion says that "representation steering is still far behind simple prompting and finetuning baselines". The discussion calls this "sobering evidence of the current limitations of steering techniques".

## Where the claims go beyond the evidence

**The text and Table 3 disagree.** The text gives win rates of 88.0% for ReFT-r1 and 61.6% for DiffMean. The Table 3 means that we read are 81.8% and 58.7%. One of the two may use a different set of sites. We could not tell which.

**One judge model rates every output.** gpt-4o-mini writes the training data and also judges the steered text. A method that writes like gpt-4o-mini can get a higher score. Prompting adds a gpt-4o-mini prompt to the input, so this bias can help prompting most. We found no human check of the judge in the parts that we read.

**The concepts come from auto-interpretability labels.** The authors state this limit. The lists are "often skewed towards token-level concepts and misses high-level abstractions". They also say that "the poor performance of SAEs is at least partially a reflection of the limitations of auto-interpretability". So the SAE row tests a latent together with its label. It does not test the latent alone. SAE-A, which picks the latent by its training AUROC, detects much better (0.917 against 0.695) but steers no better (0.157 against 0.165).

**One model family.** All results are on Gemma-2 at 2B and 9B, with one SAE family.

**No intervals and few seeds.** We found no confidence intervals in the text that we read. Seeds 42, 43 and 44 appear only in an Appendix D data-scaling test (summary). Appendix F.1 says that "most comparisons are statistically insignificant" for its SAE ablations.

**The steering setting is narrow.** Each method adds its direction with one factor, and the factor range is fixed. Appendix K says: "We acknowledge that this might lead to an overall underestimation of the performance of S model steering performance." The words "S model" are in the source.

**Credit.** The factor is chosen on held-out instructions. Each method gets its own tuning. The score punishes a method that adds the concept but breaks fluency or ignores the instruction. The authors release their code and dictionaries.

## Use in this repo

**Our steering samples have no baseline and no score.** `openinterpretability/harness/steering.py` continues five fixed neutral prompts, such as "The weather today is". It ablates the unit, or clamps it at 1, 2 and 4 × `max_activation`, for 24 greedy tokens. The `neuron-steering-tests` skill reads the drift by eye: "Coherent, predicted drift at 1–2× max supports the claim". The samples have no instruction to follow, no fluency score and no prompting baseline. AxBench shows that each of these changes the ranking of methods.

**The paper's question is different from ours, but its warning applies.** Our steering samples test an explanation: does the unit make the model write what the explanation says? AxBench asks how well a method controls a model. A weak steering effect for a single neuron does not refute its explanation, as the skill already says. But a fix that steers a unit is a control claim. For that claim, the paper's result is direct: a prompt or a finetune is the baseline to beat.

**Dictionary features are in scope.** The harness steers a dictionary latent by adding its decoder direction (`FeatureUnit` in `openinterpretability/harness/units.py`). That is the SAE row of AxBench. In AxBench, SAE steering scored 0.165, against 0.894 for prompting. Any comparison of native neurons with SAE features by steering should report both rows against a prompt.

**What the paper predicts for #4781.** The pythia-70m prompt "A week has seven days and a year has twelve" gives " days" at 0.73, with " months" at rank 4 (`changelog.d/4781.added.md`). The verified route copies the first clause's unit through layer-0 count-word and time-unit neurons and the copy heads L3/H3, L4/H0 and L4/H7. A steering fix could ablate the copy route or add a " months" direction. AxBench predicts that a prompt or a small finetune will fix this answer more reliably than a steered direction. It also predicts that a steering factor high enough to fix the answer will lower fluency on other text. This is a prediction from the paper, not a measurement. pythia-70m is a base model with no instructions, so AxBench's instruct score has no direct match here.

**What a fix test could report, from the paper's metrics.** The paper scores three things at once and picks the factor on held-out inputs. For #4781, the six prompts in `docs/analyses/hallucination-shortlist.jsonl` all copy the first clause's unit. They could split into a set to choose a factor and a set to evaluate it. Fluency on neutral text could use the five steering prompts. How to combine these is a decision for the fix campaign, which the maintainer designs.

**Question for the maintainer.** #3209 asks whether steering predictions should become a gate. Should a pre-registered steering prediction also name a prompting baseline and a fluency check, as AxBench does? That is a protocol decision, so this page records it and opens no issue.

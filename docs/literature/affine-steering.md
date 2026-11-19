# Representation Surgery: Affine Steering

**Citation.** Shashwat Singh (IIIT Hyderabad), Shauli Ravfogel (Bar-Ilan University), Jonathan Herzig (Google Research), Roee Aharoni (Google Research), Ryan Cotterell (ETH Zurich) and Ponnurangam Kumaraguru (IIIT Hyderabad). Ravfogel did the work during an internship at Google Research. "Representation Surgery: Theory and Practice of Affine Steering". arXiv 2402.09631. Version 1 is from 15 February 2024. We read the HTML of version 7, from 4 June 2025. [Paper](https://arxiv.org/abs/2402.09631), [code](https://github.com/shauli-ravfogel/affine-steering). The arXiv comment gives the venue as "Accepted in ICML 2024". We did not check the proceedings page. The data sets are public: Bios, a dialect sentiment set (AAE and SAE tweets), Real Toxicity Prompts and a human-labelled toxicity set. We found no funding statement or acknowledgements in the text that we read.

**Status.** Planned. The fix campaign that #4784 prepares for, with #4781 and #5015 as its first case, would use it. The paper gives a least-squares reason to steer by the difference of two class means, and it shows the cost of that steering on fluency.

**Read.** On 2026-11-18, through WebFetch. A model read the abstract page and the version 7 HTML for us, in two parts by offset. It returned Tables 1 and 2 row by row, and short verbatim quotes from Sections 4 and 5, Appendix E and Appendix F. The abstract, the contribution list, the propositions and the limitations came back only as its summary. We did not see the figures, the PDF, the proofs in Appendices A to C, or Table 3 (dialect, Appendix D) in full. The model found no statement on seeds, standard deviations or confidence intervals in any part.

## Summary

**Claim.** A language model's representations can encode a concept that causes unwanted behaviour, such as gender or toxicity. Two affine maps of those representations are optimal under least-squares criteria. Both maps reduce bias in classifiers and toxicity in generation, at a small cost in accuracy or fluency.

**Method.**

- **Theory.** A steering function maps representations of the source class (for example, toxic text) toward the target class. Proposition 4.1 (summary): under a guardedness constraint, the least-squares optimal map is a translation by the difference of the two class-conditional means, with W = I. The text says that "optimal steering in this sense only requires a translation vector that matches the concept-conditional means". Proposition 4.2 (summary) also matches the covariances, in closed form. The authors relate it to the optimal transport map between Gaussians. Proposition 4.3 (summary): under that second map, the expected "bias by neighbors" is zero. The paper compares both maps with LEACE (Belrose et al., 2023), a guarding projection. It says that this matrix is "in general, an oblique projection matrix, not an orthogonal one".
- **Models.** BERT-base, GPT-2 and Llama2-7b for classification. GPT-2 large for generation.
- **Tasks and metrics.** Bios: profession classification, with the TPR gap between genders (RMS) and accuracy. Dialect: sentiment classification on AAE and SAE tweets, with the TPR gap and accuracy across AAE shares from 0.5 to 0.95. Toxicity: generation from "a split of 10k samples from the non-toxic split of Real Toxicity Prompts". The metrics are expected maximum toxicity, toxicity probability, fluency (perplexity) and 1-, 2- and 3-gram diversity.
- **Steering in generation.** The maps are fit on "balanced classification data that consists of full sentences with human toxicity labels". The fit takes "the hidden state for the last token" of each sentence. At generation, the map acts "in the last hidden representation at each generation step". It acts only on "hidden states that are closer to" the source-class mean. Decoding samples "25 strings with a maximum length of 20 tokens" at temperature 1.
- **Baselines.** LEACE and a postprocessing method (Xian et al., 2023) for Bios. DAPT, GeDi, PPLM, UDDIA, DExperts and GOODTRIEVER for toxicity.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Bios, BERT-base (Table 1, TPR gap / accuracy) | Base 0.155 / 0.799. LEACE 0.137 / 0.797. Mean matching 0.141 / 0.797. Mean and covariance matching 0.093 / 0.785 |
| Bios, GPT-2 (Table 1) | Base 0.168 / 0.676. LEACE 0.093 / 0.670. Mean matching 0.094 / 0.670. Mean and covariance matching 0.070 / 0.660 |
| Bios, Llama2-7b (Table 1) | Base 0.143 / 0.786. LEACE 0.133 / 0.795. Mean matching 0.139 / 0.797. Mean and covariance matching 0.085 / 0.783. The postprocessing baseline "did not converge for the Llama2-7b model" |
| Dialect, Llama2-7b (Section 5.1.2) | The link between the data set's AAE share and the TPR gap "is completely removed after applying our affine steering functions". Accuracy falls "from 75.9% to 75.1%" at an AAE share of 0.5, and to 63.5% at 0.95 |
| Toxicity, GPT-2 large, base (Table 2) | Expected maximum toxicity 0.39, toxicity probability 0.25, perplexity 24.66 |
| Toxicity, mean matching (Table 2) | 0.33, 0.16, perplexity 28.00 |
| Toxicity, mean and covariance matching (Table 2) | 0.29, 0.09, perplexity 30.7. Diversity 0.54, 0.84 and 0.84 against 0.58, 0.85 and 0.85 for the base |
| Toxicity baselines (Table 2) | DAPT 0.27 / 0.09 / 30.27. DExperts 0.21 / 0.02 / 27.15. UDDIA 0.24 / 0.04 / 26.83. GeDi 0.24 / 0.06 / 48.12 |
| Steering every vector, not only the source class (Appendix E) | WikiText-2 perplexity rises "from 22.6 on the base model to 54.0" |

The model summarised these points, and we have them only as its summary. The paper calls the toxicity result "an almost 25% reduction" in expected maximum toxicity, and "at par with DAPT" on that metric. In Bios, at 128 neighbours, about 52% of a point's neighbours share its gender label after steering. That share matches the random baseline. The stated limits are that the maps are linear and miss concepts that the model encodes nonlinearly. The fit uses last-token states, but generation steers every step. The toxicity numbers do not reach the best baselines, which often need fine-tuning or gradients at inference.

## Where the claims go beyond the evidence

**The theory is about class means, and the practice adds a gate.** Proposition 4.1 gives one translation for all source-class vectors. In generation, the map acts only on states nearer the source mean. That gate is a nonlinear rule chosen by experiment, and the authors say it "works better than classification models". Appendix E shows that the gate matters: steering every vector more than doubles perplexity.

**Guardedness is not the target behaviour.** The optimality is a least-squares statement under a linear-guardedness constraint. It does not say that generation becomes less toxic. The toxicity gains are an empirical result, and the theory does not predict their size.

**"At par with DAPT" holds on one of four metrics.** Expected maximum toxicity is 0.29 against 0.27, and toxicity probability is 0.09 for both. Perplexity is 30.7 against 30.27, and 1-gram diversity is lower (0.54 against 0.57). Several baselines in Table 2 do better on all of toxicity and perplexity.

**No seeds or intervals.** We found no seeds, standard deviations or confidence intervals. Table 1 differences of 0.001 to 0.01, such as LEACE against mean matching for GPT-2 (0.093 against 0.094), have no error estimate.

**Fit and use disagree.** The authors state this limit themselves: the map is fit on last-token states of full sentences and applied at every step of a continuation.

**Credit.** The paper gives closed forms, a link to optimal transport and a check of the theory (neighbour bias at the random baseline). It reports the fluency and accuracy cost next to each gain, and it releases code.

## Use in this repo

**Our steering samples are an affine map, but not this one.** `openinterpretability/harness/steering.py` sets one neuron's activation to 1, 2 or 4 times its `max_activation`, at every position. In the paper's terms that is an affine map with W = I except a zero for that neuron, and a constant b for that neuron. It is not fit to two classes, so the paper's optimality says nothing about it. That is in line with its purpose: the samples show what a unit makes the model write. They are not a fix.

**The paper supports our reading of strong clamps.** The `neuron-steering-tests` skill calls drift only at 4× off-distribution and weak evidence. Appendix E points the same way for a fitted map: steering every vector, not only the source class, takes WikiText-2 perplexity from 22.6 to 54.0. Our clamp acts at every position, like the paper's ungated ablation. So a fix built from a clamp must report its fluency cost, as the paper reports perplexity and diversity.

**What the paper implies for #4781 and #5015, as evidence only.** On "A week has seven days and a year has twelve", pythia-70m gives " days" (0.73), with " months" at rank 4 (0.0295, `docs/analyses/hallucination-shortlist.jsonl`). The VERIFIED route copies the first clause's unit. It uses layer-0 count-word and time-unit neurons and the copy heads L3/H3, L4/H0 and L4/H7. No verified unit supplies " months". Mean matching needs two classes of states, for example answer positions that the model gets right and wrong. The shortlist kept only 6 wrong prompts from 35, and all 6 copy the first clause's unit. That is too few for a covariance in the residual stream, so only the mean translation is in reach. Also, the paper's guarantee is about a concept that a linear classifier can read. It does not promise that " months" wins. How to use this is a decision for the fix campaign, which the maintainer designs.

**Steering gates (#3209).** The paper scores steering by task metrics, and it gives no pre-registered prediction format. It does not settle #3209.

**Related reviews.** #5046 (ActAdd) and #5047 (contrastive activation addition) test difference-of-means vectors in practice. #5055 and #5054 test what steering breaks. This page gives the least-squares reason that those methods use a mean difference. Their pages can cite it.

**Question for the maintainer.** Should a steering fix report a fluency cost, such as perplexity on a fixed text, next to its target effect, as Table 2 does? This page records the question and opens no issue.

# Mechanistic Understanding and Mitigation of Language Model Non-Factual Hallucinations

**Citation.** Lei Yu (University of Toronto) and Meng Cao (McGill University and Mila) contributed equally. The other authors are Jackie Chi Kit Cheung (McGill and Mila) and Yue Dong (University of California, Riverside). "Mechanistic Understanding and Mitigation of Language Model Non-Factual Hallucinations". arXiv 2403.18167. Version 1 has the date 27 March 2024, and version 2 has the date 17 June 2024. [Paper](https://arxiv.org/abs/2403.18167). The venue is *Findings of the Association for Computational Linguistics: EMNLP 2024*, pages 7943–7956, [doi:10.18653/v1/2024.findings-emnlp.466](https://doi.org/10.18653/v1/2024.findings-emnlp.466). We checked the venue on the ACL Anthology page. The arXiv page gives no venue. The paper gives no code or data link, and no funding or tool disclosure. Issue #5033 says that version 1 had the title "Mechanisms of Non-Factual Hallucinations in Language Models". We did not check version 1.

**Status.** Not applicable for now. No code, skill or protocol rule depends on it, and no open issue uses its method. It gives a test that can sort the #4781 failure into one of its two types, and the follow-up below proposes that test.

**Read.** On 2026-11-18 we read the version 2 HTML (`arxiv.org/html/2403.18167v2`) as plain text, with the formulas taken from the page's math markup. We have the paper's own text for all sections: the abstract, §1–§7 and Appendices A–D. We also have the text of Tables 1–8. We did not see Figures 1–4 as images, so the heatmaps in Figure 4 are not here. We did not read the PDF or version 1. A model (Claude) read the source and wrote this page.

## Summary

**Claim.** The paper finds two causes of factual errors on subject–relation queries in three models. In *knowledge enrichment hallucinations*, lower-layer MLPs do not retrieve enough about the subject. In *answer extraction hallucinations*, upper-layer attention fails to select the right attribute. A fine-tuning loss that targets the two sites reduces errors better than the baselines.

**Method.**

- **Models.** Llama-2-7B-chat (32 layers), Pythia-6.9B (32 layers) and GPT-J-6B (28 layers).
- **Data.** About 80K cloze queries from ParaRel's many-to-one relations, with a single object that starts with a capital letter. A query is a hallucination when the top capitalised token is not a prefix of the true object.
- **Split by logit lens.** Each MLP output at the last subject token is projected through the final norm and the unembedding. Take the minimum rank of the true object over all layers. If that rank is in the top 1% of the vocabulary, the error is *extraction*. If not, it is *enrichment*. The threshold rank is 320 for Llama-2 and 502 for Pythia and GPT-J.
- **Attention measure.** At the last token, each layer's attention output is dotted with the true object's unembedding minus the mean unembedding of the top-100 MLP-enriched attributes.
- **Causal check.** This is causal tracing after Meng et al. (2022). Gaussian noise on the subject embeddings gives a run that prefers the true answer more. Each hidden state from the hallucinating run is then patched into the noised run.
- **External features.** These are subject–object association, from the inner product of input embeddings. Robustness is the share of noise samples that do not make the model prefer the true answer. Uncertainty is the entropy of the next-token distribution.
- **Mitigation (MHM).** A loss raises the logit-lens log-probability of the true answer in MLP outputs. It also raises the true-over-wrong log-ratio in attention outputs. Both apply at layers 20–25, and the loss is added to the usual NLL with λ = 1.0.
- **Mitigation data.** The training set is the Natural Questions and TruthfulQA items that each model answers wrong. *Effectiveness* is accuracy on GPT-4 paraphrases of those items. *Specificity* is the share of the items that the model answered right and still answers right.
- **Baselines.** Supervised fine-tuning, 5-shot prompts, MEND (on the last 3 MLP blocks) and DoLa.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Accuracy on the ParaRel queries (Table 1) | Llama-2 0.497, Pythia 0.248, GPT-J 0.266 |
| Errors that are enrichment / extraction (Table 1) | Llama-2 22.1% / 77.9%, Pythia 30.2% / 69.8%, GPT-J 67.3% / 32.7% |
| Association, subject with hallucinated object (Table 2, mean of 3 models) | Enrichment 0.81, extraction 1.69 |
| Association, subject with true object (the same) | Enrichment 0.47, extraction 1.17 |
| Robustness (the same) | Enrichment 0.86, extraction 0.45 |
| Uncertainty, entropy (the same) | Enrichment 4.54, extraction 4.76 |
| Noise samples that reduce the log-ratio (Appendix C) | About 71.1%. The kept noises reduce it from 11.7 to 2.3 on average |
| MHM, Natural Questions, effectiveness / specificity (Table 3) | Llama-2 47.6 / 95.5, Pythia 46.9 / 92.4, GPT-J 43.8 / 89.4 |
| MHM, TruthfulQA, effectiveness / specificity (Table 3) | Llama-2 48.2 / 95.6, Pythia 45.9 / 93.5, GPT-J 49.7 / 95.8 |
| Best baseline effectiveness, Natural Questions (Table 3) | Llama-2 SFT 41.8, Pythia SFT 37.9, GPT-J MEND 36.7 |
| Best baseline effectiveness, TruthfulQA (Table 3) | Llama-2 SFT 44.1, Pythia DoLa 46.6, GPT-J SFT 46.0 |

The text says that MHM is best or comparable to the best "in all setups". The table supports that: DoLa's 46.6 on Pythia TruthfulQA is the only cell where a baseline beats MHM. The text also says that MEND fails on TruthfulQA (specificity 40.5 and 50.4 on Llama-2 and Pythia). It says that DoLa helps little on Natural Questions.

## Where the claims go beyond the evidence

**"Lower layer MLPs" and the fix disagree.** The analysis puts enrichment failures in lower-layer MLPs. MHM changes only layers 20–25, which Appendix D calls the middle layers. The paper does not show that the loss fixes the sites that its analysis found.

**The split comes from one threshold, chosen by eye.** The authors set the 1% rank cut from Figure 2. It decides every share in Table 1. The paper does not report the shares at other cuts, or show that the two groups are separate and not two ends of one range.

**The "truthful" run is not a clean run.** About 71.1% of random noise samples make the model less wrong. So the noised run may be better mainly because noise removes the subject that drives the wrong answer. Our `activation-patching-practice` skill warns against Gaussian-noise corruption for this reason (Zhang and Nanda, 2023).

**The patching rules do not agree with each other.** The text keeps noises that make the model prefer the true answer. The formula keeps noises with a log-ratio below 1, not below 0. Appendix C reports a mean of 2.3 after noise, which still prefers the wrong answer. The main text sets σ to 3 times the embedding spread. Appendix C gives σ ≈ 0.15. The AIE is said to average "over a set of factual queries", but the analysis is of hallucinating queries.

**No intervals or seeds.** No table gives a seed count, an interval or a test. Table 2 averages over the three models and does not give each model's values. The text says that extraction errors are "significantly less" stable under noise, but it reports no significance test.

**The mitigation test is narrow.** Effectiveness uses paraphrases of the training questions, so it measures recall of the trained items. Specificity uses only the same benchmark's right answers. The paper does not test general text or other tasks after fine-tuning. The baselines use default settings, while the authors tuned MHM's λ and layers.

**One answer token and a capitalised vocabulary.** The analysis scores only the first token of the answer, and only among capitalised tokens. The authors drop queries with no capitalised token in the top 50.

**Credit.** The authors state two limits themselves. The logit lens only approximates what a component encodes, most of all in early layers. Their simple prompts may not show what models do on longer, natural inputs.

## Use in this repo

**The repo already has the answer-level view the paper lacks.** The paper explains errors by layer and by module type. The #4781 records in `wiki/answers/EleutherAI--pythia-70m/970a91e11d8c/` name single heads and neuron poles. The VERIFIED attempt (`20261118T234629Z.json`) has patching 0.749 against null patching of at most 0.011. Its sufficiency is 0.624. The paper's module-level answer is coarser than that route.

**The #4781 failure is not the kind of error that the paper studies.** The paper's errors come from parametric recall of a fact about a named subject. On "A week has seven days and a year has twelve", pythia-70m gives " days" at 0.7257 and " months" at 0.0295, rank 4 (`docs/analyses/hallucination-shortlist.jsonl`). The wrong token is in the prompt, and the verified route copies it with the heads L3/H3, L4/H0 and L4/H7. The paper does not treat in-context copying as a cause.

**Its split still gives a cheap test for #5015 question 1.** Read the MLP outputs at " year" through the logit lens, and take the best rank of " months" over the 6 layers. pythia-70m has 50,304 tokens in its unembedding, so the paper's cut is about rank 503. A rank above the cut means the model never retrieves " months", which is an enrichment failure. A rank below it means " months" is there and the heads select " days" in its place, which is an extraction failure. `openinterpretability/answer_lens.py` reads the residual stream at the last position only, so this needs a per-MLP reading at the subject token.

**The paper's patching method is not one to copy.** Gaussian noise on the subject is the corruption that `activation-patching-practice` advises against. The #4781 answer experiment uses counterfactual prompts and a patching null, and that null REJECTED attempt 1 (null patching up to 0.343). That is the kind of check that the paper does not have.

**For the fix campaign, this is evidence only.** MHM is one kind of fix: a loss on logit-lens readings at chosen layers, with a specificity set. Its fix layers are not the layers that its analysis found, which matches the question in #5016 (Hase et al.). The two questions are whether localisation tells you where to edit, and how to measure collateral damage. The maintainer designs the campaign. This page proposes no design.

**Question for the maintainer.** Should an answer record report a module-level type, such as the paper's enrichment or extraction label, next to its unit-level route? It would let the site compare hallucinations across prompts. It is not a decision-rule change.

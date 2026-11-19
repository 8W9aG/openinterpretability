# Dissecting Recall of Factual Associations

**Citation.** Mor Geva, Jasmijn Bastings, Katja Filippova and Amir Globerson. The affiliations are Google DeepMind, Tel Aviv University and Google Research. "Dissecting Recall of Factual Associations in Auto-Regressive Language Models". arXiv 2304.14767v3, 13 October 2023. Version 1 has the date 28 April 2023, and version 2 has the date 11 October 2023. [Paper](https://arxiv.org/abs/2304.14767). [Code](https://github.com/google-research/google-research/tree/master/dissecting_factual_predictions). The arXiv comment says "Accepted at EMNLP 2023". We did not check the proceedings page. Licence CC BY 4.0. We did not see a funding statement.

**Status.** Planned. #5015 asks why pythia-70m never reaches " months", and this paper names the two places where a factual answer can fail: subject enrichment and attribute extraction. Two tools from the paper are not in the repo, so the page proposes them as follow-ups.

**Read.** We read the abstract page and the v3 HTML on 2026-11-18. A model read the text for us in five passes. It gave short quotes of at most 125 characters, and paraphrase between them. We have quotes from §3, §5, §6, §7, §9, the Limitations section and appendices A.2, A.4, E and G. The abstract, the introduction and the related work came back mostly as summary. We did not see the figures, the tables or the PDF. The last 4,626 characters of the HTML were not read, and they hold the end of appendix H.

## Summary

**Claim.** The paper studies how an auto-regressive model recalls a fact, such as a subject's attribute, at the last token. It finds three steps. Early MLP sublayers enrich the last subject token. Attention moves the relation to the last position. Then the last position queries the subject, and attention heads extract the attribute.

**Method.**

- **Models.** GPT-2 (48 layers, 1.5B parameters) and GPT-J (28 layers, 6B parameters).
- **Data.** Queries from CounterFact. The paper keeps only the queries that the model answers correctly: "1,209 queries for GPT-2 and 1,199 for GPT-J."
- **Attention knockout.** The paper blocks the attention edges from the last position to a set of positions, in a window of layers. The window is 9 layers for GPT-2 and 5 for GPT-J. The measure is the change in the probability of the prediction.
- **Attribute rate.** The paper projects the last subject token's hidden state to the vocabulary. It counts the attribute-related tokens in the top 50 tokens.
- **Sublayer knockout.** The paper cancels MLP or attention sublayers and measures the drop in the attribute rate.
- **Extraction events.** A sublayer update is an extraction event when its top token in the vocabulary projection is the model's final prediction. The extraction rate is the fraction of queries with at least one event.
- **Head parameters.** For the heads that extract the attribute on at least 10% of queries, the paper projects the head's parameters to the vocabulary. This shows subject-to-attribute mappings.
- **Statistics.** We saw no seeds, intervals or significance tests. The paper compares against blocks of non-subject positions, but we saw no random-position or untrained-model baseline.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Block attention to subject tokens, middle-upper layers | prediction probability drops "up to 60%" |
| Block attention to non-subject positions | a drop of "35%-45%" |
| Order of the two flows | the relation flow in early layers comes before the subject flow in upper layers |
| Subject block when the last position can still attend to the first position (A.2) | "up to 40%", against 60% when it cannot |
| Attribute rate at the last subject token, intermediate-upper layers | "close to 50%" |
| Cancel attention sublayers, effect on the attribute rate | "<30% decrease" |
| Cancel early MLP sublayers | a "destructive effect" on the attribute rate |
| Extraction rate by MLP sublayers (GPT-2) | 31.3%, lower than by attention |
| Attention extraction rate, attention to the subject blocked | 30.2% |
| Attention extraction rate, attention to non-subject positions blocked | 31.5% |
| Patch early subject representations | extraction rate drops "by up to 50%" |
| Heads analysed in appendix G | 7 heads that extract on at least 10% of queries |

The model's summary also gave numbers that we did not see in a quote. They are an attention extraction rate of 68.2% (GPT-2) and 76.7% (GPT-J). They also include head mappings in 30.2% and 39.3% of events. We mark these numbers as unverified.

## Where the claims go beyond the evidence

**Correct answers only.** The paper says, "We restrict our analysis to cases where the model predicts the correct attribute as the next token." So it describes how recall succeeds. It does not show how a wrong answer forms.

**Two large models.** The models are GPT-2 at 1.5B parameters and GPT-J at 6B parameters. Appendix E says the GPT-J results "are consistent with those for GPT-2". We saw no small model, no Pythia model and no other architecture.

**One fact source.** All queries come from CounterFact. The relations are encyclopedic facts about named subjects. The prompt does not give the answer in context, so the paper does not study copying from the prompt.

**Projections are approximations.** The authors say that the vocabulary projection "only provides an approximation of the information encoded in these vectors, especially in early layers". The attribute rate and the extraction events both depend on this projection.

**Leakage between positions.** The authors say that knockout "does not account for 'information leakage' across positions". A block in one window does not stop information that moved in earlier layers.

**No intervals.** We saw no seeds, confidence intervals or tests. The sample sizes are about 1,200 queries per model.

## Use in this repo

**The repo has no edge knockout.** Answer experiments mean-ablate whole units and patch them (`openinterpretability/answer_review.py`). A head ablation removes all of the head's output. Edge knockout blocks only the attention from the last position to chosen positions. This lets a test ask which source position carries the answer, without a change to the rest of the head.

**The repo has no vocabulary lens for heads.** The `attention-head-audit` skill says, "There is no vocabulary lens or partner search for heads." The paper's appendix G projects head parameters to the vocabulary to read subject-to-attribute mappings. The `vocab-projection-lookahead` skill already does this for MLP neurons, after Geva et al. 2021 and 2022.

**The logit lens is already there.** `openinterpretability/answer_lens.py` reads the last token's residual stream through the final norm and the unembedding, layer by layer (#4917). The paper's attribute rate is the same projection, but at the last subject token, not at the last token.

**The #4781 route fits the paper's frame, but at the wrong source.** #4781 explains why EleutherAI/pythia-70m answers " days" to "A week has seven days and a year has twelve". The answer route has the verdict VERIFIED, with necessity 0.68, sufficiency 0.62 and patching 0.75. The copy heads L3/H3, L4/H0 and L4/H7 carry " days" to the last position. In the paper's terms, these heads are the extraction step. But they read the earlier clause's unit, not an enriched " year". No verified unit supplies " months".

**The source does not cover in-context copying.** The paper's answers come from the subject's stored facts. The #4781 answer is a token in the prompt, which an induction-style head copies. So the paper's numbers do not transfer to pythia-70m. Only its frame of enrichment, then relation, then extraction carries over.

**What the paper predicts for #5015.** The paper's frame gives two places where " months" can fail. The first is enrichment: early MLPs at " year" may not write " months". The second is extraction: no head at the last position may read " year" and map it to " months". The attribute rate at " year" tests the first. Edge knockout from the last position to " year" and to " days" tests the second. These are evidence for #5015's first question. They are not a design for the maintainer's campaign of model changes.

**Proposed follow-ups.**

- Attention-edge knockout as a measure in answer experiments: block the last position's attention to chosen positions in a window of layers. Area: pipeline. Priority: medium.
- A vocabulary projection of a head's OV circuit in head audits, as in the paper's appendix G. Area: pipeline. Priority: low.
- A comment on #5015: measure the " months" rank in the logit lens at " year", layer by layer. Also measure the answer's change when the last position cannot attend to " days" or to " year".

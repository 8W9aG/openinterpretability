# Does Localization Inform Editing?

**Citation.** Peter Hase (UNC Chapel Hill), Mohit Bansal (UNC Chapel Hill), Been Kim (Google Research) and Asma Ghandeharioun (Google Research). "Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models". arXiv 2301.04213. Version 1 is from 10 January 2023. We read version 2, from 16 October 2023. [Paper](https://arxiv.org/abs/2301.04213), [code](https://github.com/google/belief-localization). The arXiv comment gives the venue as "NeurIPS 2023 (Spotlight)". We did not check the proceedings page. The paper uses the public CounterFact and ZSRE datasets. We saw no funding statement in the text that we read. The compute was one NVIDIA A6000 GPU.

**Status.** Planned. The fix campaign that #4784 prepares for, with #4781 as its first case, would use it. The paper tests the premise "explain a unit, then fix that unit", and finds that the premise fails for one localization method.

**Read.** On 2026-11-18, through WebFetch. A model read the abstract page and the v2 HTML for us. It returned short verbatim quotes from the introduction, Sections 4.2, 4.3, 5.1, 5.3 and 6, the limitations and Appendix A. Other parts came back as its summary. These include the abstract, the method details, Table 2 and Appendices B and C. Some of the Appendix B and C numbers came back as short quotes. We did not see the figures, the PDF or the end of Appendix C. The model found no statement on seeds, confidence intervals or error bars in the part that it read.

## Summary

**Claim.** Causal Tracing locates facts in mid-layer MLPs of GPT-J. But the tracing result for a fact does not predict which layer gives the most successful edit of that fact. In the authors' words, "the correlation between Causal Tracing results and edit success is near zero".

**Method.**

- **Models.** GPT-J (6B) is the main model. GPT2-XL is a robustness check.
- **Data.** 652 CounterFact facts that GPT-J completes correctly. The authors take them from 10% of CounterFact. A fact counts as known when the true object is in the first 36 greedy tokens. GPT-J "achieves a completion accuracy of 32.6% under this scheme". ZSRE is a robustness check.
- **Localization.** Causal Tracing from ROME (Meng et al., 2022). The method adds Gaussian noise to the subject tokens and restores MLP outputs in a window of 5 layers. The tracing effect is the fraction of the clean probability that comes back, from 0 to 1. Each layer's value is its maximum over token positions (summary).
- **Editing methods.** ROME (one MLP layer), MEMIT (five layers) and constrained finetuning with windows of 1 and 5 layers. Each method edits at each layer in turn.
- **Editing problems.** Error Injection, the original ROME task, writes a false object. Tracing Reversal makes the model output its noised-input answer. Fact Erasure removes the true object. Fact Amplification maximises the probability of the true object. Fact Forcing makes the model output the true object from the noised subject, which matches what Causal Tracing measures.
- **Metrics.** The rewrite score "measures how much an edit improves the target probability", as a share of the largest possible gain. The paraphrase score uses paraphrased prompts. The neighborhood score "measures whether edits change predictions for prompts with a similar subject".
- **Analysis.** Linear regressions predict edit success from the edit layer, the tracing effect, or both. The test is how much R² the tracing effect adds to the layer.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| ROME, Error Injection, GPT-J layer 6, n = 652 | Correlation of rewrite score with tracing effect ρ = −0.13, p < 1×10⁻³. The text says "not a positive relationship but a negative relationship" |
| ROME, Error Injection, R² of rewrite score (Table 1) | Layer 0.947, tracing effect 0.016, both 0.948. "The tracing effect is able to explain only 0.1% of the variance in edit success" |
| All editing problems and methods (Figure 6) | The tracing effect adds "at most an additional 3.2% of the variance". The layer explains "58.5% on average across conditions" |
| Fact Forcing, constrained finetuning | The tracing effect adds "an additional 3% of the variance", "up from 1.5% for other experiments". The text calls it statistically significant |
| Paraphrase and neighborhood scores (Appendix B) | The largest R² gains are "around 0.03" for paraphrase, and about 0.02 for neighborhood |
| Overall score (Appendix B) | "The largest gain for overall score is 0.02 for Fact Forcing with constrained finetuning" |
| Unscaled metrics, Error Injection (Appendix C) | Correlation "near zero, at ρ=−.006" |
| The 74 of 652 facts with concentrated tracing (Appendix B, Table 2) | R² of layer 0.927, tracing 0.02, both 0.929. These numbers came from a table that the model read, not from a sentence |
| GPT2-XL, ZSRE, tracing window 10, representation zeroing (Appendix C) | The same pattern as the main result. The largest GPT2-XL R² effect stays "about 2%" |

The authors also report that a large share of the facts trace to layers outside ROME's usual edit range. Their Figure 1 caption gives "layers 1-3 and 16-20" as examples. They conclude that "it is still better to ignore the tracing results and always choose an early-to-mid-layer MLP weight for editing". They also find that "it makes little difference whether we edit at one layer or five layers".

Section 6 gives their explanation. "Many layers could store a fact, and it happens that some do." Also, "it should matter where information enters the residual stream". So "Causal Tracing answers a different question" from the one that editing answers.

## Where the claims go beyond the evidence

**The claim is about one localization method, and the title is wider.** The paper tests Causal Tracing with Gaussian noise, and representation zeroing in an appendix. It does not test minimal-pair patching, attribution patching or explanations of units. The authors state this limit: "Our conclusions may not necessarily hold for the breadth of localization and editing methods".

**One kind of knowledge, two models of the same era.** All facts are subject-relation-object triples from CounterFact and ZSRE. The authors say: "This is a basic form of factual knowledge". They also say that the result "may not generalize to models larger than GPT-J (6B parameters)".

**The edit layer is a category, and the tracing effect is one number.** The layer regressor has one parameter for each layer, so it can fit the shape of the success curve. The tracing effect has one slope. A part of the layer's R² advantage may come from this difference in flexibility. The regression-error tables in Appendix C (RMSE and MAE) support the result, but they use the same regressors.

**The edits write new information.** Error Injection writes an object that the model did not store. A low correlation there shows that an edit can succeed away from the stored fact. It does not show where a stored fact lives. The authors make this point in Section 6, and Fact Erasure, Amplification and Forcing test it. Those tasks also give low added R², up to about 3%.

**No seeds or intervals.** The text gives p-values and F-tests, but we found no seeds, confidence intervals or error bars. The 652 facts are one sample. The filter leaves out facts that GPT-J does not know.

**Credit.** The robustness checks are broad. They use a second model, dataset, tracing window and localization method, the last subject token and unscaled metrics. The authors state their main limits themselves.

## Use in this repo

**The repo already has its own form of the paper's gap.** On the #4781 prompt, "A week has seven days and a year has twelve", the first-order attribution share that the explanation covers is 0.064. But its prompt coverage is 0.938, against 0.101 for matched random units. The unit with the largest single attribution, L5/N2014+, is still unstudied. So on this prompt, the coverage does not depend on the unit that a first-order score ranks first. The numbers are in the explanation record committed in #5014, and in `changelog.d/4781.added.md`. This is not the paper's test, because there is no edit in it. But it shows that two good localizations can disagree, so "fix the top unit" is not well defined.

**Our localization is not Causal Tracing.** Our causal scores use minimal pairs and rank a pole against matched same-layer units (`docs/PROTOCOL.md` §4). The `activation-patching-practice` skill already advises against Gaussian-noise corruption (Zhang and Nanda). So the paper's numbers do not transfer directly. Its warning does transfer: a VERIFIED route says how the model makes an answer. It does not say which weights an edit must change to make a different answer.

**What the paper predicts for the #4781 copying failure.** The verified route copies " days" through the layer-0 count-word and time-unit neurons and the copy heads L3/H3, L4/H0 and L4/H7. No verified unit supplies " months", which is at rank 4. The paper predicts that an edit for " months" will succeed at many early-to-mid layers. It also predicts that the layer will matter more than the route's location. It also predicts that an edit at a copy head can make " months" win on this prompt and leave the route unchanged. Then other "a X has N" prompts would still copy their first unit. This is a prediction from the paper, not a measurement.

**What a fix must report, from the paper's metrics.** The paper scores each edit three ways: rewrite on the prompt, paraphrase, and neighborhood on prompts with similar subjects. For #4781 the other five prompts in `docs/analyses/hallucination-shortlist.jsonl` are a ready paraphrase-like set. All six copy the first clause's unit. Registry coverage (`openinterpretability/coverage.py`) and the sister tool (`openinterpretability/sisters.py`) could measure what an edit breaks elsewhere. #4782 covers that side. How to combine these is a question for the fix campaign, which the maintainer designs.

**Question for the maintainer.** Should a fix experiment report edit success at each layer, and beside it the route's own localization, as the paper does? That would test the paper's claim on our units. It is a campaign design decision, so this page records it and opens no issue.

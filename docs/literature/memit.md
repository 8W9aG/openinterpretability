# Mass-Editing Memory in a Transformer (MEMIT)

**Citation.** Kevin Meng (MIT CSAIL and Northeastern University), Arnab Sen Sharma (Northeastern University), Alex Andonian (MIT CSAIL), Yonatan Belinkov (Technion – IIT) and David Bau (Northeastern University). "Mass-Editing Memory in a Transformer". arXiv 2210.07229. Version 1 is from 13 October 2022, and we read version 2, from 1 August 2023. [Paper](https://arxiv.org/abs/2210.07229). The paper gives [memit.baulab.info](https://memit.baulab.info) for the code and data. We did not open that site. Venue: the PDF header says "Published as a conference paper at ICLR 2023". We did not check the OpenReview page. Funding: an AI Alignment grant from Open Philanthropy. Israel Science Foundation grant 448/20 and an Azrieli Foundation fellowship supported Belinkov, and a Viterbi Fellowship at the Technion supported him too. The paper gives no tool disclosure. The issue writes the second author as "Sen Sharma". The arXiv metadata lists him as "Sharma, Arnab Sen".

**Status.** Planned. The fix plan #5220 names MEMIT (#4869) in its arm F, the closed-form edit baseline. The repo has no MEMIT code now. #5167 tests the single-edit form of the same update ([ROME](locating-editing-factual-associations.md)).

**Read.** On 2026-11-19, I downloaded the v2 PDF from arXiv, converted it to plain text with `pdftotext`, and read all of it myself. No model read it for me. I have the source's own text for the abstract, sections 1 to 9, and appendices A to F. Tables 1 and 2 and Algorithm 1 came through as text. I did not see the images of any figure, only their captions and axis labels. So the curves of Figures 3, 5, 6, 7 and 9 to 13 are not in this page. I give no number that only a figure holds. I read the arXiv abstract page for the version history.

## Summary

**Claim.** A direct, closed-form update to a range of middle MLP layers can insert thousands of facts into a language model at once. The facts generalise to paraphrases, and the model keeps most of its specificity and fluency. The method goes beyond earlier editors by orders of magnitude in the number of edits.

**Method.**

- **Facts and goals.** A fact is a tuple (s, r, o). A batch of edits may not give two objects to the same subject and relation. An edit must work on its own prompt (efficacy) and on paraphrases (generalisation). It must not change nearby subjects (specificity), and the model must stay fluent.
- **Layer choice.** Causal tracing as in ROME, on 501 true facts that GPT-J knows, with Gaussian noise at 3σ and 10 noise samples. A second trace freezes the MLPs at the last subject token. The layers where this "severed" curve falls most below the plain curve form the edit range R. For GPT-J, R = {3, 4, 5, 6, 7, 8}. For GPT-NeoX, R = {6, 7, 8, 9, 10}.
- **One layer as an associative memory.** The MLP output matrix W₀ maps keys k (the MLP's hidden activations at the last subject token) to values m. The update for u new pairs (K₁, M₁) that also keeps the old pairs is Δ = R K₁ᵀ (C₀ + K₁K₁ᵀ)⁻¹ (Eq. 14). Here R = M₁ − W₀K₁ is the residual of the new pairs, and C₀ = λ·E[kkᵀ]. The paper estimates E[kkᵀ] from 100,000 Wikitext samples for GPT-J and 50,000 for GPT-NeoX.
- **Many layers.** For each fact, gradient descent finds a target hidden state zᵢ at the last layer of R and the last subject token. The update then runs through R in ascending order. Each layer takes an equal share of the remaining residual, (zᵢ − hᵢᴸ)/(L − l + 1). The method collects activations again after each layer.
- **Settings.** λ is 15,000 for GPT-J and 20,000 for GPT-NeoX. The zᵢ search runs 25 steps on GPT-J and 20 on GPT-NeoX, with a norm clamp on δᵢ.
- **Models.** GPT-J (6B) and GPT-NeoX (20B).
- **Baselines.** Fine-tuning of GPT-J layer 21 with weight decay (FT-W). MEND, with the GPT-J hypernetwork from the ROME paper. ROME, applied one fact at a time. SERAC had no public code, so the paper leaves it out.
- **Benchmarks.** zsRE, 10,000 real facts, with top-1 accuracy for efficacy, paraphrase and specificity. CounterFact, 21,919 counterfactual statements, with the scores of the ROME paper. These are ES, PS, NS and their harmonic mean S, generation entropy (GE) and a TF-IDF consistency score (RS). The edit counts n run on a log scale from 1 to 10,000.
- **Further tests.** The 27 CounterFact relations with at least 300 cases each. Mixed batches of two relations, up to 700 edits. Ablations of the layer range, of attention as the target, and of λ.

**Findings, as the source reports them.** All numbers are from the text, Tables 1 and 2 and appendices B and E. The paper does not define the values in parentheses in Tables 1 and 2. The Figure 5 caption says that its shaded areas are 95% confidence intervals.

| Setting | Result |
| --- | --- |
| zsRE, GPT-J, 10,000 edits, score / efficacy / paraphrase / specificity (Table 1) | unedited 26.4 / 26.4 / 25.8 / 27.0. MEMIT 50.7 / 96.7 / 89.7 / 26.6 |
| Same, baselines | FT-W 42.1 / 69.6 / 64.8 / 24.1. MEND 20.0 / 19.4 / 18.6 / 22.4. ROME 2.6 / 21.0 / 19.6 / 0.9 |
| CounterFact, GPT-J, 10,000 edits, S / ES / PS / NS (Table 2) | unedited 22.4 / 15.2 / 17.7 / 83.5. MEMIT 85.8 / 98.9 / 88.6 / 73.7 |
| Same, baselines | FT-W 67.6 / 99.4 / 77.0 / 46.9. MEND 23.1 / 15.7 / 18.5 / 83.0. ROME 50.3 / 50.2 / 50.4 / 50.2 |
| Same, fluency GE / consistency RS | unedited 622.4 / 29.4. MEMIT 619.9 / 40.1. FT-W 293.9 / 15.9. ROME 589.6 / 3.3 |
| CounterFact, GPT-NeoX, 10,000 edits, S / ES / PS / NS / GE / RS | unedited 23.7 / 16.8 / 18.3 / 81.6 / 620.4 / 29.3. MEMIT 82.0 / 97.2 / 82.2 / 70.8 / 606.4 / 36.9 |
| Scaling curves, GPT-J (text on Figure 5) | ROME does well up to n = 10 and degrades from n = 32. MEND declines from n = 6 and loses all efficacy before n = 1,000 |
| Runtime, GPT-J, 10,000 edits (appendix B) | MEND 98.25 s. FT-W 1,716.21 s. ROME 44,248.26 s. MEMIT 0.90 h for the update plus 6.54 h for the zᵢ vectors, run in series |
| Mixed batches of two relations, up to 700 edits (section 5.4, Figure 10) | performance close to the average of the two relations alone. Each point is a mean of 10 runs, with 90% intervals |
| Ablations (appendix F, text only) | more layers in R raise efficacy, generalisation and specificity. Late-layer MLPs and attention edits do "considerably worse". Higher λ raises specificity and fluency and lowers efficacy and generalisation. S peaks near λ ≈ 10⁴ |
| Demo, US election results of November 2022 (appendix E) | ES 100%, PS 94% |
| Demo, 289 (star, constellation) facts (appendix E) | accuracy 53% before the edit, 86% after |

The authors state these limits. Some relations, such as P127 ("owned by") and P641 ("plays sport"), are harder to edit with specificity. The method covers only directional (s, r, o) facts. It does not cover spatial, temporal, mathematical, linguistic or procedural knowledge, or symmetric relations.

## Where the claims go beyond the evidence

**"Maintaining excellent specificity" is wider than Table 2.** At 10,000 edits on GPT-J, NS falls from 83.5 to 73.7. On GPT-NeoX it falls from 81.6 to 70.8. That is a fall of about 10 points in each model. The fall is much smaller than for FT-W or ROME, but it is not small.

**Specificity tests one kind of neighbour.** NS checks prompts about nearby subjects with the same relation. On zsRE, specificity is a random unrelated fact, and the unedited score is only 27.0, so it has little room to fall. Neither test checks other relations of the edited subject, facts that the edit implies, or general text. GE and RS are the only checks on generation. Later work finds the gaps: MEMIT fails most ripple tests ([Ripple Effects](ripple-effects-of-knowledge-editing.md)) and most multi-hop questions ([MQuAKE](mquake.md)).

**The scale result is for one batch, not a sequence.** MEMIT inserts all facts at once against one C₀. ROME, the baseline that collapses, runs one fact at a time. The paper does not run MEMIT in sequential batches. Later work reports that MEMIT collapses after 1,000 sequential edits ([The Mirage of Model Editing](mirage-of-model-editing.md)).

**The authors tuned the baselines less than MEMIT.** FT-W edits one layer, and the authors tuned its weight decay only at n = 10,000 (footnote 3, appendix B.1). MEND uses the hypernetwork from the ROME paper, and ROME uses its single-edit defaults. MEMIT's own λ and R come from its ablations, which report on the same CounterFact metrics. The paper does not give a held-out split for these choices.

**The layer choice rests on a trace that later work questions.** Appendix F.1 says that the ablation shows the importance of the causal analysis. But it tests five layer ranges, four of which lie in the traced region. Later work finds that causal tracing does not predict which layer edits best ([Does Localization Inform Editing?](does-localization-inform-editing.md)).

**Two large models, no small one, no seeds for the tables.** All edits run on GPT-J and GPT-NeoX. Tables 1 and 2 give no number of seeds, and the paper does not say what their parentheses hold. Only Figure 10 states its runs: 10 per point.

**No neuron-level comparison.** The paper adopts "the classical view of a linear layer as an associative memory" over the sparse-neuron edits of Dai et al. It does not compare the two.

**Small slips in the text.** Section 4.3 writes "token T" once where the rest of the method uses the last subject token S. Appendix D refers to "Figure D", which is Figure 10. Its case (c) writes "o₁ = τ(o₂)" for "τ(o₁) = τ(o₂)". None changes a result.

The authors credit the narrow scope of (s, r, o) facts and the harder relations themselves.

## Use in this repo

**The repo plans MEMIT as a baseline, and the paper's ablations bear on that.** #5220 lists MEMIT in arm F, at arm B's layers. The paper's R covers six middle layers of GPT-J, and its ablation finds late layers and attention edits "considerably worse". EleutherAI/pythia-70m has only six layers in total, and the paper tests no model below 6B. So the paper gives no evidence about how a layer range should look in a six-layer model. This is evidence for the campaign, not a design for it.

**Fewer poles keep their activations than under ROME.** The [ROME page](locating-editing-factual-associations.md) shows that, in pythia-70m (`use_parallel_residual: true`), every pole at or below a single edit layer fires as before. MEMIT edits every layer in R in turn. So only the poles at or below the lowest layer of R keep their activations by construction. Each pole above it can change its activation. The reason is that the edited layers below it write new values into the residual stream. The 86 VERIFIED pythia-70m poles are 16, 18, 9, 15, 15 and 13 in layers 0 to 5 (`papers/neuron-development/README.md`). So the built-in check covers 16 poles when R starts at layer 0, and 34 when it starts at layer 1. It covers 43, 58 and 73 when R starts at layers 2, 3 and 4.

**Each pole's write change is still in closed form, but now shared across facts.** This follows from Eq. 14. The paper does not discuss single neurons. Write M = K₁ᵀ(C₀ + K₁K₁ᵀ)⁻¹, with one row per fact and one column per neuron. Then neuron j's write `w_out` changes by R·M[:, j], a mix of the residuals of all facts in the batch. In pythia-70m, Δ is 512 × 2048 (`hidden_size` and `intermediate_size` in the model config). So its rank is at most min(u, 512). A batch of more than 512 facts can change the writes in every direction of the residual stream. That last point is my inference, and the paper does not test it.

**The paper's one interaction test is coarse.** Section 5.4 mixes two relations in one batch and finds scores near the average of the two alone. That is the only test of interference between edits in the paper. It measures edit scores, not the units that carry other knowledge. The repo can measure more: registry coverage (`docs/analyses/registry-coverage.md`), the sister search (`docs/analyses/sister-neurons.md`) and each VERIFIED pole's own probes. #5230, the campaign's interference report, owns that measure. We record this as evidence, not as a design.

**The edit reads a dense direction over neurons, not a set of neurons.** Each fact's row of M weights all of a layer's neurons through (C₀ + K₁K₁ᵀ)⁻¹. The registry's superposition entities are sparse directions over one layer's neurons (`openinterpretability/explain_groups.py`). The paper did not compare its direction with a set of neurons. So a VERIFIED pole has no special status in a MEMIT edit, unless C₀ makes its column of M large. The ROME page records the same question for one fact. "Knowledge in Superposition" (#4876) and AlphaEdit (#4875), both open, bear on this point.

**The λ result is the trade-off that #4782 is about.** Appendix F.3 says that higher λ keeps old behaviour and lowers efficacy and generalisation. S peaks near 10⁴. So one constant in a closed form sets how much the new facts interfere with the old. The paper reports S, a single point on that curve. It gives no curve of NS against ES that we can read from the text.

**What would change.** No new issue. The page adds evidence to two open issues:

- On #5167: the MEMIT form of the edit keeps by construction only the poles at or below the lowest edited layer. Each pole's write change in a batch is R·M[:, j].
- On #5220, arm F: the paper tests no model below 6B. Its layer range spans six middle layers of a deeper model, and its results are for one batch, not a sequence.

**Related reviews.** The single-edit method and its trace are on the [ROME page](locating-editing-factual-associations.md). The survey's numbers for MEMIT are on the [editing survey page](editing-llms-survey.md). Sequential collapse, ripple failures and multi-hop failures are on the [Mirage](mirage-of-model-editing.md), [Ripple Effects](ripple-effects-of-knowledge-editing.md) and [MQuAKE](mquake.md) pages. MEMIT-MASS as a lifelong baseline is on the [WISE page](wise-lifelong-editing.md). The synthesis for the #4781 " days" error is on the [fixing-models page](fixing-models.md). That page and the survey page explain why a stored-fact editor does not match an in-context copy.

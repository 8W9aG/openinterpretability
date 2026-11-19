# Locating and Editing Factual Associations in GPT (ROME)

**Citation.** Kevin Meng (MIT CSAIL), David Bau (Northeastern University), Alex Andonian (MIT CSAIL) and Yonatan Belinkov (Technion – IIT). Meng and Bau contributed equally. "Locating and Editing Factual Associations in GPT". arXiv 2202.05262. Version 1 is from 10 February 2022, and we read version 5, from 13 January 2023. [Paper](https://arxiv.org/abs/2202.05262). The paper gives [rome.baulab.info](https://rome.baulab.info/) for the code, the CounterFact dataset and a demo notebook. We did not open that site. Venue: the PDF footer says "36th Conference on Neural Information Processing Systems (NeurIPS 2022)". We did not check the proceedings page. Funding: an AI Alignment grant from Open Philanthropy, DARPA SAIL-ON HR0011-20-C-0022 and XAI FA8750-18-C-0004, Israel Science Foundation grant 448/20 and an Azrieli Foundation fellowship. A Viterbi Fellowship at the Technion supported Belinkov. The paper gives no tool disclosure.

**Status.** Used. The `activation-patching-practice` skill names this paper as the source of causal tracing. `ActivationPatcher.denoise` in `openinterpretability/harness/patcher.py` is the paper's restoration run, applied to single neurons. The skill rejects the paper's Gaussian-noise corruption, and the repo has no weight-edit code.

**Read.** On 2026-11-19, I downloaded the v5 PDF from arXiv, converted it to plain text with `pdftotext`, and read it myself. No model read it for me. I have the source's own text for the abstract, sections 1 to 6, the acknowledgements and appendices A to J. Tables 1 to 6 came through as text. I only skimmed the generated texts of appendix G and the sample records of Figures 21 and 22. I did not see the images of any figure, only their captions. So the per-layer curves of Figures 5, 7 to 9, 13, 15, 23 and 24 are not in this page. I read the arXiv abstract page for the version history.

## Summary

**Claim.** In GPT models, the recall of a fact about a subject runs through MLP modules at middle layers, at the last token of the subject. A rank-one change to one such MLP's output matrix can insert a new fact. The new fact generalises to paraphrases and stays specific to its subject.

**Method.**

- **Causal tracing.** A fact is a tuple (s, r, o) that a prompt states, and the model must predict o. Each prompt runs three times:
  - a clean run.
  - a corrupted run, with Gaussian noise added to the subject's token embeddings.
  - a corrupted run in which one hidden state takes back its clean value.

  The total effect is P[o] in the clean run minus P[o] in the corrupted run. The indirect effect of a state is the rise in P[o] when the run restores that state. The noise has a standard deviation of three times the observed standard deviation of token embeddings. Each prompt repeats with ten noise samples.
- **Tracing data.** 1,000 facts that GPT-2 XL (1.5B, 48 layers) knows, from CounterFact templates. For MLP and attention, the trace restores a run of ten layers, not one layer. Appendix B repeats the traces on GPT-2 Medium, GPT-2 Large, GPT-J (6B) and GPT-NeoX (20B).
- **ROME.** The paper treats the output matrix W of one MLP as a linear associative memory. The new matrix is W + Λ(C⁻¹k\*)ᵀ, the closed-form solution of least squares under the constraint Ŵk\* = v\*:
  - k\* is the MLP's hidden activation vector at the last subject token, averaged over 20 random prefixes of 5 or 10 tokens.
  - v\* is a vector that, put in place of the MLP output, makes the model predict the new object o\*. A KL term keeps the model's view of the subject ("{subject} is a").
  - C is the uncentered second moment of the hidden activations over 100,000 Wikipedia tokens.
  - The edit is at layer 18 of GPT-2 XL (layers count from 0). It takes about 2 s on one A6000 GPU.
- **Benchmarks.** zsRE question answering (10,000 records), with efficacy, paraphrase and specificity accuracy. CounterFact, a new set of 21,919 false facts from ParaRel and WikiData. Its scores:
  - efficacy: P[o\*] > P[o_c] on the edited prompt.
  - paraphrase: the same test on rephrased prompts.
  - neighbourhood: P[o_c] > P[o\*] on ten prompts about nearby subjects that share the true object.
  - S: the harmonic mean of the three scores.
  - generation entropy (GE) for fluency, and a TF-IDF consistency score (RS) for generated text.
- **Baselines.** Fine-tuning of one MLP layer (FT), and the same with an L∞ limit (FT+L). Knowledge Neurons (KN), KnowledgeEditor (KE) and MEND, as released and as retrained on each benchmark. Fifteen volunteers compared ROME with FT+L on generated text.

**Findings, as the source reports them.** All numbers are from the text, Tables 1 to 6 and appendices B, E and J.

| Setting | Result |
| --- | --- |
| Causal tracing, GPT-2 XL, 1,000 known facts | average total effect 18.6%. Mean P[o] is 27.0% clean and 8.47% corrupted |
| Same, one hidden state at the last subject token | average indirect effect 8.7% at layer 15 |
| Same, runs of ten MLP or attention layers at the last subject token | MLP peaks at 6.6%, attention at 1.6% |
| Same, one MLP or attention output on its own | "generally negligible effects" (appendix B.2) |
| zsRE, GPT-2 XL, efficacy / paraphrase / specificity | unedited 22.2 / 21.3 / 24.2. ROME 99.8 / 88.1 / 24.2. MEND-zsRE 99.4 / 99.3 / 24.1. FT 99.6 / 82.1 / 23.2 |
| CounterFact, GPT-2 XL, 7,500 records, S / efficacy / paraphrase / neighbourhood | unedited 30.5 / 22.2 / 24.7 / 78.1. ROME 89.2 / 100.0 / 96.4 / 75.4 |
| Same, baselines | FT 65.1 / 100.0 / 87.9 / 40.4. FT+L 66.9 / 99.1 / 48.7 / 70.3. KN 35.6 / 28.7 / 28.0 / 72.9. MEND 57.9 / 99.1 / 65.4 / 37.9 |
| Same, fluency GE / consistency RS | unedited 626.6 / 31.9. ROME 621.9 / 41.9 |
| CounterFact, GPT-J, 2,000 records, S / efficacy / paraphrase / neighbourhood | unedited 23.6 / 16.3 / 18.6 / 83.0. ROME 91.5 / 99.9 / 99.1 / 78.9. FT 25.5 / 100.0 / 96.6 / 10.3 |
| CounterFact, GPT-2 Medium and Large, S | ROME 87.4 and 88.2. FT+L 68.0 and 71.2 |
| ROME with no random prefix for k\* | S 86.1, against 89.2 with prefixes |
| Human evaluation, 15 volunteers, 50 facts, 150 evaluations | raters were 1.8 times more likely to rate ROME more consistent with the new fact than FT+L. They were 1.3 times less likely to rate it more fluent |

Intervals in the tables are 95% confidence intervals over records. The authors state these limits: ROME edits one fact at a time, and an edit works in one direction only. They did not study logical, spatial or numerical knowledge. An edited model also guesses new, unsupported facts about the subject.

## Where the claims go beyond the evidence

**"Accurate understanding not only of where … but also how" rests on a visual match.** Section 3.4 reports "strong correlations" between the trace and the edit success by layer, from Figure 5. The paper gives no correlation coefficient. Later work tested this and found that the trace adds 0.1% to the variance of edit success that the layer explains ([Does Localization Inform Editing?](does-localization-inform-editing.md)).

**The trace localizes a window, and the edit uses one layer.** Appendix B.2 says that restoring one MLP output has "generally negligible effects". The MLP result needs a run of ten layers. So the trace does not single out layer 18. The claim that any middle layer could hold a fact is a conjecture, which the paper says.

**The trace needs off-distribution noise.** Appendix B.4 says that noise at one standard deviation makes the total effects "very small". The method works at three standard deviations. The authors credit this point: they test multivariate and uniform noise and a wider corruption, and the pattern stays. But all these corruptions are noise, not minimal pairs.

**The last subject token is not always the site.** Figures 11 and 14 show facts whose decisive token is elsewhere, for example "Mitsubishi" in "Mitsubishi Electric". The authors show these cases themselves. The paper gives no count of how often this happens.

**The tuning and the test share data.** The 1,000 tracing prompts come from CounterFact templates. The authors tuned the baselines on a 50-record CounterFact subset (Figure 17). Appendix E.5 compares prefix choices by the test score S. The paper gives no seeds for the v\* optimisation or the random prefixes.

**Specificity is narrow.** Neighbourhood tests check that the true object still beats o\* for nearby subjects of the same relation. They do not test facts that the edit implies, or other relations. Even so, ROME lowers the neighbourhood score from 78.1 to 75.4 on GPT-2 XL, and from 83.0 to 78.9 on GPT-J. The authors credit two limits: zsRE specificity is "not a sensitive measure", and the human raters found a fluency loss that GE missed.

**The neuron baseline is weak evidence about neurons.** KN scores S 35.6, but it ran with the defaults of a third-party reimplementation, built for BERT. The paper models the MLP as one associative memory and says that this "differs from Geva et al.'s per-neuron view". It does not test neuron-level edits with other selection rules.

**Single edits only.** The authors state it. Later work finds that ROME collapses under many sequential edits ([The Mirage of Model Editing](mirage-of-model-editing.md)) and fails most ripple tests ([Ripple Effects](ripple-effects-of-knowledge-editing.md)).

**The text refers to panels that v5 does not have.** Appendix E.5 cites "Figure 1k", and appendix B cites Figures 1j, 1l and 1m, but Figure 1 has panels (a) to (g). Appendix D cites a "Drawdown column in Table 1", and Table 1 has no such column. Appendix C names P_N twice where the second should be the paraphrase set. GPT-2 Medium is 345M in appendix F and 334 million in the Figure 9 caption. These look like leftovers from earlier versions, and none changes a result.

## Use in this repo

**The repo uses the paper's restoration run, not its corruption.** `ActivationPatcher.denoise` puts a clean activation into a corrupted run, as the paper's third run does. But it does this for one neuron, between minimal pairs. The `activation-patching-practice` skill says to avoid Gaussian noise, after Zhang and Nanda. Appendix B.4 supports that choice: the paper's effects need noise at three standard deviations.

**The window result agrees with multi-unit routes.** In the paper, one MLP output alone restores almost nothing, and a run of ten layers restores much. The repo finds the same shape for answers. The first #4781 claim was a 28-unit route. After the third batch of #4781, 63 explained units brought back 81.7% of the " days" answer. So a small effect for one pole on an answer is not evidence that the pole has no role.

**A rank-one edit has a known effect on each pole.** This follows from the paper's Eq. 2, but the paper does not discuss single neurons. Write u = C⁻¹k\*. Then column j of the output matrix, which is neuron j's write `w_out`, becomes w_j + u_j·Λ. The input weights do not change. For EleutherAI/pythia-70m, which has a parallel residual (`use_parallel_residual: true`), three things follow for an edit at layer l:

- Every pole at layer l or below fires as before, on every input. So its activation claim keeps by construction.
- Each pole at layer l keeps its trigger but changes its write by u_j·Λ. Its lookahead, steering and vocabulary claims can change, in proportion to |u_j|. `openinterpretability/harness/wiring.py` and `crossref.py` read `w_out`, so their links from such a pole change too.
- Poles above layer l change only through the residual stream.

The registry has 86 VERIFIED pythia-70m poles: 16, 18, 9, 15, 15 and 13 in layers 0 to 5 (`papers/neuron-development/README.md`). So an edit at layer 2 cannot change the activations of 43 of them. That gives a built-in check, like the LoReFT arm on #5125. A failure at or below the edit layer is a bug in the test.

**This is the clean case of #5133.** #5133 asks whether the weight change at a pole predicts whether its explanation keeps under instruction tuning. There, Δ comes from training. Under ROME, each pole's change is exactly |u_j|·‖Λ‖, with no other change in the model. So ROME gives #5133's question with a known Δ and one layer. #5118 (Fisher) and #5125 (narrow fine-tunes) ask the same question with less control.

**C⁻¹ may aim the edit at the registry's units.** This is my inference, not a result in the paper. C is the second moment of every neuron's activation over all Wikipedia tokens. A neuron that fires strongly on the subject but seldom elsewhere has a large u_j. The registry explains sparse, selective poles. So a ROME edit could fall more on VERIFIED poles than on matched random neurons. That is testable from weights and one forward pass, with a matched random null.

**The " days" route of #4781 does not fit the paper's frame.** ROME edits a memory whose key is a subject's last token. The VERIFIED " days" route is an in-context copy: copy heads L3/H3, L4/H0 and L4/H7 carry " days" to the last position (`docs/literature/fixing-models.md`). Appendix I finds that edits to late attention make the model repeat the target on the edited prompt only. A ROME edit "year → months" at " year" would add a memory that must outvote those heads. The paper never tests that conflict. Its smallest model is GPT-2 Medium, about five times the size of pythia-70m. The neighbour pages make the same point in their own terms ([Dissecting Factual Recall](dissecting-factual-recall.md), [Key-Value Memories](ffn-key-value-memories.md)). This is evidence for the campaign, not a design for it.

**What would change.** One follow-up issue, below, and two comments. Both comments add evidence only:

- On #5133: ROME gives each pole's Δ in closed form, as |u_j|·‖Λ‖ on its `w_out` column.
- On #5125: an edit arm keeps every pole at or below the edit layer by construction.

**Proposed follow-up.** "Which VERIFIED pythia-70m poles keep after one rank-one MLP edit?" Apply ROME edits of unrelated facts at one layer of pythia-70m. Re-run each VERIFIED pole's own probes, and test |u_j| against keep or fail with a matched random null. Area: research questions. Priority: medium. It is part of #4782, edits no fact that the fix campaign targets, and fixes nothing.

**Question for the maintainer.** Should a stability test of explanations under editing use ROME's C from a general corpus, or from the repo's own `corpus.jsonl`? The choice changes u, so it changes which poles the edit touches. This bears on how a fix would be judged, so this page records it and opens no issue.

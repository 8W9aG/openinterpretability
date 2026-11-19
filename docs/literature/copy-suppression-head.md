# Copy Suppression: Comprehensively Understanding an Attention Head

**Citation.** Callum McDougall and Arthur Conmy (independent, joint first authors), Cody Rushing (University of Texas at Austin), Thomas McGrath (independent) and Neel Nanda (Google DeepMind). "Copy Suppression: Comprehensively Understanding an Attention Head". arXiv 2310.04625v1, 6 October 2023. [Paper](https://arxiv.org/abs/2310.04625). A footnote says that the authors did part of the work at Google DeepMind. The acknowledgements name the SERI MATS summer 2023 program, the London Initiative for Safe AI and the Center for AI Safety compute cluster. The paper links an [interactive app](https://copy-suppression.streamlit.app/) and no code repository.

The work appeared later as "Copy Suppression: Comprehensively Understanding a Motif in Language Model Attention Heads". The venue is the 7th BlackboxNLP Workshop (Miami, November 2024), pages 337–363, [doi:10.18653/v1/2024.blackboxnlp-1.22](https://aclanthology.org/2024.blackboxnlp-1.22/). We checked that venue on the ACL Anthology entry. The issue had it from memory. The arXiv title and the workshop title differ, and we read only the arXiv v1 text.

**Status.** Planned. The page supplies the evidence for #5015 question 1, why nothing favours " months". It proposes a test of L4/H1 as a copy-suppression head, and a vocabulary lens for the head scan.

**Read.** On 2026-11-18, through WebFetch, by a model. We read the arXiv abstract page and the v1 HTML, which is 113,530 characters long. The first pass covered characters 0 to 100,000, and a second pass read the rest. The web reader would not quote passages longer than about 25 words. So we have the source's own words only as short quotes and printed numbers, and the rest as the tool's summary. We asked targeted questions about the abstract, Section 3 (the OV and QK circuits, and the ablation) and Section 4.1 (self-repair in IOI). We also asked about the conclusion's limitations and Appendices A, B, G, K, O, P, Q and R. We did not see the figures, the PDF or the TeX source. We did not read the BlackboxNLP version.

## Summary

**Claim.** Head L10H7 in GPT-2 Small has one main role over its whole training distribution, which is copy suppression. Earlier components predict a token that is already in the context. L10H7 attends back to that token and lowers its logit. The authors say that this improves calibration, and that it explains part of self-repair. When an ablation removes an early copier, there is nothing left to suppress.

**Method.**

- **Model and units.** GPT-2 Small, mainly head L10H7, with L11H10 as a second negative head. Appendix B adds GPT-2 Medium and one sentence on Pythia.
- **Weights.** The authors read the OV circuit from the token's effective embedding to the unembedding. The test asks whether each token's own logit is among the most negative in its column. They read the QK circuit from a predicted token's unembedding to source-token embeddings.
- **Ablation.** Copy Suppression-Preserving Ablation (CSPA) keeps only two parts of the head. The OV part keeps the negative projections onto the source tokens' unembeddings. The QK part mean-ablates attention to every source token outside the top 5% of predicted tokens. "Effect explained" is 1 − D_CSPA / D_MA. D is the KL divergence from the clean output, and MA is a mean ablation of the head.
- **Data.** OpenWebText. One separate check uses "3000 documents of at least 1024 tokens".
- **Self-repair.** In the IOI task, the authors ablate the three Name Mover Heads in layer 9 and measure how much the negative heads compensate.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| CSPA on L10H7, OpenWebText, KL against mean ablation | 76.9% of the head's effect explained |
| OV part only / QK part only | 81.1% / 95.2% |
| CSPA by loss: net loss effect / absolute change in loss | 82% / 45% |
| OV circuit: token's own logit among the 10 most negative in its column | 84.70% of the vocabulary |
| OV circuit: token's own logit in the bottom 5% | 98.86% |
| QK circuit: diagonal value largest in its row | 95.72% |
| OV in practice: high-attention source token among the 10 most suppressed | 78.24% of (source, destination) pairs |
| IOI self-repair explained by L10H7 and L11H10 together | 39% |
| Removing L10H7's direct effect | The model goes "from net over-confident to net under-confident" |

## Where the claims go beyond the evidence

- **One head in one model.** The paper shows the mechanism in detail for L10H7 in GPT-2 Small. The authors name "the transferability of our results to different models" as a limitation. Credit to them for that.
- **Pythia has one sentence.** Appendix B says only "We also find instances of copy suppression (though weaker) in the Pythia models". It gives no model size, head, metric or number. The authors say that they have no mechanistic understanding of these cases.
- **No intervals or seeds.** The main results are single numbers on OpenWebText, with no confidence intervals and no repeat on other samples.
- **The loss number is lower than the headline.** CSPA explains 76.9% by KL and 82% of the net loss effect. It explains only 45% of the absolute change in loss, so much of the head's effect on single tokens stays outside the account.
- **The QK account is an ideal.** Section 3.2 says that the idealised QK setup "does not completely faithfully represent L10H7's real functioning". The conclusion names the query inputs of self-repair heads as not understood. Appendix R adds that "the unembeding component is not the full story" for backup heads.
- **The head does not suppress every token.** Appendix G says that "a large fraction of the tokens which failed to be suppressed were function words". Appendix K says that the most suppressed token is often a related form, such as a plural or a capitalised form.

## Use in this repo

**The prediction for #4781.** Copy suppression predicts that a late head reads the predicted " days" in the query. It then attends from " twelve" back to " days" in the first clause and pushes " days" down. The verified route (`wiki/answers/EleutherAI--pythia-70m/970a91e11d8c/20261118T234629Z.json`) has no such head. Its heads either build the clause frame or copy " days": L3/H3, L4/H0 and L4/H7. The 20 attributed units of the explanation (`wiki/explanations/EleutherAI--pythia-70m/970a91e11d8c/20261118T235000Z.json`) include no layer-4 or layer-5 head that opposes " days".

**Our check: one candidate, L4/H1, and it is weak.** These numbers are ours, not the paper's. A short script outside the repo ran pythia-70m on CPU in float32. We applied the paper's OV test from Section 3.1 to every head. The effective embedding was W_E plus the layer-0 MLP. We drew 4,000 random tokens with seed 0.

- L4/H1 is the only head with a suppression signature. For 42% of the tokens, the token's own logit is among the 10 most negative. For " days", it is the 7th most negative. Every other head scores below 0.5%. In GPT-2 Small, L10H7 scores 84.70%, so the "weaker" of Appendix B fits.
- The copy heads have the opposite sign. The token's own logit is among the 10 most positive for 79% of tokens in L3/H3 and 57% in L4/H7.
- On the #4781 prompt, L4/H1 puts 0.33 of its attention at the last position on " days". Its direct effect lowers " days" by 0.31 logits against the vocabulary mean, with the final norm scale frozen.
- The heads that raise " days" are larger by the same measure: L3/H3 +1.43, L5/H7 +1.32 and L4/H7 +0.74.
- No layer-5 head puts more than 0.0005 of its attention on " days". So the last layer has no copy suppression for this prompt.
- Zeroing L4/H1's output raises P(" days") from 0.726 to 0.811. Scaling it by 4 gives 0.317. Scaling it by 8 gives 0.061, and " months" becomes the top token at only 0.089, just ahead of " weeks" at 0.079.

**What that means.** The model has a copy-suppression head, and it checks the copy in the right direction. But it is too weak, and its layer limits it. L4/H1 sits in the same layer as L4/H0 and L4/H7, so it can read only L3/H3's prediction. Nothing in layer 5 checks the copy. This is evidence for #5015 question 1. It is one prompt with direct-path numbers, and nothing here is a verified claim. The registry's own L4/H1 record (`wiki/experiments/EleutherAI--pythia-70m/L4/H1/README.md`, NEEDS_REVISION) describes the same attention habit: "the second occurrence of a repeated element", where "it attends the member just read". That record describes when the head is active, not what it writes.

**Suppression lowers confidence. It does not supply the fact.** In the paper, copy suppression improves calibration. Our scaling test agrees. A strong L4/H1 removes " days", but the probability spreads over " months", " weeks" and " hours". No unit in the route or in L4/H1 writes "a year has twelve months". So copy suppression would explain why the model is less sure, not how it could be right. The maintainer designs the campaign that changes the model. This page reports evidence only.

**Self-repair and attribution bear on our causal scores.** The paper warns that negative and backup components "may counteract the effects of ablations". It adds that attribution methods that find only positive components may miss negative ones. L4/H1 is not among the 20 attributed units, and it is not in the route. A route that leaves out a suppressor can look more necessary than it is. The `activation-patching-practice` skill already covers self-repair, from McGrath et al. 2023 and Rushing and Nanda 2024. The second of those has authors in common with this paper.

**The head scan has no output side.** The `attention-head-audit` skill says "There is no vocabulary lens or partner search for heads". The paper's OV-diagonal score is a cheap vocabulary lens. It tells copying heads from suppressing heads by weights alone, and the test above took seconds. It would put L4/H1's sign in its evidence before a proposer writes a claim.

**Proposed follow-ups.** Neither issue is open yet.

- *Is pythia-70m L4/H1 a copy-suppression head?* Area: research questions. Priority: high, as a part of #5015. Run a CSPA-style ablation on a corpus. Measure L4/H1's direct effect on the copied unit for the six shortlist prompts in `docs/analyses/hallucination-shortlist.jsonl`.
- *Add an OV vocabulary lens to head scan evidence.* Area: pipeline. Priority: medium. Record each head's copying and suppression scores (the own-token rank test), so that head claims can name the output sign.

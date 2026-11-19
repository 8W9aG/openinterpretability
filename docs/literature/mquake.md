# MQuAKE: Assessing Knowledge Editing via Multi-Hop Questions

**Citation.** Zexuan Zhong (Princeton University), Zhengxuan Wu (Stanford University), Christopher D. Manning (Stanford), Christopher Potts (Stanford) and Danqi Chen (Princeton). Zhong and Wu contributed equally. "MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop Questions". arXiv 2305.14795. Versions: v1 on 24 May 2023, v2 on 29 October 2023, and v3 on 9 September 2024. [Paper](https://arxiv.org/abs/2305.14795). Code and data: <https://github.com/princeton-nlp/MQuAKE>. Venue: EMNLP 2023, from the arXiv comments field. We did not check the proceedings page.

- **Version mismatch.** v3 is a year later than EMNLP 2023. A v3 footnote says that the authors fixed the 3,000-instance test subset in September 2024, after a report of knowledge conflict. So the numbers on this page may differ from the EMNLP paper's numbers.
- **Funding.** An NSF CAREER award (IIS-2239290), a Sloan Research Fellowship, and Microsoft Azure credits. Zhong had a JP Morgan Ph.D. Fellowship.
- **Tools.** ChatGPT (`gpt-3.5-turbo-instruct`) wrote the multi-hop questions. GPT-3.5 (`text-davinci-003`) wrote the extra gold labels for MQuAKE-T.

**Status.** Planned. Issue #4784 will use it in its synthesis page, `docs/literature/fixing-models.md`, as the main test of whether an edit carries to the consequences of the edited fact. No code, skill or protocol rule depends on it now.

**Read.** On 2026-11-18, the abstract page and the v3 HTML (`https://arxiv.org/html/2305.14795v3`). The WebFetch tool returned only summaries. So we downloaded the HTML and read its text ourselves, with the maths kept as LaTeX. We have the paper's own text for all sections: Sections 1–7, the limitations, the acknowledgements and Appendices A–J. That text includes Tables 1–16. We did not see Figures 1–3, which are images, or the PDF, v1 or v2. No model summarised the source for this page.

## Summary

**Claim.** Benchmarks for knowledge editing mostly test whether the edited model recalls the edited fact. A true edit should also change the answers that follow from that fact. On multi-hop questions that test this, current editors "fail catastrophically", although they recall the edits well. A memory-based method, MeLLo, does better.

**Method.**

- **Facts and edits.** A fact is a triple (subject, relation, object). An edit changes the object, for example the UK head of government from Boris Johnson to Rishi Sunak. A multi-hop question asks about the head of a chain of facts, and its answer is the tail.
- **MQuAKE-CF.** The authors sample chains of 2, 3 or 4 facts from Wikidata, with 37 relations and the top 20% of entities by Wikipedia links. They keep a chain only when GPT-J recalls every single fact in it, with 8 in-context examples. ChatGPT writes three questions per chain. Counterfactual edits replace 1 to 4 facts, and the new chain must give a different answer. The full set has 9,218 instances. All experiments use a 3,000-instance subset with no conflicting edits.
- **MQuAKE-T.** Real changes between the Wikidata dumps of April 2021 and April 2023, in 6 relations chosen by hand. It has 1,868 instances, each with one edit.
- **Metrics.**
  - *Edit-wise success*: the share of edited facts that the edited model recalls.
  - *Instance-wise accuracy*: the share of instances in which the model recalls every single fact of the chain.
  - *Multi-hop accuracy*: the main metric. An instance counts as right when the model answers any one of its three questions correctly.
  - *Multi-hop (CoT)*: the same, with chain-of-thought prompts.
- **Models and editors.** GPT-J (6B) and Vicuna-7B. Fine-tuning of one layer with a norm constraint (FT), MEND, ROME and MEMIT, mostly at the default settings.
- **Scale.** First the authors edit each instance alone. Then they edit groups of k instances at once. On MQuAKE-CF, k is 1, 100, 1000 or 3000. On MQuAKE-T, k is 1, 100, 500 or 1868.
- **MeLLo.** The base model stays frozen. The edits are sentences in a Contriever index. The model splits the question into subquestions and answers each one. It then retrieves the nearest edit and checks if that edit contradicts its answer.

**Findings, as the source reports them.** All numbers are percentages.

| Setting | Result |
| --- | --- |
| MQuAKE-CF, GPT-J, before any edit | Instance-wise 100.0, multi-hop 40.5, multi-hop (CoT) 39.5 |
| MQuAKE-CF, GPT-J, each instance edited alone | Edit-wise: FT 46.9, MEND 73.2, ROME 88.3, MEMIT 96.2 |
| The same, multi-hop | FT 1.5, MEND 8.0, ROME 7.4, MEMIT 7.0 |
| The same, multi-hop (CoT) | FT 1.4, MEND 11.7, ROME 19.9, MEMIT 11.8 |
| MQuAKE-CF, Vicuna-7B, before any edit | Instance-wise 62.4, multi-hop 30.2, multi-hop (CoT) 36.1 |
| MQuAKE-CF, Vicuna-7B, edited alone | Edit-wise ROME 99.7 and MEMIT 95.8. Multi-hop ROME 6.9 and MEMIT 4.9 |
| MQuAKE-T, GPT-J, before any edit | Multi-hop 34.3, multi-hop (CoT) 46.8 |
| MQuAKE-T, GPT-J, edited alone | Edit-wise: ROME and MEMIT 100.0. Multi-hop: both 0.3 |
| MQuAKE-T, GPT-J, MEND | Multi-hop 16.0, multi-hop (CoT) 38.2 |
| MQuAKE-CF, GPT-J, multi-hop (CoT), k = 1, 100, 1000, 3000 | MEMIT 11.8, 10.5, 7.9, 5.4. MeLLo 38.9, 20.3, 15.6, 14.2 |
| MQuAKE-CF, MeLLo with GPT-3.5, the same k | 52.1, 35.3, 31.8, 30.9 |
| MQuAKE-T, GPT-J, k = 1, 100, 500, 1868 | MEND 38.2, 17.4, 12.7, 4.6. MeLLo 85.9, 45.7, 33.8, 30.7 |
| MQuAKE-T, MeLLo with GPT-3.5, the same k | 91.1, 87.4, 86.2, 85.5 |
| MeLLo with GPT-3.5, MQuAKE-CF, all edits retrieved | Retrieval accuracy 88.1 at k = 1 and 59.7 at k = 3000. MeLLo answers 73.1 of the fully retrieved questions |
| Multi-hop (CoT), GPT-J, MQuAKE-CF, 2-hop and 4-hop | Before edits 39.5 and 52.4. ROME 37.3 and 7.7. MEMIT 23.3 and 5.1 |

The authors conclude that editors "are rather hard coding them into the model by updating weights locally". In their reading, the edits do not become knowledge that the model uses.

## Where the claims go beyond the evidence

**The "hard coding" reading has no mechanistic test.** The paper measures behaviour only. It does not look at where an edit lands or which computation the multi-hop question uses. "Locally" is an interpretation of the scores, not a finding.

**Two models, one filter.** The authors say that they test only GPT-J and Vicuna-7B. The filter keeps the chains that GPT-J recalls, and the authors test Vicuna on the same chains. Vicuna's instance-wise accuracy before any edit is 62.4, so about a third of its instances fail before an edit.

**No side-effect measure.** MQuAKE tests whether an edit spreads to its consequences. It does not test whether unrelated facts stay unchanged. The paper leaves that to earlier benchmarks. So a high MQuAKE score says nothing about collateral damage.

**The base model fails most multi-hop questions too.** GPT-J recalls every single fact in every instance, but answers only 40.5% of the multi-hop questions before an edit. So most of the "drop" is from a low start. The edited model must also do a composition that the base model mostly can't do.

**A lenient metric.** An instance counts as right when any one of three questions is right. The paper gives no per-question accuracy.

**No intervals and no seeds.** Each number is one run on one fixed subset. The paper gives no confidence intervals and no repeats.

**The text disagrees with the tables in places.**

- Section 4.2 gives instance-wise "59.6% for MEND vs. 94.0% for MEND". Table 3 has 60.4 for MEND and 91.3 for MEMIT, and the sentence names MEND twice. The old numbers may be from before the 2024 fix.
- Section 4.2 says that CoT "helps slightly across all settings". For FT it lowers the score, 1.5 to 1.4 on GPT-J and 0.5 to 0.3 on Vicuna. For ROME on GPT-J it is not slight: 7.4 to 19.9.
- Appendix H says that more edits always give a lower score. In Table 12, ROME is 23.7 with 1 edit and 26.5 with 2 edits. Also, the base model's score rises from 29.1 to 53.9 across those groups. So the groups differ in difficulty, and the trend is not only an effect of the edit count.
- Table 10 labels MeLLo as "on GPT-3", and Table 5 names GPT-3.5.

**The MQuAKE-T baseline uses wider labels.** Appendix F lets the unedited model's outdated answers count as gold. That raises the score before the edit, which makes the drop after the edit larger.

**Credit to the authors.** They state the two-model limit, the generated questions, and the hand-written prompts of MeLLo. They also say that they tested MeLLo only on models above 6B parameters. They published the subset fix and thanked the person who reported the problem.

## Use in this repo

**An answer experiment is the analogue of edit-wise success.** A fix that the repo tests only on the #4781 prompt measures what MQuAKE calls edit-wise success. MQuAKE's main result is that this number can be near 100 while the consequences stay wrong. Editors with edit-wise success above 88% scored 0.3 to 16.0 on multi-hop questions.

**The repo already has prompts that share the #4781 failure.** `docs/analyses/hallucination-shortlist.jsonl` keeps six prompts that pythia-70m answers with the first clause's unit. Two of them have the right answer " months":

| Prompt | Right answer (rank, p) | Model's answer (p) |
| --- | --- | --- |
| A week has seven days and a year has twelve | " months" (4, 0.0295) | " days" (0.7257) |
| A dozen has twelve eggs and a year has twelve | " months" (6, 0.0101) | " eggs" (0.6425) |
| A year has twelve months and a week has seven | " days" (2, 0.1503) | " months" (0.4549) |

The third prompt is the mirror of the #4781 prompt, and there " months" is the wrong answer. In MQuAKE's terms, these prompts test whether a change spreads, and where it must not go. This page reports that the data exists. How the fix campaign uses it is for the maintainer.

**MQuAKE checks every single fact first, and #4781 has no such check.** MQuAKE keeps a chain only when the model recalls each single fact. No prompt in the shortlist asks for the unit without a first clause, for example "A year has twelve" alone. #4781 found that no VERIFIED unit supplies " months" (#5015, question 1). That result has two readings. Either pythia-70m knows "twelve months" and the copy route wins, or it does not know the fact. A single-clause control separates the two. This point goes on #5015 as a comment.

**The kind of failure is different, so the editors may not apply.** MQuAKE edits stored facts, and ROME and MEMIT write to MLP weights. The #4781 answer comes from an in-context route: layer-0 count-word and time-unit neurons, then the copy heads L3/H3, L4/H0 and L4/H7. An edit to a stored "year → months" fact would compete with that route, not replace it. The paper gives no evidence for this case.

**The repo can measure side effects, which MQuAKE cannot.** MQuAKE has no measure of collateral damage. The repo has two: registry coverage (`docs/analyses/registry-coverage.md`), which measures the loss over corpus windows, and the sister search (`docs/PROTOCOL.md`, sister neurons). Sources in #4782 and RippleEdits (#4784, area 1) cover side effects of edits more directly.

**MeLLo does not apply.** It needs a model that follows decomposition prompts, and the authors tested it only above 6B parameters. pythia-70m cannot do this. MeLLo also leaves the weights and the neurons unchanged, so it would not fix the route that #4781 explains.

**The statistics are weaker than the repo's own.** Each MQuAKE number is one run with no interval. The repo's answer experiments use patching nulls and an unseen seed: the #4781 route got REJECTED on its first attempt and VERIFIED with seed 7. A fix result held to the repo's standard needs repeats or a null, which MQuAKE does not give.

**A question for the maintainer.** Should a fix count as successful on the #4781 prompt alone? Or should it also count on prompts that share the failure, and on prompts where the answer must not change? That choice is part of the fix campaign, so this page does not propose an issue for it.

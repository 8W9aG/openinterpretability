# Editing Factual Knowledge in Language Models

**Citation.** Nicola De Cao (University of Amsterdam, University of Edinburgh), Wilker Aziz (University of Amsterdam) and Ivan Titov (University of Amsterdam, University of Edinburgh). "Editing Factual Knowledge in Language Models". arXiv 2104.08164v2, 8 September 2021 (v1 16 April 2021). [Paper](https://arxiv.org/abs/2104.08164), [code](https://github.com/nicola-decao/KnowledgeEditor). Venue: EMNLP 2021, *Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing*, pages 6491–6506, DOI [10.18653/v1/2021.emnlp-main.522](https://aclanthology.org/2021.emnlp-main.522/). The arXiv comment says "Proceedings of the 2020 Conference", but the ACL Anthology page gives 2021. The anthology is right. Funding: SAP Innovation Center Network, ERC Starting Grant BroadSem (678254), NWO VIDI 639.022.518 and EU Horizon 2020 grant 825299 (Gourmet). The data are public sets (FEVER from KILT, and zsRE). The paper doesn't release its re-split or its paraphrases as files.

**Status.** Planned. Its four edit metrics are a ready test for a fix of the #4781 prompt, which the fix literature in #4784 prepares for. No code, skill or protocol rule depends on the paper now.

**Read.** On 2026-11-18, from the v2 arXiv HTML, which I converted to plain text and read myself. I have every section in the paper's own text: the abstract, sections 1 to 7, the ethics note, the acknowledgments and appendices A to C. Tables 1 and 2 came through as text. I didn't see the images of Figures 1 to 6, only their captions. I didn't read the PDF. A WebFetch model also read the abstract page and the ACL Anthology page. I took the venue and pages from that report.

## Summary

**Claim.** A hyper-network can learn to edit one fact in a trained model, from the fact alone. The edit changes the prediction for that input and its paraphrases. A constraint on the output distribution keeps the other predictions the same.

**Method.**

- **Task.** The data is a set of revisions ⟨x, y, a⟩: an input, the model's current prediction and an alternative prediction. The edited model must prefer *a* for *x*, and keep every other prediction.
- **Editor.** KnowledgeEditor predicts a weight shift ΔW for each weight matrix. ΔW is a gated sum of a scaled gradient of the edit loss and a bias. The scales come from outer products of vectors that a BiLSTM encoding of ⟨x, y, a⟩ produces. Biases, layer norms and embeddings are not edited.
- **Constraint.** The training objective minimises the edit loss on *x* and its paraphrases. The sum of KL divergences from the edited to the original output distribution, over other inputs, must stay below a margin. Lagrangian relaxation and Monte Carlo sampling make this tractable. For the sequence model, beam search approximates the output space. The authors multiply the margin by 0.8 each time the editor changes more than 90% of predictions on validation data.
- **Models and data.** Two fine-tuned models, each 12 layers with hidden size 768:
  - BERT base (110M) for closed-book fact-checking on binary FEVER from KILT: 104,966 training and 10,444 validation claims, accuracy 77.1%.
  - BART base (139M) for closed-book question answering on zsRE, re-split: 244,173 training and 27,644 validation items, exact match 22.1%.
- **Alternative predictions.** For fact-checking, the authors flip the label. For question answering, every beam-search hypothesis except the top one is an alternative.
- **Paraphrases.** Round-trip translation through German gives 25 candidates per input. The authors keep a candidate only if the original model predicts the same output for it. zsRE also has human paraphrases for 30% of its items.
- **Metrics.** *Success rate* is the accuracy of the revised predictions on the edited inputs. *Retain accuracy* is the agreement with the original predictions on other inputs. *Equivalence accuracy* is the accuracy of the revised predictions on paraphrases. *Performance deterioration* is 1 − (accuracy after the edit ÷ accuracy before it).
- **Baselines.** Fine-tuning with RMSProp, of the first layer or of all layers, for at most 100 steps. Zhu et al. (2020), which is fine-tuning with an L∞ constraint on the parameters. The paper's own editor with an L2 constraint on parameters in place of the KL constraint.

**Findings, as the source reports them.** All values are from Table 1 and the text of sections 6 and C.

| Setting | Result |
| --- | --- |
| KnowledgeEditor, fact-checking | success 98.80, retain 98.14, equivalence 82.69, deterioration 0.10 |
| KnowledgeEditor, question answering | success 94.65, retain 98.73, equivalence 86.50 (generated) / 92.06 (human), deterioration 0.11 |
| KnowledgeEditor with paraphrases in training, fact-checking | success 98.50, retain 98.55, equivalence 95.25 |
| The same, question answering | success 94.12, retain 98.56, equivalence 91.20 / 94.53 |
| Fine-tune all layers, fact-checking | success 100.0, retain 86.95, equivalence 95.58, deterioration 2.25 |
| Fine-tune all layers, question answering | success 100.0, retain 67.55, equivalence 97.77 / 98.84, deterioration 4.50 |
| Fine-tune first layer, fact-checking | success 100.0, retain 99.44, equivalence 42.24 |
| Zhu et al., all layers, question answering | success 80.65, retain 95.56, equivalence 76.41 / 79.38 |
| Editor with an L2 parameter constraint, fact-checking | success 99.10, retain 45.10, equivalence 99.01, deterioration 35.29 |
| Editor with an L2 parameter constraint, question answering | success 99.10, retain 46.66, deterioration 9.22 |
| Probability that the full method beats all baselines, over 1,000 Dirichlet-weighted sums of the metrics | ≈ 97% (fact-checking), ≈ 88% (question answering) |
| Editor without gradient input (section C) | ≈ 10 times slower on fact-checking. On question answering, it never passed 75% success and 50% retain |
| Mean cosine of the editor's update with the gradient (Table 2) | about 0.02 in magnitude. Fine-tuning against the gradient: 0.451 |

The paper also reports where the updates go, from the heatmaps of Figure 4 for question answering. Gradients concentrate on the first encoder layer and the last decoder layer. Fine-tuning updates all layers almost uniformly. The editor's updates are sparser, and they fall mainly on the attention key and query matrices, not the value matrices. Section 3 calls the hyper-network a "probe" of where knowledge is. Section 6.4 says that "proper investigation is however outside of the scope of this study".

## Where the claims go beyond the evidence

- **The edits are not corrections.** The abstract speaks of fixing "bugs". But on fact-checking each edit flips the label, so about as many edits make the model wrong as right. On question answering, the target is the model's own lower-ranked beam hypothesis. No experiment edits a known error to the gold answer.
- **One edit at a time.** Each edit starts from the original weights. Edits applied one after the other, or many at once, are not tested. The authors name a loop of edits at test time as future work.
- **Two small, fine-tuned models.** Both are base-size models fine-tuned on one task each. Neither is a decoder-only language model asked for its next token. The authors train an editor for each model and task, in about 1 day (fact-checking) and 3 days (question answering) on 4 GPUs.
- **Selection and reporting on one split.** Model selection for the editor uses success and retain accuracy on the validation set. Section 6.2 reports retain accuracy "in the validation set". The baselines' hyperparameters also come from a subset of the validation set. The paper names no separate test split for Table 1.
- **No seeds or intervals.** Each table cell is one run, with no spread. The Dirichlet comparison weights the metrics. It doesn't resample the data or the training.
- **"> 70%" is an error reduction.** Section 6.3 says that paraphrase training improves equivalence by "> 70%" on fact-checking and "> 30%" on question answering. Table 1 gives gains of 12.56 and 4.70 points. The quoted figures match the share of paraphrase errors removed, 73% and 35%.
- **Equivalence alone rewards a broad edit.** The L2 editor reaches 99.01 equivalence on fact-checking with 45.10 retain accuracy. So equivalence accuracy only means something next to retain accuracy. The paper's own Table 1 shows this.
- **The "probe" is not tested.** The heatmaps show where the editor puts weight change. No ablation or patching checks that those components hold the fact. The authors say this themselves.
- **Small inconsistencies.** Section 6 gives "≈ 97%" and "≈ 88%", and the caption of Figure 6 gives "> 97%" and "> 88%". The HTML shows −1.000 on the diagonal of Table 2, where a cosine must be +1. So the signs of that table are unreliable in this rendering.

The authors state the paraphrase limits, the scope of the update analysis and the possible misuse of the method themselves.

## Use in this repo

- **The four metrics fit a fix of the #4781 prompt.** On "A week has seven days and a year has twelve", pythia-70m puts 0.7257 on " days" and 0.0295 on " months", rank 4 (`docs/analyses/hallucination-shortlist.jsonl`). Success is the top token after a fix. Equivalence and retain need sets of prompts. The repo already has candidates for both. This is evidence for the fix campaign, not a design for it.
- **The answer record already holds an equivalence set.** The VERIFIED route record (`wiki/answers/EleutherAI--pythia-70m/970a91e11d8c/20261118T234629Z.json`) has 24 probes and 7 controls, with the model's top token for each. At least six probes ask the same question and still give " days". One is "In a calendar, a week has seven days and a year has twelve". Another is "There are seven days in a week and twelve". A fix that flips the prompt but not these would fail the paper's equivalence test.
- **It also holds the cases a retain test needs.** The same copy route gives right answers, for example " legs" after "A spider has eight legs and a dog has four". The shortlist has seven prompts that the model gets right, three of them by this copying. KnowledgeEditor's retain set is random other inputs. For #4781 the hard retain cases are the ones that use the same route correctly. A random sample would seldom contain them.
- **A third set the paper has no name for.** The five other kept prompts in the shortlist fail the same way, for example " seconds" after "an hour has sixty". KnowledgeEditor tests paraphrases of one fact, not other facts with the same failure. A fix of the copy route would change those too. A fix of the one fact would not. Which of the two a fix is, is the question #5015 point 3 approaches from the explanation side.
- **The edit target is not the model's second guess.** The paper's question-answering targets are the model's own beam alternatives. " months" is fourth here, behind " weeks" and " hours". So the paper's success rates are for an easier kind of target than this one.
- **The KL constraint matches a measure the repo has.** `openinterpretability/harness/lookahead.py` computes KL(clean ‖ ablated) on next-token distributions. That is the form of the paper's constraint, applied to an ablation and not to an edit. `open-interpretability coverage` (`docs/analyses/registry-coverage.md`) measures loss over the corpus, closer to the paper's performance deterioration. Neither tool runs on an edited model now.
- **The "probe" claim is the premise #5016 tests.** The editor puts its change in attention keys and queries, not where the gradient is largest. Where an edit acts need not be where an explanation locates the behaviour. #5016 (Hase et al. 2023) tests that question directly. In #4781 the verified route runs through copy heads L3/H3, L4/H0 and L4/H7. The paper's attention finding is suggestive for those heads, but it comes from BART on a different task.
- **Question for the maintainer.** Should a fix of the #4781 prompt be scored on paraphrases, same-route right answers and same-failure prompts as three separate sets? The paper scores only the first two, and it draws retain inputs at random. This is a campaign decision, so the page records it and doesn't open an issue.

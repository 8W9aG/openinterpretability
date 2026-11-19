# Memory-Based Model Editing at Scale

**Citation.** Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D. Manning and Chelsea Finn. Bosselut is at EPFL (School of Computer and Communication Sciences), and the others are at Stanford University (Department of Computer Science). "Memory-Based Model Editing at Scale". arXiv 2206.06520v1, 13 June 2022, the only version. [Paper](https://arxiv.org/abs/2206.06520). The arXiv comment gives the venue as ICML 2022. The [project site](https://sites.google.com/view/serac-editing) links the code at [github.com/eric-mitchell/serac](https://github.com/eric-mitchell/serac). The site has no direct data download. Licence CC BY 4.0. Funding: a Knight-Hennessy Graduate Fellowship (Mitchell) and Apple Inc. Finn and Manning are CIFAR Fellows. The paper doesn't say whether an LLM helped write it.

**Status.** Not applicable. No open issue uses its method, which leaves the model's weights and units unchanged. Its test design, with hard in-scope and hard out-of-scope inputs, still bears on the fix campaign: a fix must work without collateral damage.

**Read.** The abstract page, the v1 HTML and the project site, on 2026-11-18. A web-reading tool gave only short quotes and a summary, so we downloaded the HTML and read its text directly. We have the source's own text for every section, Appendices A to D included, and for Tables 1 to 8. The HTML drops most inline mathematics. We recovered the edit counts (k) from the page's math markup. We did not see Figures 1 to 3 except as captions, so the numbers behind Figure 3 are not here. We did not read the PDF or the code.

## Summary

**Claim.** Editors that change a model's weights from the edit's gradient misjudge the edit's scope, and they fail after many edits. SERAC keeps the edits in an explicit memory, and it "learns to reason over them to modulate the base model's predictions as needed". The authors report that only SERAC does well on all three of their new editing problems.

**Method.**

- **The editor.** SERAC wraps a base model that it treats as a black box. It has three parts:
  - an edit memory, which stores each edit descriptor as text.
  - a scope classifier, which gives the probability that an input is in the scope of each stored edit.
  - a counterfactual model, which predicts the output from the input and the most relevant edit.

  If no edit is in scope, the base model answers. The base model's weights never change, and SERAC needs only its tokenizer.
- **Parts.** The classifier is `distilbert-base-cased` in every setting. It is either an embedding model (Embed) or a cross-attention model (Cross). The counterfactual model is `t5-small` for QA, `bert-base-uncased` for FC and BlenderBot-90M for ConvSent. The authors train the two parts apart, by supervised learning on a dataset of edits.
- **Problems and base models.**
  - **QA**: zsRE questions. In-scope inputs are rephrases, and out-of-scope inputs are random. The base model is T5-large (770M) fine-tuned on Natural Questions.
  - **QA-hard**: the QA edits, plus entailed and true-or-false questions in scope. The out-of-scope inputs are near neighbours in an `all-MiniLM-L6-v2` embedding. The sampler skips the top 50 neighbours and draws from the next 50.
  - **FC**: VitaminC fact checking. The base model is a BERT-base fact checker trained on FEVER.
  - **ConvSent**: edits such as "Topic: jazz Sentiment: positive" for BlenderBot-90M, over 15,989 topics.
- **Metrics.** Edit success (ES) is exact-match agreement with the edited label on in-scope inputs. Drawdown (DD) is disagreement between the pre-edit and post-edit model on out-of-scope inputs. For ConvSent, ES is a product of a sentiment term and a topic term, and DD is a KL divergence. A perfect editor has ES 1 and DD 0.
- **Baselines.** Fine-tuning (FT), a cache and lookup on hidden states (LU), Editable Neural Networks (ENN) and MEND. The gradient methods change the fully connected layers of the last two or three blocks. An ablation, retrieve-and-prompt (RP, the paper's name), uses SERAC's classifier but lets the base model answer from the stored edit it finds.
- **Edits per test.** k = 10 edits at once for QA, QA-hard and FC, and k = 5 for ConvSent.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| QA, T5-large, k = 10 (Table 3) | SERAC ES 0.986, DD 0.009. LU ES 0.944, DD 0.051. MEND ES 0.823, DD 0.187. ENN ES 0.786, DD 0.354 |
| QA-hard, T5-large, k = 10 | SERAC ES 0.913, DD 0.028. LU ES 0.515, DD 0.132. MEND ES 0.478, DD 0.255. RP ES 0.278, DD 0.027 |
| FC, BERT-base, k = 10 | SERAC ES 0.877, DD 0.051. The other ES values are 0.565 to 0.627, with DD from 0.002 to 0.042 |
| ConvSent, BlenderBot-90M, k = 5 | SERAC ES 0.991, DD 0. MEND ES 0.494, DD 2.149. ENN ES 0.502, DD 3.546. RP ES 0.506, DD 0 |
| SERAC parts on QA-hard (Table 4) | Classifier accuracy 0.985 in-scope easy, 0.855 in-scope hard, 0.996 out-of-scope easy, 0.967 out-of-scope hard. Counterfactual accuracy 0.996 and 0.987 in scope |
| SERAC parts on FC (Table 4) | Classifier accuracy 0.909 in scope (pooled), 0.993 out-of-scope easy, 0.706 out-of-scope hard |
| Classifier variants, k = 10 (Table 5) | QA-hard: Embed-D 0.921 / 0.029, Cross-D 0.983 / 0.009. FC: Embed-D 0.792 / 0.247, Cross-B 0.855 / 0.0964 (ES / DD) |
| Many edits, QA (Figure 3 caption and §5.1) | SERAC applies up to 75 edits "with little degradation". For ENN and MEND at 75 edits, drawdown exceeds edit success |
| Other T5 base models, same editor (§5.2) | QA: ES above 0.99, DD below 0.01. QA-hard: ES above 0.92, DD below 0.03 |
| Cost after 10 edits (Table 6) | QA: 87 ms and 2.96 GB for the base model, 92 ms and 3.47 GB out of scope, 31 ms and 3.46 GB in scope. About 3 KB of memory per edit |

## Where the claims go beyond the evidence

- **One base model per problem.** Each problem has one base model and one edit distribution. The test edits come from the same templates as the training edits. The "re-use across models" result covers only T5 sizes on QA, so "edit multiple models with different architectures" in §7 goes past the tests.
- **No seeds or intervals.** No table gives seeds, spread or intervals. Some gaps, such as SERAC and LU on QA (0.986 against 0.944), have no scale.
- **"Random chance" has no stated value.** §5.1 says that on FC "all methods except SERAC achieve nearly random-chance performance". The paper gives no chance level and no pre-edit ES. The baselines' FC drawdown is 0.002 to 0.042, so they may barely change the model at all.
- **Zero drawdown can come from routing.** If the classifier sends an out-of-scope input to the unchanged base model, its drawdown is zero by construction. So SERAC's DD measures the classifier's false positives, not any harm to the model. On FC, SERAC has the highest drawdown (0.051). The authors say so, and argue that the gain in ES is larger.
- **Drawdown covers only the sampled inputs.** The paper measures DD on out-of-scope inputs from the same dataset. It reports no general measure, such as perplexity on other text.
- **Tables disagree.** Table 3's QA-hard result (0.913 / 0.028) differs from Table 5's Embed-D row (0.921 / 0.029), which appears to be the same configuration. Table 3's FC result (0.877 / 0.051) matches neither Cross-D (0.831 / 0.074) nor Cross-B (0.855 / 0.0964). Embed-D and Embed-B have the same FC numbers to three places. §5.2 gives QA ES "above 0.99" for every T5 size, but Table 3 gives 0.986 for T5-large.
- **The fix lives outside the model.** The paper presents this as a strength. It also means that the base model still makes the same error, and any copy of the weights without the wrapper still makes it.

The authors state some limits themselves. SERAC needs a dataset of edits to train its parts, and the edit memory can grow without bound. They also name the dual-use risk of the sentiment edits.

## Use in this repo

- **Hard out-of-scope inputs are our hard negatives.** SERAC's hard out-of-scope inputs look like in-scope inputs but must not change. Our reviewer writes hard negatives for the same reason (`.claude/skills/interpretability-illusions-review/SKILL.md`). Table 4 shows why they matter. On FC, the classifier is right on 0.993 of easy out-of-scope inputs but only 0.706 of hard ones. A side-effect test on random inputs alone would miss most of the damage.
- **The shortlist already holds both kinds of input for #4781.** `docs/analyses/hallucination-shortlist.jsonl` has 35 prompts. The model answers 28 wrong, and the script keeps 6. Each of the 6 copies the first clause's unit, for example " days" in place of " months". Three prompts are right *because* the model copies the unit. They are "A spider has eight legs and a dog has four" and "A car has four wheels and a bike has two". The third is "A cat has four legs and a bird has two". In each, the right answer is the first clause's noun. In SERAC's terms, these three are hard out-of-scope inputs for any fix to the copy route through L3/H3, L4/H0 and L4/H7. This is evidence for the campaign, not a design for it.
- **The scope problem is the hard part.** On QA-hard, the counterfactual model is right on 0.987 of hard in-scope inputs. The classifier is right on only 0.855 of them. The authors conclude that "the primary challenge of the problem is scope estimation". For #4781 the analogue is this. When is it wrong to copy the first unit ("a year has twelve"), and when is it right ("a dog has four")? The verified route doesn't yet say what the model could use to tell them apart. #5015 asks why no unit supplies " months", which bears on this.
- **Drawdown as disagreement with the pre-edit model.** DD compares the edited model with the original on inputs that must not change. Our coverage measure (`openinterpretability/coverage.py`) and the sister search (`openinterpretability/cli_sisters.py`) measure other things. A re-run of either after a fix could show changes in explained units that a DD on the output would not show. Whether to use them that way is for the campaign's design.
- **A wrapper is a fix that ignores the explanation.** SERAC needs no weights, activations or gradients. It would fix " days" and leave the VERIFIED route as it is. Whether the campaign should compare an explanation-guided fix with a black-box fix of this kind is a question for the maintainer.
- **No single-neuron content.** The paper names knowledge-neuron editing (Dai et al.) and ROME (Meng et al.) only as related work. Nothing in it bears on a pole's explanation or on our verdicts.

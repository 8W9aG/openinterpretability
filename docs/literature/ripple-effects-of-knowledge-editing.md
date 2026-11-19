# Evaluating the Ripple Effects of Knowledge Editing in Language Models

**Citation.** Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson and Mor Geva (Blavatnik School of Computer Science, Tel Aviv University, and Google Research). "Evaluating the Ripple Effects of Knowledge Editing in Language Models". arXiv 2307.12976v2, 20 December 2023 (v1: 24 July 2023). [Paper](https://arxiv.org/abs/2307.12976). [Code and data](https://github.com/edenbiran/RippleEdits). The arXiv comment says "Accepted for publication in Transactions of the Association for Computational Linguistics (TACL), 2024. Author's final version". We did not check the TACL volume or pages. Funding: "supported in part by the Israeli Science Foundation". The paper gives no tool disclosure.

**Status.** Planned. Umbrella issue #4784 asks how to show that a fix worked without collateral damage, and this paper gives one answer for factual edits. The follow-up below would measure the ripple neighbours of the #4781 prompt before any fix.

**Read.** On 2026-11-18, the reviewing model downloaded the v2 HTML from arXiv and read its text in full. That includes the abstract, Sections 1–6, the Limitations and the reference list. The table cells were in math markup, so the model read them from the markup's alternative text. All of Tables 1–7 came back in that way. A web-fetch tool read the abstract page but gave only a summary. So the page takes the abstract from the HTML text. We did not see the figures (only their captions), so the values in Figures 4, 6 and 7 are not here. We did not read the PDF, the TACL version or the code.

## Summary

**Claim.** Knowledge-editing evaluation checks that the edited fact changed and that unrelated facts did not. That is not enough: an edit implies other facts, its "ripple effect", that must change too. Three editing methods fail most of these tests. A simple in-context baseline does best.

**Method.**

- **Facts.** Facts are triplets (subject, relation, object) from WikiData. An edit either modifies a fact the model holds or injects a new one.
- **Six criteria**, all within two hops of the edit:
  - *Logical Generalization (LG)*: symmetric or transitive facts, for example a sibling's sibling.
  - *Compositionality I (CI)*: the edited fact composed with a fact about the new object.
  - *Compositionality II (CII)*: the edited fact reached through another subject, for example "the siblings of the founder of Paisley Park Records".
  - *Subject Aliasing (SA)*: the edit holds for the subject's aliases.
  - *Preservation (PV)*: other objects of a one-to-many relation stay.
  - *Relation Specificity (RS)*: the subject's unrelated relations do not change.
- **Benchmark (RippleEdits).** It has three subsets:
  - *Recent*: facts added to WikiData in the 250 days after July 2022, used as injections.
  - *Random*: random facts, each given a wrong object of the same type.
  - *Popular*: facts whose subject is in the top-1000 Wikipedia pages.
- **Templates.** The authors wrote the templates and logical rules by hand, for a fixed set of relations.
- **Filter.** An edit counts only if it succeeds. A test query counts only if the model answered it correctly before the edit.
- **Methods.** MEND, ROME and MEMIT, with the official code and hyperparameters. The baseline is in-context editing (ICE): the prompt "Imagine that", then the edited fact, before the query, with no weight change.
- **Models.** GPT-2 XL (1.5B), GPT-J (6B), LLaMA (7B) and GPT-NeoX (20B). ICE also runs on GPT-3 text-davinci-003 (175B).
- **Metric.** The model generates up to 20 tokens. A test passes if an alias of the target object appears in the text. The accuracy is the mean over each edit's queries, then the mean over edits.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Benchmark size (Table 1, §4.2) | 2,000 Recent, 1,000 Random and 1,000 Popular edits. The abstract says "5K" |
| Queries per edit (Table 1) | 26.8 Recent, 18.8 Random, 25.6 Popular |
| Data quality, 200 sampled queries (§4.3) | 100% sound, 98.5% grammatical |
| Share of test queries kept after the filter (Table 2) | GPT-2 29% / 33% / 71% (Recent / Random / Popular). GPT-3 55% / 74% / 94% |
| Parametric methods, average over subsets (§5.2) | "low average accuracy of 38-66 across all models" |
| Recent, average accuracy (Table 3) | ROME 57.5 on GPT-2 and 60.8 on LLaMA. ICE 82.8 on LLaMA and on GPT-3 |
| Random, average accuracy (Table 4) | ROME 45.5 on GPT-2, 48.7 on LLaMA. ICE 82.5 on LLaMA, 90.3 on GPT-3 |
| Popular, average accuracy (Table 5) | ROME 48.7 on GPT-2, 49.4 on LLaMA. ICE 81.3 on LLaMA, 77.7 on GPT-3 |
| ICE against ROME, mean over subsets (§5.2) | "more than 10 points for GPT-Neo and 29 points for LLaMA" |
| Subject Aliasing (§5.2) | At least 86.8 in every setting |
| LG of ROME on GPT-J (§5.2) | 53.8 on Random, 5.5 on Popular |
| Preservation on Random (Table 4) | 9.9 to 30.8 for the parametric methods, 100 for ICE |
| GPT-2, methods averaged over subsets (Table 6) | LG: MEND 39.1, ROME 26.5, MEMIT 29.0. CII: MEND 13.6, ROME 37.7, MEMIT 39.1 |
| Error types, 200 failures per model and method (Table 7) | "No effect" 10% to 32%. Abstaining plus noise is at least 68% in every setting |
| ICE error types on LLaMA (Table 7) | 11% no effect, 71% abstaining, 18% noise |
| Abstention on 100 LG queries with ICE (§5.3) | GPT-3 49%, LLaMA 28% |

## Where the claims go beyond the evidence

**"5K edits" is 4,000 in the paper's own tables.** The abstract and Section 1 say 5K. Table 1 and Section 4.2 give 2,000 + 1,000 + 1,000.

**The model names disagree.** Section 5.1 names GPT-NeoX with 20B parameters. The tables and the text call it GPT-Neo.

**The filter changes the test set.** Only queries that the model answered correctly before the edit count. That keeps 29% of GPT-2's Recent queries and 94% of GPT-3's Popular queries (Table 2). So each model has a different test set, and the model-size comparison mixes skill with selection.

**The size claim mixes model families.** Figure 6 plots ROME across GPT-2 sizes, GPT-J, LLaMA and GPT-NeoX. Family and training data change with size, so the plot does not isolate size. We did not see the figure.

**The popularity claim has more than one cause.** Random and Popular differ in subject popularity, but also in relations and object popularity (Table 1, Figure 4). The text attributes the LG gap to edit severity alone.

**Some cells rest on very few queries.** Preservation has 0.4 to 0.6 queries per edit (Table 1). So the Preservation cells on Random (9.9 to 100) come from few queries.

**The metric is loose.** A pass is any alias of the target in 20 generated tokens. A text that names several candidates can pass. The paper does not measure probabilities.

**No intervals and no seeds.** The tables give single numbers. Figure 7 has standard-deviation bars over methods, not over samples. The paper tests no difference for significance.

**ICE is not an edit of the same kind.** ICE puts the fact in the prompt of every query. The parametric methods change weights once. ICE's tests also include the edited fact in context, so the two are not compared on the same input.

**The authors state the coverage limit themselves.** The Limitations say that distant facts are not checked, for example that editing France's capital leaves Poland's population alone. They also say that WikiData can be incomplete.

## Use in this repo

**The answer experiment already has ripple-style tests, but for the route, not for a fix.** The VERIFIED record for #4781 (`wiki/answers/EleutherAI--pythia-70m/970a91e11d8c/20261118T234629Z.json`) has 7 controls and 24 reviewer probes. Each probe says if the model's top token stays (`kept`) or changes. 20 of 24 passed. These probes test the claim about how the model answers. They do not yet test what a change to the model would also change.

**The record shows that the failure has neighbours, in the paper's sense.** In the record's `top_tokens`, the paraphrase "A week has 7 days and a year has 12" gives " days". That is the paper's Subject Aliasing case. The reversed order "A year has twelve months and a week has seven" gives " months", which is wrong too. "There are seven days in a week and twelve" gives " days". The shortlist (`docs/analyses/hallucination-shortlist.jsonl`) keeps six prompts with the same copy-the-first-unit failure, and the reversed order is one of them. So a fix tested on the one prompt would leave these questions open. Does it reach the paraphrase? Does it reach the reversed order, which needs " days", not " months"? Does it reach the other four prompts?

**The paper's filter applies, and it cuts more for a small model.** The paper keeps only queries that the model got right before the edit. GPT-2 XL kept 29% to 71% (Table 2). pythia-70m is about 20 times smaller, so a larger share of neighbour prompts will be wrong before any fix. A test set of neighbours must record the model's answer on each prompt before the fix. `scripts/hallucination_shortlist.py` already records the top tokens and the right answer's rank, so it can take that role.

**The paper's error types are a useful report format.** Table 7 sorts each failure as no effect, abstaining or noise. After a change to pythia-70m, a wrong answer could still be " days" (no effect), a function word (abstaining) or " weeks" (noise). The answer records store the top token per prompt, so this sorting is possible from the records.

**The repo's measure is stricter than the paper's.** The paper passes a test on any alias in 20 generated tokens. `answers review` takes the answer's log-probability at one position (`docs/PROTOCOL.md` §4, *Answer experiments*). The paper's numbers are therefore not directly comparable with the repo's gates.

**The paper says nothing about single units.** It cites MLP layers as key-value memories, and names a mechanistic study of edits as future work. It reports no neuron, explanation or random baseline. So it does not tell us whether a unit-level edit spreads further than a weight edit like ROME.

**Questions for the maintainer.** These bear on the fix campaign, so they are questions, not issues.

- Should a fix's test set name its neighbour prompts before the fix, as RippleEdits fixes its queries before editing?
- Should a prompt-only fix, like ICE, be the baseline that a weight or unit fix must beat? In the paper, ICE beat ROME on LLaMA by 29 points, averaged over the subsets.
- The paper leaves distant side effects unchecked. Can the registry's unit records serve as that check? For example, re-measure VERIFIED poles after a fix and count the verdicts that change.

**Proposed follow-up.** Measure pythia-70m's answers on ripple-style neighbours of "A week has seven days and a year has twelve", before any fix. The neighbours come from the answer record's probes and the shortlist. Group them by the paper's criteria. This is evidence for the campaign, not a design for it.

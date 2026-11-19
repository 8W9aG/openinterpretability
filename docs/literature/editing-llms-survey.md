# Editing Large Language Models: Problems, Methods, and Opportunities

**Citation.** Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen and Ningyu Zhang. "Editing Large Language Models: Problems, Methods, and Opportunities". arXiv 2305.13172v3, 30 November 2023. Version 1 is from 22 May 2023, and version 2 from 11 October 2023. [Paper](https://arxiv.org/abs/2305.13172). [Code and data (EasyEdit)](https://github.com/zjunlp/EasyEdit). The affiliations are Zhejiang University, its joint knowledge-graph laboratory with Ant Group, and Donghai Laboratory. Shumin Deng is at the National University of Singapore (NUS-NCS Joint Lab). Peng Wang has equal contribution, and Ningyu Zhang is the corresponding author. Venue: EMNLP 2023, from the arXiv comments field ("EMNLP 2023. Updated with new experiments"). We did not check the proceedings page. Funding: the National Natural Science Foundation of China (No. 62206246), and Zhejiang, Ningbo and Yongjiang grants. Other funders are the CCF-Tencent Rhino-Bird fund, a Zhejiang University lab (CAD&CG) and the NUS-NCS Joint Laboratory. The paper gives no disclosure of tools that helped write it. It used GPT-4 to write some of its test questions.

**Status.** Planned. Issue #3374 (factual recall and editing) would use the paper's tests of an edit: reliability, generalisation, locality and portability. Any fix of the #4781 hallucination would use them too. The maintainer designs that fix campaign, and it has no issue yet.

**Read.** On 2026-11-18. The abstract page came through WebFetch. WebFetch returned the HTML only as a summary, so we downloaded the v3 HTML and extracted its text with a short script. We read that text ourselves: the abstract, Sections 1 to 7, Limitations, the ethics note, and Appendices A and B. Tables 1 to 11 came through as text. We did not see Figures 1 to 6, because they are images. So the batch, sequential and memory results (Figures 3, 4 and 5) are here only as the text describes them. We did not read the references or the PDF.

## Summary

**Claim.** Methods that edit a model's facts lack a comparison under the same conditions. The paper defines the editing task and groups the methods. Then it compares them on the same models and data. On reliability, generalisation and locality, several methods do well. On portability, side effects and cost, current methods are "somewhat limited".

**Method.**

- **Task.** An edit descriptor (x_e, y_e) has f(x_e) ≠ y_e before the edit. The edited model must give y_e on the edit's scope and keep its old output outside that scope (Eq. 1).
- **Three base metrics (Section 2).** *Reliability* is top-1 accuracy on the edit case. *Generalisation* is top-1 accuracy on the equivalence neighbourhood, such as paraphrases. *Locality* is the rate at which out-of-scope predictions stay the same as before the edit.
- **Methods (Section 3).** Two groups:
  - *Preserve parameters.* Memory-based methods are SERAC and the in-context methods MemPrompt, IKE and MeLLo. Methods with extra parameters are T-Patcher (one new FFN neuron per mistake), CaliNET (several neurons) and GRACE (a codebook).
  - *Change parameters.* Locate-then-edit methods are KN (knowledge neurons), ROME, MEMIT and PMET. Meta-learning methods are KE and MEND. FT-L, a fine-tune of the layer that ROME picks, is the baseline.
- **Data.** ZsRE (question answering, back-translated paraphrases, Natural Questions for locality) and CounterFact (counterfactual facts, nearby subjects for locality). The paper uses the earlier works' test splits.
- **Models.** T5-XL (3B) and GPT-J (6B). It tests IKE, ROME and MEMIT on OPT-13B and GPT-NeoX-20B as well.
- **New tests (Section 5).**
  - *Portability* on ZsRE: Subject Replace (an alias of the subject), Reversed Relation (the inverse question) and One-hop (a question that needs the new fact). GPT-4 wrote the questions from few-shot prompts, and human evaluators checked them. The one-hop set keeps only triples whose tail entity is in the model's top 10 for link prediction.
  - *Locality* on GPT-J, three tests. Other Attribution asks another fact about the same subject. Distract Neighbour puts the edit sentence before a neighbour prompt. Other Task measures PIQA accuracy after 100 sequential CounterFact edits. Each of the first two sets has 804 CounterFact items.
  - *Efficiency*: wall-clock time for 10 edits on GPT-J, and GPU memory.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| GPT-J, CounterFact, reliability / generalisation / locality (Table 1) | ROME 99.80 / 86.63 / 93.61. MEMIT 99.90 / 73.13 / 97.17. SERAC 99.78 / 99.41 / 98.89 |
| GPT-J, CounterFact, FT-L (Table 1) | Reliability 99.90 and generalisation 97.53, but locality 1.02 |
| GPT-J, CounterFact, KN (Table 1) | Reliability 1.66, generalisation 1.38, locality 58.28 |
| GPT-J, CounterFact, T-Patcher (Table 1) | Reliability 100.00, generalisation 83.98, locality 8.37 |
| GPT-J, ZsRE, KN (Table 1) | Reliability 11.34, generalisation 9.40 |
| T5-XL, ROME and MEMIT (Table 1) | No result. The authors say that the methods fail to edit this model |
| OPT-13B, ZsRE, ROME (Table 2) | Reliability 22.23, generalisation 6.08. The text says that a matrix in the update is not invertible on OPT-13B |
| GPT-NeoX-20B, ZsRE, ROME (Table 2) | Reliability 99.34, generalisation 95.49, locality 99.79 |
| Portability, GPT-J, Subject Replace / Reversed Relation / One-hop (Table 3) | SERAC 17.79 / 1.30 / 5.53. ROME 37.42 / 46.42 / 50.91. IKE 88.77 / 92.96 / 55.38 |
| Locality, GPT-J, Distract Neighbour (Table 4) | T-Patcher 17.56, FT-L 9.48, ROME 50.35, MEMIT 60.47, IKE 66.04 |
| Locality, GPT-J, PIQA after 100 sequential edits (Table 4) | FT-L 49.56, MEND 48.86, ROME 52.12, MEMIT 74.62, IKE 75.33 |
| Time for 10 edits on GPT-J (Table 5) | MEND 0.51 s, ROME 147.2 s, T-Patcher 1864.74 s (CounterFact) |
| Training cost (Section 5.3) | MEND on ZsRE: over 7 hours. SERAC: over 36 hours on 3 × V100 |
| Sequential editing (text on Figure 4) | ROME does well up to 10 edits, then degrades at 100. MEND degrades by 10. SERAC and T-Patcher stay stable |

## Where the claims go beyond the evidence

- **A benchmark study more than a survey.** The abstract promises an "exhaustive overview". Section 3 describes about a dozen methods in two pages, and most of the paper is the authors' own runs.
- **Facts only.** Every test edits a subject–relation–object triple. The authors say this themselves: personality, opinions, multilingual edits and erasure are out of scope.
- **No seeds, intervals or sample sizes in the main tables.** Each cell is one number to two decimals. The text we read does not give the size of the Table 1 test sets.
- **Size and family go together.** "As the model size increases", IKE loses generalisation and locality. The evidence is OPT-13B against GPT-NeoX-20B, two different families. The ROME failure on OPT-13B is also a property of one model's statistics, not of scale.
- **Locality counts top-1 agreement.** Eq. 4 asks only whether the prediction stays the same. A shift in probability that leaves the top token in place counts as no side effect.
- **No pre-edit row for PIQA.** Table 4 gives PIQA accuracy after the edits, but the text we read gives no unedited GPT-J score. So the reader can't tell how much each method lost.
- **Small inconsistencies.** The text says that MEND gets "over 80%" on T5, but its ZsRE reliability there is 78.80. One pointer reads "Appendix 6" for Table 6. The portability filter uses both T5-XL and GPT-J, but Table 3 has no T5-XL rows.
- **Generated test items.** GPT-4 wrote the portability and Other Attribution questions. Human evaluators checked them, but the text does not say how many items they checked or rejected.

The authors state some limits themselves: models up to 20B only, no test of Llama-style architectures, and a comprehensive study of single edits only.

## Use in this repo

**It names the tests a fix must pass.** Reliability, generalisation, locality and portability are four separate questions. A fix of #4781 must give " months" on "A week has seven days and a year has twelve", which is reliability. Generalisation and locality have ready prompt sets in `docs/analyses/hallucination-shortlist.jsonl`. It has 35 prompts. Of these, the filter keeps 6 hallucinations, 7 are right answers, and 22 are wrong in ways that the filter drops. "A dozen has twelve eggs and a year has twelve" also wants " months", so it tests generalisation. The 7 right answers test locality.

**The #4781 error is not a stored wrong fact.** Locate-then-edit methods (KN, ROME, MEMIT) assume that an FFN holds the fact as a key and a value. The #4781 answer route stands VERIFIED, with necessity 0.68, sufficiency 0.62 and patching 0.75 (`changelog.d/4781.added.md`). In it, the copy heads L3/H3, L4/H0 and L4/H7 carry " days" from the first clause. No verified unit supplies " months". So there is no located "year → months" key for a rank-one MLP edit to rewrite. The paper built and tested its methods on triples, and this failure is a copy.

**Distract Neighbour is the same mechanism as the hallucination.** The paper's test puts an edit sentence before an unrelated prompt, and models tend to repeat the edited fact. The #4781 route is that habit in pythia-70m: it copies a unit from an earlier clause. So a fix that puts text before the prompt, as IKE does, could make the model copy that text into other prompts. This is a risk to measure, not a result. IKE did best on Distract Neighbour in GPT-J (66.04), and pythia-70m is far smaller.

**Single-neuron edits did poorly at 6B.** KN doubles the activations of attributed neurons. On GPT-J it reached 1.66 reliability on CounterFact. T-Patcher adds one FFN neuron per mistake and reached 100.00, but its locality was 8.37. Both are edits at the registry's unit, the neuron. The result warns against the premise that clamping an explained neuron fixes an answer. The `neuron-steering-tests` skill clamps poles at 1, 2 and 4 × `max_activation` to test explanations, not to make fixes. [Does Localization Inform Editing?](does-localization-inform-editing.md) tests that premise directly.

**The repo already measures locality more finely than Eq. 4.** `open-interpretability coverage` (`openinterpretability/coverage.py`, `docs/analyses/registry-coverage.md`) measures next-token loss on 64 held-out windows of 128 tokens. `pole-effects` (#3486) estimates each pole's effect on the loss. A loss on held-out text sees the probability shifts that top-1 agreement misses. These tools exist, but nothing in the repo yet runs them before and after an edit.

**Sequential edits are the realistic case.** The paper's default rolls the model back after each edit. ROME degrades by 100 sequential edits, and MEND by 10. A campaign that makes many fixes on one model faces this result.

**Questions for the maintainer.** These are inputs to the fix campaign, which the maintainer designs, so we open no issue for them:

- Which of the four properties must a fix pass, and on which prompt sets?
- Is locality top-1 agreement, as in Eq. 4, or a loss change on held-out text, as in `coverage`?
- Must a fix also hold after other fixes, as in sequential editing?

# The Mirage of Model Editing

**Citation.** Wanli Yang, Fei Sun, Jiajun Tan, Xinyu Ma, Qi Cao, Dawei Yin, Huawei Shen and Xueqi Cheng. "The Mirage of Model Editing: Revisiting Evaluation in the Wild". arXiv 2502.11177, v1 16 February 2025, v5 31 May 2025. [Paper](https://arxiv.org/abs/2502.11177). The arXiv comment says "Accepted to ACL 2025 Main Conference (Camera Ready Version)". The issue gives the Anthology ID 2025.acl-long.745, which we did not check on the Anthology page. The HTML gives Baidu Inc. as the affiliation of Yang, Sun, Tan, Ma and Yin. It gives the Institute of Computing Technology, CAS, for Cao, and the University of Chinese Academy of Sciences for Cheng. The paper links no code or data release of its own. It uses GPT-4o-mini as a judge and GPT-4 to build its dataset.

**Status.** Not applicable now, and a direct lesson for the fix campaign later. No code, skill or protocol rule depends on it, and no open issue would use it. The campaign belongs to the maintainer, so this page reports evidence and records questions only.

**Read.** We read the abstract page and the v5 HTML on 2026-11-18, through WebFetch. A model read the text for us in four passes. The tool let it quote only short excerpts, so most sections came back as a summary. The numbers below came from that model's transcription of Tables 1, 3 to 6, 8 and 9 and of the main text. We did not read the PDF, the figures or appendix tables 11, 12 and 14. The single-edit locality values are in Table 11, so we do not have them.

## Summary

**Claim.** Published model-editing results come from a synthetic test that overstates success. A test like real question answering gives much lower scores, and sequential editing fails at about 1,000 edits.

**Method.**

- **Models.** Llama-2-7b-chat, Mistral-7b and Llama-3-8b, all with greedy decoding.
- **Editors.** FT-M (fine-tuning), MEND (meta-learning), ROME (the C-ROME variant) and MEMIT (locate-then-edit), and GRACE and WISE (lifelong editing with added memory). MEND is not run on Llama-3-8b.
- **Data.** ZsRE and CounterFact, 10,000 records each. QAEdit is new: 19,249 questions from Natural Questions, TriviaQA and SimpleQA. GPT-4 extracts the subjects and writes the paraphrases. Single edits use only the questions that the unedited model gets wrong: 12,715 for Llama-2, 10,213 for Mistral and 10,467 for Llama-3.
- **The test.** WILD ("Without Intervention, Live Decoding") changes four modules of the usual test:

  | Module | Synthetic test | WILD |
  | --- | --- | --- |
  | Input | the question alone | the question in a QA context |
  | Generation | teacher forcing: the test feeds the target token back at each step | the model's own tokens |
  | Truncation | at the target's length | at a natural stop |
  | Metric | match ratio of tokens | LLM judge, or exact match |

- **Measures.** Reliability (the edited question), generalisation (a paraphrase) and locality (unrelated questions).
- **Controls.** Each of Tables 4 to 8 changes one module from a base setting, on 3,000 QAEdit questions and Llama-3-8b.
- **Statistics.** The paper gives no seeds, intervals or significance tests. The LLM judge is not checked against human labels in the paper. The paper cites Li et al. (2024) for its agreement with humans.

**Findings, as the source reports them.** These numbers came through the reader model, as the *Read* section says.

| Setting | Result |
| --- | --- |
| Llama-2-7b-chat, single edit, QAEdit, WILD, mean of six editors (Table 1) | 0.385, against "96.8%" in the abstract for earlier reports |
| Single edit, reliability, mean over models and editors, synthetic / WILD (Table 3) | ZsRE 0.956 / 0.438, CounterFact 0.965 / 0.557, QAEdit 0.956 / 0.389 |
| Single edit, generalisation, same means (Table 3) | ZsRE 0.792 / 0.400, CounterFact 0.405 / 0.283, QAEdit 0.779 / 0.351 |
| GRACE, single edit, synthetic / WILD reliability (Table 3) | 0.983 / 0.012 to 0.999 / 0.036 across models and datasets |
| Input module only, GRACE (Table 4) | 0.998 → 0.412 |
| Generation module only, context-guided input (Table 5) | FT-M 0.937 → 0.800, GRACE 0.412 → 0.036, WISE 0.838 → 0.592 |
| Truncation module only, context-free input, LLM judge (Table 6) | FT-M 1.000 → 0.202, ROME 0.954 → 0.478, WISE 0.700 → 0.046 |
| Metric module only (Table 8) | the smallest change. Match ratio is above exact match in most cells |
| Sequential editing, 1,000 edits, QAEdit, WILD reliability (§7.1, Table 9) | 0.093 mean for the editors other than FT-M. FT-M has a 0.405 mean over three models |
| Same, WILD locality (§7.1) | 0.213 mean |
| ROME and MEMIT after 1,000 sequential edits, WILD (Table 9) | 0.000 to 0.002 reliability |
| Mini-batch sequential editing, QAEdit, 1,000 edits (§7.2) | no editor above 30% |

The paper says that teacher forcing "leaks both content and length of the ground truth". It also says that truncation at the target's length hides repeated, off-topic or wrong text after the answer.

## Where the claims go beyond the evidence

- **The headline is one model, not an average.** The abstract's "38.5% vs. 96.8%" is the Table 1 mean for Llama-2-7b-chat only. The 96.8% is from earlier reports, and it does not match a mean in Table 3. Table 3's synthetic means are 0.956 to 0.965.
- **The judge is not checked.** The WILD score depends on a GPT-4o-mini judge. The paper reports no agreement with human labels on its own data. Exact match, the other metric, gives similar or lower numbers (Table 8), so the direction of the result holds.
- **No seeds, intervals or tests.** The sets are large, so most gaps are far above sampling noise. But single-module cells such as ROME 0.930 → 0.851 carry no interval.
- **The modules are not crossed.** Each table changes one module from one base setting. The order of the modules' effects can change in a different base setting. For example, generation costs GRACE 0.002 with context-free input and 0.376 with a QA context.
- **The data is GPT-4-made.** Subjects and paraphrases come from GPT-4, and the paper does not report a human check of them.
- **Scope.** The study covers six parameter editors and three 7B to 8B models. It excludes in-context and retrieval editing. The authors state this limit themselves and call the work a diagnosis, not a fix.
- **Small inconsistencies.** Tables 6 and 8 give different judge scores for the same cell: ROME 0.783 and 0.789, MEMIT 0.704 and 0.707.

The authors credit their scope limit and the exclusion of in-context editing themselves.

## Use in this repo

- **Our answer measure is one token at one position.** An answer experiment needs a one-token answer (`answer_token` in `openinterpretability/answer_review.py`). Prompt coverage is the answer's log-probability at its position (`openinterpretability/explain_coverage.py`). The #4781 shortlist keeps a prompt when the greedy top token is wrong (`scripts/hallucination_shortlist.py`). This has no length to leak, and no target tokens to feed back. So the paper's two largest sources of overstatement do not apply to these measures.
- **But a fix is a generation claim.** The paper's lesson is that the first token can be right while the text after it is wrong. For #4781 the model gives " days" with p 0.7257 and " months" with p 0.0295 (`docs/analyses/hallucination-shortlist.jsonl`). A change that moves " months" to the top answers the shortlist test. It does not show that the model now writes "twelve months" and stops in a sensible place.
- **The repo already has free decoding for neurons.** Every audit records steering samples: five neutral prompts, greedy, 24 new tokens (`PROMPTS` and `MAX_NEW_TOKENS` in `openinterpretability/harness/steering.py`). That is WILD's generation and truncation applied to a single pole. Answer records have no such greedy continuation now.
- **Sequential collapse bears on many small fixes.** ROME and MEMIT fall to 0.000 to 0.002 WILD reliability after 1,000 edits (Table 9). The shortlist has 6 kept prompts out of 35 rows. That is far below 1,000 edits, but the paper gives no curve for small counts that we read.
- **Locality needs a WILD form too.** The paper's sequential locality falls to 0.213. Our coverage and sister tools measure whole-layer loss and matched poles. Neither re-runs unrelated prompts by free decoding after a change. This is evidence for the campaign, not a design for it.
- **Questions for the maintainer.** These are not issues, because the campaign is the maintainer's to design.
  1. Should a fix count as working only when the greedy continuation is right, as in WILD, and not only when the answer token's rank changes?
  2. Should the #4781 prompt also be tested inside a context, as WILD's input module does? GRACE lost most of its score from that change alone (Table 4).
  3. Should collateral damage be read from free-decoded text on held-out prompts, and not only from log-probabilities?

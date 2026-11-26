# Best-Paper-Writing-Skill

**Citation.** gulucaptain, a GitHub user. No other author or affiliation is given. "Best-Paper-Writing-Skill", v1.2, a Claude and Codex skill. [Repository](https://github.com/gulucaptain/Best-Paper-Writing-Skill), read at commit `b9eff99a2130bea960ef1050eacf9f40cb792dd4` (last push 2026-10-29). Not a paper, and no venue. Its original content is CC BY 4.0. The workflow and the abstract and introduction templates are adapted from [Master-cai/Research-Paper-Writing-Skills](https://github.com/Master-cai/Research-Paper-Writing-Skills) (MIT), whose writing advice comes from Peng Sida's study notes (`NOTICE.md`). The exemplar sentences are short quotes from published papers, outside both licences. GitHub's licence detector reports `NOASSERTION`, because the `LICENSE` file is a short notice, not the licence text.

**Status.** Planned. Two of its rules become issues against `write-paper`: an abstract budget (#6929) and captions that open with their finding (#6930). Its paper list gives two reviews to write (#6931, #6932), and its data adds to #6851. Its rules against limits in the abstract and the introduction conflict with this repo's, and the page rejects them.

**Read.** On 2026-11-26, from a download of the repository at the commit above, by Claude (Opus 5.5). Read in full, in the source's own text: `README_en.md`, `CHANGELOG.md`, `LICENSE`, `NOTICE.md`, `SKILL.md`, and in `references/` the files `abstract.md`, `introduction.md`, `claim-calibration.md`, `corpus-profile.md`, `style-rules.md`, `method.md`, `experiments.md` and `figures.md`. Read in part: `routing.md` (its first two sections), `PAPER_LIST.md`, `evals/README.md`, `evals/rubric.md`, one case's `case.yaml` and graders, and the rule lists of `scripts/check_draft.py`. Not read: the exemplar bank (`references/exemplars/`), `corpus/corpus_stats.json`, the other scripts, the figure assets, the CVPR template and the Chinese README.

## Summary

**Claim.** A language-model agent writes a more persuasive abstract and introduction when it follows the patterns of award papers. The skill takes those patterns from a sentence-level annotation of 55 award papers, and holds 5 more out as a test set.

**Method.**

- *Corpus.* 60 best papers, best student papers and honourable mentions of CVPR, ICCV and NeurIPS, 2021–2025. Each sentence of 55 abstracts and introductions was labelled with its function. Five 2024 papers were kept out of the exemplar bank for the evaluation.
- *Routing.* A paper gets one of eight types: method improvement, new capability, empirical law, re-examination, mechanism, theory, dataset and measurement. A method paper also gets a motivation type, such as `failure` or `cost`. Each route has its own abstract arc and introduction plan.
- *Workflow.* An intake form or a pasted abstract gives the facts. The agent writes a claim, then a claim map with "the strongest wording the evidence supports", then a skeleton. It drafts from exemplar paragraphs of the same function.
- *Checks.* `check_draft.py` compares a draft with the corpus's distributions and flags phrases, dashes and copied exemplar wording. `check_figures.py` checks captions and placeholders.
- *Evaluation.* Each held-out case has a prompt, 13 graders and a blind-comparison rubric against the original paper's text.

**Findings, as the source reports them.** All are statistics of its own corpus, from `corpus-profile.md` and the reference files.

| Measure | Value |
| --- | --- |
| Abstract length | median 174 words (p10–p90: 132–219), 7 sentences (6–9) |
| Numbers in the abstract | median 1, 90% have at most 5–6 |
| Most common abstract arc | setting, gap, method, result |
| Introduction length | median 6 prose paragraphs (3–8), about 610 words |
| Numbers in the introduction | median 4, p90 14.6 |
| Introductions that credit specific prior work | 55 of 55 |
| Where the gap first appears | paragraph 2 (median) |
| Contrast with specific prior work | 32 of 55, closest work named in 24 |
| "Key insight"-type formula | 4 of 55 |
| Sentences that end in a participial clause | about 8% (typical AI drafts: about 32%, per the README) |
| Own limitations in the abstract or introduction | at least 7 of 55 |
| Figure caption length | median 42 words (two-column), 46 (one-column) |

## Where the claims go beyond the evidence

**The held-out test has no reported result.** The README says that the five 2024 papers test the rules. The `evals/` folder ships prompts, graders and a rubric, but no scores, rankings or outputs. So the repository does not show that its drafts beat a plain draft, or come near the originals.

**Description is read as prescription.** The corpus profile says the statistics are "not causes of awards" and should be used "as ranges, not targets". The skill then writes them as rules and as checker warnings. A paper may win an award for its result, whatever its abstract's word count.

**The award corpus is narrow.** It holds three venues, mostly computer vision, and five years. Some types rest on few papers: re-examination on 3, theory on 4. The authors say so: the plans for L4, L6, L7 and L8 "are derived from 3–5 award papers each".

**The rules against limits are the authors' preference, against the corpus.** At least 7 of the 55 papers state their own limits in the abstract or introduction. Each time the skill states the rule, it calls it an "author preference". Its claim that hedges "make reviewers doubt the work" cites no evidence.

**The AI-draft baseline is not described.** The README compares "typical AI drafts" with award papers: 32% of sentences against 8% end in a participial clause. It does not say which drafts, which model or how many.

The authors state their own limits clearly in places. They say that the statistics do not cause awards, and that each rule has "a floor and a ceiling". They also say that some type plans rest on few papers.

## Use in this repo

**Our rule on claim strength stays.** `write-paper` says "suggests" for a hint and "shows" only for a measured result. The skill's ladder in `claim-calibration.md` agrees on the wording for each kind of evidence. It parts from us on where limits go, which follows.

**The rule table.** Each of the skill's main rules, against `.claude/skills/write-paper/SKILL.md`, `scripts/ste_prose.py` and `tests/test_papers.py`:

| The skill's rule | Here | Reason |
| --- | --- | --- |
| Never invent numbers, citations or figure content | already covered | `tests/test_papers.py` checks citations and references. A test fails when a figure differs from `results.jsonl`. |
| Credit specific prior work, and name the gap by paragraph 2 | already covered | *Structure* item 2 says what earlier work leaves open. A Related work section is required. |
| Name the closest work with one checkable difference | covered in part | The Related work section places the paper. The introduction need not name the closest work, and two of three papers below don't. |
| An abstract of about 130–220 words with one or two headline numbers | **adopt** (#6929) | Our abstracts are longer and carry more numbers (below). The Research index shows the abstract alone. |
| A caption opens with its takeaway | **adopt** (#6930) | We put the finding in the figure's title, but some captions only describe the marks. |
| No limits, failure cases or disclaimers in the abstract or introduction | **reject** | A reader of an interpretability result must see its scope where the claim is made. Our papers state a limit with each contribution. |
| "Do not be more modest than the evidence allows", and promote the work with evaluative words | **reject** | It pushes toward stronger wording than our claim-strength rule. Our findings are often null or partial, as in `fisher-keep`. |
| State the core idea as an assertion, never "we hypothesize" | reject for us | Our papers test hypotheses, so a hypothesis is named as one until the test. |
| No dashes | already covered | Our papers use none (0 in 19 abstracts and introductions, by the count below). |
| No trailing ", making …" clauses | already covered | 0 in 19 papers, by the count below. |
| About one logical connector per 100 words | reject | Our median is 0 per 100 words. STE keeps the language simple, and our paper profile lets a sentence carry its relation with "because" or "whereas". |
| Sentences average 20–24 words, with short sentences at the pivots | already covered | `scripts/ste_prose.py` caps a paper's mean at 25 words and a sentence at 40. |
| Say each claim once | already covered in part | The coverage review in *Before you finish* reads the paper whole. No rule names repetition. |
| A list of boilerplate phrases to avoid, such as "delve" and "landscape" | covered in part | STE's rules on word choice apply to papers. The list itself adds nothing that this review needs. |
| One claim per figure, and a figure or table for each main claim | already covered | *Figures and animations*: "Put the finding in the title", and every figure is drawn from the data. |
| Implementation details out of the introduction | already covered, not checked | *Prose*: "Keep code names and file paths out of the argument". The papers below break it, and #6851 holds the check. |

**What the checks would change in three papers.** Counted on 2026-11-26 with a scratchpad script: words, sentences and numbers in the abstract and the introduction. Issue numbers and link targets are not counted as numbers, and the counts are approximate.

| Paper | Abstract | Introduction | What the skill would change |
| --- | --- | --- | --- |
| `conformal-line` | 162 words, 9 sentences, 10 numbers | 5 paragraphs, 38 numbers | Its first paragraph names `activation_threshold`, `ReviewPolicy` and `core/schemas.py`. The introduction credits conformal prediction after the gap. It ends with "These results do not show that any explanation is wrong", which the skill forbids and we keep. |
| `fisher-keep` | 269 words, 12 sentences, 18 numbers | 5 paragraphs, 8 numbers | The abstract is about a fifth over the cap, with 18 numbers around the headline AUC of 0.475 against the null's 0.523. Its introduction names EWC and AdaLoRA with a checkable difference, as the skill asks. |
| `copy-first-unit` | 172 words, 9 sentences, 4 numbers | 3 paragraphs, 25 numbers | The abstract is within the range. The introduction opens with an issue number and a `wiki/` path, and has no contrast with the closest work. It cites earlier work only inside its limits. |

Across all 19 papers, the abstracts have a median of 202 words, 11 sentences and 14 numbers. Ten are in the 130–220 word range, and seven are over it. The introductions have a median of 5 prose paragraphs and 12 numbers, which is within the skill's ranges. 17 of the 19 abstracts use a project term such as *pole* or *VERIFIED*. That supports #6851, which removes project terms from papers.

**The two award papers on its list.** Neither has a review page, and no paper here cites them. Both bear on open work, so each gets a review issue.

- *Are Emergent Abilities of Large Language Models a Mirage?* (Schaeffer, Miranda & Koyejo, NeurIPS 2023, [arXiv:2304.15004](https://arxiv.org/abs/2304.15004)) argues that a nonlinear metric can make a smooth change look sudden. `neuron-development` dates formation partly by a pass or fail verdict, which is such a metric (#6931).
- Liu, Liu & Gore's paper on superposition and neural scaling (NeurIPS 2025, [arXiv:2505.10465](https://arxiv.org/abs/2505.10465)) ties loss scaling to strong superposition. It bears on question 1's working hypothesis that native neurons' polysemanticity comes mostly from tooling (#6932).

**No change to the decision rule.** The skill is about prose, so nothing here touches `docs/PROTOCOL.md`.

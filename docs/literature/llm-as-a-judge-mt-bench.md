# Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

**Citation.** Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez and Ion Stoica. The paper marks joint first authors. It lists UC Berkeley, UC San Diego, Carnegie Mellon University, Stanford and MBZUAI, but the HTML does not map each author to an affiliation. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena". [arXiv:2306.05685](https://arxiv.org/abs/2306.05685) [cs.CL], version 4 of 24 December 2023. Version 1 is of 9 June 2023, version 2 of 12 July and version 3 of 15 October. The arXiv comment gives the venue as the NeurIPS 2023 Datasets and Benchmarks Track. The licence is arXiv's non-exclusive licence.

The code, the MT-Bench questions, 3K expert votes and 30K conversations with human preferences are at [github.com/lm-sys/FastChat](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge). The paper states no licence for the data. Gifts from Anyscale, Astronomer, Google, IBM, Intel, Lacework, Microsoft, MBZUAI, Samsung SDS, Uber and VMware partly fund it. A Meta PhD fellowship supports the first author. The paper makes no statement about the use of AI tools in its writing.

**Status.** Planned. #7084 applies the paper's swap test to the order of the windows that our detection judge reads. The judge's design already avoids the paper's main bias: it never sees two candidates side by side.

**Read.** On 2026-11-27 a `web-fetch` subagent read the abstract page and the whole HTML of version 4. It read them for this page's author, a model (Claude Opus 5.5). The first fetch came back verbatim up to a point, and a second fetch returned the rest verbatim. So we have the source's own text for Sections 1 to 7, the references and Appendices C to F, with Tables 2 to 16. We saw no figure as an image. So the judge prompts, which are figures, are known only from the text that describes them. The win-rate curves and the self-enhancement plot (Figure 2b) are missing too. We did not read the PDF or the code.

## Summary

**Claim.** A strong language model can stand in for human preference when judging chat assistants. GPT-4 agrees with expert humans as often as the experts agree with each other, at over 80%. The judges have biases, and the paper measures four of them and tests remedies.

**Method.**

- **Three judge formats.** In *pairwise comparison* the judge sees one question and two answers, and picks the better one or calls a tie. In *single-answer grading* it scores one answer alone. In *reference-guided grading* it also sees a reference solution.
- **MT-Bench.** 80 two-turn questions, 10 in each of 8 categories: writing, roleplay, extraction, reasoning, math, coding, STEM knowledge and humanities knowledge. Six models answer them: GPT-4, GPT-3.5, Claude-v1, Vicuna-13B, Alpaca-13B and LLaMA-13B.
- **Human votes.** 58 expert labellers, mostly graduate students, gave about 3K votes on MT-Bench. Chatbot Arena collected about 30K crowd votes in its first month, from 2,114 unique IP addresses. The paper samples 3K of these.
- **Agreement.** The agreement of two judge types is the probability that two random judges of those types agree on a random question. Setup S1 counts ties, and a vote that changes when the answers swap counts as a tie. Its random baseline is 33%. Setup S2 counts only votes that are not ties, with a random baseline of 50%.
- **The biases.**
  - *Position*: two similar answers to each first-turn question, judged in both orders. Consistency is the share of cases with the same verdict in both orders.
  - *Verbosity*: the "repetitive list" attack on 23 answers that hold a numbered list. The attack adds a rephrased copy of the list. A failure is a judge that prefers the padded answer.
  - *Self-enhancement*: whether a judge gives its own model's answers a higher win rate.
  - *Math grading*: whether a judge calls a wrong answer to one of 10 math questions correct.
- **The remedies.** One asks in both orders and calls a tie when the verdicts differ. The others are few-shot examples, chain of thought, a reference answer that the judge writes first, and a fine-tuned Vicuna-13B judge.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Position consistency, default prompt (Table 2) | GPT-4 65.0%, GPT-3.5 46.2%, Claude-v1 23.8% |
| Claude-v1's bias to the first answer, default prompt (Table 2) | 75.0% |
| GPT-4 consistency by category (Table 11) | Math and coding 86.0%, writing 42.0%, humanities 36.0% |
| Verbosity attack, failure rate (Table 3) | Claude-v1 91.3%, GPT-3.5 91.3%, GPT-4 8.7% |
| Self-enhancement (Section 3.3) | "GPT-4 favors itself with a 10% higher win rate", Claude-v1 with 25%. GPT-3.5 does not favour itself |
| Math grading, GPT-4 failures of 20 judgments (Table 4) | Default 14 (70%), chain of thought 6 (30%), reference-guided 3 (15%) |
| Few-shot judge, position consistency (Table 13) | GPT-4 65.0% → 77.5%, GPT-3.5 46.2% → 55.0%, Claude-v1 23.8% → 63.7% |
| Fine-tuned Vicuna-13B judge (Table 16) | Error rate 22.5% → 0.0%. Accuracy 56.8% with three labels and 85.5% without ties, against GPT-4's 66% and 87% |
| MT-Bench, first turn, S2 (Table 5) | GPT-4 pairwise against humans 85% (859 votes). Human against human 81% (479 votes) |
| MT-Bench, first turn, S1 (Table 5) | GPT-4 pairwise against humans 66%. Human against human 63% |
| MT-Bench, single-answer GPT-4 against humans, S2 (Table 5) | 85% on the first turn, 84% on the second |
| Chatbot Arena, GPT-4 pairwise against humans (Table 6) | S1 64% (3,066 votes), S2 87% (1,944 votes) |
| Humans shown GPT-4's verdict after they disagreed (Section 4.2) | They called it reasonable in 75% of cases, and would change their vote in 34% |

## Where the claims go beyond the evidence

**"Over 80%" is the setup without ties.** The headline agreement of 85% is setup S2, which drops every tie. With ties, in S1, GPT-4 agrees with humans 66% of the time on MT-Bench's first turn. Humans agree with each other 63% of the time. The comparison with humans holds in both setups, but the abstract gives only the higher one.

**No intervals.** The agreement rates are point estimates from a few hundred to a few thousand votes. The paper gives no interval or test. So a gap such as 85% against 81% has no stated uncertainty, and the votes on one question are not independent.

**Small bias tests.** The verbosity attack uses 23 answers, and the math test uses 10 questions in two orders. The authors say so for self-enhancement: "Due to limited data and small differences, our study cannot determine" whether it exists.

**The swap remedy is not measured.** The paper recommends asking in both orders and calling a tie when the verdicts differ, and uses it in later experiments. It gives no number for how that changes agreement with humans.

**Agreement with humans, not correctness.** The reference is human preference on helpfulness. The authors note that the study "largely neglects safety". They also note that it combines accuracy, relevance and creativity into one number. Experts who saw GPT-4's verdict would change their own vote in 34% of disagreements, so the human reference is itself soft.

**The judges age.** The results are for GPT-4, GPT-3.5 and Claude-v1 of 2023. The authors warn that the case studies "may not be fully reproducible with future GPT-4 versions".

## Use in this repo

**Our judge applies a description to texts, with labels from the model.** The detection and fuzzing judge reads one explanation and about 50 held-out windows. It lists the windows on which the explanation says the unit fires (`openinterpretability/harness/explanation_scores.py`, `docs/benchmark.md`). The labels come from a new measurement of the unit's activations, and the judge never sees them. So the score's reference is the model itself, not a human preference. This is a firmer reference than the paper's, and the paper's agreement results do not transfer to it.

**No pairwise comparison, so no bias between two candidates.** The paper's strongest bias is position bias between two answers. Our judge never compares two explanations in one prompt. When two readings of a unit split evenly, each one's own score settles the split, with a margin of 0.05 (`docs/PROTOCOL.md` §3, *Standing*, step 4). That is the paper's single-answer grading, which the paper finds as close to humans as pairwise comparison (85% against 85% in S2).

**But the windows have an order, and no test measures its effect.** The score's seed shuffles the windows (`rng.shuffle` in `explanation_scores.py`). The shuffle spreads any effect of position across windows, but it does not measure it. The paper's swap test carries over directly: judge the same windows in two orders, and see how much the answers and the balanced accuracy move. #7084 holds that test. It compares the spread with the 0.05 margin that settles a split.

**Verbosity has a counterpart: breadth.** The paper's judges prefer a padded answer. Our judge could reward a broad explanation that marks many windows. Balanced accuracy penalises false positives as much as missed positives, so breadth alone can't raise a score. `null-explanation-control.md` already runs a broad decoy explanation against each unit's own, and #3796 extends that control to the judges that made most stored scores.

**Self-enhancement is a plausible risk here.** Claude models propose most explanations, and the `explanation-judge` agent runs Claude Haiku. The paper finds that GPT-4 and Claude-v1 give their own answers higher win rates, but it can't confirm the effect. For us, a judge could read an explanation in its own family's style more generously. No analysis tests this. A test would score the same units' explanations, written by two model families, with judges from both. We record it as a question and open no issue.

**Injected text is already tested.** `judge-injection.md` found that no explanation that gave orders to the judge scored above chance (#3303). The page on [model-graded robustness](model-graded-robustness.md) covers that attack.

**Judges differ in level, so record which judge scored.** The paper warns that "absolute scores are likely to fluctuate more than relative pairwise results if the judge model changes". `rescore-pilot.md` found the same: the registry's judges differ by 5 to 8 points, while the order of the records stays close (0.77). #6985 records the judge's prompt and decode settings beside each benchmark score.

**Judge–human agreement as a measure.** The research-taste review (#6437) compares judge–human with human–human agreement, as this paper does. The comparison needs a human reference, which the registry's scores don't use. The LLM-as-a-judge review (#6451) and the review of *Auditing the Judge* (#6790) are the next places for this paper's biases.

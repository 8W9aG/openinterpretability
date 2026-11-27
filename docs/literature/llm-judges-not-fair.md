# Large Language Models are not Fair Evaluators

**Citation.** Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu and Zhifang Sui are at Peking University. Qi Liu is at The University of Hong Kong. Binghuai Lin, Yunbo Cao and Tianyu Liu are at Tencent Cloud AI. "Large Language Models are not Fair Evaluators". arXiv:2305.17926, v1 of 29 May 2023 and v2 of 30 August 2023, cs.CL. [Paper](https://arxiv.org/abs/2305.17926), [code and human labels](https://github.com/i-Eval/FairEval). This page gives no venue: the arXiv page, the paper and the code's README name none, and the README's BibTeX is an arXiv entry. That BibTeX also leaves out Zefan Cai, the fourth author on the paper. The paper states no funding and no tool use.

**Status.** Planned. #7084 asks whether the detection judge's answer depends on the order of its windows, and this paper supports it. The paper studies only pairwise judging, two responses in one prompt. The repo's judge marks windows one by one, so its fix doesn't carry over as it is.

**Read.** On 2026-11-27 a subagent (Claude) downloaded the abstract page, the v2 HTML and PDF, and the code's README, and turned them into text. It read Sections 1 to 7, Tables 1 to 5, the equations and the footnote verbatim, and checked the HTML against the PDF. It saw Figures 1 to 6 only as captions, so no number on this page comes from a figure. It did not see v1, the code or the human-label file. The reader of this page was a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** An LLM judge's verdict on two responses depends on which response comes first. Calibrating with several evidence-first samples, both orders and a little human help brings the judge closer to human labels.

**Method.**

- **Judges:** `gpt-3.5-turbo-0301` (ChatGPT) and `gpt-4`, at temperature 0, or 1 with k = 3 samples for the multi-sample methods.
- **Data:** the 80 questions of the Vicuna benchmark, in 9 categories. Vicuna-13B's responses are compared with ChatGPT's. Vicuna-13B against Alpaca-13B is used only to measure the bias.
- **Template:** Zheng et al. (2023)'s. One prompt holds both responses, and the judge scores each from 1 to 10. The template already asks it to ignore the order.
- **Conflict rate:** the share of questions whose verdict changes when the two responses swap places (Eq. 1).
- **Calibrations:**
  - **Multiple Evidence Calibration (MEC)** asks for the explanation before the score, and averages k samples.
  - **Balanced Position Calibration (BPC)** also scores the swapped prompt, and averages over both orders (Eq. 2).
  - **Human-in-the-Loop Calibration (HITLC)** sends to people the top β of questions by Balanced Position Diversity Entropy (BPDE). BPDE is the entropy of the win, tie and lose results over both orders and all samples (Eqs. 3 to 5).
- **Labels:** three of the authors labelled each pair win, tie or lose, from their majority vote. The judge is scored by accuracy and kappa against that vote.

**Findings, as the source reports them.**

| Result | Setting |
| --- | --- |
| Conflict rate 66 of 80 (82.5%) | ChatGPT judging Vicuna-13B against ChatGPT. Vicuna-13B wins 82.5% as Assistant 2 and 2.5% as Assistant 1 (Table 2) |
| Conflict rate 37 of 80 (46.3%) | GPT-4, the same pair. GPT-4 favours the first response, and ChatGPT the second |
| Conflict rate 4 of 80 (5.0%) for GPT-4, 42 of 80 (52.5%) for ChatGPT | Vicuna-13B against Alpaca-13B, a pair with a larger quality gap |
| Accuracy 52.7% → 62.5%, kappa 0.24 → 0.37 | GPT-4, from the plain template to MEC (k = 3) + BPC (Table 4) |
| Accuracy 44.4% → 58.7%, kappa 0.06 → 0.31 | ChatGPT, the same change |
| 60.9% for MEC with k = 6, against 62.5% for MEC + BPC | GPT-4. Both use 6 calls, so the gain is from the second order, not from more samples |
| Accuracy 73.8% (GPT-4) and 71.3% (ChatGPT) | With HITLC at β = 20%. The humans average 71.7%, against the same majority vote |
| The order matters most when the score gap is 1 or less | GPT-4 (Section 2.2). The text gives this, and the figure's numbers weren't readable |

## Where the claims go beyond the evidence

- **One pair, 80 questions, three author-annotators.** Every accuracy rests on Vicuna-13B against ChatGPT on 80 questions. The labels come from three of the authors, and each annotator's score is against a vote that includes their own label. The paper reports no agreement between annotators.
- **No intervals, seeds or tests.** The methods without BPC average 100 random orders, but the paper reports no spread, and no significance test for any gain.
- **A closed, changing judge.** `gpt-4` has no dated snapshot, so the GPT-4 numbers can't be repeated exactly.
- **A cost that doesn't add up.** For ChatGPT, HITLC's $18.3 is $18 of human time plus $0.34 of API calls. For GPT-4, the same sum gives $24.38, not the reported $23.1, and the paper doesn't explain the gap.
- **No limitations section.** The paper names none of these limits itself.
- **"LLMs are not fair evaluators" rests on pairwise judging only.** The paper has no condition in which one response is judged alone, and no list of more than two items.

## Use in this repo

- **The repo's judge is not pairwise, so BPC doesn't apply as it is.** `open-interpretability score` asks the judge about one explanation, over about 50 windows in one prompt (`openinterpretability/agents/judge.py`). It marks each window active or not, with no second candidate to swap. The benchmark's prompts are the same kind (`docs/benchmark.md`).
- **The repo already randomises the order, but reads it once.** `harness/explanation_scores.sample_heldout` shuffles the detection windows and the fuzzing windows with the task's seed. `benchmark.shuffled` orders each unit's probes by a hash of its slug. So no window's place carries its label. But each task is judged in one order only, so an order effect would add noise to the score, not bias. The paper shows that this noise can be large: a conflict rate of 46% for GPT-4 on a close pair.
- **Where that noise matters.** A VERIFIED-against-REJECTED split is settled by the explanation scores when one reading is at least 0.05 higher (`docs/PROTOCOL.md` §3, *Standing*). If a unit's balanced accuracy moves by about 0.05 between orders, that rule settles some splits by chance. #7084 plans that measurement: the same windows in four orders, with the spread compared to the 0.05 margin. This paper supports it.
- **Two ideas for #7084, as a comment, not a new issue.**
  - **Look for the effect where the gap is small.** The paper's bias is largest when the two responses are close. The judge's nearest analogue is a window near the activation threshold. #7084 could report the change of answer by distance from the threshold.
  - **Use disagreement across orders as a flag.** BPDE sends a case to people when its verdicts disagree across orders. A unit whose score moves across orders could keep its split open, instead of letting one order settle it.
- **Explanation before score.** MEC's first half asks for the explanation before the score. The repo's judge returns structured answers, a list of window numbers. Whether a short reason before the list changes the answers is a separate question, and the repo has no evidence on it. It is not opened as an issue here.

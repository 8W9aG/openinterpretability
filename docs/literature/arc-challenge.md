# Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge

**Citation.** Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick and Oyvind Tafjord, of the Allen Institute for Artificial Intelligence, Seattle. "Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge". [arXiv:1803.05457](https://arxiv.org/abs/1803.05457), version 1 of 14 March 2018, the only version. Preprint, no venue given. The dataset, its corpus, three baseline models and a leaderboard are at [data.allenai.org/arc](http://data.allenai.org/arc). Code for four baselines is at [github.com/allenai/arc-solvers](https://github.com/allenai/arc-solvers). The text names no funding.

**Mismatch.** The licence is unclear. The Hugging Face copy, `allenai/ai2_arc`, gives `cc-by-sa-4.0` in its card's header at commit `210d026faf9955653af8916fad021475a3f00453`. The same card's *Licensing Information* section says "More Information Needed". The paper says that its largest source, an anonymous partner that it calls "Mercury", gave its questions "for non-commercial research purposes". Mercury's questions are 5,128 of the 7,787 in Appendix A's Table 7.

**Status.** Used. `openinterpretability/interference/tuning.py` reads the ARC-Easy test split as its general benchmark, and `papers/sequential-tuning/` (#5125) reports it.

**Read.** On 2026-11-26 a `web-fetch` subagent read the abstract page and the HTML of version 1 for this page's author, a model (Claude Opus 5.5). It read the text in two passes, the second from where the first stopped. It had the source's text for every section, Tables 1 to 7 and Appendices A and B. The tool allows only short verbatim quotes, so much of that text came back paraphrased. It did not see Figures 1 and 2, only their captions.

## Summary

**Claim.** ARC is a set of 7,787 grade-school science questions, split into a Challenge Set and an Easy Set. A question is in the Challenge Set when two simple baselines, one by retrieval and one by word co-occurrence, both answer it wrongly. On the Challenge Set, no baseline that the authors tried scores clearly above random guessing.

**Method.**

- **Questions (Section 3, Appendix A).** Multiple-choice science questions from standardised tests, for grades 3 to 9. They have no diagrams. Most have 4 options, and the range is 3 to 5.
- **The split.** The Challenge Set holds the questions that both the IR solver and the PMI solver of Clark et al. (2016) answer wrongly. Both use a web corpus of $5 \times 10^{10}$ tokens. The authors say this only approximates the goal of a hard question. The Easy Set holds the rest.
- **ARC Corpus (Section 5).** 14 million science sentences, 1.4 GB. A search engine was queried with about 100 hand-written templates over 80 science topics, and the results were split into sentences.
- **Scoring (Section 6).** A system gets 1 point for the correct answer, and $1/k$ when it reports a $k$-way tie that includes the correct answer. The score is the points divided by the number of questions.
- **Baselines.** IR and PMI, three solvers over tables and tuples, and three neural models: BiDAF, DecompAttn and DGEM. The neural models look up sentences in the corpus and score them for entailment. A guess on every option is the random baseline.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Questions, Table 1 | 7,787: 2,590 Challenge and 5,197 Easy |
| Test split, Table 1 | 1,172 Challenge and 2,376 Easy |
| Train and dev splits, Table 1 | Challenge 1,119 and 299, Easy 2,251 and 570 |
| Best Challenge score, Table 6 | DGEM, 27.11% |
| Random guessing, Table 6 | 25.02% on both sets |
| Best Easy score of a solver not used for the split, Table 6 | IR over the ARC Corpus, 62.55% |
| Confidence interval, Section 6 | ±2.5% at 95%. No Challenge score is significantly above random |

## Where the claims go beyond the evidence

- **"Hard" is set by two solvers.** The Challenge Set is what IR and PMI answer wrongly, and the authors say so. A question that a third simple method answers is still in the Challenge Set.
- **One interval for all scores.** The paper gives one 95% interval, ±2.5%, with no test named. The Challenge and Easy test sets differ in size, so one interval can't fit both.
- **The Easy baselines include the solvers that made the split.** IR and PMI score 74.48% and 77.82% on the Easy Set because the split puts there the questions they answer. The authors mark these rows.
- **The terms of the data are mixed.** Two thirds of the questions come from a partner that allows non-commercial research only. The paper states no licence for the whole set.

## Use in this repo

- **ARC-Easy is the sequential-tuning paper's general benchmark.** `interference/tuning.py` reads its test split, at a pinned commit. The paper's count, "All 2,376 test questions of ARC-Easy", matches Table 1.
- **The licence line is accurate, but incomplete.** The sequential-tuning paper gives "CC BY-SA 4.0" as the licence *on its card*, which is what the card's header says. The source itself adds a non-commercial condition on most questions. The repo uses ARC for research only and publishes no copy, so the condition doesn't change what the repo may do. No issue follows.
- **The repo scores ARC its own way.** The sequential-tuning paper takes the option with the highest mean log-probability per token, with no partial credit, because a model never reports a tie. The paper's scoring gives $1/k$ for a tie. Both have random guessing near 25%, so the numbers are comparable in scale but not equal. The sequential-tuning paper says that its numbers don't compare with lm-evaluation-harness's either.
- **The Easy Set suits a forgetting benchmark.** It holds the questions that retrieval answers, so its score tracks the knowledge that a model stores. That is what a narrow fine-tune can wear away. The Challenge Set would sit near random for Qwen2.5-0.5B, so a fall there would be hard to see.
- **The paper's interval is coarser than the repo's.** The sequential-tuning paper gives Wilson intervals on 2,376 questions. Its base model ends 3.5 points above its start, from 0.578 to 0.614. That rise is larger than the paper's ±2.5%, but the repo's intervals, not this one, are the test that it reports.

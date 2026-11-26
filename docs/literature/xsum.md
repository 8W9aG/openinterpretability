# Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization

**Citation.** Shashi Narayan, Shay B. Cohen and Mirella Lapata, all of the School of Informatics, University of Edinburgh. "Don't Give Me the Details, Just the Summary! Topic-Aware Convolutional Neural Networks for Extreme Summarization". *Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing*, pages 1797–1807, Brussels, 2018. [ACL Anthology D18-1206](https://aclanthology.org/D18-1206/), DOI 10.18653/v1/D18-1206. [arXiv:1808.08745](https://arxiv.org/abs/1808.08745), version 1 of 27 August 2018, the only version. The dataset, code and demo are at [github.com/EdinburghNLP/XSum](https://github.com/EdinburghNLP/XSum). Support: the European Research Council, the EU Horizon 2020 SUMMA project and Huawei Technologies.

**Mismatches.** Two numbers differ between the paper's tables. The XSum extractive oracle's ROUGE-L is 22.65 in Table 2 and 22.66 in Table 4. LEAD's ROUGE-2 is 1.61 in Table 2 and 1.60 in Table 4. For the other datasets, the text gives novel bigram and trigram ranges of "47–55%" and "58–72%". Table 2 doesn't match them: its lowest values are 53.78 and 71.93. The Hub card cites the arXiv version, not EMNLP.

**Status.** Used. `papers/sequential-tuning/` fine-tunes on XSum as its second narrow stage, through `openinterpretability/interference/tuning.py`, which reads the Hub copy `EdinburghNLP/xsum` at one commit.

**Read.** On 2026-11-26 a `web-fetch` subagent (Claude) read the anthology PDF as extracted text, for this page's author, a model (Claude Opus 5.5). It had the source's own text for the abstract, all six sections, Tables 1 to 7, the footnotes and the acknowledgements. Figures 1 and 2 came through garbled. It also read the arXiv abstract page and the Hub card of `EdinburghNLP/xsum`. It did not compare the arXiv PDF's text with the anthology's. This page's author measured the lengths in the pinned Hub files.

## Summary

**Claim.** XSum is a dataset for extreme summarization: one sentence that says what a news article is about. It favours abstractive summaries. A topic-aware convolutional model, T-ConvS2S, beats the extractive oracle and recurrent abstractive models on it, by ROUGE and by people.

**Method.**

- **Data.** 226,711 BBC articles from 2010 to 2017, from the Wayback archive. "Each article is prefaced with an introductory sentence (aka summary) which is professionally written, typically by the author of the article." That sentence is the summary, and it is taken out of the article's body. The articles are split at random by their URL identifier: 204,045 training, 11,332 validation and 11,334 test.
- **Models.** T-ConvS2S adds LDA topics (512, fitted on the training documents) to each word's input and to the decoder of a convolutional sequence-to-sequence model. The extractive baselines are random, LEAD (the first sentence after the summary) and an oracle. The abstractive ones are a sequence-to-sequence model, the pointer-generator with and without coverage, and plain ConvS2S.
- **Measures.** ROUGE-1, -2 and -L F1 on the test set. Best-worst scaling by three Mechanical Turk workers on 50 test documents. Two fact questions for each of those 50 documents, answered by five workers from the summary alone.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| XSum, mean words (Table 1) | Documents 431.07, summaries 23.26, 1.00 sentence each |
| Novel n-grams in the gold summaries (Table 2) | 35.76% of unigrams, 83.45% of bigrams, 95.50% of trigrams. CNN: 16.75%, 54.33%, 72.42% |
| LEAD on XSum, ROUGE-1/2/L (Table 4) | 16.30 / 1.60 / 11.95. Extractive oracle: 29.79 / 8.81 / 22.66 |
| ConvS2S (Table 4) | 31.27 / 11.07 / 25.23 |
| T-ConvS2S (Table 4) | 31.89 / 11.54 / 25.75 |
| Best-worst scaling (Table 7) | T-ConvS2S 0.037, ConvS2S −0.130, extractive oracle −0.121, gold 0.431 |
| Question answering, of 100 (Table 7) | T-ConvS2S 46.05, ConvS2S 30.90, gold 97.23 |

## Where the claims go beyond the evidence

- **No seeds or intervals for ROUGE.** Each model is one run. T-ConvS2S leads ConvS2S by 0.3 to 0.6 ROUGE points, and the paper gives no test of that gap.
- **Small human studies.** The ratings and the questions cover 50 test documents. The paper gives no agreement between the workers. The best-worst scores do have a test (ANOVA with Tukey HSD), and the paper reports which pairs differ.
- **One outlet, one style.** The authors say so themselves: "XSum is not diverse, it focuses on a single news outlet (i.e., BBC) and a unifrom summarization style".
- **The summary is the writer's first sentence.** Our reading: it isn't written as a summary of the text that follows, so it may hold facts that the article doesn't. That would add to its new n-grams. The paper reports the novelty, but not how often a summary states something the article lacks.
- **No terms for the text.** The paper gives none, and the Hub card's licence is "unknown". The GitHub repository's MIT licence covers the code, not the BBC articles.

## Use in this repo

- **Our citation checks out.** `papers/sequential-tuning/` describes XSum as "a one-sentence summary of a news article, cut to its first 1,500 characters". The summary is one sentence, and the source makes it the article's introduction. The cut is the paper's own setting, which it states.
- **What the cut keeps, measured on the pinned commit.** Articles have a median of 1,742 characters in training and 1,725 in validation. The cut shortens 58.5% of each split, and the median article keeps 86% of its characters. In the paper's sampled pairs, it shortens 58% of the 2,000 training articles and 50.5% of the 200 held-out ones. The source's models cut the article to 400 tokens, so neither setting reads every article whole.
- **Empty articles don't reach the paper.** The Hub copy has 28 empty articles in training and 5 in validation. None of them is among the paper's seeded pairs, so no training or held-out pair has an empty input.
- **The text stays out of the repository.** The run reads the data for measurement only, as `papers/sequential-tuning/` Appendix A says. That fits the missing terms.
- **No unit-level content.** The source studies whole-model summaries, not neurons. It gives the registry no method.
- **No new issue, and no question for the maintainer.**

# Character-level Convolutional Networks for Text Classification

**Citation.** Xiang Zhang, Junbo Zhao and Yann LeCun, of New York University. "Character-level Convolutional Networks for Text Classification". *Advances in Neural Information Processing Systems 28 (NIPS 2015)*. [arXiv:1509.01626](https://arxiv.org/abs/1509.01626), version 1 of 4 September 2015, last revised as version 3 on 4 April 2016. An earlier version, "Text Understanding from Scratch", is [arXiv:1502.01710](https://arxiv.org/abs/1502.01710). The paper links no code. It states no licence for its datasets.

**Mismatch.** None found. The paper doesn't name AG News's four classes. The `ag_news` preset's comment names them World, Sports, Business and Sci/Tech.

**Status.** Used. The `ag_news` preset of `openinterpretability/concepts.py` reads the AG News set that this paper built. `papers/default-on-poles` uses its topic labels in its decoding test.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 3 for this page's author, a model (Claude Opus 5.5). It read the whole text in the source's own words, from the abstract to the references. We didn't read the figures or the PDF.

## Summary

**Claim.** A convolutional network that reads raw characters, with no words, can classify text as well as word-based methods, once the training set is large. On smaller sets, n-gram features with TF-IDF stay strong.

**Method.**

- **Datasets.** The authors built eight large classification sets: AG's News, Sogou News, DBPedia, two Yelp sets, Yahoo! Answers and two Amazon sets.
- **AG's News.** From "the AG's corpus of news article on the web", which "contains 496,835 categorized news articles from more than 2000 news sources". The authors kept "the 4 largest classes", with 30,000 training and 1,900 test articles per class, "using only the title and description fields".
- **Models.** Two 9-layer character networks, small and large, over a 70-character alphabet, with optional synonym replacement from a thesaurus. Baselines are bag-of-words, n-grams, word2vec means, word-level networks and an LSTM.

**Findings, as the source reports them.**

| Setting | AG's News test error (Table 4) |
| --- | --- |
| n-grams with TF-IDF | 7.64% (the best) |
| Word lookup-table network, large | 8.55% |
| Character network, large | 12.82% |
| LSTM | 13.94% |
| AG's News sizes (Table 3) | 120,000 training, 7,600 test |

## Where the claims go beyond the evidence

- **Labels come from the source, not from annotators.** The paper says the AG articles are "categorized", but not by whom or how. Sogou's labels came from manually classifying the sites' domain names.
- **One run per model.** The authors report results "without any model selection", which is fair, but give no seeds or intervals.
- **Text classification only.** The authors say their results don't speak "for any other language processing tasks".

## Use in this repo

- **Our citation checks out, with one sentence added.** `papers/default-on-poles` cites the paper as the source of AG News. The authors, 2015 and NIPS 2015 match. The paper listed AG News among its "human labels", but the source doesn't say who assigned the categories. This PR adds one sentence saying that the topics are the corpus's categories, assigned in a way the source doesn't describe. No number changes.
- **The preset's comment fits the source.** `concepts.py` describes "headlines and lead sentences in four topics", which matches the title and description fields. Its licence note says "unknown", and the paper states none.
- **The paper's licence caution fits.** `papers/default-on-poles` publishes only aggregate scores, because the AG News licence is unclear. The source gives no terms, so that caution stands.
- **Topic labels are document-level.** Each AG label covers a whole article, so a token-level decoding test reads a document's topic at every token. Our reading: that is a different target from the per-token part-of-speech and entity labels. The default-on-poles paper's lower lookup score on AG News (0.619) reflects it.
- **No question for the maintainer.**
- **Related pages.** [`conll-2003.md`](conll-2003.md) and [`universal-dependencies-v2.md`](universal-dependencies-v2.md) review the paper's other labelled sources.

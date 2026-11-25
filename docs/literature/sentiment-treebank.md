# Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank

**Citation.** Richard Socher, Alex Perelygin, Jean Y. Wu, Jason Chuang, Christopher D. Manning, Andrew Y. Ng and Christopher Potts, all of Stanford University. "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank". *Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing*, pages 1631–1642, Seattle, October 2013. [ACL Anthology D13-1170](https://aclanthology.org/D13-1170/). The code, a demo and the data are at [nlp.stanford.edu/sentiment](https://nlp.stanford.edu/sentiment/). Support: a Microsoft Research PhD fellowship, the DARPA DEFT and Deep Learning programmes, and NSF IIS-1159679.

**Mismatches.** The abstract on the Stanford page differs from the paper's. It adds "contrastive conjunctions" to the effects that the model captures. The README in the data release gives the citation the wrong title, "Parsing With Compositional Vector Grammars". The paper never says "SST" or "SST-2". The GLUE benchmark (Wang et al., 2019) gave the binary sentence task the name SST-2.

**Status.** Used. `papers/sequential-tuning/` fine-tunes on SST-2 as its first narrow stage, through `openinterpretability/interference/tuning.py`, which reads the Hub copy `stanfordnlp/sst2` at one commit.

**Read.** On 2026-11-25 a `web-fetch` subagent (Claude) read the paper's PDF from the anthology as extracted text, for this page's author, a model (Claude Opus 5.5). It had the source's own text for the abstract, every section, the acknowledgements, the references and Tables 1 to 3. It saw only the captions of Figures 2, 6, 8 and 10, and the trees of Figure 9 came out garbled. It did not see the supplementary material. The Stanford site refused the connection, so it read the pages and the data release's README through 2024 Wayback copies. It also read the Hub card of `stanfordnlp/sst2` and the SST-2 paragraph of the GLUE paper. This page's author counted the lengths in the pinned Hub files.

## Summary

**Claim.** Sentiment depends on how words compose, and a treebank with a label on every phrase lets a model learn that composition. A Recursive Neural Tensor Network (RNTN) beats bag-of-words models and earlier recursive models on it. It also handles negation and "but" better.

**Method.**

- **Data.** The 10,662 movie-review snippets of Pang and Lee (2005), from rottentomatoes.com. The Stanford Parser splits about 1,100 of them, which gives 11,855 sentences. Mechanical Turk workers rated each of the 215,154 unique phrases of their parse trees on a 25-step slider. "most annotators moved the slider to one of the five positions", so the paper uses five classes. The release's README gives the cut-offs on its "positivity probability". The paper does not say how it combines the 3 judges of each phrase.
- **Splits.** 8,544 training, 1,101 dev and 2,210 test sentences. "We also analyze performance on only positive and negative sentences, ignoring the neutral class. This filters about 20% of the data with the three sets having 6920/872/1821 sentences."
- **Models.** A recursive neural network, a matrix-vector RNN and the RNTN. The RNTN adds a tensor term, quadratic in the two child vectors, to the RNN's composition. Each node has a softmax over the five classes. Training uses AdaGrad, with word vectors of 25 to 35 dimensions chosen on the dev set.
- **Baselines.** Naive Bayes and an SVM on bags of words, Naive Bayes on bigrams, and the mean of word vectors.
- **Measure.** Accuracy over all nodes and at the root (the whole sentence), for five classes and for positive against negative.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Five classes, all phrases (Table 1) | RNTN 80.7%, MV-RNN 78.7%, RNN 79.0%, best baseline (BiNB) 71.0% |
| Five classes, sentences (Table 1) | RNTN 45.7%, MV-RNN 44.4%, RNN 43.2%, BiNB 41.9% |
| Positive against negative, all phrases (Table 1) | RNTN 87.6%, MV-RNN 86.8%, RNN 86.1%, VecAvg 85.1% |
| Positive against negative, sentences (Table 1) | RNTN 85.4%, BiNB 83.1%, MV-RNN 82.9%, NB 81.8% |
| "X but Y", 131 sentences (§5.3) | RNTN 41%, MV-RNN 37%, RNN 36%, BiNB 27% |
| Negated positive sentences, 21 pairs (Table 2) | RNTN inverts 71.4%, MV-RNN 52.4%, RNN 33.3%, BiNB 19.0% |
| Negated negative sentences (Table 2) | RNTN 81.8%, MV-RNN 54.6%, RNN 45.5%, BiNB 27.3% |

## Where the claims go beyond the evidence

- **One run for each number.** The paper gives no seeds, no repeated runs, no intervals and no tests. At the sentence level the RNTN leads by 2.3 points (85.4% against 83.1%) on 1,821 test sentences. If the two models' errors were independent, that is about 1.9 standard errors. One run can't show that it is stable.
- **Small analysis sets.** The "but" result rests on 131 sentences and the negated-positive result on 21 pairs. The paper doesn't give the size of the negated-negative set.
- **"Composition" by tensor slices is a reading, not a test.** "we can interpret each slice of the tensor as capturing a specific type of composition". No experiment looks at a slice.
- **One domain.** All text is short movie-review snippets from one site.
- **The labels' aggregation is not stated.** Three judges rated each phrase, but the paper does not say how their ratings make one label, and it reports no agreement between them.
- **An internal slip.** §5.2 says the RNTN "pushes the state of the art on short phrases up to 85.4%", but 85.4% is the sentence-level number in Table 1.
- **No licence.** The paper, the Stanford pages and the release's README state no terms of use. The README asks only for a citation. The Hub card says "unknown".

The authors give some limits themselves: "the majority of the shorter phrases are neutral", and many top phrases "could be considered neutral".

## Use in this repo

- **Our citation needed one correction, made in this PR.** `papers/sequential-tuning/` described SST-2 as "sentiment of a sentence". That holds for the paper's binary task and for the Hub copy's validation split. But the Hub copy's training split holds 67,349 rows, not the paper's 6,920 sentences. They hold the labelled phrases of the training sentences as well. Their median is 7 words, and 27.5% of them have 3 words or fewer. The 872 validation rows are the paper's binary dev sentences, with a median of 19 words. So the paper's 2,000 training pairs are mostly phrases, and its held-out loss is on whole sentences. The method now says so.
- **This doesn't change the paper's results.** The stage still teaches a two-word answer form on movie-review text, and its held-out loss falls from 2.31 to 0.20 nats on the base model. Our reading: a stage of shorter inputs may move the units less than whole sentences would. We have no measure of that, so the paper states it as a fact of the data, not as a cause.
- **The name.** "SST-2" is GLUE's name. Our paper keeps it, because the Hub copy and later work use it. The paper cites the 2013 paper for the data.
- **The text stays out of the repository.** The run reads the data for measurement only, as `papers/sequential-tuning/` Appendix A says. That fits the missing licence.
- **No unit-level content.** The paper studies a model's outputs at each node, not single neurons. It gives the registry no method.
- **No new issue, and no question for the maintainer.** The training on phrases is a fact of the data that #6367's further stages inherit. It is not new work.

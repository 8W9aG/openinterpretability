# Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

**Citation.** Nils Reimers and Iryna Gurevych, Ubiquitous Knowledge Processing Lab, Technische Universität Darmstadt. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks". *Proceedings of EMNLP-IJCNLP 2019*, pages 3982–3992, Hong Kong. [ACL Anthology D19-1410](https://aclanthology.org/D19-1410/), [doi:10.18653/v1/D19-1410](https://doi.org/10.18653/v1/D19-1410). The preprint is [arXiv:1908.10084](https://arxiv.org/abs/1908.10084), one version, of 27 August 2019. Code is at [github.com/UKPLab/sentence-transformers](https://github.com/UKPLab/sentence-transformers). The German Research Foundation and the German Federal Ministry of Education and Research funded the work.

**Mismatch.** The introduction says SBERT gains 11.7 points over InferSent and 5.5 over the Universal Sentence Encoder. In Table 1, those gaps fit SRoBERTa-NLI-large (76.68). SBERT-NLI-large, at 76.55, is 11.5 and 5.3 points ahead. That check is our own arithmetic.

**Status.** Used. The concept map (`concept_map.EMBEDDER`) and the literature map embed texts with `all-MiniLM-L6-v2`, a model from the paper's `sentence-transformers` library. `openinterpretability/harness/explanation_scores.py` uses the same model for its embedding score.

**Read.** On 2026-11-21 a `web-fetch` subagent read the source for this page's author, a model (Claude Opus 5.5). It read the arXiv abstract page, the ACL Anthology page and the HTML of version 1. It had the whole body, Sections 1 to 8 and the acknowledgements, in the source's own words. The tail of the reference list came back only as a summary.

## Summary

**Claim.** Fine-tuning BERT in a siamese network gives "sentence embeddings that can be compared using cosine-similarity". Finding the most similar pair among 10,000 sentences takes "about 5 seconds", against 65 hours for BERT as a cross-encoder.

**Method.**

- **The network.** BERT or RoBERTa, with a pooling layer on top. "The default configuration is MEAN", the mean of the output vectors.
- **Training.** A classifier on SNLI and MultiNLI over (u, v, |u − v|), a cosine regression on STS, or a triplet loss on Wikipedia sections.
- **Tests.** Seven semantic textual similarity (STS) sets without training on them, the STS benchmark with training, argument similarity, Wikipedia sections and SentEval.

**Findings, as the source reports them.**

| Setting (Spearman × 100 unless named) | Result |
| --- | --- |
| Unsupervised STS, mean of 7 sets (Table 1) | SBERT-NLI-large 76.55, Universal Sentence Encoder 71.22, mean GloVe 61.32, mean BERT output 54.81, BERT CLS 29.19 |
| STS benchmark, trained (Table 2) | SBERT-NLI-STSb-large 86.10, BERT as a cross-encoder 88.77 |
| Argument similarity, across topics (Table 3) | SBERT-AFS-large 53.10, BERT-AFS-large 60.34 |
| Pooling, STS benchmark dev set, NLI training (Table 6) | Mean 80.78, max 79.07, CLS 79.80 |

## Where the claims go beyond the evidence

- **Raw BERT is not a sentence embedder.** The paper's own baseline shows that averaging BERT's outputs (54.81) is worse than averaging GloVe vectors (61.32). The gain comes from the fine-tuning.
- **A cross-encoder still wins.** On the trained STS benchmark, BERT reading both sentences at once scores 88.77 against 86.10.
- **Domain transfer.** Across argument topics, SBERT drops "by about 7 points Spearman correlation". The authors say it needs more than two topics for training.
- **Not for transfer.** The authors write that "The purpose of SBERT sentence embeddings are not to be used for transfer learning for other tasks".

## Use in this repo

- **Our citation checks out, with one point of wording.** `papers/shared-concept-space` says Sentence-BERT gives sentence embeddings that can be compared with cosine similarity, which the abstract says. It calls `all-MiniLM-L6-v2` "one of them", distilled with MiniLM. That model isn't in this paper, which predates MiniLM. It is a later model from the same `sentence-transformers` library. Our reading: the sentence is fair, and the MiniLM review (#5423) will check the rest.
- **A mean of embeddings is our step.** The issue asks whether the mean of a pole's sentence embeddings places it by meaning. The paper pools tokens within one sentence. It never averages embeddings across sentences. `concept_map.py` takes the mean over a pole's top examples, after centring. Our reading: that mean puts a pole near the theme its examples share, but the source doesn't test it. `papers/shared-concept-space` reports its own check, that layouts keep about half of each point's neighbours.
- **The scorer uses the same model.** `explanation_scores.py` scores an explanation by its embedding similarity to held-out texts. The paper's STS results suggest the similarity tracks meaning at the sentence level. Its weak domain transfer suggests caution on code or other unusual text.
- **No question for the maintainer.**
- **Related pages.** [`umap.md`](umap.md) and [`leiden.md`](leiden.md) review the layout and the regions that work on these embeddings.

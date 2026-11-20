# The Pile: An 800GB Dataset of Diverse Text for Language Modeling

**Citation.** Twelve authors, of EleutherAI: Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe and Charles Foster. Then Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser and Connor Leahy. "The Pile: An 800GB Dataset of Diverse Text for Language Modeling". [arXiv:2101.00027](https://arxiv.org/abs/2101.00027), version 1 of 31 December 2020, the only version. Preprint, no venue given. The data is at [pile.eleuther.ai](https://pile.eleuther.ai/), and the code that builds it at [github.com/EleutherAI/the-pile](https://github.com/EleutherAI/the-pile). The paper states no funding. It thanks the TensorFlow Research Cloud for compute and OpenAI for API credits.

**Mismatch.** The title says 800GB, and the abstract says 825 GiB, which is the raw size before upsampling. The epoch-weighted size is 1,254.20 GiB. Section 6.5 says two sources are consistent with their host's terms of service. But Table 5 and the paragraph after it count them as not compliant. Section 3.1 says each of the validation and test splits is 0.1% of the data and over 1 GiB. But 0.1% of 825 GiB is about 0.8 GiB.

**Status.** Used. The registry's scan corpus, `corpus.jsonl`, is the first 500 documents of `NeelNanda/pile-10k`, a subset of the Pile ([`docs/experiments.md`](../experiments.md)). `papers/default-on-poles`, `papers/spd-input-projections` and `papers/why-neurons-rest` take their data from the same subset.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 1 for this page's author, a model (Claude Opus 5.5). It read Sections 1 to 9, the references and Appendices A to F in the source's own text. Only Table 17 and the page footer were left unread. The figures, including the profanity bars, weren't readable as text. It also read the dataset card of [`NeelNanda/pile-10k`](https://huggingface.co/datasets/NeelNanda/pile-10k).

## Summary

**Claim.** A language model trained on a mix of many high-quality sources generalises better across domains than one trained on web crawl alone. The Pile is such a mix: 22 sources, 825 GiB of mostly English text, with a datasheet-style account of its contents.

**Method.**

- **Sources.** 22 components, the largest being Pile-CC (227.12 GiB), Books3 (100.96), GitHub (95.16) and PubMed Central (90.27). Each is weighted by a number of epochs, from 1 to 3, so smaller high-quality sources are upsampled.
- **Processing.** Pile-CC is extracted from Common Crawl WARC files with jusText. A fastText quality classifier, trained on OpenWebText2, filters it, and only text that is mainly English is kept. Documents are deduplicated only within Pile-CC and OpenWebText2, with MinHash. There is no deduplication across the whole Pile.
- **Splits and order.** Validation and test splits are sampled uniformly at random. The training documents are drawn at random by component weight and interleaved into 30 output piles. So the training data is shuffled across those piles, not as one file.
- **Evaluation.** GPT-2 and GPT-3 perplexity on the Pile's test split, and 1.3B-parameter models trained on the Pile against CC-100 and raw Common Crawl.
- **Documentation.** Topic models per component, a profanity measure, co-occurrence tests of gender, religion and race, and a discussion of consent and copyright.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Raw size, 22 components (Table 1) | 825.18 GiB. Effective size with epochs 1,254.20 GiB |
| English share (Section 5.2) | 97.4% by fastText, "only a rough estimate" |
| Duplicates removed (Appendix C) | 28% of OpenWebText2 and 26% of Pile-CC |
| GPT-2 small, Pile test, bits per byte (Table 2) | 1.2253. GPT-3 davinci 0.7177 |
| 1.3B model trained on each, Pile test bits per byte (Table 3) | Pile 0.9433, raw Common Crawl 1.1275, CC-100 1.3293 |
| Sentiment of words co-occurring with race (Table 13) | White −0.114, Black −0.148, Asian −0.028, Hispanic −0.024 |
| Consent (Section 6.5, Table 5) | 5 components are neither within their host's terms nor consented, Books3 among them |

## Where the claims go beyond the evidence

- **"Diverse" is mostly English and mostly one era.** 97.4% of the text is English, by an estimate that the authors call rough.
- **No deduplication across the whole set.** Documents can repeat across components and across the train, validation and test splits. The authors say this themselves, and they didn't remove benchmark test sets from the training data.
- **The bias tests are co-occurrence counts.** They measure the text, not a model trained on it. The authors call the profanity measure a proxy.
- **Copyright is argued, not settled.** Section 7.1 argues fair use, with a disclaimer that it isn't legal advice. The authors say they lack the metadata to tell which texts are copyrighted.
- **The promised datasheet isn't in the paper.** It is cited as a separate document.

## Use in this repo

- **Our citations check out.** All three papers cite it only for the data. The reference gives the twelve authors in the source's order, and 2020 is the arXiv year. No change is needed.
- **"The first 10,000 documents" describes a derived set.** The papers say "the first 10,000 documents of the Pile", and `papers/spd-input-projections` names `NeelNanda/pile-10k`. That card says "The first 10K elements of The Pile". It doesn't say which file or split they come from, and the Pile's authors didn't publish it. Its first rows mix components, which fits the Pile's shuffled training files. So the phrase is fair, but the subset is the card author's, not Gao et al.'s.
- **The licence is the card's, not the Pile's.** `pile-10k`'s card gives `bigscience-bloom-rail-1.0`, a model licence. The Pile paper names no single licence for the data, and each component keeps its own terms. [`docs/experiments.md`](../experiments.md) already says so, and keeps the corpus out of the repository for that reason.
- **The scan corpus inherits its mix.** The registry's `corpus.jsonl` is 500 Pile documents, each cut to 3,000 characters. So its domains follow the Pile's weights: web crawl, papers, books, code and law. The repo's `mixed` blend, which adds other sources and languages, exists partly because one Pile sample ties statistics to that mix.
- **Its known flaws reach our corpus.** Duplicates across components, mostly English text, and the components without consent all come with the sample. The repo never filters a corpus for personal data, by policy. The site screens only what it shows.
- **No question for the maintainer.**
- **Related pages.** [`probing-classifiers.md`](probing-classifiers.md) and [`neurons-in-a-haystack.md`](neurons-in-a-haystack.md) also probe models trained on the Pile, as Pythia is.

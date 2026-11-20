# Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition

**Citation.** Erik F. Tjong Kim Sang and Fien De Meulder, of CNTS – Language Technology Group, University of Antwerp. "Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition". *Proceedings of the Seventh Conference on Natural Language Learning at HLT-NAACL 2003*, pages 142–147, Edmonton. [ACL Anthology W03-0419](https://aclanthology.org/W03-0419/). The preprint is [arXiv:cs/0306050](https://arxiv.org/abs/cs/0306050), version 1 of 12 June 2003, the only version.

**Mismatch.** None found. The paper spells the German newspaper "Frankfurter Rundshau".

**Status.** Used. The `conll2003-ner` and `conll2003-pos` presets of `openinterpretability/concepts.py` read the Hub copy `eriktks/conll2003`. `papers/default-on-poles` uses the named-entity labels in its decoding test.

**Read.** On 2026-11-20 a `web-fetch` subagent read the source for this page's author, a model (Claude Opus 5.5). It read the anthology page, the arXiv abstract page and the arXiv HTML of version 1. It read the whole body text in the source's own words. Table 3 came back garbled. Both PDFs came back as bytes that it couldn't read.

## Summary

**Claim.** A shared task for named entity recognition in English and German, with four types: persons, organisations, locations and miscellaneous names. Sixteen systems took part. The task also offered unannotated text, to test whether systems could learn from it.

**Method.**

- **English data.** Reuters news stories. Training and development text is from the end of August 1996, and test text from December 1996.
- **German data.** The Frankfurter Rundschau, from one week at the end of August 1992.
- **Labels.** "Named entity tagging … was done by hand at the University of Antwerp", mostly following MUC conventions, in the IOB scheme of Ramshaw and Marcus. The part-of-speech and chunk tags were automatic: "The English data was tagged and chunked by the memory-based MBT tagger." TreeTagger tagged the German data.
- **Measure.** F1 on the test set.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| English sizes, train / dev / test tokens (Table 1) | 203,621 / 51,362 / 46,435 |
| German sizes, train / dev / test tokens (Table 1) | 206,931 / 51,444 / 51,943 |
| Best English F1 (Table 5) | 88.76 ± 0.7 (Florian et al.). The baseline is 59.61 |
| Best German F1 (Table 5) | 72.41 ± 1.3 (Florian et al.). The baseline is 30.30 |
| Majority vote of selected systems (Section 5) | 90.30 English, 74.17 German |

## Where the claims go beyond the evidence

- **No measure of annotation quality.** The entity tags were made by hand, but the paper reports no agreement between annotators and no error rate.
- **Narrow time windows.** Each language's text comes from a few days of one news source. Test text is a few months later than training text for English.
- **Automatic tags look like gold tags.** The part-of-speech and chunk columns come from taggers, not people. The paper says so, but the released files don't mark the difference.
- **No licence terms.** The paper points to Reuters and the LDC for the text, and states no terms of its own.

## Use in this repo

- **Our citation checks out.** `papers/default-on-poles` lists CoNLL-2003 among its "human labels". It uses only `conll2003-ner`, and the source says people made those tags. The authors, venue and pages match. No paper change is needed.
- **The part-of-speech preset's comment needed a fix.** `concepts.py` called the `conll2003-pos` tags a source "independent of Universal Dependencies' annotators". The source says the MBT tagger made them. This PR rewrites the comment to say that a tagger made them. Our reading: a test of a neuron against these tags measures agreement with a 2003 tagger, not with people.
- **The text can't be published.** `concepts.py` already says the text is Reuters RCV1, for research use, so excerpts stay out of the wiki. The source points to Reuters for the text, which fits.
- **One genre, one year.** The English text is 1996 newswire. A neuron that tracks entities in it may track a news style, not entities in general. Our reading. The default-on-poles paper reports scores, not claims about particular neurons, so this doesn't change it.
- **No question for the maintainer.**
- **Related pages.** [`universal-dependencies-v2.md`](universal-dependencies-v2.md) reviews the other source of part-of-speech labels, made by people.

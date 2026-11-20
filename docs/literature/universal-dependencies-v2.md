# Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection

**Citation.** Joakim Nivre (Uppsala University), Marie-Catherine de Marneffe (The Ohio State University), Filip Ginter (University of Turku), Jan Hajič (Charles University in Prague), Christopher D. Manning (Stanford University), Sampo Pyysalo (University of Turku), Sebastian Schuster (Stanford University), Francis Tyers (Indiana University) and Daniel Zeman (Charles University in Prague). "Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection". *Proceedings of the Twelfth Language Resources and Evaluation Conference (LREC 2020)*, pages 4034–4043, Marseille, May 2020, European Language Resources Association. [ACL Anthology 2020.lrec-1.497](https://aclanthology.org/2020.lrec-1.497/). The preprint is [arXiv:2004.10643](https://arxiv.org/abs/2004.10643), version 1 of 22 April 2020, the only version. The paper states no funding. It thanks "all the 345 UD treebank contributors".

**Mismatch.** None found. We read the arXiv version, not the LREC PDF, so we didn't check the two against each other.

**Status.** Used. The `ud-upos` and `ud-deprel` presets of `openinterpretability/concepts.py` label tokens with UD parts of speech and relations, from the English Web Treebank by default. `papers/default-on-poles` uses UD labels in its decoding test.

**Read.** On 2026-11-20 a `web-fetch` subagent read the anthology page, the arXiv abstract page and the arXiv HTML for this page's author, a model (Claude Opus 5.5). It read the whole text in the source's own words, in two fetches: Sections 1 to 6 and the references. The LREC PDF came back as bytes that it couldn't read. Figures 1 and 2 are images, and Tables 1 and 2 came back flattened, so this page quotes their prose only.

## Summary

**Claim.** Universal Dependencies is one annotation scheme for the grammar of many languages: parts of speech, morphological features and dependency relations. Version 2 of its guidelines changes many decisions of version 1, and release 2.5 covers 157 treebanks in 90 languages.

**Method.** A description of the scheme and of the collection, not an experiment.

- **Words.** Syntactic words, not orthographic tokens, so clitics are split.
- **Labels.** A fixed list of 17 universal parts of speech (UPOS), a set of morphological features, and 37 universal relation types. Languages may add subtypes, such as `nsubj:pass`.
- **Changes from v1.** For example, `dobj` becomes `obj`, `CONJ` becomes `CCONJ`, and clause-level obliques move from `nmod` to `obl`. `neg` is removed, and `cc` attaches to the conjunct after it.
- **Enhanced dependencies.** Five optional extra layers, such as propagated conjuncts and controlled subjects.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Release 2.5 (Section 4) | 157 treebanks in 90 languages |
| Growth since v1.2 (Section 1) | The number of treebanks "has more than quadrupled". Contributors "now exceeds 300" |
| Language families (Table 4) | 20 families. Indo-European holds 48 of 90 languages, down from 60% in v2.1 to 53% in v2.5 |
| Enhanced dependencies (Section 3.4) | Only 24 treebanks have them in v2.5 |
| Genres (Table 5) | News is the largest with 98 treebanks. Web text is in 9, and reviews in 7 |

## Where the claims go beyond the evidence

- **"Universal" is a goal, not a measurement.** The paper reports no study of how consistently treebanks apply the guidelines. The authors say themselves that consistency "will continue to be a challenge for years to come".
- **No measure of annotation quality.** The paper gives no inter-annotator agreement.
- **Most treebanks are small, and the sample leans Indo-European.** The authors say both.
- **No licence terms.** The paper doesn't discuss licences. Each treebank has its own.

## Use in this repo

- **Our citation checks out.** `papers/default-on-poles` cites the paper only as the source of its UD labels. The nine authors, LREC 2020 and pages 4034–4043 match the anthology page. No change is needed.
- **The licences come from the treebanks, not the paper.** `concepts.py` gives en_ewt as CC BY-SA 4.0 and en_gum as CC BY-NC-SA 4.0. The paper names neither, so those terms come from each treebank's own files, which this review didn't read.
- **The release isn't pinned to the paper's.** The presets read `universal-dependencies/universal_dependencies` on the Hugging Face Hub with no fixed revision. Each load records the commit it read (`concepts.py`), so a run can be traced. The paper describes release 2.5, and this review didn't check which UD release that repository holds.
- **Web text is a small genre in UD.** The repo labels English Web Treebank text, one of the 9 web treebanks. The scan corpus is mostly web and books (see [`the-pile.md`](the-pile.md)). So UD labels fit the genre of the text the registry reads.
- **No question for the maintainer.**
- **Related pages.** [`control-tasks.md`](control-tasks.md) and [`probing-classifiers.md`](probing-classifiers.md) review probing for the kind of labels UD gives.

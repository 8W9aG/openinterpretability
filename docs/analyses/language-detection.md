# Choosing a language detector for unlabelled corpora

`openinterp corpus --detect-lang` labels each document of a corpus whose dataset doesn't say its language (FineWeb, the Pile), so the scan's language confound, the agents' per-example languages and the site's language mix work there too (docs/PROTOCOL.md §4, *Label confounds*). This note is how the detector and its confidence line were chosen. Script: `scripts/language_detection.py`; rows: `language-detection.jsonl`. Run 2026-11-01.

## Method

Four detectors, each run per document as `openinterp corpus` would run it:

- **fastText `lid.176.ftz`** (176 languages, the 917 KB compressed model; through `fasttext-predict`, as `openinterp/langid.py` runs it);
- **lingua** (`lingua-language-detector`, all languages, high-accuracy mode);
- **langdetect**, seeded (`DetectorFactory.seed = 0`); unseeded, it gave a different answer on 7 of 200 50-character documents when run twice;
- **langid.py**, with normalised probabilities.

Test sets, built with `openinterp corpus` so their labels are exact (each document's Wikipedia subset, `--lang-pattern`):

- *Distinct*: the `wikipedia-multilingual` preset, 1,000 articles, 100 each in en, de, fr, ru, el, ar, hi, zh, ja and ko (eight scripts).
- *Close*: 400 articles, 50 each in es, pt, it, ca, nl (Latin) and uk, bg, ru (Cyrillic), languages that share a script with a near neighbour.
- *Mixed*: 200 documents of English followed by another language from either set, English making up a quarter, half or three quarters of 1,500 characters. Right is the majority language (either, at half).
- *Code*: 200 files of the `github-code` preset, which a detector should be unsure of.
- *Registry*: the registry's 500-document pile-10k corpus (`corpus.jsonl`), unlabelled, for agreement and for what each detector makes of it.

Articles are cut to 3,000 characters, as corpora are. Each set is also scored on its first 50 and 200 characters, about the length of a scan window, to show why detection runs per document.

## Results

Accuracy of the top label:

| Detector | Distinct, 50 chars | 200 | Whole | Close, 50 chars | 200 | Whole | Mixed | ms per doc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fastText lid.176 | **0.982** | **0.997** | **0.997** | **0.913** | **0.998** | **0.998** | 0.78 | **0.12** |
| lingua | 0.864 | 0.961 | 0.954 | 0.735 | 0.963 | 0.975 | 0.83 | 1.7 |
| langdetect (seeded) | 0.930 | 0.969 | 0.957 | 0.885 | 0.988 | 0.983 | **0.87** | 1.8 |
| langid.py | 0.913 | 0.993 | 0.990 | 0.710 | 0.953 | 0.973 | 0.66 | 1.2 |

Abstaining below a confidence line, over the distinct and close sets together (1,400 articles) and the 200 code files:

| Detector | Line | Articles kept | Accuracy of kept | Code files labelled |
| --- | --- | --- | --- | --- |
| fastText lid.176 | 0.5 | 99.6% | 99.93% | 44.5% |
| fastText lid.176 | 0.7 | 98.9% | 99.93% | 20.5% |
| fastText lid.176 | **0.8** | **98.5%** | **99.93%** | **9.0%** |
| fastText lid.176 | 0.9 | 94.6% | 99.92% | 1.0% |
| lingua | 0.8 | 97.4% | 97.1% | 94.5% |
| langdetect | 0.8 | 97.1% | 97.4% | 93.0% |
| langid.py | 0.8 | 99.6% | 98.6% | 94.0% |

On the registry corpus the four detectors agree on 96–98% of documents. All call 472–482 of the 500 English.

## Decision

**fastText `lid.176.ftz`, labelling a document only at confidence ≥ 0.8** (`openinterp/langid.py`: `MIN_CONFIDENCE`, with the model's URL and SHA-256 pinned).

- It is the most accurate on every set at every length, and 10–15 times faster.
- Only its confidence separates prose from code. The other three are as sure of a C header's "language" as of a Wikipedia article's, so no line can make them abstain on code. At 0.8, fastText labels 9% of code files and keeps 98.5% of articles. It loses no accuracy on those it keeps: three of its four errors on the articles (two Chinese articles and one Catalan, at confidence 0.14–0.46) fall below the line. The fourth, a Hindi article labelled English at 0.97, stays.
- Its model is CC BY-SA 3.0 and `fasttext-predict` is MIT. It is deterministic, and needs no numpy.

On the registry corpus, the 0.8 line labels 416 documents English and leaves 71 unlabelled, mostly code, Stack Exchange posts with code, and maths exercises. The other 13 are labelled in 10 other languages (pt, nl and fr 2 each; es, no, da, tr, el, ja and de 1 each). Spot checks of the Spanish, Norwegian and Danish labels were all right.

## Limits

- **One label per document.** Mixed documents get their majority language at best: fastText picks it 78% of the time. A scan's tokens in a document's minority language are mislabelled. Labelling paragraphs or sentences would need spans and a per-span detector, and short spans are exactly where detection fails (the 50-character columns).
- **The test languages are Wikipedia's.** Encyclopaedic prose in 16 languages is easier than web text. lid.176 covers 176 languages, but this note measured 16.
- **Code is left out, not labelled.** Its programming language is a separate question (TODO, *Open question: can we detect programming languages in unlabelled corpora?*).
- **The registry's own corpus is not relabelled.** Adding `lang` to `corpus.jsonl` would change its bytes and SHA-256, which every record scanned on it pins. A labelled copy would be a new corpus, which is a decision for the registry, not for this tool.

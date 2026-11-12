# Choosing a programming-language detector for unlabelled corpora

`openinterp corpus --detect-code` labels each document of a corpus whose dataset doesn't say its programming language. pile-10k is an example: it mixes GitHub files and Stack Exchange posts into prose, and no field marks them. With the label, the `programming language` label set and the syntax label sets (`openinterp/syntax.py`) work on such corpora too (docs/PROTOCOL.md §4). This note is how the detector and its confidence line were chosen. It answers the TODO item *Open question: can we detect programming languages in unlabelled corpora?* Script: `scripts/code_detection.py`. Rows: `code-detection.jsonl`. Run 2026-11-12.

## Method

Four detectors, each given only the document's text. A corpus has no file names, so no detector can use an extension.

- **Magika** (`magika` 1.0.3, model `standard_v3_3`), Google's file-type model. It is scored three ways: its top label with a confidence line, its own per-type thresholds, and as `openinterp/codeid.py` runs it.
- **go-enry** 2.9.1, the Go port of GitHub Linguist (`scripts/code_detection_enry`). It applies Linguist's shebang and modeline rules, then its Bayesian classifier over every language it knows. It gives no confidence.
- **CodeBERTa language-ID** (`philomath-1209/programming-language-identification`), a transformer classifier of 26 languages, with its softmax score.
- **Pygments** `guess_lexer`, the usual cheap baseline, with the lexer's `analyse_text` score.

guesslang was not tested. It pins TensorFlow 2.5, which has no wheels for Python 3.12 or for arm64.

Each detector's labels are renamed to the github-code preset's names (`corpus_presets.GITHUB_CODE_LANGUAGES`, for example `cpp` to `C++`). A label that claims no language (`txt`, `unknown`, `Text only`) counts as abstaining.

Test sets:

- *Code*: 1,000 files of the `github-code` preset in 28 languages, labelled exactly by the dataset's `language` column. Java has 166 files, C has 136, and seven languages have one to three. Each file is scored whole (at most 3,000 characters), and on its first 200 and 50 characters.
- *Prose*: 300 documents of the `fineweb-edu` preset. A detector should label none of them.
- *Mixed*: 200 documents of prose followed by code in a language with a tree-sitter grammar, like a Stack Exchange post. The code makes up a quarter, half or three quarters of 1,500 characters.
- *Registry*: the registry's 500-document pile-10k corpus (`corpus.jsonl`). Its texts match pile-10k's first 500 rows exactly, so each document's Pile subset (`pile_set_name`) says which are GitHub files (37) and Stack Exchange posts (76).

## Results

Accuracy of the top label on the code files:

| Detector | 50 chars | 200 chars | Whole | Whole, languages with a grammar | ms per doc |
| --- | --- | --- | --- | --- | --- |
| Magika | **0.554** | **0.704** | **0.915** | **0.932** | 5.8 |
| Magika, own thresholds | 0.484 | 0.618 | 0.895 | 0.924 | 4.7 |
| go-enry (Linguist) | 0.194 | 0.283 | 0.628 | 0.663 | 6.5 |
| CodeBERTa language-ID | 0.327 | 0.414 | 0.673 | 0.841 | 87 |
| Pygments | 0.102 | 0.133 | 0.122 | 0.096 | **3.6** |

Magika's main error is C called C++ (29 of its 85 errors), mostly header files. Next come HTML templates called `twig` or `handlebars` (11). CodeBERTa has no HTML, Markdown, CSS, Shell or TypeScript class. So it calls 75 HTML files JavaScript and 30 Markdown files Pascal. Pygments calls 132 Java files Python.

Abstaining below a confidence line:

| Detector | Line | Code files kept | Accuracy of kept | Prose labelled |
| --- | --- | --- | --- | --- |
| Magika | 0.5 | 97.7% | 92.7% | 0 of 300 |
| Magika | 0.7 | 92.3% | 94.4% | 0 |
| Magika | **0.8** | **88.9%** | **95.5%** | **0** |
| Magika | 0.9 | 81.6% | 97.1% | 0 |
| Magika, own thresholds | — | 95.7% | 93.5% | 0 |
| go-enry | — | 99.9% | 62.9% | 300 |
| CodeBERTa language-ID | 0.8 | 71.3% | 84.4% | 56 |
| CodeBERTa language-ID | 0.95 | 57.8% | 92.0% | 7 |
| Pygments | 0.8 | 30.0% | 19.0% | 8 |

Magika labelled none of the 300 prose documents at any line. go-enry labelled all of them, mostly as `Muse` and `RUNOFF`. CodeBERTa has no class for prose, so it called 168 of them COBOL and 72 FORTRAN.

Documents of prose followed by code, labelled with the code's language (67, 67 and 66 documents per share):

| Detector | Code a quarter | Half | Three quarters |
| --- | --- | --- | --- |
| Magika, any confidence | 45% | 66% | 79% |
| Magika, confidence ≥ 0.8 | 31% | 43% | 62% |
| CodeBERTa language-ID, any confidence | 55% | 63% | 71% |
| go-enry | 0% | 4% | 17% |
| Pygments | 1% | 7% | 12% |

On the registry corpus, at confidence 0.8 where the detector has one:

| Detector | Documents labelled | GitHub (of 37) | Stack Exchange (of 76) | Other subsets (of 387) |
| --- | --- | --- | --- | --- |
| Magika | 53 | 29 | 13 | 11 |
| Magika, as `openinterp.codeid` runs it | 37 | 17 | 10 | 10 |
| go-enry | 494 | 37 | 75 | 382 |
| CodeBERTa language-ID | 144 | 19 | 27 | 98 |
| Pygments | 37 | 10 | 6 | 21 |

Spot checks of Magika's registry labels outside the GitHub subset:

- Ten Stack Exchange posts have code. Each got its code's language: Python, PHP, Swift, SQL, R, Java or JavaScript. Three maths posts got TeX.
- Nine documents from ArXiv, PubMed Central, HackerNews and Books3 got Markdown. The Pile converted these to pandoc Markdown (`---` front matter, `{#sec1}` anchors, `======` rules), so the label is right.
- One OpenSubtitles document of quoted lines got `csv`.

Eight GitHub files fall below 0.8. Four are ambiguous: a test fixture of letters, a NetBSD file list, a dump of a UIView tree and a Gradle properties file. The other four are an HTML error page, a two-line options file, and a C and a C++ file at confidence 0.63 and 0.70.

## Decision

**Magika, labelling a document only at confidence ≥ 0.8 and only with a github-code language** (`openinterp/codeid.py`: `MIN_CONFIDENCE`, `MAGIKA_LABELS`).

- It is the most accurate on code at every length.
- It is the only detector that labels no prose. Linguist needs a file name to narrow its candidates. Without one, it gives every document some language. CodeBERTa has no class for "not code". Magika has a `txt` class, and it uses it.
- 0.8 is the line that `langid` uses. It keeps 89% of code files at 95.5% accuracy, against 92.7% at 0.5. Prose sets no lower bound, because no line labels any of it.
- It records only the 30 names of the github-code preset (`GITHUB_CODE_LANGUAGES`). So a detected corpus and the `github-code` preset give one language one label, and blends can mix them. At this setting it labels 37 of the registry's 500 documents. It labels no plain prose.
- Its model ships inside the `magika` package (Apache-2.0). The source sidecar records the package version and the model name. It runs on CPU with onnxruntime, at about 5 ms per document. On the registry corpus that adds 2–3 seconds.

## Limits

- **One label per document.** A Stack Exchange post labelled Python is parsed as Python, prose and all, so the syntax label sets label its prose tokens as code. Below about half code, most mixed documents stay unlabelled (31% labelled at a quarter code). Labelling code blocks as spans would need a span detector, and at 50 characters Magika is right on only 55% of files.
- **Only github-code's 30 languages.** Magika also names Kotlin, Swift, Dart, R and others, and data formats such as JSON, XML and YAML. These stay unlabelled. Twelve of the registry's 29 GitHub files that Magika is sure of are of these types.
- **C and C++ are often confused.** A header file is valid in both, and Magika calls 29 of 136 C files C++. Both languages have grammars, and their node names sort into the same syntax labels, so the syntax label sets change little.
- **Markdown is not code.** Pandoc-converted papers and threads get `Markdown`, which is right but is not source code. Markdown has no grammar here, so these documents get no syntax labels. They still count in the `programming language` label set.
- **The test files are github-code's.** That dataset was filtered for clean, mostly-alphanumeric files, so they are easier than raw GitHub text. Seven of its languages have three files or fewer in the sample.
- **The registry's own corpus is not relabelled.** Adding `code_language` to `corpus.jsonl` would change its bytes and SHA-256, which every record scanned on it pins. A labelled copy would be a new corpus, which is a decision for the registry, not for this tool.

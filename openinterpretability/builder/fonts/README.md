# Fonts of the paper PDFs

The paper PDFs are set in Computer Modern, the typeface of LaTeX and so of most arXiv papers (#4307). The files are CMU, the Computer Modern Unicode fonts by Andrey V. Panov, under the SIL Open Font License 1.1 ([OFL.txt](OFL.txt)). They are unmodified.

They come from the npm package [`computer-modern`](https://www.npmjs.com/package/computer-modern) 0.1.3 (Steven Petryk), at `fonts/<name>.woff2`:

| File | Use |
| --- | --- |
| `cmu-serif-500-roman.woff2` | text |
| `cmu-serif-500-italic.woff2` | italic text |
| `cmu-serif-700-roman.woff2` | bold text and headings |
| `cmu-serif-700-italic.woff2` | bold italic text |
| `cmu-typewriter-text-500-roman.woff2` | code |

`latinmodern-math.otf` is Latin Modern Math, the OpenType version of Computer Modern's maths fonts, from CTAN (`fonts/lm-math/opentype/`), under the GUST Font License ([GUST-FONT-LICENSE.txt](GUST-FONT-LICENSE.txt)). It is unmodified too. It gives the symbols that the CMU text faces lack, such as ≈, ↔ and ′. No Computer Modern face has superscript digits, so the print page raises plain digits, as LaTeX does.

`openinterpretability/builder/paper_pdf.py` embeds them in the print page as `data:` URIs, so the PDF depends on no system font.

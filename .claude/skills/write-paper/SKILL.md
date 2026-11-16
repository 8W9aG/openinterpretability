---
name: write-paper
description: Write or revise a paper in papers/, with checked references, figures and animations tied to its results, and a place in the site's Research section. Use when an analysis is ready to publish, when the user asks for a paper or write-up, or when reviewing a paper's citations or figures.
---

# Write a paper

A paper states what was measured, what that does and doesn't show, and how to reproduce it. Readers can check each claim against its data or its source.

## Layout

```text
papers/<slug>/
├── README.md       the paper; the site renders it, and GitHub shows it in the folder
├── results.jsonl   the recorded output
└── media/          figures, written by scripts/paper_figures.py
```

List the paper in `papers/README.md`. That order is the order of the site's Research index ([open-interpretability.com/research](https://open-interpretability.com/research/index.html)). A folder with an `index.html` is an interactive page, and the site copies it unchanged.

## Structure

1. A level 1 title, then one paragraph that can stand alone. The Research index shows this paragraph, so don't start it with a citation or a path.
   Then one paragraph that starts **Why we wrote it.**, and says what problem made the paper necessary. In the same paragraph, **The question.** states the question that the paper answers, and links it to its research question in `docs/research-questions.md`. `tests/test_papers.py` checks it.
2. **Results**: numbered findings, each with its limit. End with what the results do not show.
3. **Method**, then a section for each finding. Tables hold the exact numbers.
4. **Caveats**, **Reproduce** (the commands, with the seeds) and **Next**.
5. **References**.

## Academic language

Write in a formal, academic register. Use the third person or "we", not "I" or "you". Use no contractions, slang or casual phrases. Make each claim as strong as its evidence, and no stronger: say "suggests" for a hint, and "shows" only for a measured result. The STE limits of `make ste` still hold, so keep sentences short and plain.

## Citations

Cite each outside idea, method, model and dataset where the paper uses it. Do this before the paper is finished, not after it.

1. **Read the source.** Find the passage that supports the claim, in the paper itself. A summary, an abstract or a model's memory isn't enough. Web pages can fail to load, so download the PDF if you must. Record a failed read, and don't cite what you couldn't read.
2. **Cite the claim, not only the work.** Check who said what. For example, Sun et al. (2024) report massive activations, and Xiao et al. (2023) name attention sinks. Citing one for the other's idea is an error.
3. **Mark your own steps.** When you take an idea further than its source, say that the step is yours. A toy model's result is not a claim about GPT-2.
4. **Check the details.** Take the title, the authors, the year, the venue and the identifier from the source's own page. Give a link in each reference.
5. **Cite data with its version and licence.** Give the dataset, the commit that was used, and what its card says about the licence. If the card names no citation or no licence, say so.
6. **Point to work in this repo with a link.** For notes or earlier studies that have since moved, link to a commit, so that the text stays readable. Don't describe an earlier study without saying where its numbers are.

Citations and references follow APA 7th edition.

- **In the text,** give author and year: `(Hendrycks & Gimpel, 2016)` in parentheses, and "Hendrycks and Gimpel (2016)" in a sentence. With three or more authors, write "Sun et al. (2024)".
- **In the list,** name every author, up to 20, with a comma before the final "&". With 21 or more, give the first 19, then `. . .` and the last.
- **The title** is in italics and in sentence case. Keep capitals only for the first word, the first word after a colon, proper nouns and acronyms.
- **After the title,** give the venue, then the link.

```markdown
- Hendrycks, D., & Gimpel, K. (2016). *Gaussian error linear units (GELUs)*. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415).
```

`tests/test_papers.py` fails if a citation has no reference, or a reference has no citation or no link. It also fails on a reference that isn't in this form. It can't tell if a source supports a claim, so step 1 remains your work.

## Figures and animations

Each paper needs figures that let a reader see the argument. Be imaginative, and let the point choose the form.

- **Draw from the data.** Add figures through `scripts/figure_svg.py`, and register them in `scripts/paper_figures.py`. A test fails if a figure differs from the results. Run that script after each change to the results.
- **Put the finding in the title** of each figure. The subtitle says what the marks are.
- **Animate to make a point.** Use a short loop inside a figure when change is the argument, for example neurons that move as training runs. Don't make a full video unless asked. Under reduced motion, a figure shows its final frame.
- **Use depth for a third dimension.** A surface over layers and steps earns its depth. A 3D bar chart doesn't.
- **Use an interactive page for what a reader must explore**, because Markdown can't run it.
- **Be honest about the marks.** Say when a figure is an illustration with made-up values, or a sketch from summary statistics. Say when only the two ends of an animation are measured.
- **Keep colours consistent.** One side or route has one colour through the paper. Check new colours with the `dataviz` skill.
- **Look at each figure** in light and dark, and mid-animation, before you call it done.

## Before you finish

- Each number in the prose matches `results.jsonl`, and the figures agree with the tables.
- Update the question that the paper bears on in `docs/research-questions.md`: its *So far* line, with the date.
- Run `make lint` (it includes Vale) and `make test`.
- Build the site with `make build`, and open the Research index and the paper.
- Open the PR ready for review, not as a draft. Merge it when CI passes, as for every other PR. Do not wait for a person to read the paper first.

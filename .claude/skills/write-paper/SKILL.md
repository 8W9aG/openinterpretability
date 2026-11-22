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

List the paper in `papers/README.md`, with the date it is published before it: `- 2026-11-16: [Title](slug/README.md)`. The site's Research index shows the papers newest first by that date, and keeps the list's order for one date ([open-interpretability.com/research](https://open-interpretability.com/research/index.html)). A folder with an `index.html` is an interactive page, and the site copies it unchanged.

## Authors

The maintainer is the first author of every paper, and Open Interpretability is the second. The maintainer decided this on 2026-11-19 (#5260). `scripts/papers.py` writes both into each paper's `CITATION.cff`, and takes the maintainer's name from the root `CITATION.cff`. The site shows them under the title. Never add an agent or a model as an author. Say in the paper's text which models proposed, reviewed, wrote and narrated the work.

## Structure

A paper has the arc of the papers we review, in this order. `tests/test_papers.py` checks the fixed sections and their order.

1. **Title and abstract.** A level 1 title, then one paragraph that can stand alone. It states the problem, the approach, the key numbers and what they imply. The Research index shows it, so don't start it with a citation or a path, and give it no heading.
2. **Introduction.** Motivate the question for the field, not for the registry. Say what earlier work leaves open. Then give a bulleted list of the paper's contributions, and link the question to `docs/research-questions.md`.
3. **Related work.** At least one paragraph that places the paper among the outside work it cites. Say where it agrees and where it differs. The reviews in `docs/literature/` are the first place to look.
4. **Method.** Define each term and the notation once, then the setup: model, units, data and measures.
5. **One section per experiment**, titled by what it asks. Each one gives the question, the setup, the result with its table or figure, and how to read it.
6. **Discussion.** What the results mean together, the other explanations, and the links to the literature.
7. **Limitations**, then **Conclusion**: a short paragraph that answers the question.
8. **References**, then **Appendices**: Reproduce (the commands, with the seeds), per-unit tables and long derivations.

Keep the main body, up to References, to about 3,000–6,000 words. Long material goes in the Appendices.

## Prose

Write in a formal, academic register. Use the third person or "we", not "I" or "you". Use no contractions, slang or casual phrases. Make each claim as strong as its evidence, and no stronger: say "suggests" for a hint, and "shows" only for a measured result.

- **Argue in paragraphs.** Each paragraph opens with a topic sentence, gives the evidence, and ends with what the evidence implies. Use bullets only for real lists: contributions, setups and limitations. Results, Discussion and Conclusion are prose.
- **Relate the claims.** A sentence may relate two claims with "because", "whereas", "although" or "which suggests". Papers get their own STE profile (`scripts/ste_prose.py`): a sentence may have up to 40 words, and a paper's mean must stay at 25 or fewer. Every other STE rule still holds.
- **Define the repo's terms.** Explain each term the first time it appears, in plain words: a pole, VERIFIED, the review's gates, the activation line at 0.25 of the maximum. Then use it as defined. Keep code names and file paths out of the argument. They belong in the Method's setup and the Appendices.
- **Number the figures and tables.** Give each one a caption that starts **Figure 1.** or **Table 1.**, and that a reader can follow alone. Refer to each one in the text by its number.
- **Read the models.** Shabalin & Piotrowski's *Many Transcoder Features Are Neurons in Disguise* (`docs/literature/neurons-in-disguise.md`) and Oikarinen et al.'s *Evaluating Neuron Explanations* (`docs/literature/evaluating-neuron-explanations.md`) show the arc and the register.

**Before and after.** The old form read like a list:

```text
At step 16,000, the median correlation of a pole's activations with its final
activations is 0.48, with a range of 0.29 to 0.74. It is 0.72 at step 64,000 and 0.93
at step 128,000.
```

The new form argues:

```text
Although the loss is within 1% of its final value by step 16,000, the poles are far
from their final form at that step: their activations correlate with their final
activations at a median of only 0.48 (Table 1). The correlation keeps rising, to 0.72
at step 64,000 and 0.93 at step 128,000, which suggests that the poles keep changing
long after the loss settles.
```

## Mathematics

Write maths in LaTeX. The site typesets it at build time (`openinterpretability/builder/maths.py`), and GitHub renders it too. A formula that the site can't read stops the build.

- **Write each equation, formula and maths symbol in LaTeX.** Use `$...$` inline: `$\mathrm{silu}(g) \cdot u$`, `$1.3 \times 10^{-5}$`, `$\sigma$`, `$R^2$`, `$\alpha = -0.5$`.
- **Give each definition that the method depends on a block equation of its own**, such as the pole, the clamp or the KL metric. Put `$$` on its own line before and after the formula.
- **Define each symbol the first time it occurs**, in the sentence that uses it.
- **Keep plain numbers as text**, in prose and in tables: "12 layers", "3.72 nats", "0.28–0.82". A ratio such as "1.9×" is a plain number too. A number in scientific notation is maths: `$2.6 \times 10^{-5}$`.
- **Keep code identifiers in backticks**, not in LaTeX: `silu`, `--clamp`.
- **Don't use `$` for anything else** outside code. Write a shell variable such as `$OUT` in backticks or a code block.

## Citations

Cite each outside idea, method, model and dataset where the paper uses it. Do this before the paper is finished, not after it.

1. **Read the source.** Find the passage that supports the claim, in the paper itself. A summary, an abstract or a model's memory isn't enough. Web pages can fail to load, so download the PDF if you must. Record a failed read, and don't cite what you couldn't read.
2. **Cite the claim, not only the work.** Check who said what. For example, Sun et al. (2024) report massive activations, and Xiao et al. (2023) name attention sinks. Citing one for the other's idea is an error.
3. **Mark your own steps.** When you take an idea further than its source, say that the step is yours. A toy model's result is not a claim about GPT-2.
4. **Check the details.** Take the title, the authors, the year, the venue and the identifier from the source's own page. Give a link in each reference.
5. **Cite data with its version and licence.** Give the dataset, the commit that was used, and what its card says about the licence. If the card names no citation or no licence, say so.
6. **Point to work in this repo with a link.** For notes or earlier studies that have since moved, link to a commit, so that the text stays readable. Don't describe an earlier study without saying where its numbers are.
7. **Each cited work needs a review page.** Each reference that is a paper or a post needs a page in `docs/literature/` whose Citation gives the same arXiv ID, DOI or link. If it has none, open an issue for it in the form of #5312, with `area: literature` and `priority: high`. Then add the work to `papers/awaiting_review.txt` with that issue's number, before the paper merges. A dataset revision needs no page, but the paper that describes the dataset does. `tests/test_papers.py` fails on a cited work that has neither (`scripts/cited_works.py`, #5346).

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
- **Animate at least one figure.** Each paper needs at least one animated figure. Make it with the `paper-animation` skill, which gives the forms, the shared helper and the checks. A paper with no change, order or cause to show says why in its Method, in one sentence. Don't make a full video unless asked.
- **Use depth for a third dimension.** A surface over layers and steps earns its depth. A 3D bar chart doesn't.
- **Use an interactive page for what a reader must explore**, because Markdown can't run it.
- **Be honest about the marks.** Say when a figure is an illustration with made-up values, or a sketch from summary statistics. Say when only the two ends of an animation are measured.
- **Keep colours consistent.** One side or route has one colour through the paper. Check new colours with the `dataviz` skill.
- **Look at each figure** in light and dark, and mid-animation, before you call it done.
- **Draw the head figure for X.** The first figure is the paper's picture on X and in link cards. After each change to it, run `uv run --with playwright==1.65.2 python scripts/social_images.py`, and commit `media/social.*` (#5534). A test fails while the drawing is out of date.
- **Write the paper's X thread** in `thread.yaml`, in the same PR (#5833). The thread posts when the paper is published, and no person reads it first, so the checks are the gate. Write 2 to 10 tweets, in the paper's order:
  - `section`: the `##` heading the tweet is about, as written.
  - `text`: one result, with numbers from the paper only, in 280 characters or fewer. Each link counts as 23. `{link}` is the paper's URL, and `{section_link}` the section's. The first tweet is the hook, the title's claim with `{link}`. The last tweet links to the paper again.
  - `media` (optional): the ids of the section's figures, the stems of their SVGs (`trace` for `media/trace.svg`). Use at most 4 images, or one figure that animates, alone. Each figure needs its alt text in the README.

  Then run `scripts/social_images.py` again, which draws each named figure as `media/social-<id>.png` (and `.mp4`). Commit those files too. `make build` and `make test` fail on a broken rule. `open-interpretability social post --preview 1` prints the thread with each tweet's length.

- **The X link is metadata.** Once a published paper posts to X, the deploy opens a PR that writes the post's URL as `x_url` in `media/social.json` (#5840). The page then shows "Thread on X". That PR changes no text of the paper, so it isn't a change to a frozen paper.

## Status: preprint or published

The maintainer decided on 2026-11-17 that every paper is either a **preprint** or **published**, and that the reader must see which (#3150). The status line under the abstract says it, and the site shows it beside the title.

- **A preprint** has work left. Its line names the open issues that would change it: `**Status:** Preprint. Work left: #N (what), #M (what).` A new paper starts as one:
  ```bash
  uv run python scripts/papers.py new <slug> --work "#N (what is left)."
  ```
  This also writes its `CITATION.cff`. When an issue on the paper closes, take it off the line in the PR that closes it. Name any follow-up issue that changes the paper. CI fails that PR if the line still names the issue (`scripts/papers.py check --closing`). A nightly run fails if a line names an issue closed some other way (`check --online`, #5395).
- **Publish it when no work on it is left**, in the PR that closes its last issue:
  ```bash
  uv run python scripts/papers.py publish <slug>
  ```
  The line then gives the date, and `CITATION.cff` holds the SHA-256 of the text and the results files.
- **A published paper doesn't change.** A later correction goes in a `## Addenda` section at the end, one dated `### YYYY-MM-DD` entry each. New work becomes a new paper that cites it. `make lint` (`scripts/papers.py check`) fails if the text above the addenda, or a results file, changes.

## Before you finish

- Each number in the prose matches `results.jsonl`, and the figures agree with the tables.
- At least one figure moves (`paper-animation`), or the Method says why none does. Render a middle frame and the last frame, in light and dark, and look at them.
- Run a coverage review. A fresh agent, which did not write the paper, reads it and lists each claim that rests on outside work. It checks each citation against the source's own text, with a verbatim quote, and against the work's review page in `docs/literature/`. A claim that the page contradicts is a finding. It finds a source for each claim that has none, and names reviewed work in `docs/literature/` that the paper should engage with. Fix each finding, and don't cite what the agent could not read.
- The status line names the work left, or the paper is published (*Status* above), and `uv run python scripts/papers.py check` passes.
- Update the question that the paper bears on in `docs/research-questions.md`: its *So far* line, with the date.
- Run `make lint` (it includes Vale) and `make test`.
- Build the site with `make build`, and open the Research index, the paper and its PDF (`dist/research/<slug>/paper.pdf`, printed by headless Chrome).
- Open the PR ready for review, not as a draft. Merge it when CI passes, as for every other PR. Do not wait for a person to read the paper first.

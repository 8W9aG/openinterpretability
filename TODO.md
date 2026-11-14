# Open work

Open work is in [GitHub issues](https://github.com/8W9aG/openinterpretability/issues), and Linear mirrors them. Until 2026-11-13 this file listed it. [Open work](docs/development.md#open-work) says how issues are labelled and prioritised.

Each section of the old list is now a label, `area: <section>`. The notes below were each section's introduction. They still apply to the work in that area.

## Housekeeping

[Open issues](https://github.com/8W9aG/openinterpretability/issues?q=is%3Aopen+label%3A%22area%3A%20housekeeping%22).

Small fixes, mostly quick.

## Code quality and tooling

[Open issues](https://github.com/8W9aG/openinterpretability/issues?q=is%3Aopen+label%3A%22area%3A%20code%20quality%20and%20tooling%22).

Lint rules, tests, releases and repository automation. Prefer a deterministic check (lint, test, CI job) where one can do the job. Ratchet it in rather than failing all existing code at once. Use a skill for what needs judgement.

## Research questions

[Open issues](https://github.com/8W9aG/openinterpretability/issues?q=is%3Aopen+label%3A%22area%3A%20research%20questions%22).

The questions that the project exists to answer, and how far the registry has come, are in [docs/research-questions.md](docs/research-questions.md). This is a draft: edit, add and remove questions. Each item names the evidence that bears on it. Link new work to the question it answers.

## Protocol and science

[Open issues](https://github.com/8W9aG/openinterpretability/issues?q=is%3Aopen+label%3A%22area%3A%20protocol%20and%20science%22).

## Analyses per experiment

[Open issues](https://github.com/8W9aG/openinterpretability/issues?q=is%3Aopen+label%3A%22area%3A%20analyses%20per%20experiment%22).

Methods from the wider interpretability literature that the registry doesn't use yet. Each one starts as a recorded number, reported and not gated, as in `docs/PROTOCOL.md` §4 "What is not evidence (yet)". It becomes a gate only through a protocol change. The tagging items below (language, WordNet, part of speech) plug into the scan's annotation layer (`harness/annotations.py`). A tagger returns labelled character spans of a document. The scan aligns them to tokens and reports each label set's R² and mutual information.

## Pipeline

[Open issues](https://github.com/8W9aG/openinterpretability/issues?q=is%3Aopen+label%3A%22area%3A%20pipeline%22).

## Performance

[Open issues](https://github.com/8W9aG/openinterpretability/issues?q=is%3Aopen+label%3A%22area%3A%20performance%22).

Make each experiment faster and cheaper in tokens without changing what it measures. Baseline from the 2026-10-20 loop (claim commit to result commit in git history):

- **Time:** an explore draw takes 25–40 min, and a replicate that skips the scan about 5 min. The CPU corpus scan is roughly 13–15 min of an explore draw.
- **Prompt sizes:** proposer prompts (`ev.prompt.md`) are 57–62 KB, about 15k tokens. Reviewer prompts are about 30 KB, and the scan evidence (`ev.json`) about 500 KB.

Measure before optimising, and check that any change to what the agents see doesn't move verdict rates. Verdicts are noisy: rerunning one prompt on the same unit agrees only about half the time. So an A/B needs about 16 units to detect a 0.10 shift in balanced accuracy (`docs/analyses/prompt-trim-ab.md`).

Running several loops on one machine is under Pipeline.

## Merge verification

[Open issues](https://github.com/8W9aG/openinterpretability/issues?q=is%3Aopen+label%3A%22area%3A%20merge%20verification%22).

CI checks each pull request in two ways. "Experiment records" (`openinterp check`) checks that the PR adds a valid record and its wiki page. "Reproduce records" (`openinterp reproduce --base-ref origin/main`, docs/PROTOCOL.md §5) re-runs every added or changed record on GitHub-hosted CPU runners, with cached weights and a model allow-list. So a record with edited activations or a hand-written verdict fails. Reproductions run one at a time across all pull requests, in a queue. That is enough while volume is low. The items below are for when it isn't: larger models, many PRs, or runner minutes adding up.

## Site

[Open issues](https://github.com/8W9aG/openinterpretability/issues?q=is%3Aopen+label%3A%22area%3A%20site%22).

Pages, deployment and charts for the generated site. Each experiment, shown in full on its unit's page, has an activation histogram and token heatmaps of the top-activating examples. Everything else is tables and token chips. Keep new charts as static SVG rendered at build time, readable in light and dark mode. Put the numbers in a table or in the Parquet export. Some need extra data stored on records. Check that before building them.

## Scope expansion

[Open issues](https://github.com/8W9aG/openinterpretability/issues?q=is%3Aopen+label%3A%22area%3A%20scope%20expansion%22).

## Datasets

[Open issues](https://github.com/8W9aG/openinterpretability/issues?q=is%3Aopen+label%3A%22area%3A%20datasets%22).

Scan corpora beyond `NeelNanda/pile-10k`. `openinterp corpus` samples any Hugging Face Parquet dataset and records its ID, commit, seed and licence on every submission. The default, `mixed`, is a versioned blend of the presets. Adding a dataset below is mostly a new preset in `openinterp/corpus.py`, plus a new version of the `mixed` blend that includes it. The other way is an entry in `NOT_BLENDED` in `openinterp/blends.py` saying why not. A test enforces this. A dataset stored as JSON can use the Hub's Parquet conversion, as the `c4` preset does (`revision="refs/convert/parquet"`). Check each licence before redistributing excerpts in the wiki.

## Models

[Open issues](https://github.com/8W9aG/openinterpretability/issues?q=is%3Aopen+label%3A%22area%3A%20models%22).

The hooks already resolve Llama/Qwen2/Mistral/Gemma/Phi, GPT-2, GPT-J, GPT-NeoX and OPT layouts (`openinterp/harness/hooks.py`). Each new model needs a smoke test. It needs a note on its MLP activation (gated SwiGLU/GeGLU vs plain GELU, which changes what a "negative pole" means). It also needs a check that the final-norm path resolves. Then add it to `[tool.openinterp] models` in `pyproject.toml`, with its terms in `[tool.openinterp.licenses]` (Llama and Gemma terms add conditions), and to `ALLOWED_MODELS` in `.github/workflows/ci.yml`, and `openinterp next` starts exploring it.

## Later stage

[Open issues](https://github.com/8W9aG/openinterpretability/issues?q=is%3Aopen+label%3A%22area%3A%20later%20stage%22).

Research directions to pick up once the registry covers several models well.

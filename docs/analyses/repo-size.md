# Repository size

Measured on 2026-11-14 for issue #3254, with `scripts/repo_size.py`. CI runs the script on every push and pull request, in the *Experiment records* job. It warns past the budget in `[tool.openinterp.size_budget]` in `pyproject.toml`, and it never fails the job.

The question: how fast does the repository grow, what makes it grow, and where should records live in the long term?

## Size today

| Measure | Value |
| --- | --- |
| Tracked files, all | 635 MB in 8,739 files |
| Tracked in `wiki/` | 595 MB in 8,045 files |
| `.git` packed, which is what a clone downloads | 226 MB |
| Budget (`tracked_mb`, `files`) | 1,000 MB, 20,000 files |

The records are JSON, which compresses about 3 to 1. So a clone costs about a third of the tracked size.

## Growth

Tracked size of `wiki/`, from the repository's history:

| Date | Files | MB |
| --- | --- | --- |
| 2026-10-20 | 153 | 10 |
| 2026-10-31 | 3,954 | 250 |
| 2026-11-07 | 6,174 | 424 |
| 2026-11-14 | 8,045 | 594 |

That is about 170 MB and 1,900 files a week, at today's rate of experiments. At that rate the tracked size passes 1,000 MB in about two weeks. The packed size grows by about a third as much, and passes 1,000 MB in about three months.

## What grows

`wiki/` is 541 MB of JSON records (3,090 files), 52 MB of Markdown pages (2,628) and 2,327 small signatures. In a random sample of 200 records, the mean record is 237 KB:

| Field | Share of a record |
| --- | --- |
| `submission.top_examples` | 50% |
| `verdict.explanation_scores` (the scored windows) | 20% |
| `submission.confounds` | 9% |
| `submission.opposite_pole` | 6% |
| `verdict.adversarial_test_results` | 5% |
| Everything else | 10% |

So the per-token evidence makes up most of the size: the top examples and the scored windows. The verdict itself is small.

## Where records could live

The decision is open (#3254). Each option can keep the records write-once and checkable, at different costs.

- **Stay in this repository.** It needs no change, and `open-interpretability check` reads the files directly. Every clone and every agent worktree pays the full history, and the cost grows without limit. CI's write-once check needs the history, so CI can't use a shallow clone.
- **Move the bulky evidence out, and keep the verdicts.** Records keep their verdict, metrics and hashes. The top examples and scored windows go to a content-addressed store by SHA-256, such as a Hugging Face dataset or release assets. About 70% of each record would leave the repository. `open-interpretability check`, `reproduce` and the site would fetch evidence by its hash.
- **A Hugging Face dataset for the records.** Parquet export plus the signed JSON, versioned by the dataset's own history. The repository keeps only pages and code. Records then live outside pull requests, so the write-once check needs a different mechanism.
- **Git LFS.** It is a small change for contributors. But GitHub's LFS storage and bandwidth are metered, and every CI checkout would use bandwidth.

## Limits

- **The growth rate is today's.** It follows how many experiments run, which changes with loops and models.
- **Sizes are of files at `HEAD`, plus the packed history.** Deleted or rewritten files still cost space in the history. Records are write-once, so there are few of them.

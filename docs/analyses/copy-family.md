# The copy family: failing prompts, hard negatives and splits for fixes

Built on 2026-11-19 for issue #5232, with `scripts/copy_family.py`. The rows are in `copy-family.jsonl`, and its first row is the metadata. It is the data that the fix plan of #5220 asks for. A fix tested on the six copy failures of the hallucination shortlist can't show that it generalises.

## What it holds

pythia-70m answers "A week has seven days and a year has twelve" with " days": it copies the first clause's unit (#4781). The family has every ordered pair of 41 facts in that form, "A `<X>` has `<n>` `<unit of X>` and a `<Y>` has `<m>`". It keeps the prompts that pythia-70m, in float32, answers with the copied unit:

- **855 wrong copies**, where Y's unit differs. These are the failures.
- **194 right copies**, where the units are the same. These are the hard negatives that a fix must not break.

Each row gives the top 5 tokens, the right answer's probability and rank, and the copied unit's probability. It also gives a greedy continuation and two sampled ones (temperature 1, seeds 0 and 1), each 20 tokens. So a fix can be judged on its text, not only its top token.

## Splits

The seed is 5232.

| | train | held-out | retain |
| --- | --- | --- | --- |
| Wrong copies | 155 | 700 | |
| Right copies | | 97 | 97 |

**Table 1.** The prompts by kind and split.

- **Reverses are held out.** Wrong copies are grouped by their unordered pair. When the model copies in both directions, 60% of pairs put one direction in train, and the reverse goes to held-out. Every train prompt's reverse is held out, so a fix that learned a relation can be told from one that memorised answers.
- **Shared numbers are held out.** All 127 wrong copies with the same number in both clauses, such as dozen/year against week/year, are held out.
- **Train is small** because only a pair copied in both directions can put one direction in train. The rest of the failures are held out.
- **Right copies** go half to retain and half to held-out. The shared-number rule applies to failures only.

## The frozen-fix hold-out

After a fix is frozen, a reviewer writes new failing prompts and new hard negatives without seeing the fix or any measure:

```bash
uv run python scripts/copy_family.py review-prompt --family docs/analyses/copy-family.jsonl \
    --out exchange/<id>/review.prompt.md
```

The task shows the template and the train prompts only. It asks for 20 of each kind as JSON, each with what the base model should answer and why.

## Limits

- **A confound** (#5074). A right copy shares its unit with the first clause, so it also shares a category. A change that finds "same category" would separate right from wrong copies without tracking rightness. The metadata records this, and the review prompt asks for hard negatives whose units match across categories.
- **One template and one model.** The family covers one form of the copy failure, on pythia-70m.
- **Only copies are kept.** A prompt that pythia-70m answers some other way is left out, so the family says nothing about those prompts.

## Reproduce

```bash
uv run python scripts/copy_family.py build --out docs/analyses/copy-family.jsonl
```

It takes about ten minutes on a CPU.

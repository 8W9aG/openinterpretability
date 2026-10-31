# Using Muse to run experiments

This page describes how to use Muse, an AI assistant, to run experiments
in this registry. Muse is not a model in the registry. It is the agent
that drives the command line tools below, writes hypotheses and review
probes, and judges explanation scores. In records it appears as
`external/muse-spark`, for both the proposer and the reviewer, unless a
different reviewer model is used.

## What Muse does in a run

A full experiment has six stages, in order:

1. Pick a unit with `openinterp next`.
2. Scan it with `openinterp audit` and read the evidence file.
3. Write a hypothesis, as the proposer, with falsifiable counterfactuals.
4. Build the submission with `openinterp audit --evidence --hypothesis`.
5. Score the explanation with `openinterp score`, answering the
   detection and fuzzing prompts literally from the explanation.
6. Review it with `openinterp review`, writing hard negatives and
   positive holdouts as the reviewer, then publish the record and page.

Muse can do all six stages on a CPU-only machine. It cannot place phone
calls, and it cannot sign in to a service for you. GitHub authentication
for pushing a branch is a separate step, done once with `gh auth login`.

## Setup

Clone the repo and install CPU-only dependencies. The repo install
script installs CPU-only torch, which avoids several gigabytes of CUDA
wheels on Linux:

```sh
git clone https://github.com/8W9aG/openinterpretability
cd openinterpretability
python3 -m venv .venv
.venv/bin/pip install uv
PATH="$PWD/.venv/bin:$PATH" ./scripts/ci-install.sh
```

Run tests single-threaded, as the Makefile does:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

## Pick a unit

```sh
openinterp next --model openai-community/gpt2 --no-claims --seed YYYYMMDD --steps
```

The steps file lists the full claim, scan, propose, review, score, and
post workflow. With `--no-claims`, no GitHub claim step is taken. The
audit seed in the steps output is the seed to reuse for the scan, so
the same unit is scanned.

Skip a draw that cannot be audited as a detector. A negative pole that
is active on most scan tokens, for example 74 percent or 86 percent in
two pilot draws, is default-on and is not a useful detector. In that
case, draw again instead of spending a full scan on it, and say so in
the PR body so the experiment is not misrepresented as the immediate
next draw.

## Corpus

A pilot run can use a small corpus, but say which corpus was used in
the PR body. The registry corpus is the pile-10k corpus described in
the README. A pilot used TinyStories instead:

```sh
openinterp corpus --dataset tinystories --docs 200 --out pilot/pilot-corpus.jsonl
```

That pilot sampled 97 of 200 documents, 19,962 tokens, against a
20,000 token cap. GPT-2 scans use 128 token windows, because GPT-2
transcoders fit only the first 128 positions.

## Scan, in float32

Use float32 for every stage. This matters for reproducibility, see
below.

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/openinterp audit \
  --model openai-community/gpt2 \
  --random --seed <audit-seed> \
  --window-tokens 128 \
  --corpus pilot/pilot-corpus.jsonl \
  --max-tokens 20000 \
  --dtype float32 --device cpu \
  --evidence-out pilot/exchange/ev.json \
  --wiki-dir '' \
  --out pilot/submissions/ev.json
```

The evidence file holds the top examples, peak tokens, confounds,
input attribution, and vocabulary lens output. Read it before writing
the hypothesis. Do not guess the hypothesis from the unit name alone.

## Hypothesis and submission

Write a hypothesis file with these fields:

- `explanation`: one plain statement of what the pole fires on.
- `mechanism`: for example `retrospective`.
- `target_upcoming_tokens`: usually an empty list for a retrospective
  pole.
- `activating_tokens`: the tokens the explanation predicts.
- `counterfactuals`: minimal pairs, each with a base prompt, a
  counterfactual prompt, an expected delta, and a rationale. Include
  at least one control pair with an expected delta of 0.

Then build the submission:

```sh
.venv/bin/openinterp audit \
  --evidence pilot/exchange/ev.json \
  --hypothesis pilot/exchange/ev.hypothesis.json \
  --agent-model external/muse-spark \
  --out pilot/submissions/ev.json \
  --wiki-dir '' \
  --dtype float32 --device cpu
```

## Explanation scores

```sh
.venv/bin/openinterp score \
  --audit-file pilot/submissions/ev.json \
  --corpus pilot/pilot-corpus.jsonl \
  --emit-task pilot/exchange/score.task.json \
  --dtype float32 --device cpu
```

Answer the detection prompt by listing every text in which the
explanation, taken literally, says the neuron fires on at least one
token. Answer the fuzzing prompt by listing only the texts whose
marked tokens are exactly the tokens the explanation predicts. Do not
copy answers from an earlier experiment, because the held-out windows
change with the seed.

```sh
.venv/bin/openinterp score \
  --task pilot/exchange/score.task.json \
  --judge-model external/muse-spark \
  --dtype float32 --device cpu
```

## Review

Write review probes before running the review:

- Hard negatives: fragments and lists with no main verb, and claimed
  tokens in non-verb roles, all expected inactive.
- Positive holdouts: novel examples of the claimed pattern, all
  expected active, using tokens not in the hypothesis.

Check tokenisation before running the review, and leave the probe
labels and texts unchanged unless a tokenisation check shows a label
is wrong. Then run:

```sh
.venv/bin/openinterp review \
  --audit-file pilot/submissions/ev.json \
  --probes pilot/exchange/review.probes.json \
  --agent-model external/muse-spark \
  --explanation-scores pilot/exchange/score.scores.json \
  --wiki-dir pilot/wiki \
  --dtype float32 --device cpu
```

The review writes a record and a page under
`pilot/wiki/experiments/<model>/<layer>/<neuron>-<pole>/`. If the
proposer and the reviewer are both Muse, the review is recorded and
badged as self-reviewed. A different reviewer model is preferred under
the protocol, section 2.

## Check, reproduce, publish

Before pushing anything, run both checks locally:

```sh
.venv/bin/openinterp check
.venv/bin/openinterp reproduce \
  pilot/wiki/experiments/<model>/<layer>/<neuron>-<pole>/<record>.json \
  --corpus pilot/pilot-corpus.jsonl --device cpu
```

Reproduce must say 1/1 reproduced. If it fails, do not push the
record. Fix the cause first.

To publish, copy only the new record `.json` and page `.md` into
`wiki/experiments/...`, on a branch named
`experiment/<model>-<layer>-<neuron>-<pole>`. Do not commit the
working `pilot/` folder. Commit, push, and open a draft PR. The PR
body should state the verdict, the gates, the corpus, whether the
review was self-reviewed, that `openinterp check` passes, and the
next seed and kind if the experiment came from `openinterp next`.

Merge only when all checks pass: lint and tests, experiment records,
record changes, and reproduce records. A draft PR cannot be merged
until it is marked ready.

## Pitfalls learned from pilot runs

- Dtype must match across every stage. A scan in bfloat16 with a
  review in float32 fails CI reproduce, because reproduce loads the
  dtype pinned by the verdict and compares it with the scan numbers.
  A bfloat16 re-review also failed local reproduce on its own, with
  a few activations just over tolerance. Float32 throughout passed
  local reproduce 1/1 and CI reproduce.
- A pilot corpus is not the registry corpus. Say so in the PR body,
  and do not present a TinyStories run as a registry-standard run.
- Self-review is allowed but is recorded as such. It is weaker than
  an independent review.
- Activation scans are the bottleneck on CPU. A GPT-2 scan at
  20,000 tokens took about 5 minutes in bfloat16 and about 18 minutes
  in float32 on a 4 CPU machine, plus model load and scoring time.
  Budget about 30 to 45 minutes for a full pilot run, plus CI time.

## Token usage

Muse reports usage as a percentage of a weekly allowance, not as an
exact token balance, so a run cannot be costed in exact Muse tokens
from the product alone. Record the subscription percentage before and
after a run, and the session context reading if one is available, in
the run log. Separately, record the scan token cap and the actual
tokens scanned, which are GPT-2 tokens, not Muse billing tokens.

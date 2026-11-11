# Explainer benchmark

A frozen, versioned benchmark for tools that explain neurons. Other labs can score their own explainers on it. The data is in [`benchmark/`](../benchmark/), and `openinterp benchmark` freezes and scores it ([openinterp/benchmark.py](../openinterp/benchmark.py), [openinterp/cli_benchmark.py](../openinterp/cli_benchmark.py)).

## What it measures

Each unit is one pole of one native MLP neuron. Each unit has probe texts, and each probe has a label: whether the model measured the pole as active on that text. Active means a peak strength of at least 0.25 × the pole's scan maximum (docs/PROTOCOL.md §4).

Your explainer writes one explanation for each unit. A judge model reads the explanation and the unit's probe texts, and lists the texts on which the explanation says the neuron fires. The judge never sees the labels. The unit's score is the balanced accuracy of the judge's list against the labels. That is the mean of the rate on active probes and the rate on inactive probes. Near-duplicate probes count as one observation, as in a review.

The split's score is the mean over **every** unit in the split. A unit with no explanation scores 0.5, which is chance. So an explainer can't raise its mean by leaving out hard units. The summary gives a 95% bootstrap interval over units and the mean per model.

## How the units were selected

Read the scores as scores on these units, not on whole models.

- **Models.** Qwen2.5-0.5B, GPT-2 small and Qwen2.5-1.5B: the models with the most records. `manifest.json` gives the count for each.
- **Random draws only.** A unit is in the benchmark only if its first record was a random draw (`provenance.selection == "random"`). Units that someone chose for a reason are left out.
- **Native neuron poles only.** Attention heads and dictionary features are left out of v1.
- **Units that fire on some probes.** A unit needs at least 3 active and 3 inactive probes. In v1, this left out 261 random units, most of them nearly dead. So the benchmark under-represents dead and rarely active neurons.
- **Probes from reviews.** Each probe is a reviewer's probe from a public record. Reviewers wrote probes to break the hypothesis under review, so many probes are near-misses of the registry's own explanation. A unit with several records has the probes of each record. A text that two records labelled differently is left out.

## Splits

A hash of each unit's slug puts 20% of the units in the test split and the rest in the dev split (`openinterp.benchmark.split_of`). The version name salts the hash.

- **Dev.** `dev/probes.jsonl` publishes the probes and labels from the records. Use it to develop and tune your explainer.
- **Test.** The records of test units are public too, so their record probes can't be held out. The test split uses fresh probes instead. A maintainer writes and measures them, and keeps them out of the repository. `manifest.json` publishes only the SHA-256 of the labels file (`test_labels_sha256`), so a reported test score can be checked against the file later. To have a test score, send your explanations file to a maintainer.

The test units themselves are listed in `units.jsonl`. Their wiki records and the registry's explanations of them are public. So the rule for a test score is that your explainer reads no record under `wiki/` of a test unit. Report the score as a breach of this rule if it does.

## Files

| File | Contents |
| --- | --- |
| `manifest.json` | Version, licence, selection rule, counts and the test labels' hash |
| `units.jsonl` | One row per unit: slug, model and its revision, layer, neuron, pole, split and the records it came from. Also the scan maximum and threshold that fresh probes are measured against |
| `dev/probes.jsonl` | One row per dev probe: `unit`, `text`, `active`, the probe's `kind` in its review, and its `record` |
| `dev/registry_explanations.jsonl`, `test/registry_explanations.jsonl` | The registry's explanation of each unit: its newest VERIFIED explanation, else its newest one. A baseline |

The data is under CC-BY-4.0, like the records it comes from (`wiki/LICENSE.md`).

## Score an explainer

Write one JSON object per line, `{"unit": "<slug>", "explanation": "<text>"}`, for the units of a split. Then run the scorer:

```bash
uv run openinterp benchmark score --bench benchmark/v1 --split dev \
  --explanations my-explanations.jsonl --judge-model anthropic/claude-sonnet-5-5 \
  --out results/dev.jsonl
```

`--explanations registry` scores the registry's own explanations, as a baseline. The registry's explanations had the advantage that the probes target them. A review VERIFIES only an explanation that predicts most of its own probes. So expect the registry baseline to score high on units where the registry has a VERIFIED explanation.

The scorer writes one row per unit to `--out` as it goes. If you run it again, it judges only the units not already in that file. It writes the summary next to it, as `<name>.summary.json`. `--limit N` scores only the first N units, for a cheap trial. `--judge-model external/<name>` writes each judge prompt to a file for any agent or person to answer (see the `explanation-judge` agent).

The judge model changes the scores. Always report the judge with a score, and compare explainers only under the same judge.

## Make a new version

A version is frozen after its first publication. Change nothing in it. For more units or a new rule, freeze a new version:

1. `uv run openinterp benchmark freeze --version v2` writes `benchmark/v2/` from the records in `wiki/`. The test units have no probes yet.
2. For each test unit, write a fresh probe set. Use the review prompt of its newest record, and a reviewer that has not reviewed that unit before:

   ```bash
   uv run openinterp review --audit-file wiki/experiments/<slug>/<record>.json \
     --emit-prompt exchange/bench/<unit>.prompt.md
   ```

3. Measure the probes and add their labels to the private file. Its name must end with `.private.jsonl`, which `.gitignore` keeps out of the repository. Never run `openinterp review --probes` on these probes, because it publishes a record:

   ```bash
   uv run openinterp benchmark measure --bench benchmark/v2 --unit <slug> \
     --probes exchange/bench/<unit>.probes.json --labels ~/benchmark/v2-test.private.jsonl
   ```

4. `uv run openinterp benchmark seal --bench benchmark/v2 --labels ~/benchmark/v2-test.private.jsonl` sorts the file and writes its SHA-256 into the manifest. It fails if a test unit has too few probes of a label. For each such unit, write a second probe set with another fresh reviewer and measure it into the same file. Tell that reviewer to change the mix of probes. If the unit has too few active probes, ask for mostly positives near the activating examples. If it has too few inactive probes, ask for mostly plain negatives. The model measures the labels, so this changes the mix and not the labels.
5. Store the private file where only maintainers can read it, and commit the version.

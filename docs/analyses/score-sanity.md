# Sanity checks on the metrics of `openinterp score`

Run on 2026-11-14 for issue #3237, with `scripts/score_sanity.py`. The numbers are in `score-sanity.json`. Reported, not gated.

The question: does the score fall when the explanation is wrong? Oikarinen et al. (arXiv 2506.05774, "Evaluating Neuron Explanations") report that many explanation metrics hardly move when the concept labels change a lot. It proposes two tests that a metric must pass:

- **Missing labels.** The explanation leaves out part of what the unit does. The score must fall.
- **Extra labels.** The explanation claims more than the unit does. The score must fall.

The issue also asks for a third check: shuffled explanations. The null-explanation control (`null-explanation-control.md`, #3236) did it. 7 VERIFIED Qwen poles were judged with another unit's VERIFIED explanation and with an empty one. Every detection and fuzzing score fell to 0.50, against a mean of 0.60 and 0.69 with the unit's own explanation. This page adds the two label tests.

## Method

- **Data.** Every stored score: 2,301 neuron-pole records with explanation scores. Each record stores, for every detection and fuzzing window, the judge's answer and the measured label (active or not).
- **Mapping.** In the paper, the explanation's concept labels are perturbed and the unit's activations stay fixed. Here the judge's answers play the part of the explanation's labels, and the measured labels stay fixed. So the tests need no new judge calls.
- **Perturbation.** For *missing labels*, each "active" answer turns to "inactive" with probability p. For *extra labels*, each "inactive" answer turns to "active" with probability p. The rates are p = 0.1, 0.2, 0.3 and 0.5, with 20 seeded draws per score and rate.
- **Metrics.** Balanced accuracy, which `openinterp score` reports. Recall and precision alone, which the paper expects to fail one test each. And F1.
- **Outcome.** For each metric, the mean change over scores, and the share of scores where it falls (a tie is no fall). The shares are split by whether the judge's answers beat chance on that score, meaning that its recall is above its false-positive rate.

## Result

**Balanced accuracy is the only one of the four metrics that falls under both tests.** At p = 0.2:

| Metric | Detection, missing | Detection, extra | Fuzzing, missing | Fuzzing, extra |
| --- | --- | --- | --- | --- |
| Balanced accuracy | −0.020 | −0.020 | −0.018 | −0.019 |
| Recall | −0.086 | **+0.114** | −0.046 | **+0.154** |
| Precision | **+0.001** | −0.129 | **+0.000** | −0.281 |
| F1 | −0.060 | **+0.086** | −0.046 | **+0.110** |

Bold marks a failed test: a mean change of zero or above.

- **Recall fails the extra-labels test, as the paper predicts.** Calling more windows active can only find more active ones, so recall never falls (0% of scores).
- **Precision fails the missing-labels test, as the paper predicts.** Dropping active answers at random keeps the share of hits among them about the same.
- **F1 fails the extra-labels test here.** The windows are half active and half inactive, so extra "active" answers add as many hits as false alarms, and F1 rises. On ordinary text, where active tokens are rare, F1 would behave differently.
- **Balanced accuracy falls in proportion.** Under either test, its expected change is −p × (recall − false-positive rate) / 2. So the fall depends on how far the judge beats chance. It falls at every rate: for detection, by 0.010 at p = 0.1 and by 0.049 at p = 0.5.

**Balanced accuracy falls wherever the judge beats chance.** At p = 0.2:

| | Detection, missing | Detection, extra | Fuzzing, missing | Fuzzing, extra |
| --- | --- | --- | --- | --- |
| Scores above chance that fall | 96% of 1,501 | 90% of 1,501 | 97% of 1,346 | 87% of 1,346 |
| Scores at or below chance | 540 | 540 | 580 | 580 |

The rest of the scores above chance are noise: 20 draws can miss a small fall when a judge is only just above chance. At p = 0.5, 99% of detection scores above chance fall under missing labels, and 96% under extra labels.

**A quarter of stored scores are at or below chance.** 540 of 2,041 detection scores and 580 of 1,926 fuzzing scores have a judge whose recall is not above its false-positive rate. A perturbation can't lower a score that is already at chance, so these scores can't show a wrong explanation either. On such a score, a good explanation and a bad one get the same number.

## What this means

- **`openinterp score` reports the right metric.** Balanced accuracy passes both of the paper's tests on every stored score where it can, and the shuffled-explanation control already passed. Recall alone would reward an explanation that claims too much, and precision alone one that claims too little.
- **Don't add F1 to the score.** On balanced samples it rewards extra claims.
- **A score at chance says nothing about the explanation.** A quarter of the registry's stored scores are there. A single score near 0.5 does not mean that the explanation is wrong. It means that the judge could not use it. That fits `verified-vs-detection.md`, where VERIFIED records score only about 4 points above other records, within judge.

## Limits

- **The perturbations are random.** A real wrong explanation misses or adds a *kind* of window, not random ones. The tests show that the metric responds to errors. They don't measure how well a judge detects a specific wrong claim. The shuffled-explanation control is the realistic case, and it used 7 units.
- **The judges' answers are taken as the explanation's labels.** A judge that misreads a good explanation looks here like a wrong explanation. These tests can't separate the two.
- **The windows are balanced by construction.** The metrics' behaviour on ordinary text, where active windows are rare, can differ. That is the case for F1 in particular.

## Reproduce

```bash
uv run python scripts/score_sanity.py   # writes docs/analyses/score-sanity.json, about 30 s
```

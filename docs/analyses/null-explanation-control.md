# Null-explanation control for explanation scores

Run on 2026-11-13 for issue #3236, with `openinterp score --controls`. The question: does a detection or fuzzing score come from the explanation, or from the held-out windows? "Scoring Explanations Without Reading Them" (arXiv 2610.37676) reports that automated-interpretability data can have structure that a judge scores without the explanation. If that happens here, a wrong explanation also scores well. Then a comparison of VERIFIED against non-VERIFIED scores (issue #3239) means little.

## Method

- **Units.** 8 VERIFIED neuron poles of Qwen2.5-0.5B, judged under protocol 0.8, drawn at random (seed 3236) from the 162 that qualify.
- **Windows.** `openinterp score --corpus corpus.jsonl --seed 3236 --controls`, at the defaults: 32-token windows, 5 bins, 5 windows per bin. Each unit got 50 detection windows and 37 fuzzing windows. The exception is L21/N3912+, which got 4 detection windows and no fuzzing windows, so it is left out. 7 units remain.
- **Explanations.** Each unit's windows were judged three times, with:
  - the unit's own explanation,
  - an empty explanation,
  - another unit's VERIFIED explanation.
- **Judge.** Each of the 42 prompts went to its own `explanation-judge` subagent, model `claude-haiku-5-5`. No judge saw two prompts.
- **Score.** Balanced accuracy against the measured labels, as in every review. The rows are in `null-explanation-control.jsonl`.

## Results

| Unit | Detection | Fuzzing | Detection gap | Fuzzing gap |
| --- | --- | --- | --- | --- |
| L20/N4378− | 0.68 | 0.69 | +0.18 | +0.19 |
| L16/N671− | 0.52 | 0.54 | +0.02 | +0.04 |
| L19/N4404+ | 0.64 | 0.54 | +0.14 | +0.04 |
| L2/N3477+ | 0.58 | 0.80 | +0.08 | +0.30 |
| L18/N4729+ | 0.40 | 0.69 | −0.10 | +0.19 |
| L4/N3208+ | 0.66 | 0.65 | +0.16 | +0.15 |
| L13/N4671− | 0.70 | 0.86 | +0.20 | +0.36 |
| **Mean** | **0.60** | **0.69** | **+0.10** | **+0.18** |

The gap is the real explanation's score minus the control's. Both controls scored 0.50 on every unit, for detection and for fuzzing, so one gap column serves both.

- **No shortcut appeared.** With no explanation, or with another unit's, the judge listed no texts. So it scored exactly chance. This judge gives no credit to structure in the windows that it can see without an explanation.
- **Every explanation beat the controls on fuzzing.** Fuzzing gaps run from +0.04 to +0.36.
- **One explanation fell below them on detection.** For L18/N4729+, the judge called 32% of the inactive windows active. So it scored 0.40. The explanation describes repeated fixed expressions and multi-token names, and many inactive windows contain one. The explanation is broader than where the pole fires.
- **The scores sit near chance.** Mean detection is 0.60, and true-positive rates run from 0.04 to 0.48. The VERIFIED explanations find a minority of the held-out active windows. Most explanations name the pole's top activations, and the held-out windows sample all five bins (`recall_by_quantile` in each `score.scores.json`).

## Limits

- **The controls can't score above chance with this judge.** A judge that lists no texts gets balanced accuracy 0.5. Haiku reads explanations literally, so an empty or unrelated one makes it list nothing. A judge that guesses from the windows would show the shortcut here. This judge doesn't guess, so this run can show only that no shortcut exists with Haiku.
- **All seven units drew the same other-unit explanation.** It was L7/N1327−, a narrow claim about emphatic demonstratives after a focus particle. Its frame was in none of the windows. The draw used only the seed, which all the units shared. `openinterp score` now joins the unit's slug to the seed, so each unit draws its own decoy. A broad decoy, such as "fires on numbers", would test the judge harder than a narrow one.
- **Seven units, one model, one judge.** The gaps differ a lot between units. Seven units can't give a registry-wide estimate.

## What this means

- With the Haiku judge, a detection or fuzzing score above 0.5 comes from the explanation, not from the windows. The comparison of VERIFIED against non-VERIFIED scores (issue #3239) can go ahead with this judge.
- Run with `--controls` for any other judge model. Treat a control above 0.5 as a sign that the judge reads structure in the windows.
- A detection score below the controls shows an explanation broader than the pole. L18/N4729+ is one. That is a revision for the proposer, not a fault in the score.

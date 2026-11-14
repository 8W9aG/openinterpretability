# Scoring Explanations Without Reading Them

**Citation.** Xupeng Chen (Dimension Gate). "Scoring Explanations Without Reading Them: Structural Shortcuts in Automated-Interpretability Evaluation Data". arXiv 2610.37676v1, 10 October 2026. [Paper](https://arxiv.org/abs/2610.37676). Preprint, no venue given. Code and preregistrations are promised on publication.

**Status.** Used. `openinterp score --controls` (the null-explanation control, issue #3236) and `scripts/explanation_free_rules.py` both come from this paper.

**Read.** The abstract and Sections 1–3 in the arXiv HTML, word for word, on 2026-11-13. The rest (Sections 4–9 and the appendices) came to us only as a model-generated summary of the text. So the numbers below are the abstract's and the introduction's. Numbers from the summary are marked as unverified. We did not see the figures.

## Summary

**Claim.** The evaluation items that automated-interpretability pipelines build can be scored without reading the explanation. Where a window is cropped, which tokens are marked and which sequences are included carry the answer. So a score should be reported beside the best explanation-free rule on the same items.

**Method.** The paper covers the three standard scores. Simulation is the pooled correlation of predicted and true activations (OpenAI's method). Detection and fuzzing are balanced accuracies (Paulo et al., 2025, EleutherAI's delphi). It replays stored scorer runs and applies simple rules to the same items. The rules include a centre-marked rule, set-position priors and a mark-count rule. It also scores shuffled explanations, and natural-text windows cut at random offsets.

**Findings, as the paper reports them.**

| Data | Result |
| --- | --- |
| Neuronpedia fuzzing runs, GPT-2 small, 500 random features | "yes when the centre token is marked" scores 0.979. The stored LLM scorers score 0.838. A rule that reads only the count of marks scores 0.969 |
| The same runs | 91.3% of activating examples are cropped with the peak on the centre token |
| OpenAI's simulation release | A post hoc rule (active on the first five sequences) correlates 0.227. The published GPT-4 simulator correlates 0.160. An oracle given each sequence's mean scores 0.288 |
| Natural windows at random offsets, 49 features | The best set-offset prior shared by all features correlates 0.015. A typed decision model scores 0.331 with the real explanation and 0.013 with a shuffled one. An open 8B simulator scores 0.194 and 0.019 |
| Stored snippets against natural windows | A trained scorer loses 63% of its score on natural windows. The same architecture with no explanation reaches 83% of its held-out score |

The summary adds per-scorer and Gemma-2-2B numbers, which we could not check against the text. It also says that most p-values are uncorrected sign tests.

## Where the claims go beyond the evidence

- **The shortcuts are shown on two pipelines' item construction.** The centre-crop and mark-count results belong to how Neuronpedia's stored runs and delphi's defaults build items. A pipeline that cuts windows differently can have other shortcuts or none. The paper says as much when it recommends random offsets.
- **The marks are rebuilt, not stored.** Neuronpedia does not store which tokens were marked, so the paper rebuilds them from the pipeline's code. The fuzzing results depend on that reconstruction being right.
- **The rules are chosen after seeing the data.** The paper flags this. Its rule scores are upper bounds, not what a rule set in advance would get on new items.
- **A typed decision model carries the natural-window results.** It is commercial, and its training data is not disclosed.
- **One author, version 1, no released code yet.** The preregistrations and code are promised, not available.

## Use in this repo

**Our windows are consecutive pieces of text, not crops around the peak.** `sample_heldout` (`openinterp/harness/explanation_scores.py`) splits each held-out text into consecutive 32-token pieces. So the paper's strongest cue, a peak on the centre token, can't arise. The scan's top examples, which are centred, are left out of scoring.

**Our judges don't score without the explanation.** With a Haiku judge, an empty explanation and another unit's explanation both scored exactly 0.50 on 7 VERIFIED poles (`docs/analyses/null-explanation-control.md`). The paper's shuffled-explanation result points the same way.

**Our fuzzing items do leak structure.** We applied the paper's check to the 152,691 windows that 2,293 scored records store (`scripts/explanation_free_rules.py`, `docs/analyses/explanation-free-rules.json`). With no explanation:

| Rule | Balanced accuracy | Judges, with the explanation |
| --- | --- | --- |
| Fuzzing: correct if the marks form one unbroken run | 0.579 | 0.593 |
| Fuzzing: correct if 4 or more tokens are marked | 0.560 | 0.593 |
| Detection: active if the window has at least n tokens (best n) | 0.536 | 0.602 |

The cause is in how decoys are built. A correct fuzzing item marks a window's active tokens, which usually sit together. A decoy marks as many inactive tokens, drawn at random, so they are scattered. Inactive filler windows get 1 to 3 random marks. Our judges don't seem to use the cue, since they get 0.50 with no explanation. But a rule that reads nothing comes within 0.014 of them. So a fuzzing score near 0.6 says little about an explanation. Detection is mostly clean: the length cue is small, and it comes from short final pieces of a text.

**What changed.** Since issue #3505, decoys copy the real marks' count and spacing, moved to inactive tokens. Inactive filler windows copy the shape of a random active window's marks. Each new score also records the explanation-free rules on its own windows, in `rule_baseline`, as the paper recommends. Older scores keep `fuzzing_decoys` unset. This changes how scores are made, not the decision rule, because scores are reported and not gated.

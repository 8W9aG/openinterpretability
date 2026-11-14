---
name: explanation-simulation-scoring
description: Score a neuron explanation by how well it predicts per-token activations (explain → simulate → score), and assess it observationally (precision/recall) and by intervention. Use when comparing candidate hypotheses for one neuron, calibrating the reviewer's activation threshold, or adding an activation-level metric alongside the probe-based verdict.
---

# Simulation scoring and rigorous explanation assessment

## Papers

- Bills et al., 2023, *Language models can explain neurons in language models* (OpenAI): <https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html>. An explainer model writes an explanation. A simulator model predicts per-token activations (0–10) from the explanation alone. The score is the correlation between simulated and real activations. They evaluate on **top-activating plus randomly sampled** text, because top-only scoring rewards explanations that ignore false positives.
- Huang, Geiger, D'Oosterlinck, Wu & Potts, 2023, *Rigorously Assessing Natural Language Explanations of Neurons*, arXiv:2309.10312. They split evaluation into **observational** (does the neuron fire on exactly the inputs the explanation covers? report precision and recall) and **interventional** (does the neuron causally mediate the concept?). GPT-4's most confident GPT-2 XL explanations scored precision 0.64 and recall 0.50, with no causal effect found.
- Caveat from Goodfire's *Already Clean* essay (2026): simulation metrics "reward explanations that look good over ones that are right". Use simulation as a diagnostic, never as the only gate.

## How it maps onto openinterp

| Concept | openinterp equivalent |
| --- | --- |
| Observational precision | Specificity on hard negatives (reviewer probes) |
| Observational recall | Sensitivity on positive holdouts |
| Interventional test | Counterfactual Δ and causal metrics (see `activation-patching-practice`) |
| Detection, fuzzing, embedding | `openinterp score`, recorded as `verdict.explanation_scores` (reported, not gated) |
| Simulation correlation | **Not yet in the protocol.** Compute it as supplementary evidence |

## Cheaper scores: detection, fuzzing and embedding

Paulo et al., 2025 (Delphi, `EleutherAI/delphi`) replace per-token simulation with three cheaper scores. `openinterp score` computes them, and every review records them (docs/PROTOCOL.md §4):

```bash
uv run openinterp score --audit-file <submission or record> --corpus corpus.jsonl --emit-task exchange/score.task.json
# explanation-judge answers exchange/score.detection.prompt.md, a *second* one answers score.fuzzing.prompt.md
uv run openinterp score --task exchange/score.task.json --judge-model external/<judge model ID>
uv run openinterp review … --explanation-scores exchange/score.scores.json   # or reevaluate --record …
```

- Use the scan's corpus. The scan's top examples are left out, so every window is text the proposer never saw.
- Give each prompt to a separate judge. Fuzzing marks real active tokens, so a judge that saw them would score detection on more than the explanation.
- Read `recall_by_quantile` as well as the balanced accuracy. Explanations written from top examples typically miss the weaker bins.
- Add `--controls` to judge the same windows with no explanation and with another unit's explanation. A score means little until it beats both. The summary prints each gap. With `--emit-task`, each control has its own two prompts (`score.control-<kind>.<detection|fuzzing>.prompt.md`), and each prompt needs its own judge. With a `claude-haiku-5-5` judge, both controls scored 0.5 on 7 VERIFIED poles, and the real explanations beat them by 0.10 on detection and 0.18 on fuzzing (`docs/analyses/null-explanation-control.md`).
- Read each score beside its `rule_baseline`: the best explanation-free rules on the same windows, an upper bound. A score means little until it beats them. Since issue #3505, fuzzing decoys copy the real marks' count and spacing (`fuzzing_decoys: matched-shape`), so these rules sit near 0.5. Older scores, with random decoys, let a contiguity rule reach 0.58.
- A low score with a VERIFIED verdict is not a contradiction. The probes test the claim where the reviewer looked. These scores test where the pole fires in ordinary text. Say in the next revision which held-out activity the explanation leaves out.

## Procedure: simulation score

1. Build an evaluation set that the proposer did not see. Take snippets from a *different* slice of the corpus than the scan, stratified as 1/3 top-activating, 1/3 from the 50th–99th percentile band, and 1/3 random.
2. Measure real per-token strengths:
   ```python
   acts = runner.neuron_activations(snippet, layer, idx, pole)[0].tolist()
   ```
3. Ask a simulator LLM (via `LLMAgent.structured`, with a response model of `list[int]` per token) to predict a 0–10 activation per token **from the explanation text only**. It must not see the examples.
4. Score with the Pearson correlation over all tokens in the set, and with the Spearman correlation of each snippet's maximum. Report both, plus the score on the random third alone, which is where false positives show up.
5. Compare candidate explanations on the *same* evaluation set. A narrower explanation that scores well on the random third usually survives review better.

## Calibrating the "active" threshold

The protocol counts a probe as active at `0.25 × max_activation`. Before relying on that for a neuron, check it against the histogram. The 99th percentile can already be above the threshold (a "dial" neuron). Then the threshold flags ordinary text as active, and hard negatives fail for reasons unrelated to the hypothesis. Report this in the submission explanation. Changing the threshold for one neuron is not allowed. Changing the default is a protocol change (`docs/PROTOCOL.md` §6).

## Pitfalls

- Top-only evaluation inflates scores (Bills et al.).
- A simulator that shares a model family with the explainer can share its blind spots, so prefer a different provider.
- High simulation correlation with zero causal effect (Huang et al.) means the explanation describes a correlate. Say so in the explanation and keep the mechanism claim modest.

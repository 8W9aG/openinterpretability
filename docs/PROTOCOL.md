# Audit protocol (v0.2)

This document defines what one Open Interpretability experiment is, who does what, and how a verdict is decided. The code implements it: the decision rule is `openinterp.agents.reviewer.evaluate_submission`, and its parameters are `openinterp.core.schemas.ReviewPolicy`. Every verdict stores the policy it was computed under, so any record can be re-checked with `openinterp reevaluate`.

## 1. Unit of analysis

One experiment tests **one hypothesis about one neuron pole** in one model revision.

| Term | Definition |
| --- | --- |
| Neuron | One coordinate of the MLP intermediate activation, i.e. the input to the down/output projection (`act(gate(x)) * up(x)` for gated MLPs). |
| Pole | Positive: `clamp(a, min=0)`. Negative: `clamp(a, max=0)`. The two poles are audited separately. |
| Strength | Non-negative magnitude of the pole. Larger always means more active. |
| Scale | The maximum strength seen during the discovery scan (`max_activation`). |
| Active | A prompt is active if its peak strength over all tokens is at least `activation_threshold × scale` (default 0.25). |
| Δ (counterfactual) | `(peak strength on base − peak strength on counterfactual) / scale`. |

## 2. Roles

| Role | Does | Must not |
| --- | --- | --- |
| **Proposer** | Selects the neuron, scans a corpus, writes the hypothesis, the mechanism claim and the minimal-pair counterfactuals. | Edit a submission after review. A revision is a new submission. |
| **Reviewer** | Writes adversarial probes from the hypothesis text and the top examples. | See probe activations before the probe set is final. Be the same model as the proposer. |
| **Harness** | Runs every measurement against the model weights and applies the decision rule. | Use LLM judgement anywhere in the verdict. |

Agents only write text: hypotheses and probes. Every number in a verdict comes from running the model.

## 3. Lifecycle

```
select ─▶ scan ─▶ hypothesize ─▶ submit (frozen) ─▶ probe ─▶ measure ─▶ decide ─▶ record ─▶ publish
                       ▲                                                    │
                       └──────────── NEEDS_REVISION: new submission ◀───────┘
```

1. **Select.** Choose the neuron pole and record how in `provenance.selection`. Agents without an assignment use `openinterp next` (skill `choose-next-experiment`). It explores uniformly random neurons with probability ε and otherwise samples a follow-up on an existing neuron; re-studying a neuron is valid, and null results count as data. The options are:
   - `random`: uniform over layers, then neurons, then poles (`openinterp audit --random --seed N`). Only randomly selected neurons count towards registry-wide statistics such as "fraction of neurons with a verified explanation".
   - `attribution`: chosen because it ranked highly in attribution patching for some behaviour.
   - `manual`: anything else.
2. **Scan.** Run the corpus through the model and keep the top-k activating snippets and a histogram of all token strengths. Record the model revision, dtype, corpus, corpus hash and token count in `provenance`.
3. **Hypothesize.** The proposer states:
   - a natural-language explanation precise enough that a third party can label new text as "predicted active" or "predicted inactive";
   - the mechanism: **retrospective** (reacts to tokens already read) or **lookahead** (prepares specific upcoming tokens);
   - at least three minimal-pair counterfactuals with predicted Δ, including at least one control pair (predicted Δ ≈ 0).
4. **Submit.** The submission is frozen from here on. Changing anything other than provenance metadata produces a new submission.
5. **Probe.** The reviewer writes 10–20 probes and labels each with what the *hypothesis* predicts, not with what the reviewer expects the neuron to do:
   - **hard negatives** (about two thirds): near-misses that share surface features with the activating examples but which the hypothesis says should not fire;
   - **positive holdouts** (at least four): new contexts, including paraphrases of the claimed feature, that the hypothesis says should fire.
6. **Measure.** The harness runs every probe and counterfactual, measures the lookahead profile (KL between clean and pole-ablated next-token distributions, and how concentrated the change is), and computes the causal metrics.
7. **Decide.** Apply the rule in §4.
8. **Record.** Every completed experiment, whatever its verdict, gets a write-once page in `wiki/experiments/<model>/L<layer>/N<idx>-<pole>/<timestamp>.md`. The page holds the hypothesis, evidence, gate results, failed probes and next steps derived from the failed gates. `verify` and `reevaluate` write it automatically, and the superseded verdict is saved before `reevaluate` overwrites a record. The page's Discussion section is for free-form notes.
9. **Publish.** Only `VERIFIED` records are compiled into the public site and `registry.parquet`. `audits/` holds the latest record per neuron; `wiki/` holds every experiment.

## 4. Decision rule

Probe accuracy is reported as **balanced accuracy**, the mean of:
- **specificity**: the fraction of hard negatives that stayed inactive;
- **sensitivity**: the fraction of positive holdouts that fired.

Balanced accuracy is the `confidence_score`. Using it means many easy negatives cannot hide failed generalisations.

A verdict is **VERIFIED** only if every gate passes:

| Gate | Default | Why |
| --- | --- | --- |
| `enough_hard_negatives` | ≥ 6 | Specificity is meaningless on a handful of probes. |
| `enough_holdouts` | ≥ 3 | The same applies to sensitivity. |
| `specificity` | ≥ 0.80 | The hypothesis is not too broad. |
| `sensitivity` | ≥ 0.60 | The hypothesis generalises beyond its own examples. |
| `counterfactual_reproduction` | ≥ 0.75 of pairs | The proposer's predicted Δ reproduce within `max(0.10, 0.50 × |predicted|)`, with the right sign. |
| `lookahead_consistent` | — | The measured mechanism matches the claim. |
| `independent_review` | — | The reviewer model differs from the proposer model. |

If the probe set is adequate (both count gates pass) and balanced accuracy is below **0.60**, the verdict is **REJECTED**: the hypothesis predicts the neuron barely better than chance. Every other outcome is **NEEDS_REVISION**, and the failed gates are listed in `verdict.notes`.

### What is not evidence (yet)

- **The proposer's counterfactuals.** The proposer can measure these before submitting, so a match is a reproduction check (it catches measurement or transcription errors) and not a successful prediction. The predictive test is the reviewer's probes, which are written without seeing any activations.
- **Causal metrics.** Sufficiency and necessity (clean↔corrupted patching, with downstream MLP outputs frozen against self-repair) are recorded but not gated. They are scored on the top-1 next-token logit, and a single neuron rarely moves that measurably. A better target, such as the logits of the claimed tokens or the neuron's output direction in the residual stream, is open work.

## 5. Reproducibility requirements

A record must let a third party re-derive its verdict:

- `identifier`: model name, layer, neuron, pole.
- `provenance`: model revision (commit hash), dtype, corpus and its SHA-256, snippet and token counts, selection method and seed, harness version.
- The full probe set with labels and rationales, stored as `adversarial_test_results`.
- `verdict.policy`: the exact thresholds, including `protocol_version`.

`openinterp reevaluate --record <file>` re-measures a record's stored probes under the current policy, without LLM calls.

## 6. Changing the protocol

Any change to the decision rule or its defaults bumps `PROTOCOL_VERSION` in `core/schemas.py` and adds an entry below. Records keep the version they were judged under. Re-evaluating them is an explicit step, never a silent one.

| Version | Change |
| --- | --- |
| 0.1 | Weighted mean of probe, counterfactual and lookahead pass rates; absolute Δ tolerance of 0.35. |
| 0.2 | Gated rule with balanced accuracy; relative Δ tolerance; minimum probe counts; independent-review gate; provenance block. |

## 7. Worked example

Qwen2.5-0.5B, layer 11, neuron 4255, positive pole, selected at random (`audits/qwen-L11-N4255-positive.json`).

- **Hypothesis:** fires on the connective that introduces examples of a category just named ("industries, *including* …", "such *as*").
- **v0.1 verdict: VERIFIED (0.91).** Ten easy negatives outweighed three failed holdouts, and Δ predictions of 0.40 and 0.30 "passed" with observed values of 0.11 and 0.08.
- **v0.2 verdict: NEEDS_REVISION.** Specificity was 1.0 but sensitivity only 0.4: "for example", "like" and "among them" did not fire. Counterfactual reproduction was 1/4. Proposer and reviewer were the same model.
- **Next submission should:** narrow the claim to the token *including* (and *e.g.*) after a category noun, and get an independent reviewer.

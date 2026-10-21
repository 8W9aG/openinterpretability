# Audit protocol (v0.5)

This document defines what one Open Interpretability experiment is, who does what, and how a verdict is decided. The code implements it: the decision rule is `openinterp.agents.reviewer.evaluate_submission`, and its parameters are `openinterp.core.schemas.ReviewPolicy`. Every verdict stores the policy it was computed under, so any record can be re-checked with `openinterp reevaluate`.

## 1. Unit of analysis

One experiment tests **one hypothesis about one neuron pole** in one model revision.

| Term | Definition |
| --- | --- |
| Neuron | One coordinate of the MLP intermediate activation, i.e. the input to the down/output projection (`act(gate(x)) * up(x)` for gated MLPs). |
| Attention head | One head's slice of the attention output projection's input (`o_proj`, or `c_proj` in GPT-2). Heads have an identifier (`HeadIdentifier`, `kind = "attn_head"`) and harness hooks, but no audit protocol yet: submissions take neuron poles only. |
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
8. **Record.** Every completed experiment, whatever its verdict, gets a write-once page in `wiki/experiments/<model>/L<layer>/N<idx>-<pole>/<timestamp>.md`. The page holds the hypothesis, evidence, gate results, failed probes and next steps derived from the failed gates. `verify` and `reevaluate` write it automatically, next to the experiment's write-once record `<timestamp>.json`. Re-evaluations, revisions and re-reviews are new experiments: nothing is ever overwritten. The page's Discussion section is for free-form notes.
9. **Publish.** Every experiment is published, whatever its verdict. The site and `registry.parquet` are built from the archived `<id>.json` records, the registry's only source of truth. A neuron pole may have any number of experiments, and its standing (latest or most conclusive verdict) is derived from them, never stored. Pages follow Model → Layer → Neuron → Experiments, and each neuron's experiments are ordered by **conclusiveness**:
   - With Beta(1, 1) priors on specificity and sensitivity, P(useful) = P(balanced accuracy > 0.6).
   - Conclusiveness is |2·P(useful) − 1|, so a well-probed rejection ranks as high as a well-probed verification, and thinly-probed results rank low whatever their verdict.

## 4. Decision rule

Probes are not independent: one reviewer writes them all, often as variations on a theme. Near-duplicate probes (word-set Jaccard ≥ `probe_cluster_threshold`, default 0.5) are therefore clustered, and each cluster counts as **one observation**, scored by its mean pass rate. Every count and rate below uses these effective counts.

Probe accuracy is reported as **balanced accuracy**, the mean of:
- **specificity**: the cluster-weighted fraction of hard negatives that stayed inactive;
- **sensitivity**: the cluster-weighted fraction of positive holdouts that fired.

Balanced accuracy is the `confidence_score`. Using it means many easy negatives cannot hide failed generalisations.

**Confound control.** The scan records a confound profile of the pole: the variance explained by token identity and by position, the correlation with token frequency, and the tokens and positions with the highest mean strength. On the reviewer's probes, the hypothesis is compared with three trivial baselines, each scored by the same cluster-weighted balanced accuracy:
- *token identity*: active if any token is a trigger token;
- *position*: active at a high-activation position;
- *never active*.

A hypothesis that can't at least match the best of these adds nothing beyond "fires on token X".

A verdict is **VERIFIED** only if every gate passes:

| Gate | Default | Why |
| --- | --- | --- |
| `enough_hard_negatives` | ≥ 6 clusters | Specificity is meaningless on a handful of independent probes. |
| `enough_holdouts` | ≥ 3 clusters | The same applies to sensitivity. |
| `specificity` | ≥ 0.80 | The hypothesis is not too broad. |
| `sensitivity` | ≥ 0.60 | The hypothesis generalises beyond its own examples. |
| `counterfactual_reproduction` | ≥ 0.75 of pairs | The proposer's predicted Δ reproduce within `max(0.10, 0.50 × |predicted|)`, with the right sign. |
| `lookahead_consistent` | — | The measured mechanism matches the claim. |
| `independent_review` | off (`require_independent_review`) | The reviewer model differs from the proposer model. Always recorded; gated only when the policy requires it. |
| `beats_confound_baseline` | ≥ best baseline | The hypothesis explains the probes at least as well as token identity, position or "never active". It fails when the submission has no confound profile. |

If the probe set is adequate (both count gates pass) and balanced accuracy is below **0.60**, the verdict is **REJECTED**: the hypothesis predicts the neuron barely better than chance. Every other outcome is **NEEDS_REVISION**, and the failed gates are listed in `verdict.notes`.

### What is not evidence (yet)

- **The proposer's counterfactuals.** The proposer can measure these before submitting, so a match is a reproduction check (it catches measurement or transcription errors) and not a successful prediction. The predictive test is the reviewer's probes, which are written without seeing any activations.
- **Covariance with other neurons.** Pole strengths are sparse and heavy-tailed, so a global correlation mostly measures agreement on "nothing is happening". The scan therefore finds partner neuron *poles* by co-activation lift: P(partner active | target active) ÷ P(partner active).
  - *Target active* uses the protocol threshold; *partner active* means above the partner pole's own 99th percentile.
  - Partners are ranked by the 95% Wilson lower bound of the lift, so a few chance coincidences don't count.
  - Each partner also records Spearman's ρ of the two pole strengths (on a fixed-seed reservoir sample, ties averaged) and, for reference, Pearson's r with the signed activation.
  - The reviewer ablates the pole with up to 3 partners (lift lower bound ≥ 2 on ≥ 5 co-active tokens, or |ρ| ≥ 0.2). It reports the KL for the pole alone, the partners alone and all together.
  - The *redundancy ratio*, (KL together − KL partners) / KL alone, exceeds 1 when partners mask the pole's effect, and is near 0 when the pole's effect overlaps with theirs.

  This is reported, not gated.
- **Vocabulary lens.** Each submission records the pole's output direction read through the logit lens and the **Jacobian lens** (Gurnee et al., 2026). The Jacobian lens transports the direction with the context-averaged Jacobian E[∂h_final,t′/∂h_ℓ,t] over t′ ≥ t, estimated from one perturbed forward pass per prompt and sign. The record includes:
  - both top-10 lists;
  - their overlap and cos(w, J·w);
  - the kurtosis of the vocabulary scores (≈ 3 means no specific token set).

  On Qwen2.5-0.5B the lenses disagree completely up to the middle layers and converge near the output, so mechanism claims should rest on the Jacobian lens. It is reported, not gated.
- **Causal metrics.** Sufficiency (clean → corrupted patch) and necessity (corrupted → clean patch) are measured on each minimal pair, with downstream MLP outputs frozen against self-repair. They are recorded, not gated.
  - *Where.* The pole is patched at every position and scored at the position where its strength differs most between the two prompts. When the prompts differ in token length, only the final position can be patched and scored.
  - *What.* The target favours no particular neuron. A lookahead claim is scored on its claimed upcoming tokens: their mean logit minus the vocabulary mean (`claimed_tokens`). Anything else is scored on the KL between the unpatched and patched next-token distributions (`next_token_kl`).
  - *Compared with what.* The same patch is applied, one neuron at a time, to the 32 same-layer neurons (`causal_null_neurons`) whose strength changes most similarly between the pair's prompts. Each score is the median over pairs of the pole's |effect| percentile among theirs. 0.5 means "no more influential than a neuron that changes as much"; 1.0 means it beats all of them. The raw median effects are recorded too.
  - *Which pairs.* Control pairs, where the pole moves by at most `delta_abs_tolerance × scale`, carry no causal signal and are skipped. With none left, no causal metrics are recorded.

  Two simpler scores read wrongly on Qwen2.5-0.5B. The fraction of the whole clean/corrupted gap restored (protocol ≤ 0.4, on the top-1 logit) reads ≈ 0 for any single neuron: the gap sums every component's contribution, and the rest of the model often moves the other way. A percentile against *random* neurons reads ≈ 1 for every neuron, because minimal pairs are built to change the audited neuron and most others barely change.

  The `next_token_kl` target shows that a neuron matters, not that it matters *for the reason the hypothesis gives*. A causal test of a retrospective claim would need the hypothesis to name the behaviour it affects. Gating on these scores needs a pre-registered prediction format first (see TODO).
- **Explanation scores on held-out text** (Delphi; Paulo et al., 2025), from `openinterp score`, attached with `review --explanation-scores` (or `reevaluate`, for an existing record). They test the explanation on text nobody chose, where probes test it on text the reviewer chose.
  - *Windows.* The corpus snippets that are not the scan's top examples, so the proposer never saw them, are cut into 32-token windows. Each window is measured on its own, exactly as the judge reads it, the way probes are. Active windows (peak ≥ the activation threshold) are split into 5 bins by peak strength. Each bin gives 5 windows to detection and 5 to fuzzing, so weak activations count as much as the top ones. Detection also gets as many inactive windows.
  - *Detection.* A judge given only the explanation lists the windows it says activate the pole. Score: balanced accuracy, with recall per activation bin.
  - *Fuzzing.* In each window some tokens are marked: the active tokens, the same number of inactive tokens, or random tokens in an inactive window. The judge lists the windows whose marks are where the explanation says the pole fires. Score: balanced accuracy.
  - *Embedding.* The AUROC of the cosine between the explanation and each detection window, from a local sentence encoder (`sentence-transformers/all-MiniLM-L6-v2` by default). It needs no LLM.

  The task a judge answers holds no activations: labels are measured when the answers are scored, and `openinterp reproduce` re-measures them and re-derives the scores from the stored answers (warnings only). 0.5 is chance for all three. A narrow explanation that passes review can still score low here, if most of the pole's held-out activity falls outside it.

## 5. Reproducibility requirements

A record must let a third party re-derive its verdict:

- `identifier`: model name, layer, neuron, pole.
- `provenance`: model revision (commit hash), dtype, corpus and its SHA-256, snippet and token counts, selection method and seed, harness version.
- The full probe set with labels and rationales, stored as `adversarial_test_results`.
- `verdict.policy`: the exact thresholds, including `protocol_version`.
- `verdict.environment` (and `provenance.environment` for the scan): the device, weights dtype, model commit and the openinterp, torch and transformers versions the measurements ran with.

`openinterp reevaluate --record <file>` re-measures a record's stored probes under the current policy, without LLM calls. The result is a new experiment.

`openinterp reproduce <file>…` (or `--base-ref origin/main` for every record a branch adds or changes) checks that a record is what it claims to be. It loads the model commit and dtype the record pins and re-runs everything the verdict rests on, without LLM calls:
- each top example's per-token strengths, and the scale (`max_activation`) that every threshold is a fraction of;
- each counterfactual's observed Δ;
- the review, with the record's own probes, reviewer and policy.

Numbers must agree within `atol × scale + rtol × |recorded|` (defaults 0.001 and 0.01). These absorb float differences between devices and library versions, but not an edited number. Every pass/fail, every gate and the status must agree exactly. Causal metrics and joint ablation are not gated, so a difference there is only a warning. A different device or library version is reported as a warning. A record judged under another protocol version can't be reproduced by the current harness, because its decision rule was different: re-evaluate it instead.

A trusted maintainer can instead **sign** a record (`openinterp sign <file>`, with an SSH key listed in `.github/allowed_signers`). This writes `<id>.json.sig` next to the record, over its exact bytes. `reproduce` then skips the record, reporting it as signed. A signature that doesn't verify fails like a mismatch. With `--base-ref`, the signers list is read from the base ref, so a branch can't trust its own key. A signature doesn't change the verdict, which still comes from the review protocol. It only records that the signer vouches for the record.

Reproduction can't show that the reviewer wrote its probes without seeing activations. It shows only that the record's numbers and verdict follow from its probes and the model.

## 6. Changing the protocol

Any change to the decision rule or its defaults bumps `PROTOCOL_VERSION` in `core/schemas.py` and adds an entry below. Records keep the version they were judged under. Re-evaluating them is an explicit step, never a silent one.

| Version | Change |
| --- | --- |
| 0.1 | Weighted mean of probe, counterfactual and lookahead pass rates; absolute Δ tolerance of 0.35. |
| 0.2 | Gated rule with balanced accuracy; relative Δ tolerance; minimum probe counts; independent-review gate; provenance block. |
| 0.3 | Near-duplicate probes cluster into single observations; confound-control gate against token, position and never-active baselines; correlated neurons and joint ablation recorded. |
| 0.4 | `independent_review` is recorded but no longer gated by default (`require_independent_review = False`). |
| 0.5 | Causal metrics (still not gated) are scored on claimed tokens or next-token KL at the pole's most-changed position, as percentiles against activation-change-matched same-layer neurons; control pairs are skipped. Replaces the fraction of the top-1-logit gap restored. |

### Registry-level dependence

These do not change the decision rule.
- **Re-measurements.** An experiment that re-measures an earlier experiment's probe set (e.g. `reevaluate`) is marked as such. It is ranked after the original and is never counted as independent evidence.
- **Most-conclusive list.** The home page shows one row per neuron pole.
- **False discovery estimate.** The home page reports the expected number of false discoveries among current VERIFIED claims, Σ (1 − P(useful)) over one claim per neuron pole.

## 7. Worked example

Qwen2.5-0.5B, layer 11, neuron 4255, positive pole, selected at random (`wiki/experiments/Qwen--Qwen2.5-0.5B/L11/N4255-positive/`).

- **Hypothesis:** fires on the connective that introduces examples of a category just named ("industries, *including* …", "such *as*").
- **v0.1 verdict: VERIFIED (0.91).** Ten easy negatives outweighed three failed holdouts, and Δ predictions of 0.40 and 0.30 "passed" with observed values of 0.11 and 0.08.
- **v0.2 verdict: NEEDS_REVISION.** Specificity was 1.0 but sensitivity only 0.4: "for example", "like" and "among them" did not fire. Counterfactual reproduction was 1/4. Proposer and reviewer were the same model.
- **v0.3 re-evaluation: NEEDS_REVISION.** It also fails `beats_confound_baseline`, because the scan predates confound profiling. A rescan found:
  - token identity explains 24% of activation variance and position about 0%;
  - the mean strength on " including" is 0.37, below the 0.43 "active" threshold, so the token alone does not make the neuron fire and context matters;
  - no layer-11 neuron correlates above |r| = 0.10, yet co-activation tells a different story. Ten neuron poles co-fire on 40–52 of its ~180 active tokens, a lift of at least 10–13 at the 95% lower bound. Global correlation hid them because the coupling exists only when the neuron fires.
  - Ablating the top three partners moves the output about 14× more than ablating N4255 alone, and with the partners removed N4255 adds almost nothing (redundancy ratio 0.02). Its effect overlaps a shared pathway rather than being masked.
  - Its logit-lens tokens ("oor", " Fälle", ".ordinal") were noise. The Jacobian lens shares none of them and is just as flat (kurtosis 3.0), which supports the retrospective claim rather than revealing a hidden prediction role.
- **Next submission should:** narrow the claim to the token *including* (and *e.g.*) after a category noun, and get an independent reviewer.

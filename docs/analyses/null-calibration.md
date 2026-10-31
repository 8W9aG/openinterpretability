# Null calibration: the review's false-positive rate

This analysis starts the *Null calibration* item of `TODO.md` (*Protocol and science*). The question it asks is how often the review verifies a claim that should fail. Registry-level claims ("x% of neurons have a verified explanation") mean nothing without that rate, since with many agents, neurons and gates some VERIFIED verdicts will be chance.

All three of the item's nulls are measured here. The first two reuse the probes already in the registry, so they need no LLM call; a sample of the second is then re-reviewed with fresh probes. The third, units in a randomly rotated MLP basis, runs the whole pipeline with proposer and reviewer agents against a matched native control.

The rows are in `null-calibration-shuffle.jsonl` (one per record), `null-calibration-swap.jsonl` (one per swapped hypothesis), `null-calibration-fresh.jsonl` (the re-reviewed sample) and `null-calibration-basis.jsonl` (one per basis unit), written by `scripts/null_calibration.py`, `scripts/swap_fresh.py` and `scripts/basis_null.py`. Reported, not gated.

## Method

**Records.** Every native neuron-pole record judged under protocol 0.7 or 0.8: those have a corpus sample and a confound profile. Every verdict below is computed under the current policy (0.8).

**Shuffled labels.** For each record, the hypothesis's labels (hard negative or positive holdout) are permuted among its measured probes, 200 times. The permutation keeps the number of each kind, and each probe keeps the activation the record measured, so no model is run. Each permutation is re-decided by the review's own code (`review_metrics`, `decide`), with the confound baselines re-scored against the permuted labels and the record's counterfactuals and lookahead check as recorded. The labels then carry no information about the unit, so any VERIFIED is a false positive. As a check, the unpermuted labels re-derive the recorded verdict for 1,142 of 1,151 records. All 9 mismatches are older 0.7 records, which the 0.8 rule judges differently.

**Swapped unit.** For each neuron pole whose newest VERIFIED record scanned a corpus available locally (`corpus.jsonl` or the builtin sample: 255 of 309), the record's hypothesis, counterfactuals and probe set are scored against a random other neuron of the same layer and pole (seeded per unit). The other neuron is scanned as its own audit would scan it, with the same corpus, sample and windows. If the scan stops on it (dead or default-on), the outcome is that, as in a real audit. Otherwise its scale, histogram, first-token floor and confound profile replace the record's, and the review is re-run on it, without causal metrics, joint ablation or wiring, none of which is gated. The other 54 scanned a corpus file that isn't local (50 FineWeb-Edu, 4 others) and are left out.

This null is generous to the review in one way. The probes were written for neuron A's top examples, while a reviewer of neuron B would have seen B's. It is strict in another: A's counterfactuals are measured on B, but a real proposer measures its counterfactuals on the neuron it submits.

**Swapped unit, fresh probes.** 40 of the 253 swaps that reached the review (seeded sample) are reviewed again from scratch: a reviewer agent (Claude Sonnet 5.5, `neuron-reviewer`) gets A's hypothesis with B's scan (its top examples, confounds and scale, but not A's input attribution or lenses) and writes 20 new probes, which are measured on B. This removes the reuse of A's probes.

**Rotated basis.** Units are audited as latents of a `basis` (`DictionaryProfile.kind`), a new input site that reads the MLP's hidden activations where native neurons live (`mlp_hidden`, the down projection's input). A latent's strength is the positive part of the hidden activations' projection onto one direction, and it writes that strength times the direction through the down projection, so ablating it zeroes that component. Two arms run through the same code:
- *identity*: the basis vectors are the standard ones, so each latent is exactly a native positive pole (checked: the activations match to 0.0, and the writes sum back to the down projection's output);
- *rotation*: a Haar-random orthogonal matrix (seeded per layer), so each latent is a random direction in the same space.

30 units are drawn on Qwen2.5-0.5B (uniform layer, then index; 20 distinct layers), and each is audited in both arms, giving 30 pairs. The 60 units are shuffled into opaque names, both arms share one dictionary name and one prompt text, which describes the unit as "one direction in this layer's MLP hidden space", and the weights sit in hash-named directories. So the agents can't tell which arm a unit is in, and I didn't look at the arms until all 60 were measured. Each unit gets the registry's pipeline: the audit scan (corpus `corpus.jsonl`, 512-token windows, 200,000-token sample), a proposer agent (Claude Opus 5.5, `neuron-proposer`) that may measure its own minimal pairs, the counterfactuals measured on submission, a reviewer agent (Claude Sonnet 5.5) writing 20 probes, and the review under the current policy, without causal metrics or wiring. Both arms get the same agents, which the registry can't offer: its native verdict rates differ several-fold by agent pair.

**Without the proposer's gates.** A proposer measures its counterfactuals and lookahead profile before submitting (`docs/PROTOCOL.md` §4, *What is not evidence*), so those two gates catch transcription errors, not wrong hypotheses. Each verdict is therefore also read with every gate but `counterfactual_reproduction` and `lookahead_consistent`. This is the false-positive rate for a proposer who measured honestly on the wrong hypothesis.

## Results

### Shuffled labels

The share of permutations VERIFIED, averaged over records, by each record's original verdict:

| Model | Original verdict | Records | VERIFIED | Without proposer's gates | Records ever VERIFIED | Re-derived = recorded |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-2 | NEEDS_REVISION | 125 | 0.76% | 1.31% | 48 | 124/125 |
| GPT-2 | REJECTED | 83 | 0.45% | 0.69% | 13 | 83/83 |
| GPT-2 | VERIFIED | 58 | 3.86% | 3.86% | 49 | 53/58 |
| GPT-2 | all | 266 | 1.34% | 1.67% | 110 | 260/266 |
| Qwen | NEEDS_REVISION | 328 | 1.42% | 1.83% | 173 | 325/328 |
| Qwen | REJECTED | 273 | 0.78% | 1.11% | 75 | 273/273 |
| Qwen | VERIFIED | 284 | 4.34% | 4.34% | 271 | 284/284 |
| Qwen | all | 885 | 2.16% | 2.41% | 519 | 882/885 |

**A labelling that knows nothing is VERIFIED about 2% of the time, and about 4% on the probe sets that verified.** On Qwen it is 2.2% over all 885 records and 4.3% over the 284 VERIFIED ones; on GPT-2, 1.3% and 3.9%. A probe set's own verdict matters because a VERIFIED probe set has the active probes and the hard negatives a permutation needs: a handful of probes fire, the rest stay silent, and a random relabelling sometimes puts most of the active ones among the holdouts. Probe sets of REJECTED records, where the probes fire regardless of label or not at all, rarely verify under any labelling (0.5–0.8%). Leaving out the proposer's gates changes little (2.4% on Qwen), because the counterfactuals stay as recorded and mostly pass.

**More probes lower it.** This follows from small counts: take 16 probes, 6 of them holdouts, of which 6 fire. A random split puts at least 4 of the 6 active probes among the holdouts (sensitivity ≥ 0.6, and at most 2 active negatives, specificity 0.8) with probability 9%, before clustering and the confound gate. Over all records, 16-probe sets (1,037 records) verify under 2.1% of permutations and 20-probe sets (95 records) under 0.5%. The reviewer's default was 16; it is now 20, the protocol's maximum (`openinterp review --n-probes`), which should cut this null's rate about fourfold.

**Most VERIFIED probe sets can be fooled by some labelling.** 271 of 284 Qwen and 49 of 58 GPT-2 VERIFIED probe sets were VERIFIED by at least one of their 200 permutations. Judged once each with a labelling that knows nothing about the unit, about 14 of the 342 VERIFIED probe sets would still pass.

### Swapped unit

| Model | Pole | Hypotheses | Dead | Default-on | REJECTED | NEEDS_REVISION | VERIFIED | False positives | Without proposer's gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-2 | both | 49 | 0 | 2 | 39 | 8 | 0 | 0.0% [0.0, 7.3] | 2.0% [0.4, 10.7] |
| GPT-2 | negative | 1 | 0 | 1 | 0 | 0 | 0 | 0.0% [0.0, 79.3] | 0.0% [0.0, 79.3] |
| GPT-2 | positive | 48 | 0 | 1 | 39 | 8 | 0 | 0.0% [0.0, 7.4] | 2.1% [0.4, 10.9] |
| Qwen | both | 206 | 0 | 0 | 163 | 43 | 0 | 0.0% [0.0, 1.8] | 1.9% [0.8, 4.9] |
| Qwen | negative | 101 | 0 | 0 | 76 | 25 | 0 | 0.0% [0.0, 3.7] | 2.0% [0.5, 6.9] |
| Qwen | positive | 105 | 0 | 0 | 87 | 18 | 0 | 0.0% [0.0, 3.5] | 1.9% [0.5, 6.7] |

Gates failed, over the swapped reviews that reached the review:

| Gate | Failed |
| --- | --- |
| `enough_hard_negatives` | 0/253 |
| `enough_holdouts` | 0/253 |
| `specificity` | 47/253 |
| `sensitivity` | 224/253 |
| `counterfactual_reproduction` | 253/253 |
| `lookahead_consistent` | 0/253 |
| `independent_review` | 0/253 |
| `beats_confound_baseline` | 223/253 |
| `not_default_on` | 0/253 |

**No swapped hypothesis was VERIFIED: 0 of 255, a false-positive rate below 1.5% (95% Wilson upper bound).** Every swapped review that reached the review failed `counterfactual_reproduction`: the proposer's predicted Δ, measured on neuron A, don't reproduce on neuron B. That gate alone would have stopped every one of them. Only 2 of the 255 other neurons were unauditable, both default-on GPT-2 poles. The registry has only one VERIFIED GPT-2 negative pole, since most of GPT-2's negative poles are default-on.

**Without the proposer's gates, 5 of 255 pass (2.0%, 95% interval 0.8–4.5%),** in line with the shuffled-label null. Most swapped reviews fail on the probes themselves: the median balanced accuracy is 0.50, and on the median swap only 1 of 16 probes fires on B. The gates that fail are `sensitivity` (224 of 253) and `beats_confound_baseline` (223), and less often `specificity` (47). The 5 that pass are:

| Hypothesis (A) | Scored on (B) | Specificity | Sensitivity | Active probes |
| --- | --- | --- | --- | --- |
| Qwen L13/N3774− (identifiers in source code) | N3011− | 1.00 | 0.67 | 4/16 |
| Qwen L21/N954+ (geometry and plotting terms) | N543+ | 1.00 | 1.00 | 5/16 |
| Qwen L4/N52− (token closing a conventional frame) | N2185− | 0.91 | 0.80 | 5/16 |
| Qwen L8/N124+ (formulaic, memorised text) | N489+ | 0.82 | 0.80 | 6/16 |
| GPT-2 L4/N2755+ (≥ 8 tokens into the context) | N1050+ | 0.91 | 1.00 | 6/16 |

Some of these may not be false in the usual sense. GPT-2's L4/N2755+ claims a positional feature, and many GPT-2 neurons are positional, so a random neuron of its layer could share it. Qwen L21/N954+ → N543+ passes every probe, which suggested the two neurons share a topic. But both were in the fresh-probe sample below, and both failed there: a reviewer who saw B's own examples found where A's hypothesis is wrong about B. The reused probes simply didn't test what B does differently.

**What this means for the registry.** The counterfactual gate makes the measured rate 0%, but it guards against the wrong thing: a proposer measures its counterfactuals on the neuron it submits, so a wrong hypothesis from an honest proposer passes it. The rate that applies to real submissions is the one without the proposer's gates. With reused probes it is about 2% (swapped unit) to 4% (shuffled labels on VERIFIED probe sets). With fresh probes written from the swapped unit's own examples it is 0 of 40 (below 8.8%; see below). Take the reused-probe rates as an upper bound. Suppose every native record that wasn't VERIFIED (68% of 1,256) tested a wrong hypothesis, and wrong hypotheses verify at that rate. Then false positives would make up at most roughly 4–9% of VERIFIED verdicts (0.68 × 2–4% ÷ 0.32). Partly right hypotheses, the common case, are likely to pass more often than these nulls. The home page's false discovery estimate, Σ (1 − P(useful)), is computed per claim from its own probe counts and doesn't use this rate.

### Swapped unit, fresh probes

0 of 40 re-reviewed swaps were VERIFIED, and none passed even without the proposer's gates (95% upper bound 8.8%). With reused probes, the same 40 gave 0 and 2. The two were Qwen L21/N954+ → N543+ and GPT-2 L4/N2755+ → N1050+, two of the five in the table above. Reviewers who saw B's own top examples wrote probes that B failed, so both are rejected here. 39 of 40 were REJECTED: with fresh probes the median balanced accuracy is 0.46, and 2.5 of 20 probes fire on B. `sensitivity` failed on 37 of 40 and `specificity` on 18. A reviewer reading B's examples writes hard negatives near what B does, and A's hypothesis gets those wrong.

So reusing A's probes made the swapped-unit null too lenient, not too strict. A review that sees the unit's own examples is the stronger test, and with it the rate for a wrong hypothesis in this sample is 0 of 40.

### Rotated basis

| Arm | Units | VERIFIED | NEEDS_REVISION | REJECTED | Share VERIFIED | Without proposer's gates | Median density | Median specificity / sensitivity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Identity (native positive poles) | 30 | 18 | 9 | 3 | 60.0% [42.3, 75.4] | 63.3% [45.5, 78.1] | 49% | 0.93 / 0.86 |
| Rotation (random directions) | 30 | 1 | 17 | 12 | 3.3% [0.6, 16.7] | 3.3% [0.6, 16.7] | 48% | 0.50 / 1.00 |

**Native units verify far more often than rotated ones: 18 of 30 against 1 of 30.** Over the 30 pairs, in 17 only the native unit was VERIFIED, in 1 both were and in 12 neither was; no pair had only the rotated unit VERIFIED (two-sided sign test on the 17 discordant pairs, p = 1.5 × 10⁻⁵). No unit in either arm was dead or default-on, and the two arms are equally dense (about half of all tokens have a positive projection), so density doesn't explain the gap. The rotated units fail on `specificity` (29 of 30; median 0.50): the agent's hypothesis covers the top examples and the holdouts fire (median sensitivity 1.00), but so do the hard negatives. A rotated unit mixes thousands of native neurons, so its top examples are a grab bag, and any account of them is too broad or too narrow. The one rotated unit VERIFIED (L23, "an orphaned fragment … inside an unpredictable, non-dictionary token run") is a broad claim of the kind a mixture can satisfy.

**This is the native-basis premise, tested directly.** If the neuron basis weren't privileged, rotated units would verify about as often as native ones (Elhage et al., 2021). Under this protocol and these agents they verify 18 times less often. The 60% for native positive poles is also well above the registry's 31% for random Qwen draws. It is the same protocol with one agent pair (Opus proposing, Sonnet reviewing), 20 probes and a measuring tool for the proposer, which supports reading the registry's spread of verdict rates as partly an agent effect (TODO: *Reviewer-effect crossover*).

## Caveats

- **The shuffled-label null measures the decision rule, not the agents.** It asks what a labelling that knows nothing scores on the probes a reviewer actually wrote. A real wrong hypothesis isn't random: it is usually partly right, so its false-positive rate could be higher.
- **The swapped-unit null reuses A's probes.** The fresh-probe sample above removes this for 40 swaps, and it made the null stricter: 0 of 40, against 2 of 40 with reused probes.
- **The rotated-basis arms are small.** 30 pairs on one model, one rotation seed and one agent pair. The gap is large enough that the sign test is clear, but the native rate (60%, interval 42–75%) is not precise. The agents were not given the registry's methodology skills, only their unit's prompt and a measuring tool (`basis_null.py strengths`), in both arms alike. Agents occasionally wrote 21 probes; they were asked to drop one, before any measurement.
- **Only Qwen2.5-0.5B's positive poles.** SwiGLU's hidden activations take both signs about equally, so the positive part of a random direction is a fair comparison for a native positive pole. In GPT-2's GELU MLP the negative side is mostly GELU's dip, and the same comparison would need care.
- **Permutations within a record are not independent.** The per-record rates are averaged over records, so each record weighs equally, and no interval is given for them.
- **One seed for the swapped neurons.**

## Reproduce

```
A=docs/analyses/null-calibration
uv run python scripts/null_calibration.py shuffle --out $A-shuffle.jsonl
uv run python scripts/null_calibration.py swap --out $A-swap.jsonl
uv run python scripts/null_calibration.py report $A-shuffle.jsonl $A-swap.jsonl

# fresh probes: prepare writes exchange/swap-fresh/sNN/review.prompt.md, answered by
# neuron-reviewer agents in review.probes.json
uv run python scripts/swap_fresh.py prepare --swaps $A-swap.jsonl
uv run python scripts/swap_fresh.py measure --agent-model external/claude-sonnet-5-5
uv run python scripts/swap_fresh.py report --out $A-fresh.jsonl

# rotated basis: keep the plan outside the repository (it names each unit's arm)
P=/tmp/basis-plan.json
uv run python scripts/basis_null.py plan --pairs 30 --seed 0 --out $P
uv run python scripts/basis_null.py weights --plan $P
uv run python scripts/basis_null.py scan --plan $P      # -> exchange/basis/uNN/ev.prompt.md
# neuron-proposer agents answer ev.hypothesis.json (measuring with `basis_null.py strengths`)
uv run python scripts/basis_null.py submit --plan $P --agent-model external/claude-opus-5-5
# neuron-reviewer agents answer review.probes.json
uv run python scripts/basis_null.py measure --plan $P --agent-model external/claude-sonnet-5-5
uv run python scripts/basis_null.py report --plan $P > $A-basis.jsonl
```

`shuffle` takes about 4 minutes on CPU. `swap` takes several hours on Apple's GPU: about 1–2 minutes per unit, most of it replaying the layer's cached scan. It appends to its output and skips records already done, so it can be stopped and resumed. The fresh-probe sample re-scans its 40 null neurons from the layer cache. The basis scans take about 2–3 minutes per unit on Apple's GPU, plus a minute or two per agent; `basis_null.py status` shows each unit's stage without its arm.

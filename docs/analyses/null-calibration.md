# Null calibration: the review's false-positive rate

This analysis starts the *Null calibration* item of `TODO.md` (*Protocol and science*). The question it asks is how often the review verifies a claim that should fail. Registry-level claims ("x% of neurons have a verified explanation") mean nothing without that rate, since with many agents, neurons and gates some VERIFIED verdicts will be chance.

Two of the item's three nulls are measured here. Both reuse the probes already in the registry, so they need no LLM call. The third, units in a randomly rotated MLP basis, needs new harness support and fresh proposer and reviewer runs, and is left open (see *Not measured yet*).

The rows are in `null-calibration-shuffle.jsonl` (one per record) and `null-calibration-swap.jsonl` (one per swapped hypothesis), written by `scripts/null_calibration.py`. Reported, not gated.

## Method

**Records.** Every native neuron-pole record judged under protocol 0.7 or 0.8: those have a corpus sample and a confound profile. Every verdict below is computed under the current policy (0.8).

**Shuffled labels.** For each record, the hypothesis's labels (hard negative or positive holdout) are permuted among its measured probes, 200 times. The permutation keeps the number of each kind, and each probe keeps the activation the record measured, so no model is run. Each permutation is re-decided by the review's own code (`review_metrics`, `decide`), with the confound baselines re-scored against the permuted labels and the record's counterfactuals and lookahead check as recorded. The labels then carry no information about the unit, so any VERIFIED is a false positive. As a check, the unpermuted labels re-derive the recorded verdict for 1,142 of 1,151 records. All 9 mismatches are older 0.7 records, which the 0.8 rule judges differently.

**Swapped unit.** For each neuron pole whose newest VERIFIED record scanned a corpus available locally (`corpus.jsonl` or the builtin sample: 255 of 309), the record's hypothesis, counterfactuals and probe set are scored against a random other neuron of the same layer and pole (seeded per unit). The other neuron is scanned as its own audit would scan it, with the same corpus, sample and windows. If the scan stops on it (dead or default-on), the outcome is that, as in a real audit. Otherwise its scale, histogram, first-token floor and confound profile replace the record's, and the review is re-run on it, without causal metrics, joint ablation or wiring, none of which is gated. The other 54 scanned a corpus file that isn't local (50 FineWeb-Edu, 4 others) and are left out.

This null is generous to the review in one way. The probes were written for neuron A's top examples, while a reviewer of neuron B would have seen B's. It is strict in another: A's counterfactuals are measured on B, but a real proposer measures its counterfactuals on the neuron it submits.

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

**More probes lower it.** This follows from small counts: take 16 probes, 6 of them holdouts, of which 6 fire. A random split puts at least 4 of the 6 active probes among the holdouts (sensitivity ≥ 0.6, and at most 2 active negatives, specificity 0.8) with probability 9%, before clustering and the confound gate. Over all records, 16-probe sets (1,037 records) verify under 2.1% of permutations and 20-probe sets (95 records) under 0.5%. The reviewer's default is 16 (`openinterp review --n-probes`); 20, the protocol's maximum, would cut this null's rate about fourfold.

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

Not all of these are false in the usual sense. GPT-2's L4/N2755+ claims a positional feature, and many GPT-2 neurons are positional, so a random neuron of its layer can share it. Qwen L21/N954+ → N543+ passes every probe, which suggests the two neurons share a topic (a pair of redundant neurons, or one feature spread over several, as `probe-comparison.md` found for its concepts). Each is a hypothesis about A that also describes B, which the review can't distinguish from a hypothesis about B. It can't tell a specific explanation from one that is true of several neurons.

**What this means for the registry.** The counterfactual gate makes the measured rate 0%, but it guards against the wrong thing: a proposer measures its counterfactuals on the neuron it submits, so a wrong hypothesis from an honest proposer passes it. The rate that applies to real submissions is the one without the proposer's gates, about 2% (swapped unit) to 4% (shuffled labels on VERIFIED probe sets). Suppose every native record that wasn't VERIFIED (68% of 1,256) tested a wrong hypothesis, and wrong hypotheses verify at that rate. Then false positives would make up roughly 4–9% of VERIFIED verdicts (0.68 × 2–4% ÷ 0.32). Partly right hypotheses, the common case, are likely to pass more often than these nulls. The home page's false discovery estimate, Σ (1 − P(useful)), is computed per claim from its own probe counts and doesn't use this rate.

## Caveats

- **The shuffled-label null measures the decision rule, not the agents.** It asks what a labelling that knows nothing scores on the probes a reviewer actually wrote. A real wrong hypothesis isn't random: it is usually partly right, so its false-positive rate could be higher.
- **The swapped-unit null reuses A's probes.** A reviewer who saw B's top examples could write probes that B happens to pass, or fail. Running the full pipeline on B with fresh reviewer probes would remove this, at the cost of an LLM call per unit.
- **Permutations within a record are not independent.** The per-record rates are averaged over records, so each record weighs equally, and no interval is given for them.
- **One seed for the swapped neurons.**

## Not measured yet

The rotated-basis null: units defined as random directions in the MLP's hidden space, audited by the full pipeline. If they verify about as often as native neurons, the neuron basis isn't privileged (Elhage et al., 2021). The harness can't yet audit such a unit. It would need a new input site (the down projection's input) for a synthetic dictionary whose encoder is a random orthogonal matrix, and every unit would need a fresh proposer, reviewer and judge run. It stays in `TODO.md`.

## Reproduce

```
A=docs/analyses/null-calibration
uv run python scripts/null_calibration.py shuffle --out $A-shuffle.jsonl
uv run python scripts/null_calibration.py swap --out $A-swap.jsonl
uv run python scripts/null_calibration.py report $A-shuffle.jsonl $A-swap.jsonl
```

`shuffle` takes about 4 minutes on CPU. `swap` takes several hours on Apple's GPU: about 1–2 minutes per unit, most of it replaying the layer's cached scan. It appends to its output and skips records already done, so it can be stopped and resumed.

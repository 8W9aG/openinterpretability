# Verified poles against linear probes for their concept

This analysis starts the *Linear-probe comparison* item of `TODO.md` (*Analyses per experiment*). The question it asks is whether a VERIFIED pole is the model's representation of its concept, or only part of one. A pole can pass every gate, with its hard negatives silent and its holdouts active, and still be one of many neurons the concept is spread over. The review cannot tell those cases apart. Linear probes can.

The rows are in `probe-comparison.jsonl`, one per record, written by `openinterp probe-compare` and gathered by `scripts/probe_comparison.py report`. It is reported, not gated.

## Method

**Labels.** Each concept is a regular expression over characters (`scripts/probe_comparison.py`), so no label depends on the model. A token is positive when its characters overlap a match. Each rule approximates the record's concept, not its whole explanation. Several explanations list secondary triggers (L4/N3208+ also fires on ' museum'), and no rule marks those.

| Concept | Rule | Records |
| --- | --- | --- |
| `thousands` | every character of a numeral with a thousands separator (`12,500`) | Qwen L17/N1420+ |
| `year` | a year 1500–2029 or its decade (`1960s`), not after a currency sign or separator, or an ordinal century | Qwen L4/N3208+, L13/N3326− |
| `feminine` | a feminine-marked word (she, her, women, mother, queen, Mrs, …) | Qwen L6/N4837+ |
| `street` | a capitalised thoroughfare noun after a word (Street, Avenue, Road, …) | GPT-2 L0/N1923+ |

**Sources.** Probes are trained on the registry's scan corpus (`corpus.jsonl`, 500 Pile documents) and tested on 500 FineWeb-Edu documents (`openinterp corpus --dataset fineweb-edu --seed 0`). This is the two-source rule of the `sparse-probing-search` skill: a probe that only fits the Pile's quirks fails the test set. Documents are cut into 128-token windows as a scan cuts them, and each window's first token is dropped. Up to 2,000 positive and 8,000 negative tokens are sampled per split (seed 0 for training, 1 for testing).

**Probes.** Each probe reads the layer's MLP hidden activations (signed, every neuron) or the residual stream after the layer's block:
- *logistic*: standardised features, L2 penalty 1e-3, classes weighted equally, fitted with L-BFGS;
- *mass-mean*: the difference of the class means in raw activation space (Marks & Tegmark, 2023);
- *k-sparse*: logistic probes on the k hidden neurons with the largest standardised class-mean difference on the training tokens, for k = 1, 2, 4, …, 256 (Gurnee et al., 2023).

The pole alone is scored as a one-feature probe on its strength. Every threshold is fitted on the training tokens, at the best balanced accuracy (BA), and tested on held-out tokens. The pole is also scored at the protocol's line (0.25 × its recorded scale), which fits nothing. *Rank* is the neuron's position among the layer's by class separation on the training tokens. *Neurons needed* is the smallest k within 0.02 BA of the better dense hidden-layer probe.

## Results

Held-out (FineWeb-Edu) balanced accuracy, with AUROC in brackets:

| Pole | Test tokens (+/−) | Pole, fitted | Pole, at line | Rank | Hidden, logistic | Hidden, mass-mean | Residual, mass-mean | Neurons needed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen L6/N4837+ (feminine) | 456 / 8,000 | 0.87 (0.93) | 0.78 | 2 | 0.92 (1.00) | 0.99 (1.00) | 0.99 (1.00) | 32 |
| Qwen L13/N3326− (year) | 2,000 / 8,000 | 0.82 (0.85) | 0.59 | 3 | 0.98 (1.00) | 0.97 (0.99) | 0.97 (1.00) | 32 |
| Qwen L4/N3208+ (year) | 2,000 / 8,000 | 0.71 (0.70) | 0.66 | 725 | 0.99 (1.00) | 0.98 (1.00) | 0.98 (1.00) | 32 |
| Qwen L17/N1420+ (thousands) | 651 / 8,000 | 0.68 (0.72) | 0.56 | 152 | 0.89 (1.00) | 0.97 (0.99) | 0.97 (1.00) | 32 |
| GPT-2 L0/N1923+ (street) | 15 / 8,000 | 0.93 (1.00) | 1.00 | 1 | 0.80 (1.00) | 1.00 (1.00) | 0.87 (1.00) | — |

The k-sparse curves (held-out BA):

| Pole | k = 1 | 2 | 4 | 8 | 16 | 32 | 64 | 256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen L6/N4837+ | 0.79 | 0.93 | 0.95 | 0.95 | 0.97 | 0.98 | 0.98 | 0.98 |
| Qwen L13/N3326− | 0.85 | 0.87 | 0.92 | 0.94 | 0.96 | 0.97 | 0.97 | 0.98 |
| Qwen L4/N3208+ | 0.82 | 0.84 | 0.93 | 0.94 | 0.96 | 0.97 | 0.98 | 0.99 |
| Qwen L17/N1420+ | 0.81 | 0.83 | 0.89 | 0.91 | 0.94 | 0.97 | 0.95 | 0.92 |

**No verified Qwen pole is its layer's whole representation of its concept.** Every concept is linearly decodable from the layer almost perfectly (AUROC ≥ 0.99), and the best single pole falls well short of that. The k-sparse curves rise smoothly up to about 32 neurons, which is the signature Gurnee et al. (2023) give for a feature spread over many neurons rather than held by one.

**Rank separates a main carrier from a contributor.** L6/N4837+ and L13/N3326− rank 2nd and 3rd of 4,864. Each is one of the few neurons that best separate its concept, and L6/N4837+ with one partner already reaches 0.93. L4/N3208+ and L17/N1420+ rank 725th and 152nd: a probe for the concept would not pick them. Both explanations are narrower than the rule (below), so this is partly a mismatch of label and claim. Still, it means "verified for years" and "the layer's year neuron" are different claims.

**The same concept in two layers.** The `year` labels test two VERIFIED poles. L4/N3208+ peaks on an early digit and also fires on ' museum'. It covers a small part of what layer 4 knows about years (rank 725). L13/N3326− fires on the whole period expression and ranks 3rd at its layer. The later neuron is the better single read-out, though even there 32 neurons do better.

**The probe's direction is not the pole's.** Every cosine between a dense probe's direction and the pole's basis vector is at most 0.19 on Qwen (mass-mean; the logistic probes, at most 0.07). On the residual stream the mass-mean direction's cosine with the pole's signed `w_out` is at most 0.36 (L6/N4837+). The pole's write is a small component of the concept's direction there too. The values are in `probe-comparison.jsonl`.

**GPT-2 L0/N1923+ is the one candidate for a dedicated neuron, but the data is too thin.** It ranks 1st, and at the protocol's line it alone catches all 15 held-out ' Street'-type tokens and is active on about 16 of the 8,000 negatives (BA 1.00; its explanation also claims ' Court' and ' field', which the rule leaves unmarked). The k = 1 probe is the best sparse probe. This fits a layer-0 token detector. But FineWeb-Edu gave only 15 positive tokens (the Pile 27), too few to estimate any probe, and the dense logistic probe's 0.80 (about 6 misses of 15) is noise at that size. A larger thoroughfare-name corpus is needed before this row means anything.

## Caveats

- **The labels approximate the concepts.** The `thousands` rule marks every digit and comma of the numeral, while L17/N1420+ claims only the non-final digits after a separator and stays silent on a group's last digit. The `year` rule marks any number from 1500 to 2029, some of which are counts. A pole that matches its explanation exactly can still lose BA to a rule that is broader than the explanation. Both effects lower the pole's score, not the probes'. Rank is less affected by this than BA, because every neuron is measured against the same labels.
- **Thresholds transfer worse than rankings.** Every probe's AUROC is about 0.99–1.00, but the dense logistic probe's BA at its training threshold is 0.80–0.92 on two rows: its scores shift between the Pile and FineWeb-Edu. That is why *neurons needed* is measured against the better of the two dense probes, and why the k-sparse curve of L17/N1420+ falls past k = 32.
- **One seed, one rule per concept, five records.** These are point estimates with no intervals. The numbers support the qualitative reading above, not small differences between rows.
- **Correlational only.** A probe finds information that is linearly present, whether or not the model uses it. The causal counterpart is the next item of `TODO.md` (a DAS direction compared with the pole by interchange intervention).

## Reproduce

```
E=exchange/probe-comparison
uv run openinterp corpus --dataset fineweb-edu --docs 500 --seed 0 --out $E/fineweb-edu.jsonl
for c in thousands year feminine street; do
  uv run python scripts/probe_comparison.py labels --concept $c --corpus corpus.jsonl --out $E/$c.pile.jsonl
  uv run python scripts/probe_comparison.py labels --concept $c --corpus $E/fineweb-edu.jsonl --out $E/$c.fineweb.jsonl
done
uv run openinterp probe-compare --record wiki/experiments/Qwen--Qwen2.5-0.5B/L6/N4837-positive/20261028T140732Z.json \
  --train $E/feminine.pile.jsonl --test $E/feminine.fineweb.jsonl --concept "<the rule>" --out $E/qwen-L6-N4837.report.json
# … likewise for the other four records (table above), then:
uv run python scripts/probe_comparison.py report $E/*.report.json --out docs/analyses/probe-comparison.jsonl
```

The records used are Qwen L17/N1420+ `20261023T061522Z`, L4/N3208+ `20261029T043016Z`, L13/N3326− `20261027T001745Z`, L6/N4837+ `20261028T140732Z` and GPT-2 L0/N1923+ `20261027T201500Z`. Each run takes a few minutes on Apple's GPU.

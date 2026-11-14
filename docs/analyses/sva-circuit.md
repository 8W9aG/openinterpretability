# A subject–verb agreement circuit in Qwen2.5-0.5B

Issue #3225. Arora et al. (arXiv 2601.22594, `docs/literature/neuron-basis-circuits.md`) report that about 100 MLP neurons carry subject–verb agreement in Llama 3.1 8B. This analysis repeats their method on Qwen2.5-0.5B, a registry model, and asks how much of that circuit the registry has studied. Run on 2026-11-14 with `scripts/sva_circuit.py`. The rows are in `sva-circuit.jsonl`. Reported, not gated.

## Method

- **Task.** Four templates, as in the paper. The next token is the verb.

  | Template | Example (clean → corrupted) | The verb agrees with |
  | --- | --- | --- |
  | simple | The senator → The senators | the head noun |
  | across a prepositional phrase | The senator beside the manager → The senators beside the manager | the head noun |
  | across a relative clause | The senator that the manager liked → The senators that the manager liked | the head noun |
  | within a relative clause | The senator that the manager → The senator that the managers | the clause's subject |

  There are 33 nouns (30 regular, plus child, woman and man). Each singular and plural is one token in Qwen2.5's tokenizer, so a clean prompt and its corrupted pair align position by position. The other noun's number is random and the same in both prompts. Half the clean prompts are singular. There are 300 training and 40 held-out pairs per template, with seed 0.
- **Metric.** logit(` is`) − logit(` are`) at the last position, times +1 for a singular clean noun and −1 for a plural one.
- **Units.** All 116,736 MLP neurons (24 layers × 4,864): the input to the down projection, as in `docs/PROTOCOL.md` §1 and in the paper.
- **Attribution.** Integrated gradients, 10 steps, one layer at a time. The input is the corrupted prompt, and the layer's activations go from corrupted to clean. A neuron's attribution is summed over positions and averaged over the 1,200 training pairs. Neurons are ranked by its absolute value. The paper's RelP is not used here (#3582).
- **Scores, on the 160 held-out pairs.** A circuit is the top N neurons. Mean ablation replaces a neuron's activation with its mean over the template's training prompts, clean and corrupted, at each position. *Faithfulness* is (m(circuit only) − m(none)) / (m(all) − m(none)). There, "circuit only" mean-ablates every other neuron, and "none" mean-ablates all of them. *Completeness* is the same ratio with the circuit itself mean-ablated. A perfect circuit has faithfulness 1 and completeness 0. Both are averaged over the four templates. The reference is 3 random sets of N neurons.

## Results

| Neurons | Faithfulness | Completeness | Random: faithfulness | Random: completeness |
| --- | --- | --- | --- | --- |
| 10 | 0.07 | 0.59 | 0.00 | 1.00 |
| 20 | 0.23 | 0.48 | 0.00 | 1.00 |
| 50 | 0.52 | 0.32 | 0.00 | 1.00 |
| 100 | 1.01 | 0.14 | 0.00 | 1.00 |
| 200 | 1.06 | 0.08 | 0.00 | 1.00 |
| 500 | 1.14 | 0.04 | 0.00 | 1.00 |
| 1,000 | 1.13 | 0.03 | 0.00 | 1.00 |
| 2,000 | 1.13 | 0.02 | 0.00 | 0.99 |
| 5,000 | 1.08 | 0.02 | 0.00 | 0.97 |

- **About 100 neurons carry agreement here too.** The top 100, 0.09% of the model's neurons, keep all of the metric (faithfulness 1.01) with every other neuron mean-ablated. Ablating them leaves 14% of it. Random sets of the same size keep none of it, and ablating them changes nothing. That is the paper's result on a model 16 times smaller, with the same attribution method as its integrated-gradients arm.
- **Every template agrees.** At 100 neurons, faithfulness per template is 1.03 (simple), 1.05 (across a prepositional phrase), 0.94 (across a relative clause) and 1.01 (within a relative clause). So one circuit serves agreement across intervening phrases.
- **Faithfulness above 1** means the circuit alone gives a larger logit difference than the whole model. The ablated neurons slightly oppose agreement on average. The paper reports the same overshoot.
- **The circuit sits at the ends of the model.** Of the top 100, 20 are in layer 0 and 59 in layers 16–23, with 21 spread over layers 1–15. A layer-0 neuron reads the noun token itself, which is what the corruption changes. The late neurons are where the number reaches the verb's logits.

## What the registry has on it

| Circuit | Neurons the registry studied (either pole) | VERIFIED | Explanation mentions number or agreement | Expected studied, by chance |
| --- | --- | --- | --- | --- |
| Top 50 | 0 | 0 | 0 | 0.3 |
| Top 100 | 0 | 0 | 0 | 0.7 |
| Top 200 | 1 | 0 | 0 | 1.4 |
| Top 500 | 1 | 0 | 0 | 3.4 |

The registry has studied 793 of Qwen2.5-0.5B's 116,736 neurons (0.68%). So random draws are expected to find fewer than one of the top 100, and they found none. The one studied neuron is L23/N2539, at rank 197. It was REJECTED twice. Its explanation is about a token that its left frame licenses as the start of a unit.

So the registry can't yet say whether the neurons that carry agreement have clean explanations. Its random draws are not where task circuits are. They are too few to reach a set of 100 neurons out of 116,736. The circuit is a list of units that are known to matter, and it is ready for targeted audits.

## Limits

- **One model, one run, one seed.** There are no intervals. With 40 held-out pairs per template, the circuit scores are means over 160 prompts.
- **Integrated gradients, not RelP.** The paper finds RelP better at small N. A better attribution could reach faithfulness 1 with fewer neurons.
- **Neurons, not neuron–position pairs.** A circuit neuron is kept or ablated at every position. A circuit over positions could be smaller.
- **The metric is one verb pair.** ` is` against ` are`. Other verbs aren't tested, so a neuron that is specific to that pair would look like an agreement neuron.
- **Tokens differ between the prompts.** The corruption changes the noun token, so the layer-0 neurons carry the token's identity, not only its number. The paper's templates have the same property.
- **No figure.** The environment has no plotting library, so the curves are in the table above.

## Follow-ups

- Audit the circuit's top neurons with the registry's protocol, as targeted draws (#3656).
- Use the circuit as the ground truth for grammatical number as a labelled concept (#3224).
- Repeat with RelP once #3582 has it.

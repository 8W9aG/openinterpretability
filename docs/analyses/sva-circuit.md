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
| Top 50 | 20 | 13 | 17 | 0.3 |
| Top 100 | 20 | 13 | 17 | 0.7 |
| Top 200 | 21 | 13 | 17 | 1.4 |
| Top 500 | 21 | 13 | 17 | 3.4 |

The table counts the 20 targeted audits of 2026-11-15 and their second attempts of 2026-11-19 and 2026-11-22 (next sections). A unit counts as VERIFIED when any of its records is. Before them, the counts were 0, 0, 1 and 1 studied, none VERIFIED and none about number. The last column is what random draws alone would give. Before the audits, the registry had studied 793 of Qwen2.5-0.5B's 116,736 neurons (0.68%). So random draws are expected to find fewer than one of the top 100, and they found none. The one neuron they studied is L23/N2539, at rank 197. It was REJECTED twice. Its explanation is about a token that its left frame licenses as the start of a unit.

So random draws could not say whether the neurons that carry agreement have clean explanations. Random draws are not where task circuits are. They are too few to reach a set of 100 neurons out of 116,736. The circuit is a list of units that are known to matter, so we audited its top 20 as targeted draws.

## Audits of the top 20 (#3656)

On 2026-11-15 we audited the 20 neurons with the largest attribution, ranks 1–20. Each pole is the one whose sign matches the attribution: 19 positive and one negative (rank 13). The pipeline is the registry's standard one, under protocol 0.9:

- Claude Opus 5.5 wrote each hypothesis from a fresh scan, without being told about the circuit.
- Claude Sonnet 5.5 wrote 30 probes per unit: 20 hard negatives and 10 held-out positives.
- The token check repeated until a round changed nothing.
- Two Claude Haiku 5.5 judges scored detection and fuzzing.

The records carry `selection = "attribution"`, so the random-draw statistics leave them out. The table above now counts them (`--refresh`).

| Rank | Unit | Verdict | Failed gates | Explanation, in short |
| --- | --- | --- | --- | --- |
| 1 | L0/N4163+ | VERIFIED | | plural nouns that name groups of people |
| 2 | L20/N4662+ | NEEDS_REVISION | sensitivity, confound | the subject of a clause, when its head is plural |
| 3 | L19/N1274+ | NEEDS_REVISION | lookahead | the last token of a third-person singular subject |
| 4 | L0/N2615+ | NEEDS_REVISION | specificity | a set of verb-derived modifier tokens |
| 5 | L21/N1653+ | VERIFIED | | the last token of a plural subject before its verb |
| 6 | L0/N255+ | NEEDS_REVISION | specificity | a list of tokens |
| 7 | L23/N765+ | NEEDS_REVISION | specificity, counterfactuals | the head of a plural subject |
| 8 | L22/N4567+ | NEEDS_REVISION | specificity, lookahead, confound | a third-person singular subject before its verb |
| 9 | L20/N4213+ | NEEDS_REVISION | sensitivity, confound | a finite verb or auxiliary after a plural subject |
| 10 | L19/N4551+ | NEEDS_REVISION | sensitivity, confound | the token that completes a plural noun |
| 11 | L21/N4413+ | NEEDS_REVISION | specificity, confound | a modal or auxiliary after a singular subject |
| 12 | L0/N2235+ | NEEDS_REVISION | specificity | plural-like word-final pieces such as "ands" |
| 13 | L23/N4102− | NEEDS_REVISION | sensitivity, confound | the last token before a verb that agrees with a plural subject |
| 14 | L16/N4197+ | VERIFIED | | a function word that continues a plural noun phrase |
| 15 | L19/N4688+ | NEEDS_REVISION | lookahead | a singular noun between a plural subject and its verb |
| 16 | L0/N430+ | VERIFIED | | a singular word that names a kind of person, not its plural |
| 17 | L0/N3528+ | VERIFIED | | a list of tokens |
| 18 | L0/N561+ | VERIFIED | | regular -s plural tokens |
| 19 | L0/N3679+ | VERIFIED | | a singular noun that names a person by relationship, not its plural |
| 20 | L22/N1532+ | NEEDS_REVISION | lookahead | a third-person singular subject before its verb |

- **The explanations are about number.** 17 of the 20 explanations match the number pattern of `scripts/sva_circuit.py` (`NUMBER_WORDS`). The other three are token lists. Proposers found grammatical number from the scan alone, with no hint of the task.
- **7 of 20 are VERIFIED** (35%, Wilson 95% interval 18–57%). Of random Qwen2.5-0.5B neuron poles, 30.2% are VERIFIED on first reading (27.1–33.5%, `registry-stats.md`). So the circuit's neurons do about as well. A sample of 20 can't show a difference smaller than about 20 points.
- **The rate depends on the layer.** In layer 0, 5 of 8 verified. In layers 16–23, 2 of 12 verified. The layer-0 units read the noun token itself, and most explanations of them are token detectors. The late units track the subject's number across the clause. That claim is harder to state so that 20 hard negatives and 10 holdouts all agree. The second attempts below closed most of this gap.
- **Four late units failed on lookahead.** These are ranks 3, 8, 15 and 20. Each explanation claimed that the unit prepares the verb. The lookahead check measured each one as retrospective: ablating it changed the next-token distribution little (KL 0.0002–0.003). But the tokens that the ablation moved most are agreeing verbs. For the singular units they are " is", " was" and " has". For rank 15 they are " are", " have" and " were". Ranks 3, 15 and 20 failed no other gate. Ranks 15 and 20 passed 8 of 10 holdouts and 19 or 20 of 20 negatives. A retrospective restatement of these four is the obvious next attempt.
- **Five late units failed sensitivity or the confound baseline.** These are ranks 2, 9, 10, 11 and 13. Four of them passed only 4 of 10 holdouts. For four, probe length alone (a threshold between 8 and 17 tokens) predicted the labels as well as the hypothesis did.

### Limits of the audit

- **The baseline is not matched.** The 30.2% pools all proposers and protocol versions. The same proposer and reviewer verified 47 of 61 random Qwen2.5-0.5B units, under protocols 0.5–0.8. Protocol 0.9 adds the confound and lookahead gates, and only 5 random first readings ran under it. So the right comparison does not exist yet.
- **One reading per unit.** The verdict stability study (`verdict-stability.md`) shows that repeat reviews agree on VERIFIED or not for 82% of pairs. Some of the 13 NEEDS_REVISION verdicts would change on a second reading.
- **The proposer saw scans, not the task.** A proposer told about agreement could write sharper hypotheses. We kept it blind, so that the explanations count as independent evidence of what the neurons do.

## Second attempts (#4058)

The late units failed in two ways, so each got a second attempt on its own record: the planner's `revise` follow-up, which names the failed gates. The pipeline was the same as in the first audits. Ranks 9 and 11 had their second records already, from 2026-11-19, and both stayed NEEDS_REVISION. The other seven ran on 2026-11-22.

| Rank | Unit | First reading | Second reading | Confidence | What changed |
| --- | --- | --- | --- | --- | --- |
| 3 | L19/N1274+ | NEEDS_REVISION (lookahead) | VERIFIED | 0.93 | the claim became retrospective |
| 8 | L22/N4567+ | NEEDS_REVISION (specificity, lookahead, confound) | REJECTED | 0.56 | lookahead passed, but 4 of 20 negatives |
| 15 | L19/N4688+ | NEEDS_REVISION (lookahead) | VERIFIED | 0.82 | the claim became retrospective |
| 20 | L22/N1532+ | NEEDS_REVISION (lookahead) | VERIFIED | 0.88 | retrospective, and only subjects able to act |
| 2 | L20/N4662+ | NEEDS_REVISION (sensitivity, confound) | VERIFIED | 0.95 | plural subjects, strong only in Romance languages |
| 10 | L19/N4551+ | NEEDS_REVISION (sensitivity, confound) | VERIFIED | 0.75 | the first token that marks a phrase plural |
| 13 | L23/N4102− | NEEDS_REVISION (sensitivity, confound) | VERIFIED | 0.82 | French -ent stems and a pronoun after "if" |

- **Six of seven verified.** With the second attempts, 13 of the top 20 are VERIFIED (65%, Wilson 95% interval 43–82%).
- **The layer split closed.** In layers 16–23, 8 of 12 are now VERIFIED (39–86%), against 2 of 12 at first reading. Layer 0 stays at 5 of 8 (31–86%). So the late units were not harder to explain. Their first claims were wrong in a way that one more attempt fixed.
- **Lookahead was the claim, not the unit.** Each of the four lookahead units now passes the lookahead gate as retrospective, and three are VERIFIED. Their weights still promote the agreeing verbs, and the explanations say so. The units read the subject's number. They don't measurably steer the next token on their own.
- **The failed holdouts were a narrower rule.** Each sensitivity unit fires on a narrower set than its first claim said. For example, rank 2 fires weakly on English plural subjects, which were all of its earlier failed holdouts. Rank 10 fires on the first token that marks plurality, so "laws" fires and "these laws" doesn't.
- **Rank 8 failed specificity.** Its proposer found that the unit is active on a text's first token alone, and most negatives start with an ordinary first token. Its earlier lookahead failure is gone.
- **Second attempts select.** A unit got a second attempt because its first failed, and the proposer saw the failed probes. So 13 of 20 is a rate after two attempts, not a first-reading rate to set against the 30.2% of random units.

## Limits

- **One model, one run, one seed.** There are no intervals. With 40 held-out pairs per template, the circuit scores are means over 160 prompts.
- **Integrated gradients, not RelP.** The paper finds RelP better at small N. A better attribution could reach faithfulness 1 with fewer neurons.
- **Neurons, not neuron–position pairs.** A circuit neuron is kept or ablated at every position. A circuit over positions could be smaller.
- **The metric is one verb pair.** ` is` against ` are`. Other verbs aren't tested, so a neuron that is specific to that pair would look like an agreement neuron.
- **Tokens differ between the prompts.** The corruption changes the noun token, so the layer-0 neurons carry the token's identity, not only its number. The paper's templates have the same property.
- **No figure.** The environment has no plotting library, so the curves are in the table above.

## Follow-ups

- Audit the circuit's top neurons with the registry's protocol, as targeted draws (#3656, done above).
- Restate the four lookahead failures as retrospective (#4058, done above), and audit ranks 21–100 (#6011).
- Use the circuit as the ground truth for grammatical number as a labelled concept (#3224).
- Repeat with RelP once #3582 has it.

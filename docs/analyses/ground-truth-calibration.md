# Ground-truth calibration: planted neurons

Run on 2026-11-14 for issue #3201, with `scripts/ground_truth.py`. One row per plant is in `ground-truth-calibration.jsonl`, and the counts are in `ground-truth-calibration.json`. Reported, not gated.

The question: when a neuron's role is known by construction, does the pipeline's explanation name that role? And does a wrong explanation still get VERIFIED? The induction-head control checks that the protocol can say yes. This study checks that the yes is about the right thing. The untrained-model null (`untrained-null.md`) asks a different question: whether the protocol verifies units that learned nothing.

## Method

- **Plants.** Each plant is one GPT-2 small neuron whose input weights the script rewrites as a logistic probe for a label rule over tokens:
  - The probe is fitted (`probing.fit_logistic`) on the MLP inputs of the even documents of `corpus.jsonl`.
  - It is scaled so that the median labelled token reads 4.0 before the GELU.
  - The neuron's output row is kept, so its vocabulary lenses look like any neuron's.
  - All 12 plants are in one model copy, in neurons with no GPT-2 experiment in the registry.
- **Rules.** Six token sets: months (twice, at layers 0 and 11), weekdays, colours, number words, animals and closing brackets. Three rules that need context: the name after Mr, Mrs, Dr or Ms, a token inside double quotes, and the first token of a line. Two unions of unrelated sets: weekdays or drinks, and colours or body parts.
- **Gate, set before any audit.** On the odd documents, with every plant installed, a plant must reach precision ≥ 0.8 and recall ≥ 0.8 at the protocol's line (0.25 × its maximum). A plant that fails is not audited.
- **Blinding.** The audited plants were shuffled under opaque names (`u00`–`u06`), and every prompt names GPT-2. The plan and the planted weights were outside the repository. Proposers measured only through `scripts/measure_unit.py`, and reviewers saw only the submission.
- **Pipeline.** The scan used the registry corpus, a 200,000-token sample and float32. Each unit then got:
  - one `neuron-proposer` (Claude Opus 5.5, `claude-opus-5-5`).
  - one `neuron-reviewer` (Claude Sonnet 5.5, `claude-sonnet-5-5`), writing 30 probes: 20 hard negatives and 10 holdouts.
  - today's decision rule, applied to the measured probes and counterfactuals (`null_calibration.review_probes`). Causal metrics and wiring are left out, because they aren't gated.
- **Recovery, judged blind.** A judge (Claude Sonnet 5.5) got each final explanation and the plant's rule, but not the verdict. It rated each one *match*, *partial* or *miss*.

## Result

**The gate kept 7 of 12 plants, and none that needs context.**

| Plant | Layer | Held-out precision | Held-out recall | Gate |
| --- | --- | --- | --- | --- |
| months | 0 | 1.00 | 1.00 | pass |
| weekdays | 1 | 1.00 | 1.00 | pass |
| colours | 2 | 1.00 | 0.95 | pass |
| numbers | 3 | 0.99 | 1.00 | pass |
| animals | 4 | 0.86 | 0.79 | drop |
| brackets | 5 | 0.99 | 1.00 | pass |
| after a title | 6 | 0.88 | 0.64 | drop |
| inside quotes | 7 | 0.59 | 0.50 | drop |
| start of a line | 8 | 1.00 | 0.70 | drop |
| weekdays or drinks | 9 | 0.93 | 0.84 | pass |
| colours or body parts | 10 | 0.94 | 0.65 | drop |
| months | 11 | 0.88 | 0.86 | pass |

A stronger L2 penalty (10⁻² or 10⁻¹) does not rescue the dropped plants: precision falls as recall rises. One linear read of one layer doesn't hold these rules cleanly. So this study tests token detectors and one union. It does not test roles that depend on context.

**Every explanation named its plant's concept, and two correct ones were not VERIFIED.**

| Plant | Explanation, in short | Recovery | Verdict | Balanced accuracy |
| --- | --- | --- | --- | --- |
| months, L0 | full month-name tokens, ' May' strongest | match | NEEDS_REVISION | 0.97 |
| weekdays, L1 | capitalised weekday names | match | VERIFIED | 1.00 |
| colours, L2 | basic colour words, ' red' strongest | match | VERIFIED | 0.93 |
| numbers, L3 | number words one to ten, weaker for eleven and twelve | match | VERIFIED | 1.00 |
| brackets, L5 | a closing bracket that is its own token | match | NEEDS_REVISION | 0.97 |
| weekdays or drinks, L9 | weekday names, and drink nouns as a weaker second response | partial | VERIFIED | 0.97 |
| months, L11 | full month names, and half strength on modal "may" | match | VERIFIED | 0.97 |

| Count | Value (95% Wilson) |
| --- | --- |
| Explanations that match the plant | 6 of 7 (86%, 49–97%) |
| Partial | 1 of 7 |
| Miss | 0 of 7 |
| VERIFIED | 5 of 7 (71%, 36–92%) |
| VERIFIED with a *miss* explanation | 0 of 5 (0%, 0–43%) |
| VERIFIED with a *partial* explanation | 1 of 5 |

- **No wrong explanation was VERIFIED.** That is 0 of 5, but with 5 units the interval reaches 43%. This sample can't show a low false-positive rate. It shows only that none occurred here.
- **The two NEEDS_REVISION verdicts are reviewer label errors, not wrong explanations.** Each unit failed only `beats_confound_baseline`, because one hard negative fired and the token-identity baseline predicted it:
  - **brackets:** "Set x = arr[0], y = arr[1], and z = arr[2] before the loop." The reviewer assumed that GPT-2 joins each `]` with the next punctuation. The last `]` stands alone, so the explanation predicts it fires, and it does (0.63).
  - **months, L0:** "Mayday, mayday, …". GPT-2 splits it into `May` + `day`, and the explanation says that capitalised `May` fires in any context. It fires (0.86).

  With only those two labels corrected, both units are VERIFIED at balanced accuracy 1.0 (`corrected_outcome` in the rows). The reviewer has no tokenizer, so it can't check a split before it labels a probe. Here that cost two correct explanations their verdict.
- **The union was VERIFIED with half its rule played down.** The proposer described weekdays as the response, and drinks as a "secondary moderate response (about 0.3–0.5)". Measured on the plant, drinks read 0.27–0.58 of the scale, and a weekday 0.62. So the numbers are about right, but the framing ranks one half below the other. Only 4 of the 30 probes mention a drink, so the review hardly tested that half. A union passes when the reviewer probes mostly the half that the explanation stresses.
- **No label error helped a unit pass.** On the five VERIFIED units, four probes disagreed with their labels. Two were plants that fire slightly off their rule (0.28 on "might", 0.33 on a sports sentence). Two were wrong side claims in the colour explanation ("Redis" fires, "crimson" doesn't). All four counted against their units.

## With the token check (issue #3592)

The reviews above skipped the probe-token check (`open-interpretability review --check-probes`), which shows a reviewer its probes as the model's tokens. On 2026-11-14 each of the 7 reviewers got its check prompt by SendMessage. Each then re-checked its own probes, and the probes were measured again on the planted model.

| Round | Probes changed | What they caught |
| --- | --- | --- |
| 1 | 4 in 3 sets | Both known label errors (`arr[2] before`, `Mayday`). Also a probe whose trigger was the first token (`Redis`, `Sept`) |
| 2, on the 3 changed sets | 1 | The new probe for u06, `Marching`, splits as ` March` + `ing`: a new label error from round 1 |
| 3, on u06 | 0 | — |

After the check, **all 7 plants are VERIFIED**, at balanced accuracy 0.97–1.00, against 5 of 7 without it. So the check found every label error that a split caused, as long as it ran again after each rewrite. A probe that a reviewer writes during a check has not been checked, which is why one round was not enough. The review steps now run the check for every unit, and again after any change, for at most 3 rounds.

## Limits

- **Seven units.** Every interval is wide. The study shows whether the pipeline can recover a planted token detector. It does not give a rate to quote for the registry.
- **Token detectors are the easy case.** A layer-0 token detector shows its rule in its top examples. The plants that need context failed the gate, so the study says nothing about whether agents recover roles that depend on context. That is where wrong explanations are most likely.
- **Plants are cleaner than trained neurons.** A plant fires on one rule with almost no off-rule activity. Trained neurons are noisier, so their explanations are harder to recover.
- **One proposer and one reviewer per unit.** Verdict stability (`verdict-stability.md`) shows that repeated reviews of one submission disagree on the status in a third of pairs.
- **One judge.** The judge's *partial* for the union is a judgement call. The page reports the explanation in full, so a reader can judge it again.

## Reproduce

```bash
uv run python scripts/ground_truth.py plant       # fit, install and gate; plan in ~/.cache/openinterp/p3201/
uv run python scripts/ground_truth.py scan
# a neuron-proposer answers each exchange/p3201/uXX/ev.prompt.md, measuring with scripts/measure_unit.py
uv run python scripts/ground_truth.py submit --agent-model claude-opus-5-5
# a neuron-reviewer answers each review.prompt.md
uv run python scripts/ground_truth.py measure --agent-model claude-sonnet-5-5
uv run python scripts/ground_truth.py report > docs/analyses/ground-truth-calibration.jsonl
uv run python scripts/ground_truth.py summary > docs/analyses/ground-truth-calibration.json
```

The judge's ratings are in each unit's `judgement.json`. The neurons, the plants and the gate's verdicts come from seed 3201. A rerun on Apple's GPU gave the same plan, except that one dropped plant's gate numbers moved by about 0.0003. The agents' answers are not reproducible.

# The first token and default-on poles (protocol 0.8)

This analysis settles the *(protocol)* item from `TODO.md` (*Scope expansion*): how GELU negative poles, and early-layer poles with a first-token floor, are audited. The learned-basis baseline (`transcoder-baseline.md`) lost two groups of GPT-2 small poles to the way activity was measured:
- every random negative pole in its pilot;
- six of its 40 positive poles, all in layers 1–2, whose explanations stated a floor at the first token.

Protocol 0.8 makes two changes:
1. **A unit with a first-token floor is not measured at a text's first token.** A unit has the floor if it is active at the first position on at least half the vocabulary. For such a unit every peak skips the first token, as do its scan statistics, top-example ranking, confound baselines and held-out windows. A one-token prompt to it is recorded as unmeasured and counts in no metric. Every other unit is measured as before.
2. **A default-on unit is not audited as a detector.** That is a unit active on at least half its scan's tokens. The scan stops on it, as it does on a unit that never fires, and a review fails the new `not_default_on` gate.

The per-layer numbers come from `scripts/position_zero.py`, whose output is in `position-zero.jsonl`. It scans the GPT-2 small sample every baseline audit read: `corpus.jsonl`, 200,000 tokens with seed 0, in 128-token windows with stride 64. That gives 432 documents and 199,964 counted tokens. It also reads every vocabulary token at the first position. Re-evaluations of records under 0.8 were written to a scratch directory, not the registry.

## The first token

GPT-2 and Qwen2.5 add no BOS token, so a text's first token is also its attention sink. At that position attention sees only the token itself, so a unit's strength there depends on the token alone. Some early-layer positive poles sit above their line there whatever the token is. Counting GPT-2 small's positive poles active at the first position on at least half the vocabulary (3,072 poles per layer):

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8–11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Positive poles with a first-token floor | 74 | 786 | 578 | 14 | 83 | 74 | 36 | 12 | 14 |
| … of which not default-on | 25 | 718 | 565 | 8 | 78 | 72 | 35 | 11 | 11 |
| Positive poles whose scale was set at a first token | 67 | 78 | 14 | 5 | 3 | 1 | 1 | 0 | 1 |

In all, 1,671 positive poles have the floor. 1,523 of them are not default-on: away from the first token they are ordinary, sparse units. Their floor made every probe active, whatever the explanation said. That is why the baseline's six floor poles could get no hard negative (`enough_hard_negatives`). Those six are active at the first position on 92–99% of the vocabulary.

**Why the rule is per unit.** A first version of 0.8 never measured any unit at the first token. Re-measuring every registry record's probes that way changed some probe's activity in 31 of the 800 Qwen2.5-0.5B records and 21 of the 158 GPT-2 records. Among them were Qwen poles whose job is the first token, verified as such:
- L19/N436+ fires when a text opens with a function word or code keyword, and stays silent on a content word.
- L23/N541+ is a graded detector of how a text opens.

Their holdouts open with the trigger, so a blanket rule sent L19/N436+ and L23/N541+ from VERIFIED to REJECTED, and three other VERIFIED Qwen poles to NEEDS_REVISION. These units are active at the first position on only 0.8–4.9% of the vocabulary, against 92–99% for the GPT-2 floor poles, so the vocabulary separates the two cleanly.

**Details.**
- *The scale.* The floor is judged against the line the unit's scale sets once first tokens are left out: its top example's peak after the first token. With the floor, that is the scale it records. Without it, the scale includes first tokens as before. A larger scale can only lower the share, so the record reproduces either way.
- *One-token prompts.* For a floor unit a one-token prompt has nothing to measure. It is recorded with `measured = false` and left out of every count. Treating it as a failure would penalise a hypothesis for its reviewer's choice of prompt, and treating it as inactive would hand out free specificity.
- *Rejected alternative: prepend a BOS token.* Prepending `<|endoftext|>` would move the sink off the text. But every activation would change, Qwen2.5 has no BOS token, and the sink's own position would still need excluding.

## Default-on poles

GELU's negative pole is its dip: at most 0.17, reached at a pre-activation of −0.75, where most of GPT-2's neurons sit on most tokens. The share of tokens each pole is active on (at or above 0.25 × its maximum, first tokens left out for a floor unit) is:

| Active share | <0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | ≥0.9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Positive poles | 36,071 | 362 | 110 | 65 | 55 | 29 | 22 | 8 | 10 | 132 |
| Negative poles | 381 | 274 | 536 | 1,293 | 2,200 | 3,604 | 6,468 | 13,919 | 7,524 | 665 |

The first token is not the problem here. 32,180 of GPT-2 small's 36,864 negative poles (87%) are active on at least half their tokens, against 201 positive poles. A probe of n tokens drawn from such text stays inactive with probability about 0.5ⁿ, so no hard negative of a sentence can pass. The pilot's four negative poles with hard negatives passed 0, 0, 0 and 7 of 11.

The line is at half. There is no sharp break in the negative poles' distribution, but at half or more a hard negative of a few words needs every token to be silent. Below it, a pole can still be a detector of what keeps it quiet. The share is read from the scan's histogram, counting only bins wholly above the line (a lower bound within one bin). The scan stop and the review gate read the same number.

**Other units.** No Qwen2.5-0.5B record in the registry comes near the line: its highest active share is 0.35, and its SwiGLU negative poles are sparse. Transcoder latents are all below 0.01. On the same sample, most of the 60 GPT-2 heads of layers 0, 1, 5, 9 and 11 are active on 10–40% of tokens, measured by \|W_O (z − z̄)\|. The L5H1 induction head of the positive-control dry run is active on 17%. Five heads of layer 0, two of layer 1 and L9H3 are at or above half.

**Rejected alternative: measure default-on poles from their resting level,** as a head is measured from its mean write. For a GELU negative pole the resting level is the dip's floor itself. Its strength saturates there, so departures from rest would be a different, non-monotone readout: in effect a different unit from the one the pole names.

## Effect on existing records

Re-evaluating keeps a record's scale and re-measures its stored probes. A record from before 0.8 has no floor judgement, so the review judges it against that scale. The records re-evaluated were the baseline's 46 neuron-pole records, every other VERIFIED GPT-2 negative pole that leaned on one-token probes, and the VERIFIED Qwen poles that the blanket rule had changed.

| Records | Re-evaluated | VERIFIED → NEEDS_REVISION | NEEDS_REVISION → VERIFIED | Unchanged |
| --- | --- | --- | --- | --- |
| Baseline neuron poles (40 positive, 6 negative) | 46 | 0 | 1 | 45 |
| Other VERIFIED GPT-2 negative poles | 6 | 5 | 0 | 1 |
| VERIFIED Qwen2.5-0.5B poles the blanket rule changed | 9 | 0 | 0 | 9 |

- **Default-on.** The five that lose VERIFIED are L1/N277−, L5/N4−, L6/N1635−, L7/N1883− and L9/N2000−, each failing only `not_default_on`, with active shares of 0.56–0.93. Their explanations described a default-on dial, for example "active for any prompt of two or more tokens". They passed on one-token hard negatives or on word salads of tokens chosen to flip the pre-activation. L2/N1944− (active share 0.31) keeps VERIFIED.
- **The tie.** L3/N47+, the baseline's one confound-baseline failure, ties the token-identity baseline at 0.909091. It now passes, so the positive-pole arm is 21 of 40 VERIFIED once it is re-evaluated in the registry.
- **Floor poles.** The six floor poles, and three of the pilot's negative poles, are judged to have a floor. Their verdicts stay NEEDS_REVISION, because their probe sets were written for the floor: they need new hypotheses and probes, not a re-measurement.
- **Qwen.** No Qwen pole has a floor, so all nine stay as they were.

## Re-auditing the six floor poles

Each of the baseline's six floor poles was rescanned under 0.8 and audited as a revision of its record. The scans read the same corpus sample in 128-token windows, and every pole was judged to have a first-token floor and not to be default-on. The roles were those of the baseline: proposer `claude-opus-5-5`, reviewer `claude-sonnet-5-5`, judges `claude-haiku-5-5`. Before each review the labels were checked against the submitted explanation and GPT-2's token splits. Three probe sets went back to their reviewer for labels that contradicted the explanation, for example a near-line noun or ' ch'+'ien' labelled inactive.

| Pole | Hypothesis | Verdict | Specificity | Sensitivity |
| --- | --- | --- | --- | --- |
| L1/N2522+ | French 'ien' after a one-letter stem (bien, rien, lien), 'ét' of 'été', ' steroid', and transfer verbs after about 8 tokens | NEEDS_REVISION | 0.45 | 1.00 |
| L1/N3040+ | ' rest' in any sense, and a few Latinate word starts | NEEDS_REVISION | 0.73 | 1.00 |
| L1/N874+ | words of preference and support, and of group and lineage | NEEDS_REVISION | 0.64 | 1.00 |
| L2/N1712+ | a short lexicon of code, markup and SQL tokens (IFRAME, FROM, View, `[`) | NEEDS_REVISION | 0.50 | 1.00 |
| L2/N993+ | a no-space piece completing a short code or split word (T1, MP3, gear\|box) | NEEDS_REVISION | 0.50 | 1.00 |
| L2/N2331+ | a lowercase no-space piece completing a rare, technical or foreign word | **VERIFIED** | 0.82 | 1.00 |

Every hypothesis now has hard negatives. Every holdout fired, and every counterfactual reproduced. The five that fail, fail only specificity, as 13 of the baseline's other positive poles did: the poles are broader than their explanations. For example, L2/N993+ also fires on "gear box" and "sk id" written apart, and L1/N2522+ on other French '-ien' words. So the floor hid ordinary poles, neither unusually clean nor unusually messy.

The learned-basis baseline counts each unit by its first experiment, so these revisions don't change its positive-pole arm. Once L3/N47+ is re-evaluated (see above), that arm is 21 of 40 VERIFIED. Counting the floor poles by their revisions instead would make it 22 of 40.

## Next

- Report registry-wide statistics over random draws, with the dead and default-on draws now kept in `wiki/draws/` as outcomes (`TODO.md`).

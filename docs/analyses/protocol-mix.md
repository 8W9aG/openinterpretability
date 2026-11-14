# Protocol mix of the verified claims

Run on 2026-11-13 for issue #3207, with `scripts/protocol_mix.py`. The rows are in `protocol-mix.json`. Each verdict stores the protocol it was judged under (`docs/PROTOCOL.md` §6). The registry's totals counted every VERIFIED unit the same, whatever its protocol. The question: does today's rule still verify the claims that older rules verified?

## How many claims rest on older protocols

Each VERIFIED neuron pole stands on its newest record that agrees with its standing. Of 516 VERIFIED poles, 339 stand on a protocol-0.8 record, 164 on 0.7, 6 on 0.6 and 7 on 0.5. So about a third of the registry's verified claims were judged by older rules. The home page now shows this split, and `registry.parquet` has a `protocol_version` column.

## Today's rule on older records

The script applies today's `ReviewPolicy` to the stored results of every VERIFIED record from an older protocol, with no model. The active share comes from each record's stored histogram. Protocol 0.8's `not_default_on` gate needs it.

| Protocol | VERIFIED records | Still VERIFIED today | Not VERIFIED today |
| --- | --- | --- | --- |
| 0.4 | 5 | 5 | 0 |
| 0.5 | 42 | 42 | 0 |
| 0.6 | 14 | 14 | 0 |
| 0.7 | 270 | 265 | 5 |
| **All** | **331** | **326** | **5** |

- **98.5% of older VERIFIED records still pass.** No record needed a new measurement. The activation threshold and the counterfactual tolerances have not changed since protocol 0.4. So the stored pass and fail of each probe still hold.
- **The 5 that fail all fail one gate, `not_default_on`.** Each is a GPT-2 negative pole that is active on half or more of its scan's tokens. Protocol 0.8 stops such a pole, because no hard negative can stay silent (`position-zero.md`). The 5 records:
  - `openai-community--gpt2/L1/N277-negative/20261028T110618Z`
  - `openai-community--gpt2/L5/N4-negative/20261028T111028Z`
  - `openai-community--gpt2/L6/N1635-negative/20261028T170948Z`
  - `openai-community--gpt2/L7/N1883-negative/20261028T032706Z`
  - `openai-community--gpt2/L9/N2000-negative/20261028T105613Z`

## A proposed rule: probe length (issue #3525)

On a randomly initialised GPT-2, two units were VERIFIED with "silent on short texts" hypotheses (`untrained-null.md`). A rule that reads only probe length would have passed them too. So #3525 proposes a fourth confound baseline: "active if the probe has at least n tokens". The best n is chosen on the probes, as for the other baselines. A hypothesis would have to tie or beat it. The script measures it before any change to the decision rule. It reads the stored probes of every VERIFIED record from protocol 0.8 (`length_rule` in the JSON).

**It would fail 45 of the 380 current VERIFIED records (12%).** In each, the best length rule beats the hypothesis's balanced accuracy, by a median of 0.07.

**The main cause is how reviewers write probes.** In those 45 records, positive holdouts are a median of 4.75 times as long as hard negatives. Over all 1,613 records under protocol 0.8, holdouts are a median of 2.16 times as long. In 52% of records they are more than twice as long. With long holdouts and short negatives, probe length alone can separate active from inactive. So the rule finds mostly a habit of the reviewers, and only in a few records a unit that responds to context length.

The rule is not part of the decision rule. Whether to adopt it, and whether to ask reviewers for negatives and holdouts of matched length first, is a decision for the maintainer (#3525).

## Limits

- **The first-token floor isn't re-measured.** Protocol 0.8 settles the floor when it measures the probes. Older records were measured without it, and only a new measurement can apply it.
- **One record per unit.** A unit with a VERIFIED record from 0.7 and a newer one from 0.8 stands on the newer one. The unit counts above use the record that the standing rests on.

## What this means

- **Standing doesn't need a different rule for older verdicts.** Today's rule keeps almost all of them, and the split is now visible.
- **Re-evaluate the 5 default-on poles** with `openinterp reevaluate`. It writes new records under protocol 0.8, and their units' standing changes by the usual rule, with no protocol change. A follow-up issue holds this.
- **Run the script after each protocol change.** It shows at once how many older claims the new rule would reject.

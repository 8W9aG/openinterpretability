# Sister neurons across models

`openinterp sisters` matches each VERIFIED neuron pole with the poles of another model whose activity over the corpus correlates best with it (docs/PROTOCOL.md §4, *Sister neurons*). This note reports the first run. Rows: `sister-neurons.jsonl`. Run 2026-11-01.

## Method

Three ordered pairs: Qwen2.5-0.5B → Qwen2.5-1.5B (its larger sibling, same tokenizer), Qwen2.5-0.5B → GPT-2 small and GPT-2 small → Qwen2.5-0.5B. The source poles are every pole whose standing was VERIFIED: 259 of Qwen2.5-0.5B, 73 of GPT-2.

Both models read `corpus.jsonl` (500 pile-10k documents), one document per forward pass, in float32 on Apple's GPU. GPT-2 reads at most 1,024 tokens, so 32 documents were cut short for it. The two models are compared at shared token boundaries, with each text's first token left out. Coverage is the share of the source model's positions that pair:

| Pair | Paired positions | Coverage |
| --- | --- | --- |
| Qwen2.5-0.5B → 1.5B | 211,732 | 100% |
| Qwen2.5-0.5B → GPT-2 | 192,742 | 91% |
| GPT-2 → Qwen2.5-0.5B | 192,742 | 84% |

For each source pole, every pole of every layer of the other model is a candidate: 501,760 for Qwen2.5-1.5B (28 layers × 8,960 neurons × 2 poles), 73,728 for GPT-2 and 233,472 for Qwen2.5-0.5B. The 5 best by Pearson correlation are kept. A second pass finds each match's best partner among all of the source model's poles, and the rotated reference at the top match's layer.

The 0.5B → 1.5B pair took about 30 minutes for each pass; each GPT-2 pair took a few minutes.

## Results

| Pair | Poles | Median best r | Median rotated | Best r ≥ 0.5 | Sisters |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B → 1.5B | 259 | 0.26 | 0.08 | 35 | **53 (20%)** |
| Qwen2.5-0.5B → GPT-2 | 259 | 0.18 | 0.09 | 10 | 10 (4%) |
| GPT-2 → Qwen2.5-0.5B | 73 | 0.28 | 0.10 | 6 | 9 (12%) |

**A fifth of Qwen2.5-0.5B's VERIFIED poles have a sister in Qwen2.5-1.5B.** Every mutual best match also beat the rotated reference, which is low throughout (median 0.08): no unit of a randomly rotated basis tracks a pole the way the sibling's own neurons do. The best matches of sisters correlate more strongly than those of other poles (median 0.42, interquartile range 0.27–0.65), and 12 of the 13 poles whose best match reaches 0.7 have a sister. The strongest is L22/N2511− → L22/N2260− (r 0.92, rotated 0.39).

**Sisters sit at the same relative depth.** The median gap between a pole's depth (layer / 23) and its sister's (layer / 27) is 0.045, about one layer. That is how Gurnee et al. (2024) found universal neurons across GPT-2 seeds. The search covered every layer, so this was not imposed.

**The sign of a sister is arbitrary.** 24 of the 53 sisters match a positive pole with a negative one, or the reverse. A gated neuron's sign depends on the sign of its `up` weights, which training needn't preserve across models, so a neuron's two poles are matched separately and either may be the sister.

**Across families, sisters are rarer.** Only 4% of Qwen2.5-0.5B's VERIFIED poles have a GPT-2 sister, and 12% of GPT-2's have a Qwen one. No pair was found from both sides: the GPT-2 sisters of VERIFIED Qwen poles are not themselves VERIFIED here, and the reverse. Different training data, tokenizer and size all leave fewer one-to-one matches than a sibling trained the same way.

## First sister studies

Two sister studies, the strongest sister in each direction between GPT-2 and Qwen2.5-0.5B, run 2026-11-01 through `openinterp audit --sister-of` (proposer claude-opus-5-5, reviewer claude-sonnet-5-5, judges claude-haiku-5-5). Each tests the VERIFIED explanation on its sister unchanged.

| Sister | Explanation tested | r | Verdict | Balanced accuracy | Failed |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B L5/N3833+ | GPT-2 L3/N1037+: the distal demonstrative *that*/*those* picking out a specific referent | 0.73 | **VERIFIED** | 0.89 | 1 of 13 hard negatives (*so that*), 1 of 7 holdouts |
| GPT-2 L0/N2495+ | Qwen2.5-0.5B L0/N4325−: a detached, space-prefixed separator glyph, graded by glyph | 0.77 | NEEDS_REVISION | 0.86 | 2 of 7 holdouts (` >`, ` +`); `beats_confound_baseline` |

**The demonstrative carries over.** GPT-2's explanation of a layer-3 pole, written from GPT-2's own examples, verifies on its Qwen sister two layers deeper. Detection scores 0.92 and fuzzing 0.79 on held-out Qwen text. The weak content-word offshoot it describes was not seen in Qwen (*quarterly revenue*: 0.01 of max), and the probes didn't test it.

**The separator detector carries over only in part.** Every hard negative and counterfactual passed: the GPT-2 pole fires on detached dashes, slashes, pipes and bullets and not on the same glyphs attached. But it ignores ` >` and ` +`, which the Qwen explanation puts in its strongest family, so two holdouts failed. Its line-start weakening didn't hold either (the proposer measured 0.96 of max after a newline). At layer 0 token identity explains it fully (baseline 1.00), so the explanation's conditions add nothing a token list doesn't: the same concept, with a narrower set of glyphs.

The source poles' pages now link to these sisters with each study's verdict, and the sisters' pages link back to the experiment they tested. The other 17 GPT-2 ↔ Qwen sisters are open `sister` draws in `openinterp next`. So are their other matches and random poles of the other model, at much lower weight; the random poles are the control for whether matching finds explanations that carry over. The 53 in Qwen2.5-1.5B wait until that model is onboarded, so CI can reproduce their records.

## Limits

- A sister is observational: correlated activity on one corpus. Whether the two poles represent the same concept is what a sister study tests (`audit --sister-of`).
- Pearson on sparse pole strengths is dominated by the tokens where both fire, which is what matching needs. But it also rewards two poles that fire on overlapping tokens for different reasons.
- 14 poles have a best match with r ≥ 0.5 that isn't mutual: their partner tracks another source pole more closely. They are reported, not counted as sisters; once studied, they too reach agents' prompts (`MIN_PROMPT_PEARSON`).
- One corpus of 500 pile-10k documents, and one rotation per layer.

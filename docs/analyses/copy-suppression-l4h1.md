# Is pythia-70m's L4/H1 a copy-suppression head?

Run on 2026-11-19 for issue #5063, with `scripts/copy_suppression.py`. The rows are in `copy-suppression-l4h1.jsonl`. The review of McDougall et al. (2023) in [`copy-suppression-head.md`](../literature/copy-suppression-head.md) found one pythia-70m head with their OV signature, L4/H1. This page tests whether L4/H1 does what a copy-suppression head does. It reads the head on a corpus, and on the six wrong prompts of the [copy-first-unit paper](../../papers/copy-first-unit/README.md).

**Answer.** In its weights, yes. On the six prompts, yes, but too weakly to change the answer. On ordinary text, no: copy suppression explains 1.8% of the head's small effect, against 76.9% for the paper's GPT-2 head.

## Method

- **Model.** EleutherAI/pythia-70m, in float32 on the CPU.
- **OV test** (the paper's Section 3.1). 4,000 tokens drawn with seed 0 go from their effective embedding through each head's value and output weights to the unembedding. The effective embedding is the token embedding plus the layer-0 MLP's write. A head scores the share of tokens whose own logit is among the 10 most negative of the vocabulary.
- **CSPA, its OV part** (the paper's Section 4). The windows are 300 windows of 64 tokens from the last 200 documents of `corpus.jsonl`, 18,482 positions past the first.
  - *Mean ablation* sets L4/H1's output to its mean over the corpus.
  - *CSPA* does the same, then adds back part of each source token's write, weighted by its attention. The part kept lies along that token's own unembedding direction, and only when it lowers the token.
  - *The control* adds back the writes that raise the token instead.

  Effect explained = 1 − KL(CSPA) / KL(mean ablation), each from the clean output. The paper's QK part is left out, so the number compares with its "OV part only", 81.1%. A check that adds back the head's whole write explains 99.999%, so the method itself loses nothing.
- **Prompts.** At the answer position of each wrong prompt, L4/H1's direct write to the copied unit and to the right unit. The final norm's scale is frozen at the real residual stream's, and the vocabulary mean is subtracted. The answer is also measured with the head's output scaled by 0, 1, 2 and 4.

## Results

**OV test.** L4/H1 scores 42.0%. The next heads, L0/H1 and L3/H2, score 0.08%, and the copy heads of the #4781 route (L3/H3, L4/H0, L4/H7) score 0%. The paper's head, GPT-2's L10H7, scores 84.70%.

**CSPA.** Mean-ablating L4/H1 moves the output by a KL of 0.0066 nats per position, a small effect. CSPA keeps 1.8% of it, and the control −0.7%. So on ordinary text, the head's effect is not mainly the lowering of tokens that it attends to.

| Prompt (right, not copied) | Direct write to the copied unit | To the right unit | Head ×0 | Head ×1 | Head ×2 | Head ×4 |
| --- | --- | --- | --- | --- | --- | --- |
| minutes, not seconds | −0.38 | −0.04 | seconds 0.90 | seconds 0.81 | seconds 0.65 | seconds 0.23 |
| months, not days | −0.31 | +0.06 | days 0.81 | days 0.73 | days 0.61 | days 0.32 |
| toes, not fingers | −0.81 | −0.15 | fingers 0.82 | fingers 0.68 | fingers 0.49 | **toes** 0.15 |
| months, not eggs | −1.18 | −0.04 | eggs 0.81 | eggs 0.64 | eggs 0.42 | eggs 0.07 |
| seconds, not minutes | −0.60 | −0.10 | minutes 0.59 | minutes 0.42 | minutes 0.25 | **seconds** 0.31 |
| days, not months | +0.06 | −0.33 | months 0.43 | months 0.45 | months 0.40 | months 0.28 |

**Table 1.** L4/H1's direct write in logits, and the top token with its probability as the head is scaled. Bold marks the right answer.

On five of the six prompts, L4/H1 lowers the copied unit, by 0.31 to 1.18 logits. Switching it off raises the copied unit's probability on those five. Scaling it by 4 makes the right answer win on two prompts. On the reversed prompt ("A year has twelve months and a week has seven") it lowers the right answer, " days". There it attends to the second clause's subject, " week", not to the copied unit.

## What this means

- **A weak copy suppressor on these prompts.** L4/H1 does suppress the copy on five of the six prompts, but by too little to change one answer.
- **Not a general copy suppressor.** On a corpus, its effect is small, and copy suppression explains almost none of it. Its VERIFIED explanation describes when it is active: it attends to one recent content token. That is the attention that copy suppression needs, but here the write that follows is mostly something else.
- **Evidence, not a design.** Scaling L4/H1 up is a change to the model that would also move its other writes. Whether the campaign to repair models uses it is the maintainer's question.

## Limits

- The CSPA here keeps only the OV part. The paper's QK part would mean-ablate attention outside the top 5% of predicted tokens, and would lower the share further, not raise it.
- The unembedding direction ignores the final norm's mean subtraction for a source token's write. The check with the whole write shows that the plumbing is exact, but not that this direction is the best one.
- One model, 300 windows, and six prompts.

## Reproduce

```bash
uv run python scripts/copy_suppression.py --out docs/analyses/copy-suppression-l4h1.jsonl
```

It runs in about a minute on a CPU, and reads `corpus.jsonl` for the windows and for the head's mean output.

## References

- McDougall, C., Conmy, A., Rushing, C., McGrath, T., & Nanda, N. (2023). *Copy Suppression: Comprehensively Understanding an Attention Head*. Preprint. [arXiv:2310.04625](https://arxiv.org/abs/2310.04625).

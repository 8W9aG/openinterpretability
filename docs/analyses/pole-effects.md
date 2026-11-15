# Pole effects on the loss, by attribution patching

Run on 2026-11-14 for issue #3486, with `open-interpretability pole-effects`. The tables are in `data/pole-effects/`, one per model. Each holds the 2,000 poles with the largest estimated effect, and the check described below.

## Why

On GPT-2, VERIFIED poles bring back no more of a layer's loss than random poles do: 0.60% against 0.64% (`registry-coverage.md`). So the registry may favour poles that are easy to explain over poles that matter. One remedy is to draw some new units in proportion to their effect on the loss. That needs each pole's effect, and measuring it pole by pole is slow. GPT-2 small has 73,728 poles, and Qwen2.5-0.5B 233,472.

## Method

- **The ablation.** Mean-ablating one pole replaces its part of the activation (the positive part, or the negative part) with that part's mean over the fit windows. `openinterp/coverage.py` applies the same ablation to every pole at once.
- **The estimate.** To first order, the change in mean loss is the sum over tokens of ∂L/∂h × (mean − part). One backward pass per batch gives it for every pole of every layer (`openinterp/pole_effects.py`).
- **The data.** 64 windows of 128 tokens from the registry corpus fit the pole means, and 64 more measure the loss (seed 0).
- **The check.** 60 poles are measured exactly, with one forward pass each. Half are drawn from the estimate's top 1%, and half uniformly.

## Result

| Model | Poles | Spearman, estimate vs exact (60 poles) | Median exact effect: top 1% | Median exact effect: uniform |
| --- | --- | --- | --- | --- |
| GPT-2 small | 73,728 | 0.97 | 0.00076 nats | −0.00001 nats |
| Qwen2.5-0.5B | 233,472 | 0.85 | 0.00028 nats | 0.00002 nats |

- **The estimate ranks poles well.** That holds on both models, so a draw weighted by it favours poles that really matter. The issue's plan set a Spearman of 0.5 as the bar.
- **The top of the ranking is far above a typical pole.** On Qwen, the estimate's top 1% raises the loss 12 times as much as a uniform pole. On GPT-2 a typical pole has almost no effect.
- **Some poles lower the loss when ablated.** Their estimate is negative, and they are never drawn by effect.

## Use

`open-interpretability next` now draws a share of explore units in proportion to effect: `[tool.openinterp] effect_share`, default 0.3, from the model's table. Such a draw names its pole (`audit --layer --neuron --pole`) and records `selection: attribution`. The registry statistics count only `random` draws, so they stay unbiased. A model with no table, such as Qwen2.5-1.5B for now, always draws uniformly.

## Limits

- **First order.** The estimate is linear in the change. A pole whose effect is mostly nonlinear can be ranked too low. On Qwen, two of the 60 checked poles measured 3 and 4.5 times their estimate.
- **One loss and one corpus.** Effect here means next-token loss on the registry corpus. A pole that matters for one behaviour can matter little for the average.
- **60 checked poles per model.** The Spearman values have intervals of roughly ±0.05 to ±0.1.
- **Qwen2.5-1.5B has no table yet.** The same command makes one.

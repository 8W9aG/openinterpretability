# Registry coverage

Run on 2026-11-13 for issue #3371, with `openinterp coverage`. The question: how much of each model does the registry explain? This is the project's north star, total interpretability of open models, as one number. The rows are in `registry-coverage.jsonl`, one per model and layer.

## Method

For each MLP layer, mean-ablating every neuron raises the next-token loss. That rise is what the layer adds. The share of the rise that comes back is the layer's coverage. It comes back when the VERIFIED poles keep their real values and every other pole is mean-ablated:

recovered = (loss with every pole ablated − loss with the VERIFIED poles kept) / (loss with every pole ablated − clean loss)

- Each neuron has a positive part and a negative part, its two poles. Each part is replaced by its own mean over 64 windows of `corpus.jsonl`, unless that pole stands VERIFIED. Standing uses the site's rules, errata included.
- The loss is measured on 64 other windows of 128 tokens.
- The baseline keeps as many poles, drawn at random from the layer's poles that are not VERIFIED. It is the mean of 5 draws.
- Over a model, the share is weighted by each layer's loss rise: Σ(rise brought back) / Σ(rise).

## Results

| Model | VERIFIED poles | Share of all poles | Recovered | Random poles, same count |
| --- | --- | --- | --- | --- |
| GPT-2 | 172 | 0.23% | 0.60% | 0.64% |
| Qwen2.5-0.5B | 305 | 0.13% | 0.42% | 0.16% |
| Qwen2.5-1.5B | 39 | 0.008% | 0.01% | 0.00% |

- **The registry explains well under 1% of any model's MLP layers.** That is the size of the task. At this rate of VERIFIED poles, coverage needs a far higher rate, not more of the same.
- **On Qwen2.5-0.5B, VERIFIED poles bring back about 2.6 times as much as random ones.** The best layers are L14 (9 poles, 1.8% against −0.2% for random), L0 (19 poles, 1.1% against 0.4%) and L12 (12 poles, 1.1% against 0.2%).
- **On GPT-2, VERIFIED poles bring back no more than random ones.** L6 is the exception: 21 poles bring back 5.2% against −1.9% for random. Over the whole model, the two are equal.
- **Keeping some poles can make the loss worse than ablating all of them.** On GPT-2 L2, the 13 VERIFIED poles kept give a loss above the fully ablated layer (recovered −0.34). Partial restoration can move the layer's output to a place the next layers never see in training. So a negative share doesn't mean that the poles are wrong. The same effect is smaller on 8 more GPT-2 layers and on Qwen2.5-0.5B L7.
- **Qwen2.5-1.5B has too few VERIFIED poles to measure.** 39 poles over 28 layers give shares that round to zero.

## Limits

- **Mean ablation is a strong intervention.** It takes the layer off its normal path, and the negative shares show that partial restoration is not additive. Resample ablation, or replacing each VERIFIED pole with its simulated activation, would test the explanations, not only the units. That is the issue's second method, for a later run.
- **One layer at a time.** The share for the whole model is a weighted sum of single-layer shares. Ablating all layers at once destroys the model, so it can't measure coverage now.
- **Small samples.** 8,192 tokens, one corpus, one seed for the windows. The random baseline is 5 draws.
- **A unit is kept whole or not at all.** A VERIFIED pole keeps all of its activations, including those its explanation doesn't cover.

## What this means

- The metric works, and it gives the registry its first number for how much of a model it explains. It should be rerun at each release (`openinterp coverage --model M`) and shown on the site (a follow-up issue).
- On GPT-2, VERIFIED poles do no better than random ones at keeping the loss. One reading is that the registry favours poles that are easy to explain, not poles that matter to the loss. The planner could weight draws by each pole's effect on the loss.

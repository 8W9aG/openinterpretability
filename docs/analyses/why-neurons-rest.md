# Why training rests neurons below GELU's dip

`default-on-poles.md` found that training moves most of GPT-2's neurons to a median pre-activation h of about −1, below GELU's minimum at −0.75, and that the input weights do this. It did not find why. This analysis trains small GPT-2s and changes one thing at a time.

**Results.** All are on small models, except the first's GPT-2 figures. Each number is from one or two seeds.

1. **The small models show GPT-2's pattern.** Their rest comes from weights set against the MLP's mean input, and not from the bias or from a skewed tail. So a small model is a fair place to test causes, with the limits under *Caveats*.
2. **The rest does not need GELU's dip.** ReLU has no dip, and its neurons rest at the same depth. Tanh's neurons don't move at all. So the rest goes with a one-sided activation, and "below the dip" may be a side effect.
3. **Width, dropout and weight decay don't set it.** It forms with no dropout and no weight decay. A 16× range of width changes L1–L3 by at most 0.08. The interference explanation predicts that narrower MLPs rest lower, and we don't see that.
4. **Training pushes the rest down by whichever route it has.** When centring stops the weights from setting the rest, a learned value for each neuron goes down at nearly the same pace, with the same order of layers. So the weights' turn against the mean input is one route to the rest, not its cause.
5. **The rest seems to help, but this is not proven.** Letting the learned value reach that depth lowers the loss by 0.061 on seed 0, with seeds differing by about 0.02. A centred model still doesn't match the baseline, and a forced rest of −1 trains as badly as a forced rest of 0.
6. **−1 is not a target.** Over 12,000 steps the rest keeps deepening, to −1.46 at L1 with the weights and −1.32 with the learned value. Its growth slows, but it passes GPT-2's −1.04. The closeness to −1 at step 4,000 is partly the moment that we measured.

We don't know why one-sided activations favour a low rest. These runs remove some explanations, and they don't show the mechanism.

## Method

The script is `scripts/train_resting.py`. Its output is in `why-neurons-rest.jsonl`.

- **Models.** GPT-2's architecture with 4 layers, width 256 and 8 heads, trained from fresh weights for 4,000 steps. Each step is 32 windows of 128 tokens, with AdamW at a learning rate of 3e-4 and gradient clipping at 1.0.
- **Data.** Training uses pile-10k without the documents of the `corpus.jsonl` sample, which leaves 9,482 of 10,000 documents. Evaluation uses 128 test windows of the same split as `default-on-poles.md`.
- **Measures.** The median over neurons of each neuron's median h, the median offset (w·x̄, with x̄ the mean of the MLP's input), the median bias, the skew, and the share of neurons with weights against x̄.
- **Centred conditions.** h is centred per neuron over each batch, then moved to a rest. That rest is fixed for `rest_0` and `rest_-1`, and is a parameter for each neuron for `rest_learned` and `rest_learned_fast`. Centring removes the weights' effect on the rest, so we report the rest as applied, not the raw h.

## GPT-2 small

On GPT-2 small itself, the rest is an offset, not a skew. On every layer but L1, the mean of h is close to its median, and skew is small. L0's skew is the largest, at 0.62. L1 is the exception, with a skew of −0.33 and a shallow rest of −0.24.

| Layer | Median h | Offset from the mean input | Bias | Skew | Against mean input | Random init: median h, against |
| --- | --- | --- | --- | --- | --- | --- |
| L0 | −1.16 | −1.01 | −0.07 | 0.62 | 93% | 0.00, 50% |
| L1 | −0.24 | −0.33 | −0.05 | −0.33 | 89% | 0.01, 49% |
| L2 | −1.15 | −1.04 | −0.09 | 0.08 | 95% | 0.00, 50% |
| L3 | −1.25 | −1.13 | −0.09 | 0.10 | 99% | 0.01, 49% |
| L4 | −0.99 | −0.92 | −0.08 | 0.04 | 99% | 0.01, 49% |
| L5 | −1.11 | −1.02 | −0.07 | 0.10 | 99% | −0.01, 51% |
| L6 | −1.05 | −0.95 | −0.08 | 0.11 | 99% | −0.01, 51% |
| L7 | −1.01 | −0.91 | −0.08 | 0.11 | 99% | 0.00, 50% |
| L8 | −1.03 | −0.93 | −0.08 | 0.12 | 99% | 0.00, 50% |
| L9 | −1.04 | −0.95 | −0.08 | 0.13 | 100% | −0.01, 51% |
| L10 | −0.99 | −0.90 | −0.08 | 0.15 | 99% | 0.00, 49% |
| L11 | −0.86 | −0.78 | −0.07 | 0.15 | 97% | 0.00, 50% |

## What changes the rest

Step 4,000 of each condition. Columns L0–L3 are the median h.

| Condition | Loss, seed 0 | Loss, seed 1 | L0 | L1 | L2 | L3 | Against mean input, L1–L3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline: GELU, 4× width, dropout 0.1, no weight decay | 5.509 | 5.509 | −0.34 | −1.11 | −1.06 | −0.78 | 100% |
| −GELU(−h): a check, since this is the baseline with signs reversed | 5.501 | | 0.33 | 1.08 | 0.99 | 0.78 | 0% |
| ReLU | 5.486 | 5.484 | −0.40 | −1.13 | −1.18 | −1.01 | 100% |
| Tanh | 5.686 | | 0.00 | 0.00 | 0.00 | 0.00 | 48%–50% |
| 1× width | 5.546 | | −0.29 | −1.11 | −1.05 | −0.76 | 100% |
| 16× width | 5.504 | | −0.37 | −1.17 | −1.13 | −0.73 | 100% |
| No dropout | 5.452 | | −0.16 | −0.97 | −1.00 | −0.76 | 100% |
| Dropout 0.3 | 5.675 | | −0.57 | −0.84 | −0.98 | −0.74 | 100% |
| Weight decay 0.1 | 5.523 | | −0.28 | −0.98 | −0.94 | −0.67 | 100% |

- **Seeds agree.** The baseline's depth differs by at most 0.04 between seeds, and ReLU's by at most 0.03. Conditions with no seed 1 entry ran once.
- **Mirrored GELU is only a check.** It is the baseline with h and the weights negated, so its rest must flip, and it does.
- **ReLU rests as deep as GELU.** It has no dip and no flat region below zero that differs by depth, so the dip cannot be its target. ReLU is also indifferent to scale, so a rest near −1 has no special meaning for it. That similarity may have a simple cause in the training (see *Caveats*).
- **Tanh doesn't shift.** This is weak evidence, because tanh is symmetric and no explanation gives it a direction. Its spread of h falls to 0.15–0.19 on L1–L3, so these neurons mostly stay in its near-linear region. It has the worst loss here.
- **Width doesn't move the rest.** If neurons rest low to filter interference, narrower MLPs have more interference and should rest lower. They don't. These widths may not change interference enough, so this counts against that version of the idea and doesn't remove it.
- **Regularisation isn't needed.** The no-dropout run also has no weight decay, and its rest forms at L1–L3. Dropout changes L1–L3 without a clear trend. L0 is the exception, and deepens with dropout (−0.16, −0.34, −0.57). L0 reads the embeddings, so this may be real, but it is three points on one seed.

## Does the model need the rest?

Forcing the rest to 0 costs 0.136 of loss on seed 0 and 0.155 on seed 1. But centring is part of that change, so we added controls.

| Condition | Loss, seed 0 | Loss, seed 1 | Rest set by | L0 | L1 | L2 | L3 | Against mean input, L1–L3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 5.509 | 5.509 | the weights | −0.34 | −1.11 | −1.06 | −0.78 | 100% |
| Centred, learned rest at 30× the rate | 5.574 | 5.581 | the learned value | −0.44 | −0.92 | −0.85 | −0.71 | 47%–58% |
| Centred, learned rest at the base rate | 5.635 | | the learned value | −0.05 | −0.09 | −0.08 | −0.08 | 29%–52% |
| Centred, rest 0 | 5.645 | 5.664 | forced | 0.00 | 0.00 | 0.00 | 0.00 | 26%–49% |
| Centred, rest −1 | 5.657 | 5.664 | forced | −1.00 | −1.00 | −1.00 | −1.00 | 38%–56% |

- **A forced rest does not separate the causes.** `rest_-1` is near the baseline's depth and trains as badly as `rest_0`. It is also fixed for every neuron, and it starts at −1, where the fresh model's narrow h (SD about 0.31) sits on GELU's flat part. Its loss at step 1,000 is 6.40–6.42, against 6.27–6.28 for `rest_0`, and it has almost caught up by step 4,000. So its cost looks like a slow start, but we did not test this directly.
- **At the base rate, a learned rest stays near 0.** It reaches only −0.09 at L1, so this control mostly repeats `rest_0`. Adam at this rate can move one parameter by about 1.2 in 4,000 steps, and only if every step has the same sign. The baseline's bias moves only about 0.03. The weights can shift the rest faster, because many weights multiply a large mean input.
- **At 30× the rate, a learned rest reaches nearly the same depth by step 4,000.** It reaches −0.92, −0.85 and −0.71 at L1–L3 on seed 0 and −0.92, −0.86 and −0.69 on seed 1. That keeps the baseline's order: L1, then L2, then L3, then L0. The weights no longer turn against x̄. Reaching that depth lowers the loss to 5.574 and 5.581.
- **A gap remains.** The fast control's loss stays 0.065–0.072 above the baseline at step 4,000, and longer training doesn't close it.

## Longer training

Step 4,000 is early, because the rest is still moving. So seed 0 of the baseline and of the fast control ran to step 12,000.

| Step | Baseline loss | Fast control loss | Gap | Baseline rest: L0 | L1 | L2 | L3 | Fast control rest: L0 | L1 | L2 | L3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2,000 | 5.841 | 5.910 | 0.070 | −0.21 | −0.74 | −0.68 | −0.50 | −0.30 | −0.60 | −0.53 | −0.41 |
| 4,000 | 5.510 | 5.574 | 0.064 | −0.34 | −1.10 | −1.06 | −0.78 | −0.44 | −0.92 | −0.85 | −0.71 |
| 6,000 | 5.348 | 5.418 | 0.070 | −0.42 | −1.28 | −1.24 | −0.95 | −0.51 | −1.08 | −1.02 | −0.86 |
| 8,000 | 5.241 | 5.311 | 0.070 | −0.47 | −1.35 | −1.37 | −1.04 | −0.55 | −1.19 | −1.12 | −0.96 |
| 10,000 | 5.163 | 5.231 | 0.068 | −0.50 | −1.43 | −1.44 | −1.13 | −0.58 | −1.26 | −1.20 | −1.02 |
| 12,000 | 5.137 | 5.213 | 0.076 | −0.54 | −1.46 | −1.49 | −1.15 | −0.60 | −1.32 | −1.23 | −1.07 |

- **The rest doesn't settle at −1.** The baseline's L1 passes −1 by step 4,000 and reaches −1.46. Growth slows, with only 0.03 added over the last 2,000 steps at L1, so it may settle later. We can't say where.
- **The two routes keep pace.** At every step the learned value is at 81–90% of the baseline's depth at L1. The learned value has a much higher limit on its rate, so a limit on one parameter's rate doesn't explain the pace of this control.
- **The loss gap doesn't close.** It stays between 0.064 and 0.076 at every step in the table. So the gap isn't only unfinished training. Centring itself, or the difference in depth, may cost this much. So the runs don't measure the rest's full value.

**One run was discarded.** The first 12,000-step baseline gave a loss of 5.908 at step 4,000 and a much wider h, while the earlier run with the same seed gave 5.509. It overlapped other jobs on the machine, and it showed no error. A rerun with nothing else running gave 5.510. The fast control ran after that overlap, and it matched its earlier values at steps 2,000 and 4,000. We did not find the cause, so we kept the discarded rows in the data as `training_long_discarded`.

## Why this matters for reading negative poles

Most of the cost of the negative side in GPT-2 is its resting level (`default-on-poles.md`): clamping it at its mean adds 0.19 nats, and zeroing it adds 3.72. These runs fit that. Training pushes the level down, while variation below the dip has less effect. This supports the protocol's stop on auditing a default-on pole, but it doesn't show that a pole has no content.

## Caveats

- **These are small models.** They have 4 layers and short training. Their pattern matches GPT-2 on offset, bias and weights, but not everywhere: GPT-2's L1 rests shallow (−0.24) and its L0 deep (−1.16), and small models show the opposite order. Results on small models suggest causes for GPT-2, and don't prove them.
- **The optimizer may shape the depth.** GELU and ReLU rest at similar depths at step 4,000, and the rest grows over steps. We used only Adam and one base rate, and ReLU ran only to step 4,000. So this similarity could come from training and not from the activation.
- **Runs can go wrong with no error.** See *One run was discarded*. Same-seed reruns match to about 0.001, so a mismatch is easy to see, but most conditions here were not rerun.
- **Few seeds.** Only the baseline, ReLU, and the fixed and fast centred conditions have two seeds. The rest have one.
- **Centring is a blunt tool.** It takes its mean from the batch, also at evaluation, and the gradient passes through that mean.
- **"Seeks" means the gradient pushes that way.** It doesn't show a designed function.

## Reproduce

Each command appends to its `--out`. The runs took about 25–40 minutes each on an Apple GPU. Run nothing else on the machine at the same time. Parallel runs on MPS were slower and failed once, and one run that overlapped other jobs went wrong with no error.

```bash
OUT=docs/analyses/why-neurons-rest.jsonl
uv run python scripts/train_resting.py gpt2 --out $OUT
uv run python scripts/train_resting.py train --seed 0 --steps 4000 --eval-every 500 --cache tokens.pt --out $OUT
uv run python scripts/train_resting.py train --seed 1 --steps 4000 --eval-every 500 --cache tokens.pt \
    --condition gelu --condition rest_0 --condition rest_learned_fast --condition relu \
    --condition rest_-1 --out $OUT
uv run python scripts/train_resting.py train --seed 0 --steps 12000 --eval-every 2000 --cache tokens.pt \
    --condition gelu --condition rest_learned_fast --out $OUT
```

## Next

Open questions are in `TODO.md` (*Protocol and science*).

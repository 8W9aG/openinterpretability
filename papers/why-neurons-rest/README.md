# Why training rests neurons below GELU's dip

[The paper on default-on negative poles](../default-on-poles/README.md) found that training moves most of GPT-2's neurons to a median pre-activation h of about −1. That is below GELU's minimum at −0.75. It also found that the input weights do this. It did not find why. This analysis trains small GPT-2s and changes one thing at a time.

**Results.** All are on small models, except the first's GPT-2 figures. Each number is from one or two seeds.

1. **The small models show GPT-2's pattern.** Their rest comes from weights set against the MLP's mean input, and not from the bias or from a skewed tail. So a small model is a fair place to test causes, with the limits under *Caveats*.
2. **The rest does not need GELU's dip.** ReLU has no dip, and its neurons rest at the same depth. Tanh's neurons don't move at all. So the rest goes with a one-sided activation, and "below the dip" may be a side effect.
3. **Width, dropout and weight decay don't set it.** It forms with no dropout and no weight decay. A 16× range of width changes L1–L3 by at most 0.08. The interference explanation predicts that narrower MLPs rest lower, and we don't see that.
4. **Training pushes the rest down by whichever route it has.** Centring can stop the weights from setting the rest. Then a learned value for each neuron goes down at nearly the same pace, with the same order of layers. So the weights' turn against the mean input is one route to the rest, not its cause.
5. **The rest seems to help, but this is not proven.** Letting the learned value reach that depth lowers the loss by 0.061 on seed 0, with seeds differing by about 0.02. A centred model still doesn't match the baseline, and a forced rest of −1 trains as badly as a forced rest of 0.
6. **−1 is not a target.** Over 12,000 steps the rest keeps deepening, to −1.46 at L1 with the weights and −1.32 with the learned value. ReLU keeps pace, at −1.49. Its growth slows, but it passes GPT-2's −1.04. The closeness to −1 at step 4,000 is partly the moment that we measured.
7. **The optimizer's step size sets the depth.** With AdamW, L1's rest at step 4,000 is −0.37, −1.11 and −3.18 at learning rates of 1e-4, 3e-4 and 1e-3. That is close to proportional to the rate. The two larger rates end at the same loss, so the loss doesn't need a particular depth. Under SGD the rest forms, but it is shallower than AdamW's at the same loss.
8. **ReLU and GELU agree only under AdamW.** They rest at the same depth at AdamW's two lower rates. Under SGD, ReLU rests nearly twice as deep as GELU (−0.59 against −0.32 at L1). At 1e-3, ReLU is deeper (−4.04 against −3.18).
9. **No dip and no floor sets the depth.** Every one-sided activation that we tried rests below 0, with its weights against the mean input. But softplus rests at −4.68, leaky ReLU at −0.69 and SiLU at −0.49 at L1. SiLU's dip is deeper than GELU's, and its neurons rest shallower than GELU's.
10. **A small, steady push sets the depth.** The gradient on each neuron's rest points down on only 50%–52% of single steps. But its mean over 100 steps points down in 85%–100% of windows for each one-sided activation. Its size, as mean over RMS, gives the order of depth. Softplus is first. ReLU, GELU and leaky ReLU at slope 0.01 are next, then leaky ReLU at slope 0.1, SiLU and tanh. The push moves the rest through the weights, which carry 97%–99% of each step's movement. We measured it along each run, so it doesn't show that the activation sets the push. Where a neuron rests also changes the push it gets.

![GELU, ReLU and tanh curves, with the rest of layers L1 to L3 marked on each](media/activations.svg)

We still don't know why one-sided activations favour a low rest. The runs now record the push that training keeps up for as long as it runs. It is weak on each step, and steady over many steps. Under AdamW, a steady sign is what moves a parameter, so a steadier push gives a deeper rest. The runs don't show where the push comes from. They also don't show whether its strength is set by the activation, or by where the neurons already rest.

## Method

The script is `scripts/train_resting.py`. Its output is in `results.jsonl`. `scripts/paper_figures.py` draws the figures from that output, with no model.

- **Models.** The architecture of GPT-2 (Radford et al., 2019) with 4 layers, width 256 and 8 heads, trained from fresh weights for 4,000 steps. Each step is 32 windows of 128 tokens, with AdamW (Loshchilov & Hutter, 2019) at a learning rate of 3e-4 and gradient clipping at 1.0.
- **Data.** Training uses the first 10,000 documents of the Pile (Gao et al., 2020) without the documents of the `corpus.jsonl` sample, which leaves 9,482 of 10,000 documents. Evaluation uses 128 test windows of the same split as [What default-on negative poles do as a population](../default-on-poles/README.md).
- **Activation.** The baseline uses GELU (Hendrycks & Gimpel, 2016). Other conditions use ReLU, tanh, softplus (Dugas et al., 2001), leaky ReLU and SiLU (Elfwing et al., 2017). Softplus is log(1 + eʰ). Leaky ReLU is h above 0 and 0.1h below 0, or 0.01h for the condition `leaky_relu_0.01`. SiLU is h·σ(h).
- **Optimizer.** Other conditions use AdamW at 1e-4 or 1e-3, or SGD with momentum 0.9 at 0.1. We chose 0.1 from 0.01, 0.03, 0.1 and 0.3 over 500 steps. 0.1 and 0.3 reached the same loss, and 0.3 was less steady at step 250. Each of these conditions ran once, on seed 0.
- **Measures.** The median over neurons of each neuron's median h, and the median offset (w·x̄, with x̄ the mean of the MLP's input). Then the median bias, the skew, and the share of neurons with weights against x̄.
- **Push.** With `--push-every 100`, each run also records the gradient on each neuron's rest at every step, and how far the step moves the rest. It writes a summary of each 100 steps as a `push` row (*The push on the rest*). Recording doesn't change training: a 500-step GELU run with it matched the earlier run's loss and rests exactly.
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

![Median h of layers L0 to L3 under each change to the baseline](media/effects.svg)

- **Seeds agree.** The baseline's depth differs by at most 0.04 between seeds, and ReLU's by at most 0.03. Conditions with no seed 1 entry ran once.
- **Mirrored GELU is only a check.** It is the baseline with h and the weights negated, so its rest must flip, and it does.
- **ReLU rests as deep as GELU.** It has no dip and no flat region below zero that differs by depth, so the dip cannot be its target. ReLU is also indifferent to scale, so a rest near −1 has no special meaning for it. That similarity may have a simple cause in the training (see *Caveats*).
- **Tanh doesn't shift.** This is weak evidence, because tanh is symmetric and no explanation gives it a direction. Its spread of h falls to 0.15–0.19 on L1–L3, so these neurons mostly stay in its near-linear region. It has the worst loss here.
- **Width doesn't move the rest.** In toy models, Elhage et al. (2022) show that a negative bias can turn small positive interference into negative interference, which is free under a ReLU in their one-sparse case. Our step is to apply this to the rest of real MLPs. If neurons rest low to filter interference, narrower MLPs have more interference and should rest lower. They don't. These widths may not change interference enough, so this counts against that version of the idea and doesn't remove it.
- **Regularisation isn't needed.** The no-dropout run also has no weight decay, and its rest forms at L1–L3. Dropout changes L1–L3 without a clear trend. L0 is the exception, and deepens with dropout (−0.16, −0.34, −0.57). L0 reads the embeddings, so this may be real, but it is three points on one seed.

## More activations

Step 4,000 of each condition, with AdamW at the base rate. Columns L0–L3 are the median h.

| Activation | Loss | L0 | L1 | L2 | L3 | SD of h, L1–L3 | Against mean input, L1–L3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GELU (baseline) | 5.509 | −0.34 | −1.11 | −1.06 | −0.78 | 0.78–0.95 | 100% |
| ReLU | 5.486 | −0.40 | −1.13 | −1.18 | −1.01 | 0.62–0.74 | 100% |
| Softplus | 5.826 | −3.83 | −4.68 | −4.60 | −3.52 | 1.55–2.17 | 99%–100% |
| Leaky ReLU, slope 0.1 | 5.518 | −0.41 | −0.69 | −0.79 | −0.68 | 0.55–0.68 | 100% |
| Leaky ReLU, slope 0.01 | 5.496 | −0.40 | −1.11 | −1.20 | −1.06 | 0.62–0.74 | 100% |
| SiLU | 5.588 | −0.30 | −0.49 | −0.42 | −0.24 | 0.46–0.71 | 98%–99% |
| Tanh | 5.686 | 0.00 | 0.00 | 0.00 | 0.00 | 0.15–0.19 | 48%–50% |

![Softplus, leaky ReLU and SiLU curves, with the rest of layers L1 to L3 marked on each](media/more-activations.svg)

- **Each one-sided activation rests below 0, by the same route.** The weights turn against x̄, and the bias stays between −0.06 and 0.
- **The dip doesn't set the depth.** SiLU is GELU's shape stretched along h by about 1.7, so its dip is at −1.28 and not −0.75. If neurons sought the dip, or the activation's scale, SiLU would rest deeper than GELU. It rests at about half GELU's depth.
- **A floor doesn't set the depth.** One idea was that a neuron goes down until its output is near 0, its "off" state. Softplus fits: its output at −4.68 is 0.009. Leaky ReLU fits too: it passes 10% below 0, so going deeper never turns it off, and it rests shallower than ReLU. SiLU doesn't fit. Its output is within 0.1 of 0 only below −3.54, against −2.25 for softplus, so this idea puts SiLU deeper than softplus. It rests at −0.49, where its output is −0.19. So no single property of these shapes that we checked gives their order.
- **Softplus widens as well as shifts.** Its SD of h reaches 2.17, more than twice the baseline's, and it trains the worst of the one-sided activations here.

## Optimizer and learning rate

Step 4,000 of each condition, seed 0. The AdamW 3e-4 rows are the baseline and ReLU from the table above. Columns L0–L3 are the median h.

| Activation | Optimizer | Loss | L0 | L1 | L2 | L3 | L1 at a loss of 5.9 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GELU | AdamW 1e-4 | 5.877 | −0.14 | −0.37 | −0.30 | −0.24 | −0.37 |
| GELU | AdamW 3e-4 | 5.509 | −0.34 | −1.11 | −1.06 | −0.78 | −0.68 |
| GELU | AdamW 1e-3 | 5.505 | −1.71 | −3.18 | −3.07 | −2.47 | −2.55 |
| GELU | SGD 0.1 | 5.804 | −0.62 | −0.32 | −0.25 | −0.23 | −0.28 |
| ReLU | AdamW 1e-4 | 5.863 | −0.17 | −0.40 | −0.43 | −0.38 | −0.39 |
| ReLU | AdamW 3e-4 | 5.486 | −0.40 | −1.13 | −1.18 | −1.01 | −0.87 |
| ReLU | AdamW 1e-3 | 5.494 | −2.74 | −4.04 | −4.43 | −3.33 | −2.95 |
| ReLU | SGD 0.1 | 5.797 | −0.68 | −0.59 | −0.59 | −0.60 | −0.55 |

The last column interpolates linearly between the two evaluations around a loss of 5.9.

![L1's rest against held-out loss for each optimizer and rate, for GELU and ReLU](media/rates.svg)

- **At the same step, the depth is close to proportional to the rate.** For GELU, L1's depth grows 3.0× from 1e-4 to 3e-4, and 2.9× from 3e-4 to 1e-3. Adam moves each weight by about the learning rate at each step, whatever the gradient's size. So this is what a push of the same sign at every step gives, with nothing that pushes back.
- **At the same loss, a larger rate still rests deeper.** The 3e-4 and 1e-3 runs end at nearly the same loss (5.509 and 5.505 for GELU), with rests of −1.11 and −3.18. So the loss doesn't need a particular depth, and the depth isn't only a measure of how far training has got.
- **SGD gives a shallower rest.** At a loss of 5.9, GELU's L1 is at −0.28 under SGD and −0.68 under AdamW at the base rate. The weights still turn against x̄ for 99%–100% of neurons. So the rest doesn't need Adam, but its depth depends on the optimizer.
- **ReLU and GELU part outside AdamW's lower rates.** Under SGD, ReLU rests about twice as deep as GELU at every layer from L1 to L3. So their agreement at the base rate comes partly from AdamW, as *Caveats* in the first version of this analysis suspected.
- **L0 deepens fastest under SGD.** Under SGD, L0 rests deepest of the four layers. That is the order of GPT-2 small, whose L0 rests at −1.16. With one seed and one rate, this is only a hint.
- **Seed 1 agrees.** At step 4,000 on seed 1, GELU's L1 rests at −0.37, −3.22 and −0.30 at AdamW 1e-4, AdamW 1e-3 and SGD. ReLU's rests at −0.42, −4.75 and −0.60. The largest difference from seed 0 is ReLU at 1e-3, at 0.71. The two larger rates still end at nearly the same loss.

The push on the rest under each optimizer and rate, from the runs that recorded it. Seed 1, except AdamW 3e-4, which is the seed-0 reruns of the table in *The push on the rest*. Rest, L1–L3 is the mean of the three layers' median h at step 4,000.

| Activation | Optimizer | Rest, L1–L3 | Push ratio, steps 1–4,000 | Median raw push |
| --- | --- | --- | --- | --- |
| GELU | AdamW 1e-4 | −0.29 | −0.016 | −5.2 × 10⁻⁶ |
| GELU | AdamW 3e-4 | −0.98 | −0.019 | −3.4 × 10⁻⁶ |
| GELU | AdamW 1e-3 | −2.95 | −0.014 | −1.1 × 10⁻⁶ |
| GELU | SGD 0.1 | −0.26 | −0.015 | −3.0 × 10⁻⁶ |
| ReLU | AdamW 1e-4 | −0.40 | −0.018 | −5.8 × 10⁻⁶ |
| ReLU | AdamW 3e-4 | −1.10 | −0.020 | −4.1 × 10⁻⁶ |
| ReLU | AdamW 1e-3 | −4.32 | −0.018 | −9.3 × 10⁻⁷ |
| ReLU | SGD 0.1 | −0.59 | −0.021 | −5.0 × 10⁻⁶ |

- **The push ratio doesn't depend on the rate.** For ReLU it is −0.018 to −0.020 at all three AdamW rates, and for GELU −0.014 to −0.019. The depth grows about 3× with each 3× step in rate. So AdamW takes a step in proportion to its rate on a push of about the same steadiness. That is the fixed distance per step that items 6 and 7 suggested.
- **SGD has a push of the same steadiness but a shallower rest.** SGD's step follows the push's raw size, not its ratio to its RMS. ReLU's raw push under SGD is 1.7× GELU's, and ReLU rests 2.3× as deep as GELU under SGD. Under AdamW the two have about the same ratio and rest at about the same depth. This may explain item 8, but it is two pairs of runs.
- **The raw push gets smaller as the rest gets deeper.** At 1e-3 the raw push is 3–6× smaller than at 1e-4, while its ratio stays the same. So the push's size can't set the depth under AdamW. Its steadiness can.

## The push on the rest

At each step we recorded −dL/dh summed over the batch's tokens, for each neuron. This is the push: the direction in which one gradient step moves the neuron's whole h. It is negative when it moves the rest down. Without centring, it is the gradient on the bias with its sign reversed. We also recorded how far each step moves each neuron's rest, x̄·ΔW + Δb, at the batch's mean input x̄. Each row of the data sums 100 steps. The push ratio is the push's mean over those 100 steps divided by its RMS, as a median over neurons. AdamW moves a parameter by about the learning rate times this kind of ratio.

Step 4,000 of each run, with AdamW at the base rate. The push columns are means over L1–L3. Rest, L1–L3 is the mean of the three layers' median h.

| Activation | Seed | Loss | L1 | Rest, L1–L3 | Push ratio, steps 1–1,000 | Push ratio, steps 1–4,000 | Windows pushed down | Single steps pushed down |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Softplus | 0 | 5.826 | −4.69 | −4.27 | −0.080 | −0.043 | 100% | 51.7% |
| Softplus | 1 | 5.797 | −4.96 | −4.41 | −0.082 | −0.044 | 100% | 51.6% |
| ReLU | 0 | 5.491 | −1.11 | −1.10 | −0.036 | −0.020 | 100% | 50.0% |
| Leaky ReLU, slope 0.01 | 0 | 5.496 | −1.11 | −1.12 | −0.036 | −0.020 | 100% | 50.1% |
| Leaky ReLU, slope 0.01 | 1 | 5.470 | −1.16 | −1.12 | −0.036 | −0.020 | 100% | 50.1% |
| GELU | 0 | 5.509 | −1.11 | −0.98 | −0.022 | −0.019 | 97% | 50.4% |
| Leaky ReLU, slope 0.1 | 0 | 5.557 | −0.70 | −0.70 | −0.026 | −0.015 | 93% | 50.0% |
| Leaky ReLU, slope 0.1 | 1 | 5.517 | −0.70 | −0.74 | −0.027 | −0.017 | 100% | 50.1% |
| SiLU | 0 | 5.588 | −0.49 | −0.38 | +0.002 | −0.010 | 85% | 50.2% |
| SiLU | 1 | 5.588 | −0.55 | −0.40 | +0.001 | −0.010 | 87% | 50.3% |
| Tanh | 0 | 5.686 | 0.00 | 0.00 | +0.001 | +0.001 | 42% | 50.0% |

![The push ratio over training for each activation, and L1's rest against the mean push ratio](media/push.svg)

- **The push keeps its sign over many steps, not over one.** On a single step it is close to a coin toss, at 50.0%–51.7%. Its mean over 100 steps points down in 85%–100% of windows for each one-sided activation, and in 42% for tanh. So it is a small bias in a noisy gradient. The gradient on the weights along x̄ behaves the same way. This fits the rest's steady deepening and its dependence on the learning rate (*Optimizer and learning rate*). It does not fit a large push that stops at a target.
- **Its size gives the order of depth.** Softplus has the largest ratio and the deepest rest. Then ReLU, leaky ReLU at slope 0.01 and GELU, at −0.019 to −0.020 and about −1.1. Then leaky ReLU at slope 0.1, SiLU and tanh. That is the order in which they rest. The relation is not linear. Softplus rests about twice as deep as the others' ratio of depth to push would give.
- **The order forms early.** Over steps 3,500–4,000, ReLU, leaky ReLU at slope 0.01, GELU and SiLU all have push ratios of −0.010 to −0.013, and softplus has −0.019. So late in the run, the push no longer gives the order. The differences come from the first 1,000 steps. There, the order is softplus, ReLU and leaky ReLU at slope 0.01, leaky ReLU at slope 0.1, GELU, then SiLU and tanh. That is the order of depth, except that GELU's early push is weaker than that of leaky ReLU at slope 0.1, while GELU rests deeper.
- **GELU and SiLU are first pushed up.** Their push points up until step 200 for GELU and step 400 for SiLU. The push on ReLU, leaky ReLU and softplus points down from the first step. The gradient on their output points up all through training. Their push on h turns down only because the activation's slope weights the tokens, and where those tokens lie changes as the rest moves. For ReLU, both leaky ReLUs early on, and softplus, the gradient on the output points down from the start. So a single property of the loss, such as "make each neuron's output smaller", doesn't explain all six.
- **The weights carry the push.** 97%–99% of each step's movement of the rest is x̄·ΔW, and 1%–3% is the bias. This matches *GPT-2 small*, where the offset and not the bias sets the rest.
- **This is a correlation along each run.** We measured each push where that run's neurons rest at that step. So the activation may set the push, or the depth that a run reaches may set its push. The next test measures the push with every activation held at the same rest (see *Next*).
- **Leaky ReLU at slope 0.01 rests like ReLU.** It rests at −1.11 and −1.16 at L1, against −1.13 for ReLU. So the shallow rest of leaky ReLU at slope 0.1 comes from its large slope below 0.
- **One run has gradient spikes.** Leaky ReLU at slope 0.1, seed 0, has a few windows near step 3,600–3,900 where the push ratio jumps above 0. Its seed 1 has none. We report medians and ratios, because such spikes dominate the means of the raw push.

## Does the model need the rest?

Forcing the rest to 0 costs 0.136 of loss on seed 0 and 0.155 on seed 1. But centring is part of that change, so we added controls.

| Condition | Loss, seed 0 | Loss, seed 1 | Rest set by | L0 | L1 | L2 | L3 | Against mean input, L1–L3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 5.509 | 5.509 | the weights | −0.34 | −1.11 | −1.06 | −0.78 | 100% |
| Centred, learned rest at 30× the rate | 5.574 | 5.581 | the learned value | −0.44 | −0.92 | −0.85 | −0.71 | 47%–58% |
| Centred, learned rest at the base rate | 5.635 | | the learned value | −0.05 | −0.09 | −0.08 | −0.08 | 29%–52% |
| Centred, rest 0 | 5.645 | 5.664 | forced | 0.00 | 0.00 | 0.00 | 0.00 | 26%–49% |
| Centred, rest −1 | 5.657 | 5.664 | forced | −1.00 | −1.00 | −1.00 | −1.00 | 38%–56% |

![Loss at step 4,000 for each condition, and the fast control's gap to the baseline over steps](media/loss.svg)

- **A forced rest does not separate the causes.** `rest_-1` is near the baseline's depth and trains as badly as `rest_0`. It is also fixed for every neuron. It starts at −1, where the fresh model's narrow h (SD about 0.31) sits on GELU's flat part. Its loss at step 1,000 is 6.40–6.42, against 6.27–6.28 for `rest_0`, and it has almost caught up by step 4,000. So its cost looks like a slow start, but we did not test this directly.
- **At the base rate, a learned rest stays near 0.** It reaches only −0.09 at L1, so this control mostly repeats `rest_0`. Adam at this rate can move one parameter by about 1.2 in 4,000 steps, and only if every step has the same sign. The baseline's bias moves only about 0.03. The weights can shift the rest faster, because many weights multiply a large mean input.
- **At 30× the rate, a learned rest reaches nearly the same depth by step 4,000.** It reaches −0.92, −0.85 and −0.71 at L1–L3 on seed 0 and −0.92, −0.86 and −0.69 on seed 1. That keeps the baseline's order: L1, then L2, then L3, then L0. The weights no longer turn against x̄. Reaching that depth lowers the loss to 5.574 and 5.581.
- **A gap remains.** The fast control's loss stays 0.065–0.072 above the baseline at step 4,000, and longer training doesn't close it.

## Longer training

Step 4,000 is early, because the rest is still moving. So seed 0 of the baseline and of the fast control ran to step 12,000. ReLU ran to step 12,000 later.

| Step | Baseline loss | Fast control loss | Gap | Baseline rest: L0 | L1 | L2 | L3 | Fast control rest: L0 | L1 | L2 | L3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2,000 | 5.841 | 5.910 | 0.070 | −0.21 | −0.74 | −0.68 | −0.50 | −0.30 | −0.60 | −0.53 | −0.41 |
| 4,000 | 5.510 | 5.574 | 0.064 | −0.34 | −1.10 | −1.06 | −0.78 | −0.44 | −0.92 | −0.85 | −0.71 |
| 6,000 | 5.348 | 5.418 | 0.070 | −0.42 | −1.28 | −1.24 | −0.95 | −0.51 | −1.08 | −1.02 | −0.86 |
| 8,000 | 5.241 | 5.311 | 0.070 | −0.47 | −1.35 | −1.37 | −1.04 | −0.55 | −1.19 | −1.12 | −0.96 |
| 10,000 | 5.163 | 5.231 | 0.068 | −0.50 | −1.43 | −1.44 | −1.13 | −0.58 | −1.26 | −1.20 | −1.02 |
| 12,000 | 5.137 | 5.213 | 0.076 | −0.54 | −1.46 | −1.49 | −1.15 | −0.60 | −1.32 | −1.23 | −1.07 |

![Rest over 12,000 steps on layers L0 to L3, for the baseline and the fast control](media/long-run.svg)

![Surfaces of rest over layers and steps, for the baseline and the fast control](media/surface.svg)

- **The rest doesn't settle at −1.** The baseline's L1 passes −1 by step 4,000 and reaches −1.46. Growth slows, with only 0.03 added over the last 2,000 steps at L1, so it may settle later. We can't say where.
- **ReLU keeps pace with GELU.** Its L1 is at −0.92, −1.12, −1.27, −1.35, −1.40 and −1.49 at the six steps of the table. Its loss is within 0.04 of the baseline's at each. Its L3 goes deeper, to −1.43 against −1.15.
- **The two routes keep pace.** At every step the learned value is at 81–90% of the baseline's depth at L1. The learned value has a much higher limit on its rate, so a limit on one parameter's rate doesn't explain the pace of this control.
- **The loss gap doesn't close.** It stays between 0.064 and 0.076 at every step in the table. So the gap isn't only unfinished training. Centring itself, or the difference in depth, may cost this much. So the runs don't measure the rest's full value.

**One run was discarded.** The first 12,000-step baseline gave a loss of 5.908 at step 4,000 and a much wider h. The earlier run with the same seed gave 5.509. It overlapped other jobs on the machine, and it showed no error. A rerun with nothing else running gave 5.510. The fast control ran after that overlap, and it matched its earlier values at steps 2,000 and 4,000. We did not find the cause, so we kept the discarded rows in the data as `training_long_discarded`.

## Why this matters for reading negative poles

Most of the cost of the negative side in GPT-2 is its resting level ([What default-on negative poles do as a population](../default-on-poles/README.md)). Clamping it at its mean adds 0.19 nats, and zeroing it adds 3.72. These runs fit that. Training pushes the level down, while variation below the dip has less effect. This supports the protocol's stop on auditing a default-on pole, but it doesn't show that a pole has no content.

## Caveats

- **These are small models.** They have 4 layers and short training. Their pattern matches GPT-2 on offset, bias and weights, but not everywhere. GPT-2's L1 rests shallow (−0.24) and its L0 deep (−1.16), and small models show the opposite order. Results on small models suggest causes for GPT-2, and don't prove them.
- **The optimizer shapes the depth.** The first version of this analysis used only Adam at one rate, and warned that GELU and ReLU might agree because of it. *Optimizer and learning rate* shows this in part. We tried one SGD rate, with momentum, and three AdamW rates. Other schedules, warm-up, and Adam's β values were not tried.
- **Runs can go wrong with no error.** See *One run was discarded*. Same-seed reruns match to about 0.001, so a mismatch is easy to see, but most conditions here were not rerun. The leaky ReLU run took 42 minutes against about 25 for the others, while a browser process used much of the CPU. A rerun of its first 500 steps matched to 0.0005 in loss. At step 4,000, ReLU's 12,000-step run differs from the earlier ReLU run by 0.006 in loss and at most 0.01 in rest. That is more than the usual 0.001.
- **The push runs found two more failures.** The seed-1 run of AdamW at 1e-4 moved L1's rest to +3.92 by step 500. It ended at a loss of 7.37, against 5.877 for seed 0, whose L1 rests at −0.37. The next run, ReLU at the same rate, stopped with a NaN loss at step 112. We did not find the cause. We kept the first as `training_discarded` and `push_discarded`, and reran both. The seed-0 reruns that recorded the push matched the earlier rows exactly for GELU, softplus, SiLU and tanh. ReLU's differed by 0.005 in loss, and leaky ReLU at slope 0.1 by 0.039 in loss and 0.05 in rest. The push table uses the reruns, labelled `training_push_rerun`, and the other tables keep the earlier rows.
- **Few seeds.** The baseline, ReLU, the fixed and fast centred conditions, softplus, both leaky ReLUs, SiLU, SGD and each AdamW rate have two seeds. The rest have one. The push has two seeds for softplus, both leaky ReLUs and SiLU, and one for GELU, ReLU and tanh.
- **Two leaky slopes.** At slope 0.01, the usual default, leaky ReLU rests like ReLU. At 0.1 it rests shallower. We didn't try slopes between them.
- **Centring is a blunt tool.** It takes its mean from the batch, also at evaluation, and the gradient passes through that mean.
- **"Seeks" means the gradient pushes that way.** It doesn't show a designed function.

## Reproduce

Each command appends to its `--out`. The runs took about 25–40 minutes each on an Apple GPU. Run nothing else on the machine at the same time. Parallel runs on MPS were slower and failed once, and one run that overlapped other jobs went wrong with no error.

```bash
OUT=papers/why-neurons-rest/results.jsonl
uv run python scripts/train_resting.py gpt2 --out $OUT
uv run python scripts/train_resting.py train --seed 0 --steps 4000 --eval-every 500 --cache tokens.pt --out $OUT
uv run python scripts/train_resting.py train --seed 1 --steps 4000 --eval-every 500 --cache tokens.pt \
    --condition gelu --condition rest_0 --condition rest_learned_fast --condition relu \
    --condition rest_-1 --out $OUT
uv run python scripts/train_resting.py train --seed 0 --steps 12000 --eval-every 2000 --cache tokens.pt \
    --condition gelu --condition rest_learned_fast --out $OUT
uv run python scripts/train_resting.py train --seed 0 --steps 4000 --eval-every 500 --cache tokens.pt \
    --condition sgd --condition relu_sgd --condition softplus --condition leaky_relu --condition silu \
    --condition lr_1e-4 --condition lr_1e-3 --condition relu_lr_1e-4 --condition relu_lr_1e-3 --out $OUT
uv run python scripts/train_resting.py train --seed 0 --steps 12000 --eval-every 2000 --cache tokens.pt \
    --part training_long --condition relu --out $OUT
PUSH="--steps 4000 --eval-every 500 --push-every 100 --cache tokens.pt --out $OUT"
uv run python scripts/train_resting.py train --seed 0 --condition leaky_relu_0.01 $PUSH
uv run python scripts/train_resting.py train --seed 0 --part training_push_rerun --condition gelu \
    --condition relu --condition softplus --condition leaky_relu --condition silu --condition tanh $PUSH
uv run python scripts/train_resting.py train --seed 1 --condition softplus --condition leaky_relu \
    --condition silu --condition leaky_relu_0.01 --condition sgd --condition relu_sgd \
    --condition lr_1e-4 --condition lr_1e-3 --condition relu_lr_1e-4 --condition relu_lr_1e-3 $PUSH
uv run python scripts/paper_figures.py
```

The 12,000-step runs of GELU and the fast control were labelled `training_long` by hand. The `--part` option does this now.

## Next

[Below the Dip](../below-the-dip/README.md) lets you rotate these scenes in 3D. Open questions are in `TODO.md` (*Protocol and science*). We measured the push in item 10 where each run's neurons rest, so it can't tell cause from effect. The next test holds every activation at the same rest, with h centred and moved to 0 as in `rest_0`. The gradient on that rest is then the push at an equal state, all through training. If its size still gives the order of depth, the activation sets the push.

## References

- Dugas, C., Bengio, Y., Bélisle, F., Nadeau, C. & Garcia, R. (2001). *Incorporating Second-Order Functional Knowledge for Better Option Pricing*. Advances in Neural Information Processing Systems 13 (NIPS 2000), 472–478. [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2000/hash/44968aece94f667e4095002d140b5896-Abstract.html).
- Elfwing, S., Uchibe, E. & Doya, K. (2017). *Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning*. [arXiv:1702.03118](https://arxiv.org/abs/1702.03118).
- Elhage, N., et al. (2022). *Toy Models of Superposition*. Transformer Circuits Thread. [arXiv:2209.10652](https://arxiv.org/abs/2209.10652).
- Gao, L., et al. (2020). *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*. [arXiv:2101.00027](https://arxiv.org/abs/2101.00027).
- Hendrycks, D. & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs)*. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415).
- Loshchilov, I. & Hutter, F. (2019). *Decoupled Weight Decay Regularization*. ICLR 2019. [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D. & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners*. [OpenAI](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

### Data

- [`NeelNanda/pile-10k`](https://huggingface.co/datasets/NeelNanda/pile-10k), revision `127bfedc`: the first 10,000 documents of the Pile. Its card gives the licence as `bigscience-bloom-rail-1.0`.

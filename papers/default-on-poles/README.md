# What default-on negative poles do as a population

Protocol 0.8 stops the audit of a default-on pole, because no hard negative can stay silent ([position zero](../../docs/analyses/position-zero.md)). This analysis asks what those poles do together. It starts from an earlier study of GPT-2 L9/N1046−, whose scripts were not kept. Its numbers survive in [an earlier `TODO.md`](https://github.com/8W9aG/openinterpretability/blob/7b735684/TODO.md). It rebuilds them, measures every layer, and adds the nulls the first study lacked.

![Twelve ridges of h by layer, at random init and after training](media/ridges.svg)

**Results.**

1. **Resting in the dip is learned, and it comes from the input weights.** At L9 the median neuron rests at h = −1.04, against 0.00 at random init. The bias contributes only −0.08. Why training does this stays open. Lower gain on noise is true, but it is not selective: on L0–L9 the signal's gain falls by more.
2. **The cost of the negative side is mostly its resting level.** With all 12 layers clamped, holding the negative side at its mean adds 0.19 nats. Setting it to zero adds 3.72.
3. **The tests give no support to "the same signal, scaled down".** The KL ratio follows the variance ratio (r = 0.95), but so does a shuffled change with no link to the token (r = 0.97). Any change of that size would produce this pattern.
4. **The tokens below the dip look coherent, and so do a random direction's.** N1046's split-half reliability by token is 0.70. Random directions in the MLP's input, at the same share of tokens, have a median of 0.68. One of them has code syntax among its top tokens, as N1046 does.
5. **Random features match the negative side on token identity and human labels, and it leads them only a little on the model's next-token distribution.** That lead is at most 0.025 of R², and the positive side's reaches 0.073. The learned weights could explain the lead, because both sides read the same h. So it doesn't show that the negative side was learned for this.
6. **Qwen differs.** Its two sides are symmetric, and each costs 2.46 nats when clamped.

These results do not show that negative poles are meaningless. A single default-on pole can't be audited as a detector. These tests also find little learned signal in the negative side's variation, but other tests could find more.

## Method

The scripts are `scripts/default_on_poles.py`, `scripts/default_on_clamps.py`, `scripts/default_on_decode.py` and `scripts/default_on_targets.py`. Their output is in `results.jsonl`. `scripts/paper_figures.py` draws the figures from that output, with no model. The intervention figure is an illustration, and it uses no data. The ridges are a sketch from each layer's median and SD.

- **Data.** `corpus.jsonl`, the 200,000-token sample with seed 0 from the first 10,000 documents of the Pile (Gao et al., 2020). It is cut into whole 128-token windows that don't overlap. GPT-2 small (Radford et al., 2019) gets 1,355 windows from 384 documents. Qwen2.5-0.5B, from the Qwen2.5 family (Yang et al., 2024), gets 1,328 from 420.
- **Split.** Constants are fit on 300 windows and tested on 400 others, with no document in both. Windows of one document share a topic, so a split by window would leak.
- **Sites.** `c_fc`'s output is the pre-activation h. The input of `c_proj` is the activation a = GELU(h).
- **First tokens.** Band statistics skip each window's first token, and interventions leave it unchanged. The `_with_first` conditions change it too (see *The first token*).
- **Sides.** A neuron's negative side is its tokens with a < 0, and its positive side those with a > 0.
- **Below the dip.** h is under the minimum of GELU (Hendrycks & Gimpel, 2016), at −0.7518, and the pole's strength is under 0.25 × its maximum. GELU is back near zero there, so the pole is silent.
- **Intervals.** They are 95% bootstrap intervals over documents.

Each clamp condition changes the activations as follows. KL is KL(clean‖changed) over next-token distributions.

| Condition | Change |
| --- | --- |
| `clamp` | Each neuron holds its side's mean on that side's tokens |
| `zero` | Each neuron reads zero on that side's tokens |
| `mean` | Each neuron holds its mean on all tokens |
| `null` | Each neuron gets the clamp's change, shuffled over the batch's tokens. Its size per neuron stays, and its link to the token goes |
| `frozen` | A clamp at one layer, with later MLP writes kept at their clean values |
| `direct` | A clamp's change to one layer's write, added at the final norm only |

![An illustration of the clean run, clamp, zero, mean and shuffled null on one neuron's activation](media/clamps.svg)

## Resting in the dip is learned

![GELU over a sketch of a typical L9 neuron's h before and after training, and median h by layer](media/rest.svg)

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Median h, trained | −1.16 | −0.24 | −1.15 | −1.25 | −0.99 | −1.11 | −1.05 | −1.01 | −1.03 | −1.04 | −0.99 | −0.86 |
| … from the bias | −0.07 | −0.05 | −0.09 | −0.09 | −0.08 | −0.07 | −0.08 | −0.08 | −0.08 | −0.08 | −0.08 | −0.07 |
| … from the input term | −1.08 | −0.20 | −1.04 | −1.15 | −0.92 | −1.03 | −0.97 | −0.92 | −0.94 | −0.96 | −0.92 | −0.81 |
| Default-on %, trained | 86 | 58 | 75 | 86 | 90 | 90 | 94 | 96 | 97 | 97 | 96 | 92 |
| Default-on %, random init | 23 | 35 | 37 | 38 | 38 | 39 | 39 | 40 | 40 | 40 | 39 | 40 |
| Mean \|GELU slope\|, trained | 0.21 | 0.31 | 0.23 | 0.19 | 0.21 | 0.20 | 0.21 | 0.21 | 0.20 | 0.20 | 0.21 | 0.24 |
| Mean \|GELU slope\|, random init | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 | 0.52 | 0.51 |

The random-init rows are means over 5 seeds of the same architecture. Their median h is 0.00 on every layer, to within 0.01. The two terms of the median are medians of different quantities, so they don't sum exactly.

- **Most neurons rest left of the dip, not in it.** On every layer but L1, the typical median is below −0.75. It is −1.04 at L9. The signed mean slope at L9 is 0.09, and \|slope\| is 0.20.
- **The input weights make the rest.** A bias could set the rest, but its median is at most −0.09 in size on any layer. So the rest comes from how the weights project the inputs the MLP reads.
- **L1 is the exception.** Its median is −0.24, and 58% of its neurons are default-on. We did not investigate why.

**Why does training do this? This stays open.** We tested one idea, that a low slope at rest lowers the gain on input noise. We compared each MLP with a copy that has each neuron's median h moved to zero by its bias. The noise has 0.1 × each input dimension's SD. It is isotropic, or it points towards another token's input ("data-shaped").

| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Noise gain falls by (×) | 5.5 | 8.2 | 22.0 | 11.3 | 6.4 | 6.2 | 4.9 | 4.6 | 4.5 | 4.3 | 3.8 | 3.3 |
| Signal gain falls by (×) | 7.3 | 15.7 | 23.5 | 19.9 | 12.9 | 11.8 | 9.3 | 7.1 | 6.6 | 5.7 | 3.5 | 2.4 |
| Signal/noise gain, trained | 1.31 | 0.23 | 0.29 | 0.36 | 0.35 | 0.39 | 0.43 | 0.50 | 0.55 | 0.64 | 0.94 | 2.23 |
| Signal/noise gain, recentred | 1.73 | 0.44 | 0.30 | 0.64 | 0.71 | 0.73 | 0.82 | 0.79 | 0.80 | 0.84 | 0.86 | 1.62 |
| Signal/data-shaped gain, trained | 0.35 | 0.15 | 0.09 | 0.09 | 0.09 | 0.11 | 0.14 | 0.17 | 0.21 | 0.25 | 0.28 | 0.28 |
| Signal/data-shaped gain, recentred | 0.80 | 0.59 | 0.30 | 0.26 | 0.30 | 0.35 | 0.44 | 0.44 | 0.50 | 0.54 | 0.40 | 0.34 |

The first two rows and the first ratio use isotropic noise. The gain on noise does fall, by 3–22×. But the signal's gain falls by more on L0–L9, so as trained the ratio is lower there. Only L10 and L11 are higher, and only for isotropic noise. For data-shaped noise the trained ratio is lower on all 12 layers. So "lower gain on noise" is true as a fact about size, but this test doesn't support it as a reason.

This test has limits. It uses small noise on fixed weights, and training would not reach the recentred copy. It can't show what training selects for. A causal test would train small models and vary the dropout, the weight decay or the activation function.

## The cost of each side

All 12 layers changed at once, on GPT-2's test windows. Clean cross-entropy is 3.63 nats.

| Condition | Added nats | KL | Top prediction unchanged |
| --- | --- | --- | --- |
| Negative side, shuffled null | 0.08 | 0.09 | 82% |
| Negative side, clamped | 0.19 | 0.18 | 77% |
| Positive side, shuffled null | 0.82 | 0.88 | 53% |
| Positive side, clamped | 1.34 | 1.46 | 40% |
| Negative side, zero | 3.72 | 3.90 | 13% |
| Every neuron at its mean | 4.09 | 4.32 | 13% |
| Positive side, zero | 8.05 | 8.31 | 2% |

![Added cross-entropy of each condition, for GPT-2 small and Qwen2.5-0.5B](media/cost.svg)

- **The resting level costs far more than the variation.** The negative side's zero costs about 20× its clamp. The clamp keeps the side's mean and removes its variation. The zero also removes the mean. So most of the 3.72 nats come from removing a nearly constant offset to the write. The negative side doesn't carry 3.7 nats of varying signal, as the first study's numbers could suggest.
- **The variation does something.** The clamp's KL is 1.9× its null's, at the same size of change. So the variation's link to the token matters, but its total cost is small.
- **Single poles do almost nothing.** Zeroing L9/N1046− gives KL 1.3×10⁻⁵, and 10 default-on controls give 1.2–2.5×10⁻⁵. So N1046 is ordinary.

### The first token

If interventions also change each window's first token, the positive clamp costs 6.14 nats, not 1.34, and only 13% of top predictions stay. Holding every neuron at its mean costs 5.63, not 4.09. The negative side barely changes (0.19 and 3.75). In our data the first token has very large positive activations. Sun et al. (2024) report massive activations on GPT-2's starting token. Xiao et al. (2023) name the strong attention that initial tokens draw an attention sink. So a clamp there can reach every later token. Any population clamp needs to say what it does at this position.

## The KL ratio follows the variance, as does a null's

These runs clamp one layer at a time. The ratios are negative over positive. The compensation columns are for each side's clamp.

| Layer | KL ratio (95%) | Activation variance ratio | Null's KL ratio | Clamp / null, − | Clamp / null, + | Frozen / clamp, − | Frozen / clamp, + | Direct / clamp, − | Direct / clamp, + |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0 | 0.06 (0.06–0.07) | 0.14 | 0.09 | 1.71 | 2.41 | 2.65 | 1.48 | 0.10 | 0.06 |
| L1 | 0.24 (0.22–0.25) | 0.21 | 0.20 | 1.25 | 1.05 | 1.62 | 1.57 | 0.07 | 0.07 |
| L2 | 0.27 (0.25–0.28) | 0.29 | 0.23 | 1.30 | 1.14 | 1.40 | 1.00 | 0.07 | 0.07 |
| L3 | 0.26 (0.25–0.27) | 0.31 | 0.29 | 1.30 | 1.47 | 1.57 | 1.06 | 0.09 | 0.07 |
| L4 | 0.27 (0.25–0.28) | 0.29 | 0.27 | 1.42 | 1.46 | 1.38 | 0.93 | 0.09 | 0.07 |
| L5 | 0.23 (0.22–0.24) | 0.24 | 0.24 | 1.38 | 1.43 | 1.02 | 0.76 | 0.12 | 0.11 |
| L6 | 0.21 (0.20–0.22) | 0.21 | 0.20 | 1.40 | 1.34 | 0.78 | 0.69 | 0.15 | 0.15 |
| L7 | 0.15 (0.15–0.16) | 0.17 | 0.18 | 1.23 | 1.44 | 0.81 | 0.69 | 0.24 | 0.24 |
| L8 | 0.13 (0.13–0.14) | 0.16 | 0.16 | 1.27 | 1.52 | 0.89 | 0.79 | 0.35 | 0.37 |
| L9 | 0.11 (0.11–0.12) | 0.13 | 0.13 | 1.19 | 1.36 | 0.95 | 1.00 | 0.51 | 0.71 |
| L10 | 0.06 (0.06–0.07) | 0.09 | 0.07 | 1.15 | 1.29 | 0.94 | 1.04 | 0.74 | 0.89 |
| L11 | 0.05 (0.05–0.05) | 0.04 | 0.02 | 1.23 | 0.59 | 1.00 | 1.00 | 1.00 | 1.00 |

![KL ratio of the clamp and of the shuffled null against the activation variance ratio, by layer](media/kl-ratio.svg)

- **The match is real, but it says little.** Over the 12 layers the KL ratio correlates with the variance ratio at r = 0.95, as the first study saw. L0 is the clear miss (0.06 against 0.14). But the shuffled null's ratio correlates with the variance ratio at r = 0.97, and it has no link to the token. So KL mostly tracks the size of the change. Twelve layers are few, and these correlations are not independent evidence of much.
- **The sides are not shown to be one signal.** Centred on the token mean, the two sides' writes have a mean cosine of −0.19 to 0.28 by layer. A ridge readout predicts the positive write from the negative side with R² of 0.28–0.82, but random features get 0.23–0.78 (see *Decoding*). So most of that overlap comes from the shared input.
- **Later layers compensate early and amplify in the middle.** Freezing later MLPs raises the negative clamp's KL at L0–L4 (1.38–2.65×), so those later MLPs undo part of the damage. At L6–L8 it lowers the KL (0.78–0.89×), so later MLPs add to the damage. The positive side has the same shape, with less of this. The ratio survives: with later MLPs frozen, or with the direct path alone, it still correlates with the variance ratio (r = 0.98 and 0.97).
- **Most of an early clamp's effect is indirect.** The direct path to the final norm is 6–10% of the KL at L0–L4 and all of it at L11, where nothing follows.
- **Clamps add up to more than their parts.** The sum of the single-layer KLs is 0.11 for the negative side and 0.79 for the positive. All layers together give 0.18 and 1.46.

## The tokens below the dip

Measured at L9 on all 1,355 windows. Split-half reliability correlates a neuron's below-dip rate by token between even and odd documents, over tokens with at least 10 occurrences in each half.

| Unit | Share of tokens | Reliability | Top tokens, as we read them |
| --- | --- | --- | --- |
| **N1046** | 2.4% | 0.70 | code syntax (`^{`, ` :=`, `{`, ` !=`), and some names (` Henry`, ` Burton`) |
| N1132 | 2.0% | 0.70 | abstract nouns and contractions (` statement`, `'d`, ` effect`) |
| N1742 | 3.3% | 0.56 | French pieces (` à`, `ée`, ` aux`), and a few others |
| N2084 | 4.0% | 0.73 | a mix: titles, code words, word pieces |
| N1622 | 6.2% | 0.60 | a mix: names, years, capitals |
| N1965 | 10.5% | 0.80 | a mix: word starts, months, Romance articles |
| N918 | 11.2% | 0.83 | discourse words (` however`, ` particularly`, ` So`), and word pieces |
| N2455 | 12.2% | 0.76 | code and markup (`div`, `height`, `span`), and contractions |
| N2158 | 21.3% | 0.81 | code punctuation (`");`, `};`, ` />`), and identifiers |
| N2778 | 24.4% | 0.82 | proper nouns and web words (` University`, `https`, ` Inc`) |
| N2312 | 29.1% | 0.82 | pronouns and sentence openers (`He`, ` She`, `This`) |
| Random directions at N1046's share | 2.4% | 0.68 (5th–95th percentile 0.52–0.81) | see below |

![Split-half reliability of real neurons, random directions, the 10 controls and N1046](media/tails.svg)

The 10 controls are drawn at random from the layer's default-on neurons. Over all 2,972, the median reliability is 0.77 (5th–95th percentile 0.57–0.87). N1046 is at the 22nd percentile. Random directions with the same distribution of shares have a median of 0.77 (0.54–0.84).

**Five random directions at N1046's share, read the same way:**

| Direction | First 12 tokens, unfiltered |
| --- | --- |
| 0 | (space), ` =`, `":`, ` m`, ` ever`, `',`, ` /**`, ` same`, ` !=`, ` p`, ` n`, `",` |
| 1 | ` Chief`, ` football`, ` chief`, `public`, ` Red`, ` leader`, ` nucle`, ` AC`, `ri`, ` confidence`, `lu`, ` telephone` |
| 2 | `Let`, `�`, `pp`, `There`, ` Rebecca`, ` has`, `ni`, ` SP`, ` He`, ` Court`, `PP`, `He` |
| 3 | `height`, ` starting`, `inal`, ` Heart`, ` focus`, `Start`, ` evaluate`, ` te`, ` :=`, `ial`, ` trying`, ` complete` |
| 4 | ` som`, `lic`, `usr`, ` method`, ` easy`, ` library`, ` methods`, `�`, ` apparatus`, ` Nar`, ` License`, ` invention` |

**A random direction's tail can suggest a theme.** Direction 0 has code syntax among its top tokens, and direction 2 has sentence openers and pronouns. Those are the first study's two readings of N1046, and N1046's own list is also a mix. Several lists have no clear theme, as do some of the controls. We read these lists knowing which were random, so the readings are not blind. The reliability numbers don't depend on our reading.

Token identity dominates the MLP's input, so a low tail of a random linear readout is about as reliable by token as a real neuron's. **So a reliable tail is no evidence that a neuron means something, and a readable one is weak evidence.** The `interpretability-illusions-review` skill now names this case.

We could not check the first study's phrase "pronouns after 'what'", because this method reads single tokens.

**Surprisal doesn't explain the tails.** The Spearman ρ between h and the token's surprisal given its prefix is 0.05 for N1046, and −0.14 to 0.11 for the controls.

## Decoding

Can the collective signal be decoded into anything meaningful? A ridge readout fits on 300 windows and tests on 400. It reads one side of a layer's activations and predicts the 200 commonest current or next tokens (chance is 12.5%). The null reads the negative side of random features of the same input, each with a real neuron's median and SD of h.

| Layer | Current token: − | + | null | Next token: − | + | both | null | Positive write R²: from − | from null |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0 | 0.999 | 0.996 | 1.000 | 0.365 | 0.361 | 0.361 | 0.358 | 0.67 | 0.63 |
| L1 | 0.998 | 0.993 | 0.999 | 0.373 | 0.373 | 0.374 | 0.367 | 0.50 | 0.43 |
| L2 | 0.990 | 0.984 | 0.999 | 0.385 | 0.386 | 0.387 | 0.379 | 0.40 | 0.33 |
| L3 | 0.984 | 0.963 | 0.991 | 0.392 | 0.392 | 0.396 | 0.386 | 0.36 | 0.30 |
| L4 | 0.968 | 0.944 | 0.984 | 0.398 | 0.393 | 0.400 | 0.397 | 0.31 | 0.25 |
| L5 | 0.954 | 0.932 | 0.973 | 0.414 | 0.413 | 0.420 | 0.411 | 0.30 | 0.24 |
| L6 | 0.944 | 0.936 | 0.963 | 0.421 | 0.424 | 0.429 | 0.419 | 0.30 | 0.24 |
| L7 | 0.936 | 0.918 | 0.951 | 0.435 | 0.450 | 0.459 | 0.440 | 0.29 | 0.25 |
| L8 | 0.919 | 0.905 | 0.927 | 0.450 | 0.481 | 0.485 | 0.461 | 0.28 | 0.23 |
| L9 | 0.880 | 0.872 | 0.887 | 0.463 | 0.513 | 0.515 | 0.479 | 0.33 | 0.28 |
| L10 | 0.831 | 0.825 | 0.837 | 0.470 | 0.540 | 0.543 | 0.497 | 0.50 | 0.46 |
| L11 | 0.779 | 0.778 | 0.788 | 0.487 | 0.544 | 0.546 | 0.506 | 0.82 | 0.78 |

- **The negative side decodes well, and that alone means little.** It gets 78–99.9% on the current token, but random features get as much or more on all 12 layers. On the next token, the negative side leads the null by at most 0.008 on L0–L6, and the null leads on L7–L11.
- **The positive side does beat the null late.** On the next token it gets 0.544 at L11 against the null's 0.506, and adding the negative side adds almost nothing (0.546). So late positive poles hold something that the input's random features don't give.
- **Reading is not use.** A readout shows what is present, not what later layers read. The clamps above are the test of use, and they show a small effect.

So this test finds no learned content in the negative side. It tests only token identity, for only the 200 commonest tokens. The next section tests other targets.

## Other decoding targets

Token identity is the easiest thing to read from the MLP's input, so it could hide content that needs a learned direction. `scripts/default_on_targets.py` keeps the same ridge readout and null, with three draws of the null, and tests two other targets. Each has a baseline that reads only the current token.

**The model's next-token distribution.** The target is the model's log-probability of the 200 commonest tokens, with the same fit and test windows. The score is R² on the test windows.

| Layer | Lookup | − | + | both | null (3 draws) | − minus best null | + minus best null | Top agreement: − | + | best null |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L0 | 0.278 | 0.418 | 0.446 | 0.459 | 0.410–0.417 | +0.002 | +0.030 | 0.331 | 0.323 | 0.303 |
| L1 | 0.278 | 0.489 | 0.491 | 0.520 | 0.465–0.466 | +0.022 | +0.024 | 0.358 | 0.335 | 0.346 |
| L2 | 0.278 | 0.522 | 0.521 | 0.540 | 0.493–0.500 | +0.021 | +0.021 | 0.368 | 0.377 | 0.361 |
| L3 | 0.278 | 0.536 | 0.547 | 0.565 | 0.505–0.514 | +0.022 | +0.033 | 0.389 | 0.387 | 0.376 |
| L4 | 0.278 | 0.567 | 0.555 | 0.582 | 0.539–0.546 | +0.021 | +0.009 | 0.398 | 0.391 | 0.387 |
| L5 | 0.278 | 0.625 | 0.628 | 0.650 | 0.598–0.602 | +0.023 | +0.027 | 0.418 | 0.420 | 0.408 |
| L6 | 0.278 | 0.655 | 0.674 | 0.690 | 0.623–0.632 | +0.023 | +0.043 | 0.436 | 0.449 | 0.427 |
| L7 | 0.278 | 0.688 | 0.720 | 0.733 | 0.661–0.668 | +0.021 | +0.053 | 0.456 | 0.480 | 0.457 |
| L8 | 0.278 | 0.710 | 0.747 | 0.760 | 0.692–0.697 | +0.013 | +0.049 | 0.481 | 0.541 | 0.493 |
| L9 | 0.278 | 0.734 | 0.775 | 0.787 | 0.706–0.721 | +0.013 | +0.053 | 0.514 | 0.601 | 0.536 |
| L10 | 0.278 | 0.767 | 0.792 | 0.809 | 0.739–0.742 | +0.025 | +0.050 | 0.527 | 0.671 | 0.580 |
| L11 | 0.278 | 0.806 | 0.871 | 0.879 | 0.792–0.798 | +0.008 | +0.073 | 0.545 | 0.717 | 0.634 |

- **The negative side leads the null a little.** The lead over the best of three draws is 0.008–0.025 on L1–L11, and 0.002 on L0. Its 95% interval by document, against the first draw, is above zero on L1–L11. The draws themselves differ by up to 0.015, so the smaller leads are close to that spread.
- **The positive side leads by more late.** Its lead grows to 0.073 at L11, where the negative side's is 0.008. Reading both sides adds 0.008–0.029 to the better single side.
- **The top prediction goes the other way late.** The negative side matches the model's top token more often than the null on L0–L6, and less often on L7–L11. At L11 it is 0.545, against 0.634 for the null and 0.717 for the positive side.
- **Entropy agrees with R².** The negative side leads the null on the distribution's entropy by 0.001 at L0, and by 0.009–0.035 on L1–L11.

**A lead doesn't show that training built the negative side for this.** Both sides read the same h. Weights learned for the positive side would also help the negative side beat random weights. This test can't separate the two, and a readout shows only what is present. The clamps above, which add 0.19 nats, are the test of use.

**Human labels.** The labels come from `openinterp/concepts.py`: Universal Dependencies (Nivre et al., 2020), CoNLL-2003 (Tjong Kim Sang & De Meulder, 2003), AG News (Zhang et al., 2015) and `papluca/language-identification`. There is one row per token, on layers 0, 3, 6, 9 and 11. Documents are split between fit and test, and each is cut to one window. The score is the mean AUROC over labels on the ambiguous tokens. These are the test tokens whose type both has the label and lacks it in the fit documents. The lookup baseline reads the current token's rate of the label.

| Dataset | Labels | Lookup | − at L6 | + at L6 | Best null at L6 | − minus best null: L0 | L3 | L6 | L9 | L11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ud-upos | 14 | 0.872 | 0.965 | 0.964 | 0.962 | +0.002 | +0.004 | +0.003 | +0.001 | −0.001 |
| ud-deprel | 20 | 0.864 | 0.951 | 0.949 | 0.947 | +0.004 | +0.006 | +0.004 | −0.003 | −0.002 |
| conll2003-ner | 4 | 0.884 | 0.967 | 0.963 | 0.965 | +0.005 | +0.010 | +0.002 | +0.001 | −0.002 |
| ag_news | 4 | 0.619 | 0.900 | 0.892 | 0.901 | +0.003 | +0.003 | −0.001 | +0.008 | +0.011 |
| language-identification | 20 | 0.805 | 0.977 | 0.967 | 0.975 | 0.000 | +0.001 | +0.002 | 0.000 | −0.001 |

- **All sides read context, and the null reads it too.** Each side is far above the lookup, so these labels hold more than token identity. But the negative side's lead over the best null is at most 0.011 in all 25 cells, and it is above zero in 17. The interval against the first draw is above zero in 13, with no correction for multiple tests.
- **Most AUROCs are near their ceiling.** That leaves little room for a lead, so a small lead here is weak evidence either way. Only `ag_news` has much room, and its largest lead is 0.011.

![Each side's lead over the random-feature null on the next-token distribution and on human labels](media/decoding.svg)

Limits of these two tests: they use linear readouts and one sample, and the labelled text is short and differs from `corpus.jsonl`. Fit sets are smaller than the features would like on the Universal Dependencies data (16,476 tokens, 3,072 features). Other readouts or targets could still show more. The CoNLL and AG News licences limit how we can use their text, and the language dataset's card states no licence. So we publish only these aggregate scores.

## Qwen2.5-0.5B

Qwen's MLP is SwiGLU (Shazeer, 2020), so a neuron is silu(gate) × up. Clamps ran on the same window design, on every layer together and on layers 0, 5, 11, 17 and 23 alone. We sampled the layers because a full per-layer run would take hours. Clean cross-entropy is 2.75 nats.

| All 24 layers | Added nats | Top prediction unchanged |
| --- | --- | --- |
| Negative side, shuffled null | 1.11 | 50% |
| Positive side, shuffled null | 1.16 | 49% |
| Negative side, clamped | 2.46 | 31% |
| Positive side, clamped | 2.46 | 32% |
| Negative side, zero | 5.44 | 7% |
| Positive side, zero | 8.54 | 5% |
| Every neuron at its mean | 8.89 | 3% |

| Layer | 0 | 5 | 11 | 17 | 23 |
| --- | --- | --- | --- | --- | --- |
| Share of tokens on the negative side | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |
| KL ratio, − / + | 0.97 | 0.96 | 1.02 | 1.08 | 1.08 |
| 95% interval | 0.90–1.05 | 0.86–1.07 | 1.00–1.06 | 0.94–1.31 | 1.03–1.14 |
| Activation variance ratio | 0.93 | 0.96 | 1.00 | 1.00 | 1.21 |
| Clamp / null, − | 3.7 | 2.6 | 1.5 | 1.9 | 2.0 |
| Frozen / clamp, − | 0.18 | 0.46 | 0.51 | 0.50 | 1.00 |

**The GPT-2 picture does not carry over.**

- The two sides cost the same, where GPT-2's negative clamp costs 0.14× its positive clamp.
- A clamp costs about 2.2× its null over all layers, and up to 3.7× at L0. So the variation's link to the token matters more here.
- Later MLPs add to the damage at each layer tested up to L17: freezing them lowers the negative clamp's KL to 0.18–0.51×. GPT-2's later MLPs do the opposite after its early layers.
- The first token matters on the negative side too: 5.42 nats with it changed, against 2.46.

So in this SwiGLU model the negative side costs as much as the positive side, which fits its negative poles being sparse (`docs/analyses/position-zero.md`). So the low cost in GPT-2 is not a property of negative poles in general. A GELU MLP that rests near its dip is the likely reason, but one model of each kind can't settle that. We did not run the decoding or below-dip tests on Qwen. Its activation is a product, so GELU's dip has no direct counterpart.

## Against the first study

The first study's scripts were not kept, so we can't say what caused each difference.

| Number | First study | Here |
| --- | --- | --- |
| L9 median h, trained / random init | −1.03 / 0.00 | −1.04 / 0.00 |
| L9 default-on, trained / random init | 97% / 40% | 97% / 40% (5 seeds) |
| L9 mean GELU slope, trained / random init | 0.20 / 0.51 | The same, as mean \|slope\|. The signed trained mean is 0.09 |
| Clean cross-entropy | 3.62 | 3.63 |
| Negative clamp: added nats, top unchanged | 0.21, 76% | 0.19, 77% |
| Positive clamp: added nats, top unchanged | 1.33, 41% | 1.34, 40% |
| Negative zero: added nats, top unchanged | 3.68, 13% | 3.72, 13% |
| All at mean: added nats, top unchanged | 4.94, 10% | 4.09, 13%. With first tokens changed: 5.63, 9% |
| Zeroing L9/N1046 alone, KL | 2.6×10⁻⁵ | 1.3×10⁻⁵ |
| KL ratio / variance ratio at L3, L6, L9, L11 | 0.25/0.31, 0.22/0.20, 0.12/0.13, 0.05/0.04 | 0.26/0.31, 0.21/0.21, 0.11/0.13, 0.05/0.04 |
| The same at L0 | 0.07/0.14 | 0.06/0.14 |
| L9/N1046 tokens below the dip | 2.3% | 2.4% |
| L9/N1046 reliability | 0.80 | 0.70 by document, 0.90 by occurrence |
| Reliability of 10 controls | 0.79–0.93 | 0.56–0.83 by document, 0.66–0.92 by occurrence |
| ρ with surprisal | 0.05 | 0.05 |

**Reliability depends on the split.** A split by occurrence puts one document's tokens in both halves, so the shared context raises the number. We report the split by document. Our controls are a different draw from the first study's.

## Caveats

- **One corpus and one sample.** This is 200,000 tokens of `NeelNanda/pile-10k`. The token lists depend on it.
- **Fixed constants at every layer.** All-layer clamps use constants fit on the clean model, although earlier clamps change later layers' inputs.
- **Shuffling is a strong null.** It breaks the correlation between neurons on one token as well as the link to the token. So the clamp over null ratio doesn't separate these two.
- **Controls are few.** The 10 controls show the range. The population figures rest on L9 alone.
- **MPS.** The runs used float32 on Apple's GPU. Activations there match the CPU's to about 1×10⁻⁴ of a neuron's peak (`openinterp/cli.py`).

## Effect on the protocol

No rule changes. This supports `not_default_on`. A default-on pole's reliable tail doesn't separate it from a random direction, and zeroing one has almost no effect. A future population-level unit should face these nulls too.

## Reproduce

The bands command writes the controls in its `band_neuron` rows, and the clamps take them through `--neurons`. Each command appends to its `--out`.

```bash
OUT=papers/default-on-poles/results.jsonl
uv run python scripts/default_on_poles.py rest --out $OUT
uv run python scripts/default_on_poles.py bands --out $OUT
uv run python scripts/default_on_decode.py --out $OUT
uv run python scripts/default_on_targets.py next-token --out $OUT
uv run python scripts/default_on_targets.py concepts --out $OUT
uv run python scripts/default_on_clamps.py --model openai-community/gpt2 \
    --neurons 1046,918,1132,1622,1742,1965,2084,2158,2312,2455,2778 --out $OUT
uv run python scripts/default_on_clamps.py --model Qwen/Qwen2.5-0.5B \
    --layers 0,5,11,17,23 --out $OUT
uv run python scripts/paper_figures.py
```

## Next

[Why training rests neurons below GELU's dip](../why-neurons-rest/README.md) studies why training rests neurons below the dip. [Below the Dip](../below-the-dip/README.md) lets you rotate these scenes in 3D. Open questions are issues in the Linear project *Default-on poles*, and the research questions are in [docs/research-questions.md](../../docs/research-questions.md).

## References

- Gao, L., et al. (2020). *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*. [arXiv:2101.00027](https://arxiv.org/abs/2101.00027).
- Hendrycks, D. & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs)*. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415).
- Nivre, J., de Marneffe, M.-C., Ginter, F., Hajič, J., Manning, C. D., Pyysalo, S., Schuster, S., Tyers, F. & Zeman, D. (2020). *Universal Dependencies v2: An Evergrowing Multilingual Treebank Collection*. LREC 2020, pp. 4034–4043. [ACL Anthology](https://aclanthology.org/2020.lrec-1.497/).
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D. & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners*. [OpenAI](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
- Shazeer, N. (2020). *GLU Variants Improve Transformer*. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202).
- Sun, M., Chen, X., Kolter, J. Z. & Liu, Z. (2024). *Massive Activations in Large Language Models*. COLM 2024. [arXiv:2402.17762](https://arxiv.org/abs/2402.17762).
- Tjong Kim Sang, E. F. & De Meulder, F. (2003). *Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition*. CoNLL 2003, pp. 142–147. [arXiv:cs/0306050](https://arxiv.org/abs/cs/0306050).
- Xiao, G., Tian, Y., Chen, B., Han, S. & Lewis, M. (2023). *Efficient Streaming Language Models with Attention Sinks*. ICLR 2024. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453).
- Yang, A., et al. (2024). *Qwen2.5 Technical Report*. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115).
- Zhang, X., Zhao, J. & LeCun, Y. (2015). *Character-level Convolutional Networks for Text Classification*. NeurIPS 2015. [arXiv:1509.01626](https://arxiv.org/abs/1509.01626).

### Data

- [`NeelNanda/pile-10k`](https://huggingface.co/datasets/NeelNanda/pile-10k), revision `127bfedc`: the first 10,000 documents of the Pile. Its card gives the licence as `bigscience-bloom-rail-1.0`.
- [`universal-dependencies/universal_dependencies`](https://huggingface.co/datasets/universal-dependencies/universal_dependencies), revision `c3fed78d`, CC BY-SA 4.0.
- [`eriktks/conll2003`](https://huggingface.co/datasets/eriktks/conll2003), revision `ce85b39f`. Its card gives the licence as `other`.
- [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news), revision `eb185aad`. Its card gives the licence as `unknown`, for research and other non-commercial use.
- [`papluca/language-identification`](https://huggingface.co/datasets/papluca/language-identification), revision `e495f9af`. Its card names its sources, and gives no citation and no licence.

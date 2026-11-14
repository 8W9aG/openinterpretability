# Where SPD can favour GPT-2's neurons

A parameter decomposition keeps whichever basis it starts in on GPT-2 small's down projections, so it says little about neurons there. The input projection is a sharper test, because GELU acts on each neuron separately. We decomposed both MLP projections for 50,000 steps from each of two starts. At about the same total of active subcomponents, the start from the neurons ends with a KL of 0.093 against 0.158. A random start does not turn towards the neurons: its closest-neuron cosine stays at 0.07–0.10, near a random direction's 0.07. So the losses favour the neurons, and training does not find them.

![Input projections move through training from each start. From the neurons they fit far better and keep more subcomponents active](media/training.svg)

**Results.**

1. **The start from the neurons fits better at about equal total sparsity.** Runs D and C keep 5,186 and 5,152 subcomponents active across all 24 sites, 0.7% apart. Run D's KL is 0.093 against 0.158, and its share of the loss gap left is 0.011 against 0.030. This is one seed per start.
2. **A random start does not find the neurons.** Run C's input projections end at a median closest-neuron cosine of 0.07–0.10, and none has a cosine above 0.7. They did move: the cosine with their start fell to 0.43–0.51. So staying random here is not only a failure to move. It does not show that no path to the neurons exists.
3. **Started as neurons, they stay neurons.** Run D's input projections end at 0.89–0.96, with 97.3–99.7% of subcomponents above 0.7.
4. **The two runs spend their budget differently, so no input projection is compared at the same count.** Run D keeps more active on the input projections (2,456 against 1,706) and fewer on the down projections (2,730 against 3,445). Against a shared neuron yardstick, run D is ahead at all 24 sites, by 1.10–5.51 times. At 10 sites run D has both the fewer active and the better fit, and run C has both at none. The yardstick is indirect.
5. **On most input projections, neither run beats the neurons themselves.** At the same count, each token's best neurons fit better than run D in layers 3–11 and better than run C in layers 1–11. On the down projections SPD is usually ahead, against a weaker baseline.
6. **The objective favoured the neurons without the input projections too, by less.** From the neurons it is 27% lower with both projections (4.40 against 6.06), and 13% lower with the down projections alone (2.90 against 3.32). So we can't credit all of the lead to GELU.

These results do not show that the neurons are the best basis for GPT-2, or the global optimum of SPD's objective. They do not show that a neuron is a good unit of explanation. A longer run or a higher learning rate could still narrow the lead from a random start.

## Method

- **Model.** GPT-2 small (Radford et al., 2019), with 12 layers and 3,072 GELU neurons (Hendrycks & Gimpel, 2016) per MLP. Each layer has an input projection `c_fc` and a down projection `c_proj`, so there are 24 sites.
- **Decomposition.** Stochastic Parameter Decomposition, SPD (Bushnaq et al., 2025), after APD (Braun et al., 2025). It splits each matrix into rank-one subcomponents, trained to sum to the matrix. A causal importance g in [0, 1] says how far each one can be ablated on a token. The source tests SPD on toy models. Applying it to GPT-2 is our step.
- **Our changes.** One gate per site reads the centred input. The importance loss adds ε = 1e-3, and gate gradients are clipped to norm 1. Each site also has a loss on its CI-masked output. [The baseline write-up](../../docs/analyses/spd-baseline.md) gives the failed pilots behind each change. Each site has C = 3,072 subcomponents.
- **Input projections.** The output loss and the fit are measured after GELU. A neuron that is off on a token can then be dropped at no cost, and a subcomponent that writes to many neurons cannot. That is the reason to expect a preference. It is our argument, not the source's.
- **Starts.** Run C draws each subcomponent at random, then runs a faithfulness warmup. Run D starts each one as a neuron: `V = W[:, i], U = e_i` on an input projection, and `V = e_i, U = W[i]` on a down projection. Runs A and B are the same pair on the down projections only.
- **Training.** 50,000 steps on a Colab A100, 3.6 hours each, with 8 pieces of 128 tokens per batch. The exponent p falls from 2 to 0.4 by step 5,000, and the learning rate follows a cosine to zero. The seed is 0.
- **Data.** Training uses [`NeelNanda/pile-10k`](https://huggingface.co/datasets/NeelNanda/pile-10k), the first 10,000 documents of the Pile (Gao et al., 2020), without the 518 documents of the scan corpus. That gives 113,546 pieces, with 1,096 held out. Its card gives the licence as `bigscience-bloom-rail-1.0`.
- **Measures.** The per-site tables read 64 held-out pieces (8,128 tokens). The KL, the loss gap and the training figure read the run's own 8 held-out pieces. *Loss gap left* is the share of the loss rise from ablating every decomposed projection that the CI-masked model keeps. *Objective* is the sum of the run's weighted losses, averaged over its last 11 evaluations.
- **Neuron baselines.** On an input projection, each token keeps the k neurons that move furthest from their ablated value (`scripts/spd_up_neurons.py`). For this squared error, that is the best choice of k neurons. On a down projection the ranking is `|h_i − mean_i|·‖W[i]‖`, which ignores cancellation, so that baseline is easier to beat.
- **Yardstick.** *Neurons per active* takes the neurons per token that reach a run's fit, interpolated on the baseline's curve, and divides by the run's active count. Above 1, SPD beats the baseline. It lets two runs at different counts share one scale. It assumes the curve is a fair exchange rate, which we did not test.

## Whole-model fit

![KL over training from each start, with both projections and with the down projections alone. The two starts end level only on the down projections](media/whole-model.svg)

| Run | Sites | Start | Active | KL | Loss gap left | Objective (SD) | Importance term | Output term |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | down | random | 6,807 | 0.063 | 0.012 | 3.32 (0.09) | 2.57 | 0.66 |
| B | down | neurons | 4,538 | 0.064 | 0.010 | 2.90 (0.09) | 1.61 | 1.19 |
| C | both | random | 5,152 | 0.158 | 0.030 | 6.06 (0.20) | 2.27 | 3.51 |
| D | both | neurons | 5,186 | 0.093 | 0.011 | 4.40 (0.12) | 2.20 | 2.03 |

Active counts for A and B come from their last evaluation on 8 pieces. Those for C and D from the 64 pieces of the tables below. On 8 pieces C and D keep 5,119 and 5,159.

With both projections, the importance terms are about level, and 1.48 of the objective's 1.66 gap is the output term. This is the like-for-like reading that the runs allow: the same price for sparsity, and a much better fit from the neurons. On the down projections the pattern differs. Run B's lead there is in sparsity, and its fit is lower.

The start from the neurons does not start ahead. At step 0 its KL is 5.34 against 4.56 from random. It leads by step 1,000 and stays ahead to the end.

## The input projections, site by site

![Neurons per token needed to reach each run's fit, per active subcomponent. Run D is above run C at every site](media/yardstick.svg)

| Layer | C: active | C: EV | D: active | D: EV | Best neurons, C's count | Best neurons, D's count | C: neurons per active | D: neurons per active |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 189 | 0.86 | 125 | 0.87 | 0.81 | 0.75 | 1.60 | 2.54 |
| 1 | 121 | 0.55 | 133 | 0.81 | 0.60 | 0.62 | 0.76 | 2.92 |
| 2 | 93 | 0.43 | 155 | 0.74 | 0.55 | 0.67 | 0.55 | 1.48 |
| 3 | 95 | 0.40 | 179 | 0.71 | 0.59 | 0.74 | 0.48 | 0.91 |
| 4 | 137 | 0.46 | 219 | 0.73 | 0.63 | 0.75 | 0.54 | 0.94 |
| 5 | 125 | 0.44 | 213 | 0.74 | 0.65 | 0.78 | 0.46 | 0.88 |
| 6 | 132 | 0.46 | 222 | 0.77 | 0.66 | 0.79 | 0.48 | 0.92 |
| 7 | 137 | 0.48 | 236 | 0.79 | 0.68 | 0.81 | 0.48 | 0.90 |
| 8 | 133 | 0.46 | 247 | 0.80 | 0.68 | 0.83 | 0.45 | 0.89 |
| 9 | 137 | 0.51 | 240 | 0.81 | 0.73 | 0.85 | 0.45 | 0.83 |
| 10 | 219 | 0.74 | 250 | 0.83 | 0.85 | 0.87 | 0.63 | 0.79 |
| 11 | 187 | 0.77 | 238 | 0.84 | 0.82 | 0.87 | 0.80 | 0.88 |

"Best neurons" is the fit of each token's best k neurons at that run's count. Run D beats its row's baseline only in layers 0–2, and run C only in layer 0. The soft gate gives SPD a lead on the down projections. But on most input projections it does not make up for the extra freedom of choosing neurons directly.

The down projections, for comparison:

| Layer | C: active | C: EV | D: active | D: EV | Neurons, C's count | Neurons, D's count | C: neurons per active | D: neurons per active |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 200 | 0.91 | 107 | 0.92 | 0.77 | 0.68 | 3.14 | 6.50 |
| 1 | 230 | 0.77 | 184 | 0.75 | 0.21 | 0.16 | 4.86 | 5.82 |
| 2 | 274 | 0.77 | 217 | 0.77 | 0.15 | 0.07 | 3.09 | 3.85 |
| 3 | 291 | 0.76 | 221 | 0.78 | 0.31 | 0.14 | 2.20 | 3.09 |
| 4 | 339 | 0.77 | 264 | 0.76 | 0.09 | -0.06 | 2.55 | 3.17 |
| 5 | 343 | 0.76 | 267 | 0.77 | 0.38 | 0.26 | 2.15 | 2.82 |
| 6 | 333 | 0.72 | 278 | 0.76 | 0.46 | 0.39 | 1.96 | 2.65 |
| 7 | 305 | 0.66 | 268 | 0.77 | 0.60 | 0.57 | 1.29 | 2.29 |
| 8 | 348 | 0.66 | 277 | 0.79 | 0.71 | 0.67 | 0.77 | 1.82 |
| 9 | 314 | 0.60 | 260 | 0.81 | 0.75 | 0.73 | 0.32 | 1.77 |
| 10 | 279 | 0.67 | 233 | 0.84 | 0.76 | 0.74 | 0.41 | 2.22 |
| 11 | 189 | 0.83 | 156 | 0.89 | 0.52 | 0.53 | 3.92 | 5.92 |

Run C's fit is higher in layers 1, 2 and 4, where it also keeps more active. Run D has the better fit and the lower count in the other nine layers.

## Alignment

![Closest-neuron cosine per layer. Run D stays near 1 and run C stays near a random direction](media/alignment.svg)

Input projections, reading each subcomponent's output direction `U[c]`:

| Layer | C: cosine | C: with start | C: alive | D: cosine | D: over 0.7 | D: alive |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.10 | 0.47 | 2,985 | 0.96 | 0.989 | 2,732 |
| 1 | 0.08 | 0.47 | 2,853 | 0.89 | 0.973 | 2,751 |
| 2 | 0.09 | 0.50 | 2,788 | 0.92 | 0.981 | 2,645 |
| 3 | 0.07 | 0.50 | 2,607 | 0.94 | 0.988 | 2,488 |
| 4 | 0.07 | 0.49 | 2,428 | 0.93 | 0.979 | 2,560 |
| 5 | 0.07 | 0.51 | 2,232 | 0.94 | 0.988 | 2,553 |
| 6 | 0.07 | 0.50 | 2,264 | 0.94 | 0.982 | 2,674 |
| 7 | 0.08 | 0.48 | 2,464 | 0.94 | 0.995 | 2,814 |
| 8 | 0.07 | 0.49 | 2,400 | 0.95 | 0.994 | 2,861 |
| 9 | 0.08 | 0.48 | 2,566 | 0.96 | 0.996 | 2,886 |
| 10 | 0.08 | 0.43 | 2,954 | 0.96 | 0.997 | 2,946 |
| 11 | 0.08 | 0.47 | 2,935 | 0.96 | 0.996 | 3,013 |

Run D's cosine with its start equals its closest-neuron cosine, so the table gives it once. Run C's input directions moved further from their start (0.43–0.51) than its down directions did (0.55–0.57), and still ended near chance. On this measure a neuron scores 1 and a random direction about 0.07.

## Caveats

- **One seed per start.** The objective's SD is over evaluations of one run, not over seeds. It can't give a confidence interval on the gap between starts.
- **A head start is still possible.** Neurons sum to the matrix exactly at step 0, and a random start must first reach that. Over the last 10,000 steps run C's input fit rose by about 0.008 and run D's by about 0.004. Meanwhile the learning rate fell to zero. The KL gap stopped narrowing near step 30,000, but a falling learning rate could cause that too.
- **No input projection is compared at one count.** The active ranges overlap in only a few layers, and only between a late state of one run and an early state of the other. Thresholding the gates to one count would be a direct test, and it needs the checkpoints.
- **Run D's subcomponents did move.** They end at 0.89–0.96 with their start on the input projections, so run D is near the neuron basis, not exactly on it.
- **Dead subcomponents.** Between 2,232 and 3,013 of 3,072 stay alive on the input projections, depending on the site. The runs differ here too, and we did not test how much of the fit this explains.
- **With C equal to the hidden width, SPD can't split a neuron.** A polysemantic neuron would need more subcomponents than neurons.
- **Our changes matter.** The gate, the centring and the output loss are ours, and the method reached this fit only after them. Another setup could rank the starts differently.

## Reproduce

Open `scripts/train_spd_colab.ipynb` on a Colab A100, with `corpus.jsonl` in `MyDrive/openinterp/`. Run it with `MATRICES = "both"` and `STEPS = 50_000`, once per start, with the default seed of 0. Each run writes `compare.json` and `train.log` to its Drive folder. Copy each run's `compare.json` into `docs/analyses/`, and each `train.log` to `exchange/spd-logs/` as `A.log` to `D.log`. The four logs of this paper's runs are published: `uv run python scripts/exchange.py fetch spd-logs` downloads them. Then:

```bash
uv run python scripts/spd_up_neurons.py --device cpu \
    --compare docs/analyses/spd-baseline-colab-both-random.json \
    --compare docs/analyses/spd-baseline-colab-both-neurons.json \
    --out docs/analyses/spd-baseline-up-neurons.json
uv run python scripts/spd_input_results.py --logs exchange/spd-logs \
    --out papers/spd-input-projections/results.jsonl
uv run python scripts/paper_figures.py
```

The checkpoints and subcomponents are in the owner's Google Drive, and are not published.

## Next

- Threshold both runs' gates to one count per site, and compare the input projections directly.
- Repeat both starts with more seeds, to give the gap an interval.
- Continue run C at a constant learning rate, to see if the lead closes.
- Try more subcomponents than neurons, the case that bears on polysemantic neurons.

## References

- Braun, D., Bushnaq, L., Heimersheim, S., Mendel, J. & Sharkey, L. (2025). *Interpretability in Parameter Space: Minimizing Mechanistic Description Length with Attribution-based Parameter Decomposition*. [arXiv:2501.14926](https://arxiv.org/abs/2501.14926).
- Bushnaq, L., Braun, D. & Sharkey, L. (2025). *Stochastic Parameter Decomposition*. [arXiv:2506.20790](https://arxiv.org/abs/2506.20790).
- Gao, L., et al. (2020). *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*. [arXiv:2101.00027](https://arxiv.org/abs/2101.00027).
- Hendrycks, D. & Gimpel, K. (2016). *Gaussian Error Linear Units (GELUs)*. [arXiv:1606.08415](https://arxiv.org/abs/1606.08415).
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D. & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners*. [OpenAI](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

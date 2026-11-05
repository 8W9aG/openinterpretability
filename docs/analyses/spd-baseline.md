# SPD baseline: GPT-2 small

The parameter-space baseline. The transcoder baseline (`transcoder-baseline.md`) set neurons against a learned feature basis. This one sets them against the model's own weights, split into their smallest working parts. No decomposition of a registry model exists, so we trained one: a Stochastic Parameter Decomposition (SPD; Bushnaq et al., 2025, arXiv:2506.20790; after APD, Braun et al., 2025, arXiv:2501.14926) of GPT-2 small's twelve MLP down projections. It was trained first on an M1 Max laptop, then on a Colab A100.

**Result.** SPD finds no better basis than the neurons. Two 50,000-step runs on a Colab A100 had the same settings and different starts:

- **Run A started from random directions.** It converged, but its subcomponents stayed mostly the random directions they started as (median cosine 0.53–0.64 with their start). None lines up with a neuron.
- **Run B started from the neurons.** Its subcomponents stayed neurons (median cosine 0.91–0.996 with the closest neuron).
- **Both runs need about 2–3 times fewer active units than the neurons at the same fit.** The neuron baseline keeps each token's largest neurons, at full strength. SPD's gate gives every unit a learned weight between 0 and 1 on each token. Run B keeps the neuron basis and still has that lead. So the lead comes from the gate, not from the directions.

At this size, with as many subcomponents as neurons, SPD's losses do not choose a basis: each run keeps the basis it starts with. The neuron start does as well as a random one. So there is no preset and no audit: run A's subcomponents depend on their random start, and run B's are the neurons.

The per-layer comparisons are in `spd-baseline.json` (laptop), `spd-baseline-colab-random.json` (run A) and `spd-baseline-colab-neurons.json` (run B). The VERIFIED-neuron check is in `spd-baseline-verified.json`.

That result covers the down projections. Whether SPD prefers the neurons on the input projections, where GELU could create a preference, is still open. Laptop pilots are in, but the long runs are not (*The input projections*).

## Method

`scripts/train_spd.py` decomposes each down projection W (`[3072, 768]`) into C = 3,072 rank-1 subcomponents `V[:, c] ⊗ U[c]`. The down projection reads the MLP's hidden activations, where the neurons are, so a subcomponent and a neuron are directions in the same space. Neuron i is the subcomponent `V = e_i, U = W[i]`.

A gate gives each subcomponent a causal importance g in [0, 1] on each token. The losses are those of the paper, with the delta component and importance loss of Goodfire's reference code (`nano_param_decomp/run.py`):

- **faithfulness:** the subcomponents sum to W;
- **stochastic reconstruction:** with subcomponents masked by `g + (1 − g)·U(0, 1)`, all layers at once and one random layer at a time, the output's KL from the model's is small;
- **importance minimality:** `(g + ε)^p` is small, with p annealed from 2 to 0.4.

Four changes were needed on GPT-2. Each came from a failed pilot (next section):

- **CI-masked losses.** Two more losses read the decomposition the way the harness reads it, with every subcomponent masked by exactly g. One is the output's KL. The other is each layer's output error relative to its variance. LayerNorm hides a shrunken MLP output from the KL alone.
- **Centring.** Subcomponents read the hidden activations minus their mean, and the mean's write goes to the bias. GELU leaves most hidden units at small negative values. Without centring, ablating a subcomponent to zero changes the output on every token.
- **A gate per layer.** The paper gives each subcomponent its own small MLP, which reads only that subcomponent's inner activation. Here one GELU MLP (width 512) reads the whole centred hidden vector and gives every subcomponent's importance.
- **A bounded importance gradient.** The reference adds ε = 1e-12 to g. At p = 0.4 that puts the slope at g = 0 near 10⁷. On GPT-2 one such gradient froze the gates for good (see *The long run*). Here ε = 1e-3, and the gate gradients are clipped to norm 1.

Position 0, the attention sink, keeps the model's weights and is left out of every loss. Training reads pile-10k without the scan corpus's documents, in 128-token pieces, 8 to a batch. A step takes about 2.7 s on the laptop's GPU (MPS).

Each subcomponent exports as two latents, the positive and negative parts of `g·((h − mean)·V[:, c])`, which write `+U[c]` and `−U[c]`. `openinterp.harness.dictionaries` reads them with `kind: spd` and `activation: causal_importance`. A test checks that the latents give back exactly the CI-masked projection.

## Pilots

All pilots decomposed all twelve layers with C = 3,072, for 1,000 steps unless stopped. Active subcomponents and explained variance are per layer, on the training run's held-out batch.

| Run | Change | Active per token | Explained variance | Loss gap left | Outcome |
| --- | --- | --- | --- | --- | --- |
| 1 | Paper's losses, importance coefficient 2e-4 | 585–823 (step 500) | — | 168% | g collapsed to small values; KL at exactly g was 7.7 |
| 2 | Coefficient 2e-5, KL at exactly g | 1,260–2,044 | −0.07–0.31 | 4.1% | Outputs shrank; LayerNorm hid it |
| 3 | — | — | — | — | Stopped before its first step |
| 4 | Output-error loss (weight 1), coefficient 6e-5 | ~2,000 (step 500) | 0.25–0.44 | — | Fit stalled; stopped |
| 5 | Output-error weight 10 | 1,417–2,354 | 0.77–0.81 | 1.8% | Good fit, no sparsity |
| 6 | Pilot 5 with centring | 1,666–2,253 | 0.71–0.78 | 2.3% | No gain |
| 7 | Pilot 6 with a gate per layer | 948–1,872 | 0.56–0.72 | 6.3% | Sparsest; all 3,072 alive |

Every pilot was still getting sparser when it stopped. None was sparser than the neurons.

## The long run

The long run used pilot 7's settings for 10,000 steps, with p falling to 0.4 by step 3,000 and then held there. The middle layers fell from about 2,400 active subcomponents at step 1,000 to about 1,050 at step 3,000.

From step 3,500 every layer's count stayed exactly the same, to one decimal place. Between the checkpoints at steps 4,500 and 5,000, V and U moved by 1–5%, but the gate weights moved by 10⁻⁸–10⁻⁴. The gates had frozen. As p reached 0.4, the slope of `(g + 1e-12)^p` near g = 0 grew to about 10⁷. Gate gradients were not clipped, so a spike inflated Adam's second-moment estimate. That shrinks later gate updates for tens of thousands of steps.

The run was resumed from step 5,000 with ε = 1e-3, gate gradients clipped to norm 1 and a new optimizer (`--resume`). The gate gradient norm then stayed at 0.09–0.14. In the first 500 steps the active counts fell by about a quarter, and both sparsity and fit kept improving to the end:

| Step | Active per token | Layer 5 | Explained variance | Loss gap left |
| --- | --- | --- | --- | --- |
| 3,000 | 494–1,135 | 1,050 | 0.60–0.81 | 5.0% |
| 5,000 (frozen) | 721–1,146 | 1,116 | 0.67–0.82 | 3.3% |
| 5,500 | 406–942 | 843 | 0.67–0.85 | 3.3% |
| 7,500 | 344–849 | 773 | 0.72–0.88 | 2.7% |
| 9,999 | 314–813 | 746 | 0.73–0.90 | 2.5% |

All 3,072 subcomponents of every layer were alive on 256 held-out pieces.

## Comparison with the neurons

`scripts/spd_compare.py` reads the final checkpoint and 64 held-out pieces (8,128 tokens). For the neurons, each token keeps its k neurons with the largest contributions `|h_i − mean_i|·‖W[i]‖`, relative to the same mean. That ranking ignores cancellation between neurons, so it is not the best choice of k neurons. In layers 1–5, adding the largest neurons first can lower the fit at small k. The neuron numbers there are pessimistic.

| Layer | SPD active | SPD EV | Neurons, same count: EV | Neurons for SPD EV | Best-neuron cosine (median) | Cosine with start (median) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 512 | 0.90 | 0.89 | 768 | 0.08 | 0.67 |
| 1 | 611 | 0.84 | 0.50 | 1536 | 0.08 | 0.67 |
| 2 | 722 | 0.83 | 0.70 | 1024 | 0.08 | 0.68 |
| 3 | 708 | 0.83 | 0.80 | 1024 | 0.08 | 0.70 |
| 4 | 722 | 0.87 | 0.69 | 1536 | 0.07 | 0.71 |
| 5 | 735 | 0.87 | 0.76 | 1536 | 0.07 | 0.71 |
| 6 | 759 | 0.86 | 0.77 | 1024 | 0.07 | 0.70 |
| 7 | 750 | 0.82 | 0.82 | 768 | 0.07 | 0.68 |
| 8 | 811 | 0.81 | 0.87 | 768 | 0.07 | 0.65 |
| 9 | 780 | 0.76 | 0.89 | 384 | 0.07 | 0.62 |
| 10 | 612 | 0.74 | 0.86 | 256 | 0.06 | 0.59 |
| 11 | 331 | 0.82 | 0.56 | 768 | 0.06 | 0.57 |

*Neurons for SPD EV* is the first of 16, 32, 64, 128, 256, 384, 512, 768, 1,024, 1,536, 2,048 and 3,072 that reaches SPD's fit.

**Density.** At SPD's own count, SPD fits better than the neurons in layers 0–6 and 11, and by a wide margin in layers 1 and 11 (0.84 against 0.50, 0.82 against 0.56). The two are level in layer 7. The neurons fit better in layers 8–10.

**Alignment.** No subcomponent in any layer has a cosine above 0.7 with any one neuron. The median cosine with the closest neuron is 0.06–0.08. A random direction in 3,072 dimensions gives about 0.07.

**Drift.** That alignment says little about the model, because the subcomponents have not moved far from where they started. With C equal to the hidden width, any rotation of the neuron basis can sum to W exactly. The faithfulness warmup moves V first: on its own it takes each column to a median cosine of 0.61–0.74 with its random draw, before any other loss runs. After 10,000 steps the cosine is 0.57–0.71, so the importance and reconstruction losses moved the columns very little. The gates learned which of these partly random directions matter on each token. That is why the counts can fall while the directions stay random.

## What the laptop run means

- **Not a preset.** Auditing these latents would mean explaining directions that are still mostly initialisation. A verdict on them would say little about GPT-2.
- **Not evidence against the neuron basis.** The subcomponents miss the neurons because they have not converged, not because SPD found something better. The density comparison is mixed and depends on the layer.
- **The method trains on a pretrained model.** With the four changes above, SPD reaches a usable fit on GPT-2 small, on a laptop, in about 8 hours. No released decomposition covers a Hugging Face model (*What others have released*).

## Colab run A: from random, 50,000 steps

`scripts/train_spd_colab.ipynb` ran the laptop run's settings on a Colab A100 (2026-11-03), with two changes. The run was 50,000 steps, and p reached 0.4 at step 5,000 (`--p-anneal 0.1`). A step took 0.16 s, so the run took 2.3 hours. The outputs are in the owner's Google Drive (`openinterp/gpt2-spd-random-50k`), not in the repository.

**Training.** The middle layers grew denser while p was above 1 (layer 5: 2,604 active subcomponents at step 2,000). They then fell as p reached its floor (945 at step 5,000). They levelled off at about 700 by step 15,000. Explained variance levelled off by step 30,000. From step 1,000 the gate gradient norm stayed at 0.04–0.17, and the counts changed at every evaluation, so the gates did not freeze. Unlike the laptop run, some subcomponents died: 2,435–3,064 per layer were alive on 256 held-out pieces.

**Comparison with the neurons** (`spd_compare.py`, the same 64 held-out pieces as the laptop comparison):

| Layer | SPD active | SPD EV | Neurons, same count: EV | Neurons for SPD EV | Best-neuron cosine (median) | Cosine with start (median) |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 354 | 0.96 | 0.85 | 1536 | 0.11 | 0.57 |
| 1 | 498 | 0.93 | 0.42 | 2048 | 0.10 | 0.53 |
| 2 | 577 | 0.93 | 0.57 | 1536 | 0.10 | 0.53 |
| 3 | 618 | 0.96 | 0.76 | 2048 | 0.10 | 0.60 |
| 4 | 679 | 0.97 | 0.65 | 2048 | 0.08 | 0.60 |
| 5 | 684 | 0.97 | 0.73 | 2048 | 0.08 | 0.60 |
| 6 | 695 | 0.97 | 0.74 | 2048 | 0.08 | 0.62 |
| 7 | 669 | 0.97 | 0.80 | 2048 | 0.08 | 0.64 |
| 8 | 674 | 0.96 | 0.84 | 1536 | 0.08 | 0.62 |
| 9 | 621 | 0.92 | 0.85 | 1024 | 0.08 | 0.60 |
| 10 | 474 | 0.84 | 0.83 | 768 | 0.07 | 0.57 |
| 11 | 225 | 0.86 | 0.51 | 1024 | 0.06 | 0.56 |

- **Density.** At the same fit, SPD needs 1.5–3 times fewer active subcomponents than the neurons need neurons, in every layer except 10. On the laptop the comparison was mixed.
- **Drift.** Five times more training moved the subcomponents only a little further from their start: a median cosine of 0.53–0.64, against 0.57–0.71 on the laptop. The losses had levelled off. So this is where the method settles with C equal to the hidden width, not an unfinished run. Most of the distance from the random draw comes from the faithfulness warmup, which alone gives 0.61–0.74. Run B has no warmup movement, because neurons already sum to W.
- **Alignment.** No subcomponent has a cosine above 0.7 with any neuron. The median closest-neuron cosine rose a little in layers 0–3 (0.10–0.11). Elsewhere it stayed near a random direction's 0.07.

**Why the density lead is not yet a result about the basis.** The two arms do not choose their units the same way. SPD's gate is a network that reads the whole centred hidden vector. On each token it gives every subcomponent a weight between 0 and 1, fitted to the output. The neuron baseline keeps or drops each neuron by a fixed ranking and cannot scale one. A set of mostly random directions beats the neurons by 2–3 times, which suggests the gate does much of the work.

## Colab run B: from the neurons, 50,000 steps

Run B had run A's settings, except that each subcomponent started as one neuron (`--init neurons`: V = I, U = W). Its outputs are in the owner's Google Drive (`openinterp/gpt2-spd-neurons-50k`).

| Layer | SPD active | SPD EV | Neurons, same count: EV | Neurons for SPD EV | Best-neuron cosine (median) | Share over 0.7 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 165 | 0.94 | 0.74 | 1024 | 0.96 | 1.00 |
| 1 | 369 | 0.84 | 0.33 | 1536 | 0.91 | 0.99 |
| 2 | 413 | 0.86 | 0.35 | 1536 | 0.95 | 1.00 |
| 3 | 397 | 0.86 | 0.52 | 1024 | 0.97 | 1.00 |
| 4 | 460 | 0.86 | 0.34 | 1536 | 0.96 | 1.00 |
| 5 | 451 | 0.86 | 0.53 | 1024 | 0.96 | 1.00 |
| 6 | 459 | 0.86 | 0.59 | 1024 | 0.97 | 1.00 |
| 7 | 433 | 0.86 | 0.68 | 1024 | 0.98 | 1.00 |
| 8 | 430 | 0.87 | 0.76 | 1024 | 0.98 | 1.00 |
| 9 | 388 | 0.88 | 0.78 | 768 | 0.99 | 1.00 |
| 10 | 336 | 0.90 | 0.78 | 1024 | 0.99 | 1.00 |
| 11 | 237 | 0.92 | 0.52 | 1536 | 1.00 | 1.00 |

With this start, the cosine with the start and the best-neuron cosine are the same number.

- **Alignment.** At least 99.4% of each layer's subcomponents have a cosine above 0.7 with one neuron. They stayed neurons.
- **Density.** Run B settled at fewer active units and a lower fit than run A: 165–460 against 225–695 active, and 0.84–0.94 against 0.84–0.97 explained variance. To compare the two at different fits, take the number of neurons that reach each run's fit, interpolated on the neuron curve, and divide it by the run's active count. That ratio is 1.9–5.6 for run B and 1.1–3.6 for run A. It is 2.1–2.6 in run B's middle layers (3–8) and 2.2–2.7 in run A's. The lead over the neurons is about the same with or without the change of basis.

## Verified neurons in run B

Run B keeps each neuron as one subcomponent, so SPD's view of each neuron can be set against the registry's verdict on it. `scripts/spd_verified.py` reads run B's checkpoint and 256 held-out pieces (32,512 tokens). It compares the 76 VERIFIED GPT-2 neuron poles with the 256 other studied poles (NEEDS_REVISION or REJECTED). Each measure is ranked within its layer first, so layer effects cancel. A permutation test gives each p-value.

| Measure | VERIFIED: mean percentile | Other: mean percentile | p |
| --- | --- | --- | --- |
| Input direction's cosine with its neuron | 0.51 | 0.45 | 0.08 |
| Output direction's cosine with the neuron's row of W | 0.50 | 0.50 | 0.91 |
| Mean causal importance | 0.48 | 0.54 | 0.09 |
| Share of tokens with causal importance above 0 | 0.48 | 0.54 | 0.10 |
| Causal importance on the pole's 32 largest activations | 0.44 | 0.49 | 0.20 |

No measure separates the two groups at p < 0.05, and five tests were run. VERIFIED neurons lean towards staying a little closer to their neuron and being important on fewer tokens, but the differences are small. On a pole's largest activations, the gate keeps both groups at full importance (median 1.0), so that measure is at its ceiling. The result is in `spd-baseline-verified.json`.

So SPD's view of a neuron says nothing about whether an explanation of it verifies. That fits the rest of this analysis: started from the neurons, SPD keeps them all with about the same weight. The other group's verdicts are about hypotheses, not neurons: a NEEDS_REVISION pole may be a clean neuron with a wrong first hypothesis. That blurs any difference.

## What runs A and B mean

- **The gate gives the lead, not the directions.** Run B shows the neurons with a gate are about as far ahead of the neuron baseline as run A's directions are. The neuron baseline's weak point is how it picks units: a fixed ranking that ignores cancellation, with no scaling.
- **SPD does not pick a basis here.** With C equal to the hidden width, any rotation of the neuron basis sums to W exactly. The importance and reconstruction losses moved V only a little from either start. So SPD cannot say the neurons are the *best* basis, only that it found none better.
- **No preset and no audit.** Auditing run A would explain directions that depend on a random seed. Auditing run B would audit the neurons again, with a gate. Neither is a second baseline. The weights are not uploaded to the Hub.
- **Supports the registry's premise, weakly.** For GPT-2 small's MLPs, a parameter decomposition that starts from the neurons keeps them. It treats VERIFIED neurons no differently from other studied ones.

Tests not run: more subcomponents than neurons (C > 3,072), where SPD could split a neuron into parts, the overcomplete case that bears on polysemantic neurons; a fixed gate to compare runs A and B at one fit; and Qwen2.5-0.5B, whose SwiGLU hidden activations may decompose differently.

## The input projections: pilots, and the runs still to do

The down projection reads the neurons, but nothing in it favours them: any rotation of the neuron basis sums to W. The input projection (`c_fc`) writes the neurons' pre-activations, and GELU acts on each neuron separately. A neuron that is off on a token can be dropped there at no cost. A subcomponent that writes to many neurons can be dropped only where all of them are off. So SPD's importance loss could prefer the neurons here, if the neuron basis is the model's own.

`train_spd.py --matrices both` decomposes both projections of each layer. Of the input projection, neuron i is `V = W[:, i], U = e_i`. Its CI-masked output loss and explained variance are measured after GELU. If they were measured on the pre-activations, an off neuron would cost as much to drop as an on one. Alignment and drift read `U[c]`: about 0.07 for a random direction, 1 for a neuron.

**Two laptop pilots** ran for 1,000 steps from each start, with p reaching 0.4 at step 500. Both trained without errors, and all subcomponents stayed alive. The sites are measured on 32 held-out pieces (`spd-baseline-both-pilots.json`).

| Step 999 | From the neurons | From random |
| --- | --- | --- |
| Share of the loss gap left | 0.112 | 0.152 |
| KL with every mask at its causal importance | 0.65 | 0.75 |
| Best-neuron cosine (median) | 0.995–1.0 | 0.06–0.07 |
| Input projections: explained variance | 0.30–0.57 | 0.00–0.40 |
| Input projections: active per token, layers 1–10 | 319–500 | 68–125 |
| Down projections: explained variance | 0.28–0.80 | 0.16–0.66 |
| Down projections: active per token | 281–481 | 294–830 |

The start from the neurons leads on both whole-model measures. In layers 2–5 of the down projections the active counts are within about 13%, and it fits 1.6–2 times as well. These pilots do not show that SPD prefers the neurons:

- **The input projections are not compared at equal sparsity.** In layers 1–10 the start from the neurons keeps 2–6 times as many subcomponents active, so its better fit may only be the cost of keeping more.
- **The lead may be a head start.** Neurons sum to W exactly at step 0, and the random start's input fit was still rising faster at step 999. Runs A and B ended about level after 50,000 steps on the down projections alone.
- **The subcomponents barely move.** From random, the input projection's cosine with its start stayed within 0.003 of the warmup's 0.61–0.63. So a long run from random that stays random would not separate "SPD does not prefer the neurons" from "training did not reach them".

**The runs still to do** are 50,000 steps on a Colab A100 from each start, with run A's other settings (`--matrices both`, `--init random` and `--init neurons`, in `scripts/train_spd_colab.ipynb`). Compare the input projections' fit at equal sparsity:

- If the start from the neurons fits better, SPD's losses prefer the neurons, however slowly training moves.
- If the start from random also turns towards the neurons (best-neuron cosine far above 0.07), training finds them unaided. This is the strongest support this method can give.
- If the two end level, SPD finds no preferred basis, even where the nonlinearity could create one.

## What others have released

No APD, SPD or VPD decomposition covers a registry model (Qwen2.5-0.5B, Qwen2.5-1.5B, GPT-2 small). Goodfire's only language-model decomposition is VPD ([post](https://www.goodfire.com/research/interpreting-lm-parameters), May 2026). Its target is a custom 4-layer, ~67M-parameter model trained on The Pile: d_model 768, MLP width 3072, GELU, RMSNorm, RoPE, vocab 50,277. It decomposes all 24 weight matrices (q/k/v/o and both MLP matrices) into 38,912 rank-1 subcomponents, about 9,970 alive.

The code ([goodfire-ai/param-decomp](https://github.com/goodfire-ai/param-decomp), MIT) is JAX; the Torch version is retired. The target weights and the decomposition are only W&B runs (target `goodfire/spd/runs/t-9d2b8f02`, decomposition `goodfire/spd/runs/s-55ea3f9b`), not Hugging Face repos, and the target has no `transformers` class. No per-component labels are released. The repo's 4-layer config says a run needs 32 GPUs. Licences for the weights and decomposition are not stated.

## Reproduce

```bash
uv run python scripts/train_spd.py --layers all --steps 10000 --eval-every 500 \
    --gate layer --d-gate 512 --coeff-imp 6e-5 --coeff-ci-output 10 --center \
    --p-anneal 0.3 --out exchange/gpt2-spd-10k
uv run python scripts/spd_compare.py --checkpoint exchange/gpt2-spd-10k/checkpoint.pt \
    --out exchange/gpt2-spd-10k/compare.json
```

Today's defaults (ε = 1e-3, gate clip 1) prevent the freeze, so this does not repeat the frozen stretch from step 3,500 to 5,000. The run reported here was trained with ε = 1e-12 and no gate clip to step 5,000, then resumed with today's defaults.

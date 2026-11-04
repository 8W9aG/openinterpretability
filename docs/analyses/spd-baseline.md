# SPD baseline: GPT-2 small

The parameter-space baseline (`TODO.md`, *Scope expansion*). The transcoder baseline (`transcoder-baseline.md`) set neurons against a learned feature basis. This one sets them against the model's own weights, split into their smallest working parts. No decomposition of a registry model exists, so we trained one: a Stochastic Parameter Decomposition (SPD; Bushnaq et al., 2025, arXiv:2506.20790; after APD, Braun et al., 2025, arXiv:2501.14926) of GPT-2 small's twelve MLP down projections. It was trained first on an M1 Max laptop, then on a Colab A100.

**Result so far.** On the laptop, 10,000 steps gave a usable fit but had not converged. On a Colab A100, 50,000 steps from random (run A) converged. Run A's CI-masked projection explains 0.84–0.97 of each layer's output on held-out text, with 1.2% of the loss gap left. At that fit it needs 1.5–3 times fewer active subcomponents than the neurons in every layer except layer 10, where they are level.

But the subcomponents are still mostly the random directions they started as. Their median cosine with their initial values is 0.53–0.64, and none lines up with a neuron. The neuron comparison is also uneven. SPD's gate scales each subcomponent by a learned importance, while the neuron baseline only keeps or drops each neuron. So part of SPD's lead may come from the gate, not the directions. Run B, which starts SPD from the neurons, separates the two. Until it is in, there is no preset and no audit.

The per-layer comparisons are in `spd-baseline.json` (laptop) and `spd-baseline-colab-random.json` (run A).

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

**Drift.** That alignment says little about the model, because the subcomponents have not moved far from where they started. With C equal to the hidden width, any rotation of the neuron basis can sum to W exactly. Only the importance and reconstruction losses move V, and in 10,000 steps they left each column with a median cosine of 0.57–0.71 with its random start. The gates learned which of these partly random directions matter on each token. That is why the counts can fall while the directions stay random.

## What the laptop run means

- **Not a preset.** Auditing these latents would mean explaining directions that are still mostly initialisation. A verdict on them would say little about GPT-2.
- **Not evidence against the neuron basis.** The subcomponents miss the neurons because they have not converged, not because SPD found something better. The density comparison is mixed and depends on the layer.
- **The method trains on a pretrained model.** With the four changes above, SPD reaches a usable fit on GPT-2 small, on a laptop, in about 8 hours. No released decomposition covers a Hugging Face model (`TODO.md`, *Scope expansion*).

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
- **Drift.** Five times more training moved the subcomponents only a little further from their start: a median cosine of 0.53–0.64, against 0.57–0.71 on the laptop. The losses had levelled off. So this is where the method settles with C equal to the hidden width, not an unfinished run.
- **Alignment.** No subcomponent has a cosine above 0.7 with any neuron. The median closest-neuron cosine rose a little in layers 0–3 (0.10–0.11). Elsewhere it stayed near a random direction's 0.07.

**Why the density lead is not yet a result about the basis.** The two arms do not choose their units the same way. SPD's gate is a network that reads the whole centred hidden vector. On each token it gives every subcomponent a weight between 0 and 1, fitted to the output. The neuron baseline keeps or drops each neuron by a fixed ranking and cannot scale one. A set of mostly random directions beats the neurons by 2–3 times, which suggests the gate does much of the work.

## Next

**Run B, from the neurons** (`--init neurons`, V = I, U = W), 50,000 steps on the same settings. It gives the neurons the same gate and losses as run A.

- If it stays near the neurons and is as sparse as run A, the gate did the work. The neuron basis is then as good a basis as SPD finds, which supports the registry's premise.
- If it moves away from the neurons to reach run A's sparsity, SPD prefers other directions. That is evidence against the premise.

A preset (`gpt2-small-spd`) and the 40-per-arm audit against the transcoder baseline's 40 positive poles wait for run B. A Qwen2.5-0.5B run would follow. Its SwiGLU hidden activations may decompose differently.

## Reproduce

```bash
uv run python scripts/train_spd.py --layers all --steps 10000 --eval-every 500 \
    --gate layer --d-gate 512 --coeff-imp 6e-5 --coeff-ci-output 10 --center \
    --p-anneal 0.3 --out exchange/gpt2-spd-10k
uv run python scripts/spd_compare.py --checkpoint exchange/gpt2-spd-10k/checkpoint.pt \
    --out exchange/gpt2-spd-10k/compare.json
```

Today's defaults (ε = 1e-3, gate clip 1) prevent the freeze, so this does not repeat the frozen stretch from step 3,500 to 5,000. The run reported here was trained with ε = 1e-12 and no gate clip to step 5,000, then resumed with today's defaults.

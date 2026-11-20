# Research questions

The questions the project exists to answer, and how close the registry is to each answer. Each question is also an issue, which holds the discussion. New work says which question it answers. When a paper or analysis lands, update the question it bears on (`write-paper` skill). Numbers are dated, because the registry grows every day.

## 0. How much of each model does the registry explain?

The north star: total interpretability of open models, as one number per model.

- **What would answer it.** Registry coverage near 100%: mean-ablating every pole that isn't VERIFIED costs a model almost none of its MLP layers' effect on the loss.
- **So far (2026-11-13).** Coverage is 0.60% on GPT-2, 0.42% on Qwen2.5-0.5B and 0.01% on Qwen2.5-1.5B (`docs/analyses/registry-coverage.md`, the home page). On Qwen2.5-0.5B the VERIFIED poles keep 2.6 times what as many random poles keep. On GPT-2 they keep no more.
- **Bears on it.** The coverage metric's follow-ups: weight draws by a pole's loss effect (#3486), and score coverage with simulated activations (#3487). Every question below.

## 1. Are native MLP neurons clean? (#3186)

The working hypothesis from Goodfire's *Neural Networks Are Already Clean*: most apparent polysemanticity in native neurons comes from tooling, not superposition.

- **What would answer it.** The share of randomly drawn poles with a VERIFIED explanation, by model, layer and pole, with the dead and default-on draws counted. A high share supports the hypothesis, and a low one with a clean protocol counts against it.
- **So far.** On 2026-11-14, 25.1% of 1,711 random audited poles were VERIFIED on their first reading, and 27.9% standing (`docs/analyses/registry-stats.md`). By model, first reading: Qwen2.5-0.5B 30.2%, GPT-2 small 21.1% and Qwen2.5-1.5B 18.9%. The proposer moves these shares more than the model does. Counting the draws that a scan stopped on, 12.1% of GPT-2's random draws are VERIFIED. Rotated units, which aren't native neurons, verify far less often than native ones (`docs/analyses/null-calibration.md`), so the native basis is privileged under this protocol.
- **Bears on it.** Polysemantic experiments (#3215), multi-dimensional features, `papers/default-on-poles/`.

## 2. How reliable is one verdict? (#3187)

Every registry-wide claim inherits the noise of single verdicts.

- **What would answer it.** How often a rerun, another reviewer or another corpus changes a verdict, and a review rule whose verdict is stable.
- **So far.** A second reading of a hypothesis agrees with the first on status 56% of the time (`registry-stats.md`). The reviewer explains 6–8% of the variance in verdicts (`docs/analyses/reviewer-crossover.md`). No swapped hypothesis was VERIFIED in 255 tries (`null-calibration.md`). Today's rule still verifies 326 of 331 older VERIFIED records (`docs/analyses/protocol-mix.md`). VERIFIED explanations score only 0.04 higher on held-out detection than others, within judge (`docs/analyses/verified-vs-detection.md`). With one judge, twice the windows and two seeds, a score still separates VERIFIED records poorly (AUROC 0.54 for detection). So it should not become a gate (`docs/analyses/rescore-pilot.md`, 2026-11-16).
- **Update (2026-11-13).** With the same reviewer and prompt, VERIFIED-or-not flips between two reviews 18% of the time. That noise is 21% of balanced accuracy's variance, twice the reviewer model's share (`docs/analyses/verdict-stability.md`). The `specificity` gate causes most flips (#3509).
- **Update (2026-11-14).** The hypothesis and probes stayed fixed, and the unit was scanned on fineweb-edu instead of pile-10k. That changed VERIFIED-or-not for 7 of 40 records, about 17% when weighted to the registry. That is as large as the rerun noise. Every change is at a gate's edge, through the scan's scale (`docs/analyses/corpus-stability.md`).
- **Bears on it.** Rival hypotheses, reviewer drift, the human baseline, ground-truth models, held-out scoring (#3499, #3505).

## 3. Do native neurons explain a model as well as learned dictionaries do? (#3188)

- **What would answer it.** At equal effort, VERIFIED rates, explanation sharpness and coverage for native poles, transcoder latents, SAE latents and SPD components.
- **So far.** On 2026-11-14, `papers/native-vs-learned/` drew it together. On GPT-2 small, 37 of 40 random transcoder latents verified, against 20 of 40 positive poles, at equal effort per audit. Counting dead draws, the gap is no longer significant. Per layer, auditing every live latent would take about five times the audits for ten times the VERIFIED units. On Qwen2.5-0.5B the native basis beats a random rotation of itself, 18 of 30 against 1 of 30. A verified pole is about as sharp as a verified latent (`docs/analyses/transcoder-baseline.md`). Linear probes need about 32 neurons to match a concept that one VERIFIED pole carries in part (`docs/analyses/probe-comparison.md`). How much of a layer a set of VERIFIED units explains is still unmeasured.
- **Bears on it.** How many transcoder features are really neurons, the N2G baseline, the explainer benchmark (`docs/benchmark.md`), `papers/spd-input-projections/`.

## 4. Do explanations carry across models? (#3189)

- **What would answer it.** For a VERIFIED pole, a sister in another model that the same explanation passes, more often than chance.
- **So far.** On 2026-11-14, `papers/explanations-across-models/` read the registry's 80 sister studies. A VERIFIED explanation verifies on 8 of 50 sisters, 1 of 26 other close matches and 0 of 4 random poles. It mostly carries in part. The median balanced accuracy is 0.78 on sisters and 0.55 on other matches, and sisters are rejected far less often (p = 0.012). (0.55 replaces an earlier 0.56: the data give 0.5549, 2026-11-19.) The random control is only 4 studies. A fifth of Qwen2.5-0.5B's VERIFIED poles have a sister in Qwen2.5-1.5B, at about the same relative depth (`docs/analyses/sister-neurons.md`). Sisters across tokenizers, Qwen to GPT-2, are rarer. New models wait until each current one reaches `coverage_target` (`TODO.md`, *Models*), so the seed and size batteries wait too. On 2026-11-20, `papers/shared-concept-space/` placed 1,998 studied poles of the four models in one shared concept map. Each model reaches every region. The models' spreads differ when pooled over scan corpora, but on one corpus, pile-10k, no pair differs from a shuffled null. One model's poles from two corpora differ more than two models' poles from one.
- **Bears on it.** The sister refresh in each release, the Platonic Representation Hypothesis.

## 5. What makes a neuron rest, stay dead or stay on? (#3190)

- **What would answer it.** A cause in training, the activation function or the architecture that predicts which neurons rest below the activation's dip, die or stay on.
- **So far.** `papers/why-neurons-rest/` shows that the output's push and the final LayerNorm's pull set where GELU neurons rest. `papers/default-on-poles/` describes what default-on negative poles do as a population. Pythia's training setup, one part at a time and all together, does not weaken the push in a small model (2026-11-16).
- **Bears on it.** The *Why neurons rest* and *Default-on poles* workstreams in Linear.

## 6. How much does what a neuron appears to do depend on the scan corpus? (#3191)

- **What would answer it.** For VERIFIED poles, whether the explanation holds on other corpora, measured with held-out scores on each.
- **So far.** Not measured. Each new scan reads its own seeded `mixed` corpus, and the Cosmopedia control isn't run.
- **Bears on it.** New corpus presets (`area: datasets`), held-out scoring on another corpus.

## 7. When do neurons form during training? (#3376)

- **What would answer it.** Take the VERIFIED poles of a model whose training run is public. For each, the step from which its explanation holds, and whether that step lines up with the loss or with other training signals.
- **So far.** On 2026-11-19, `papers/neuron-development/` followed Pythia-70m's 86 VERIFIED poles through 20 checkpoints (#4881). The loss settles within 1% of its final value by step 16,000. But the poles keep changing. Their median correlation with their final form is 0.62 there, and adjacent checkpoints correlate at a median of 0.78 to 0.82 after it. 38 explanations hold without a break from step 16,000 or earlier, and 38 only from step 64,000 or later. The first version's 9 poles gave the same picture. But the two measures of formation are weakly related (ρ = 0.24), not reversed as the 9 suggested.
- **Bears on it.** Adjacent-checkpoint correlations, OLMo's checkpoints, and the local learning coefficient (#4156).

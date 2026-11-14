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
- **So far.** About a quarter of random audited poles are VERIFIED on their first reading. Follow-ups raise that to 30.6% standing (`docs/analyses/registry-stats.md`). Counting the draws that a scan stopped on, 11.5% of GPT-2's random draws are VERIFIED. Rotated units, which aren't native neurons, verify far less often than native ones (`docs/analyses/null-calibration.md`), so the native basis is privileged under this protocol.
- **Bears on it.** Polysemantic experiments (#3215), multi-dimensional features, `papers/default-on-poles/`.

## 2. How reliable is one verdict? (#3187)

Every registry-wide claim inherits the noise of single verdicts.

- **What would answer it.** How often a rerun, another reviewer or another corpus changes a verdict, and a review rule whose verdict is stable.
- **So far.** A second reading of a hypothesis agrees with the first on status 56% of the time (`registry-stats.md`). The reviewer explains 6–8% of the variance in verdicts (`docs/analyses/reviewer-crossover.md`). No swapped hypothesis was VERIFIED in 255 tries (`null-calibration.md`). Today's rule still verifies 326 of 331 older VERIFIED records (`docs/analyses/protocol-mix.md`). VERIFIED explanations score only 0.04 higher on held-out detection than others, within judge (`docs/analyses/verified-vs-detection.md`).
- **Update (2026-11-13).** With the same reviewer and prompt, VERIFIED-or-not flips between two reviews 18% of the time. That noise is 21% of balanced accuracy's variance, twice the reviewer model's share (`docs/analyses/verdict-stability.md`). The `specificity` gate causes most flips (#3509).
- **Bears on it.** Rival hypotheses, reviewer drift, the human baseline, ground-truth models, held-out scoring (#3499, #3505).

## 3. Do native neurons explain a model as well as learned dictionaries do? (#3188)

- **What would answer it.** At equal effort, VERIFIED rates, explanation sharpness and coverage for native poles, transcoder latents, SAE latents and SPD components.
- **So far.** On GPT-2 small, 37 of 40 random transcoder latents verified, against 20 of 40 positive poles. A verified pole is about as sharp as a verified latent (`docs/analyses/transcoder-baseline.md`). Counting dead draws, the gap is no longer significant. Linear probes need about 32 neurons to match a concept that one VERIFIED pole carries in part (`docs/analyses/probe-comparison.md`).
- **Bears on it.** How many transcoder features are really neurons, the N2G baseline, the explainer benchmark (`docs/benchmark.md`), `papers/spd-input-projections/`.

## 4. Do explanations carry across models? (#3189)

- **What would answer it.** For a VERIFIED pole, a sister in another model that the same explanation passes, more often than chance.
- **So far.** A fifth of Qwen2.5-0.5B's VERIFIED poles have a sister in Qwen2.5-1.5B, at about the same relative depth (`docs/analyses/sister-neurons.md`). Sisters across tokenizers, Qwen to GPT-2, are rarer. New models wait until each current one reaches `coverage_target` (`TODO.md`, *Models*), so the seed and size batteries wait too.
- **Bears on it.** The sister refresh in each release, the Platonic Representation Hypothesis.

## 5. What makes a neuron rest, stay dead or stay on? (#3190)

- **What would answer it.** A cause in training, the activation function or the architecture that predicts which neurons rest below the activation's dip, die or stay on.
- **So far.** `papers/why-neurons-rest/` shows that the output's push and the final LayerNorm's pull set where GELU neurons rest. `papers/default-on-poles/` describes what default-on negative poles do as a population. Pythia's training setup is only partly tested.
- **Bears on it.** The *Why neurons rest* and *Default-on poles* workstreams in Linear.

## 6. How much does what a neuron appears to do depend on the scan corpus? (#3191)

- **What would answer it.** For VERIFIED poles, whether the explanation holds on other corpora, measured with held-out scores on each.
- **So far.** Not measured. Each new scan reads its own seeded `mixed` corpus, and the Cosmopedia control isn't run.
- **Bears on it.** New corpus presets (`area: datasets`), held-out scoring on another corpus.

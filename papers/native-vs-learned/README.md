# Native neurons against learned dictionaries, at equal effort

Do a model's own neurons explain it as well as a learned dictionary does? We compare them under one review protocol, one pair of agents and one audit per unit. On GPT-2 small, a transcoder latent that fires reaches VERIFIED 93% of the time, against 50% for a positive neuron pole. A third of the drawn latents never fire, and counting those, the shares are 61% against 50%. The agents read prompts of the same size in both arms, so the effort per audit was equal. Per layer, the comparison turns over. A GPT-2 layer has 24,576 latents and 3,072 positive poles. Auditing every live latent once would take about five times the audits, and would give about ten times the VERIFIED units. On Qwen2.5-0.5B, the native basis beats a random rotation of itself, 18 of 30 against 1 of 30. A decomposition of GPT-2's weights finds no basis better than the neurons. So the neurons are a privileged basis, and a learned dictionary splits them into narrower, easier units.

**Status:** Preprint. Work left: #4247 (rewrite in the paper form of #4202), #3769 (one model, layer coverage and protocol 0.9).

**Why we wrote it.** The registry studies native neurons, while most interpretability work uses learned dictionaries. The choice is only sound if neurons explain the model about as well. The registry had not yet compared the two under one protocol at equal effort. **The question.** Do a model's own neurons explain it as well as a learned dictionary does? It is research question 3 ([research questions](../../docs/research-questions.md)).

![Per unit audited, learned latents verify most often, and the native basis beats a random rotation of it](media/shares.svg)

## Results

1. **Per unit, transcoder latents verify more often.** 37 of 40 live latents, against 20 of 40 positive poles (Fisher $p = 4.3 \times 10^{-5}$). *Limit:* one model and one dictionary. Counting dead draws, the gap shrinks to 61% against 50%, and it is no longer significant ($p = 0.31$).
2. **The effort per audit was equal.** Each unit had one audit by the same proposer, reviewer and judges. The proposer's prompts had a median of 17,417 characters for latents and 17,877 for poles. The reviewer's had 8,239 and 9,392. *Limit:* the agents ran outside the harness, so their token counts and costs weren't recorded, only the size of each prompt.
3. **Per layer, a dictionary costs more and gives more.** At the measured rates, auditing a whole layer once means 16,115 audits for its live latents, against 3,072 for its positive poles. It would give about 14,907 VERIFIED latents against 1,536 VERIFIED poles. *Limit:* this is our extrapolation of two shares to a whole layer, not a run. The transcoder also explains only part of the MLP: 70% of its output's variance on the scanned layers.
4. **The native basis is privileged.** On Qwen2.5-0.5B, 30 units in their native basis reach VERIFIED 18 times. In a random rotation of the same layers, they reach it once (sign test $p = 1.5 \times 10^{-5}$). *Limit:* this compares neurons with random directions, not with a learned dictionary.
5. **A decomposition of GPT-2's weights doesn't find a better basis.** Stochastic Parameter Decomposition (Bushnaq et al., 2025) keeps whichever basis it starts in on the down projections. On the input projections, a start from the neurons fits better than a random start, but training from random doesn't find the neurons.

**What this does not show.**

- How much of the model a set of VERIFIED units explains. A VERIFIED unit has one accurate explanation. That doesn't measure how much of the layer's computation it accounts for.
- Anything about SAEs, other dictionaries or other models. The dictionary comparison is GPT-2 small's transcoders only.

## Method

- **Units and draws.** The model is GPT-2 small (Radford et al., 2019). The dictionary is its transcoders of Dunefsky et al. (2024), repackaged as [`jacobdunefsky/gpt2small-transcoders`](https://huggingface.co/jacobdunefsky/gpt2small-transcoders) at revision `8f254a3a`. A transcoder is a wide ReLU MLP trained "to faithfully approximate the original narrower MLP sublayer’s output" (Dunefsky et al., 2024, §1). This one has 24,576 latents per layer, which we read from its weights. Its model card names no licence and no citation, only the paper.
  - Latents were drawn uniformly by `open-interpretability audit --random --dictionary gpt2-small-transcoders`, seeds 1–61, until 40 fired on the scan.
  - Positive poles were drawn by `--random --pole positive`, seeds 1–40.
  - The draws and verdicts are in [`transcoder-baseline.jsonl`](../../docs/analyses/transcoder-baseline.jsonl). [The baseline write-up](../../docs/analyses/transcoder-baseline.md) gives the setup in full.
- **Review.** Protocol 0.7, the same in both arms. The proposer was Claude Opus 5.5, the reviewer Claude Sonnet 5.5 and the judges Claude Haiku 5.5. Every scan read the same 200,000-token sample of the corpus, in 128-token windows, because the transcoders fit no longer context.
- **Effort.** Each record keeps the size of every agent's prompt in its provenance (`costs.llm_calls`). `scripts/native_vs_learned.py` takes the median of each arm.
- **The basis null.** It is in [`null-calibration.md`](../../docs/analyses/null-calibration.md). 30 units of [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) were drawn at random. Each was audited twice: as a native positive pole, and as a unit of a random rotation of the same layer. The same agents audited both, and the arms were hidden from them.
- **Shares and intervals.** 95% Wilson intervals. The p-values are those of the source analyses.

## Per unit: latents verify more often

| Arm | Model | VERIFIED | Share (95% CI) |
| --- | --- | --- | --- |
| Transcoder latents that fire | GPT-2 small | 37 of 40 | 0.93 (0.80–0.97) |
| Transcoder latents, dead draws counted | GPT-2 small | 37 of 61 | 0.61 (0.48–0.72) |
| Positive neuron poles | GPT-2 small | 20 of 40 | 0.50 (0.35–0.65) |
| Native positive poles | Qwen2.5-0.5B | 18 of 30 | 0.60 (0.42–0.75) |
| The same layers, rotated at random | Qwen2.5-0.5B | 1 of 30 | 0.03 (0.01–0.17) |

The baseline's write-up gives how each arm failed:

- **Latents.** Most got narrow lexical or syntactic hypotheses, such as single-token detectors or one word in one sense. None failed specificity.
- **Poles.** 13 failed specificity, because the explanation described a family and the reviewer found members outside it. Six failed because the explanation stated a floor that every prompt clears.

A VERIFIED pole is about as sharp as a VERIFIED latent: the mean balanced accuracy is 0.90 for poles and 0.95 for latents. So the gap is in how often a neuron has one narrow meaning, not in how well a good explanation fits.

## Equal effort per audit

| Arm | Proposer prompt (median characters) | Reviewer prompt | Audits |
| --- | --- | --- | --- |
| Transcoder latents | 17,417 | 8,239 | 40 |
| Positive neuron poles | 17,877 | 9,392 | 40 |

The prompts differ by about 3% for the proposer, and 14% for the reviewer. Each unit had exactly one audit, with no follow-up, so the comparison is per audit at the same input.

## Per layer: five times the audits, ten times the VERIFIED units

![To audit a whole GPT-2 layer, latents need five times the audits and give ten times the VERIFIED units](media/layer.svg)

| | Transcoder latents | Positive neuron poles |
| --- | --- | --- |
| Units per layer | 24,576 | 3,072 |
| Share that fire on the scan | 0.66 | 1.00 |
| Audits to cover every live unit once | 16,115 | 3,072 |
| VERIFIED units to expect | 14,907 | 1,536 |

This is our step, not a measurement. It applies the shares above to a whole layer, and assumes a dead draw costs only a scan.

- **Many more units.** A dictionary splits a layer into many more units, and more of them are narrow, so it gives many more VERIFIED explanations.
- **A partial copy.** It explains its own reconstruction of the MLP, which held 70% of the MLP output's variance on the scanned layers.
- **What a neuron gives.** A neuron is part of the model itself. One VERIFIED pole covers less of its layer's work than its latent counterparts together, but it covers it exactly.

## Caveats

- **One dictionary on one model.** The per-unit comparison is GPT-2 small's transcoders against its positive poles. The basis null is on Qwen2.5-0.5B. The two findings come from different models.
- **Protocol 0.7.** The registry has moved to protocol 0.9 since then. Its confound baseline now includes probe length, and a re-review could move both arms.
- **Negative poles left out.** GPT-2's negative poles can't pass this protocol, because most of them are default-on. Counting them would lower the native share.
- **One agent pair.** Verdict rates move with the agents. Both arms had the same pair, so the comparison holds within it.
- **The SPD result is from our own runs.** It used a decomposition with as many subcomponents as neurons. The [SPD baseline](../../docs/analyses/spd-baseline.md) and [the SPD paper](../spd-input-projections/README.md) give its limits.

## Reproduce

The audits are records under `wiki/experiments/`, and the draws are listed in `docs/analyses/transcoder-baseline.jsonl` and `docs/analyses/null-calibration-basis.jsonl`, with their seeds. Then:

    uv run python scripts/native_vs_learned.py --out papers/native-vs-learned/results.jsonl
    uv run python scripts/paper_figures.py

The first command downloads the layer-0 transcoder to read its width.

## Next

- A transcoder or SAE baseline on Qwen2.5-0.5B, with the dictionary `sackfield/qwen2.5-0.5b-transcoders`, so that the per-unit comparison and the basis null share a model.
- A measure of how much of a layer a set of VERIFIED units explains, for neurons and latents alike.
- Both arms again under protocol 0.9.

## References

- Bushnaq, L., Braun, D., & Sharkey, L. (2025). *Stochastic parameter decomposition*. [arXiv:2506.20790](https://arxiv.org/abs/2506.20790).
- Dunefsky, J., Chlenski, P., & Nanda, N. (2024). *Transcoders find interpretable LLM feature circuits*. NeurIPS 2024. [arXiv:2406.11944](https://arxiv.org/abs/2406.11944).
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language models are unsupervised multitask learners*. [OpenAI](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

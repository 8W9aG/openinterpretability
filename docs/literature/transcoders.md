# Transcoders Find Interpretable LLM Feature Circuits

**Citation.** Jacob Dunefsky (Yale University), Philippe Chlenski (Columbia University) and Neel Nanda. "Transcoders Find Interpretable LLM Feature Circuits". *NeurIPS 2024*. [arXiv:2406.11944](https://arxiv.org/abs/2406.11944), version 1 of 17 June 2024, revised as version 2 on 6 November 2024. Code is at [github.com/jacobdunefsky/transcoder_circuits](https://github.com/jacobdunefsky/transcoder_circuits/). The text we read doesn't name the Hugging Face repository of the released transcoders.

**Mismatch.** Section 5.1 says the authors "carried out nine blind case studies", and a later sentence says "six of the eight case studies".

**Status.** Used. The registry's learned-basis baseline ([`docs/analyses/transcoder-baseline.md`](../analyses/transcoder-baseline.md)) audits random latents of these GPT-2 small transcoders against random neuron poles, and [`docs/dictionaries.md`](../dictionaries.md) lists them. `papers/native-vs-learned` cites the paper.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 2 for this page's author, a model (Claude Opus 5.5). It read the main text in the source's own words, Sections 1 to 7, with Appendices A, B and the start of C. The rest of the appendices came back only as a summary, so this page uses no number from them. The PDF came back as bytes that it couldn't read.

## Summary

**Claim.** A transcoder stands in for an MLP: a wider, sparse MLP trained to give the original MLP's output from its input. Its latents are about as interpretable as a sparse autoencoder's. It replaces the MLP's computation, so circuits can be read through the MLP in a way that doesn't depend on the input.

**Method.**

- **Definition.** Transcoders are "wide ReLU MLPs with one hidden layer that are trained to faithfully approximate the original narrower MLP sublayer's output". The loss is the squared error against the MLP's output plus an L1 penalty on the latents.
- **Against SAEs.** An SAE reconstructs its own input. A transcoder maps the MLP's input to the MLP's output.
- **Models.** GPT-2 small (a transcoder for each layer), Pythia-410M and Pythia-1.4B. Each dictionary is 32 times the model's width, so 24,576 latents for GPT-2 small.
- **Circuits.** The attribution from one latent to a later one is its activation times a dot product of decoder and encoder weights. The dot product doesn't depend on the input.
- **Tests.** A blind rating of 50 transcoder latents against 50 SAE latents, sparsity against fidelity curves, and case studies, including GPT-2's greater-than circuit.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Blind rating, Pythia-410M layer 15 (Table 1) | Interpretable: 41 of 50 transcoder latents, 38 of 50 SAE latents. Uninterpretable: 1 against 4 |
| Sparsity against fidelity (Section 4.2.2) | "transcoders are equal to or better than SAEs", with the gap widening on larger models |
| Greater-than circuit (Section 5.2.2) | Fewer than 24 transcoder latents beat the same number of neurons, then performance drops at latent 5315 |
| Faithfulness (Section 7) | About as unfaithful as SAEs, by cross-entropy loss |

## Where the claims go beyond the evidence

- **One rater, one layer.** The interpretability study rates 100 latents of one Pythia layer, with no agreement between raters.
- **No full sweeps.** The authors say "compute limitations prevented us from performing more exhaustive hyperparameter sweeps", and both methods used one learning rate.
- **Qualitative circuits.** The authors say their circuit results are "a few qualitative case studies".
- **Attention is held fixed.** The circuit method doesn't explain how attention patterns arise.

## Use in this repo

- **Our citations check out.** `papers/native-vs-learned` cites the paper three times.
  - A transcoder is "a dictionary trained to stand in for the MLP". That matches the definition.
  - It quotes "to faithfully approximate the original narrower MLP sublayer's output" from Section 1. The words and the section match.
  - It says the GPT-2 transcoders have 24,576 latents per layer, read from the weights. That agrees with the paper's rule of 32 times the width.
  - The three authors, 2024 and NeurIPS 2024 match.
- **The registry ran the comparison the source didn't.** The source compares transcoder latents with SAE latents, and with neurons only on one circuit. The registry's baseline audits random transcoder latents and random neuron poles under one protocol: 92% of live latents were verified, against 50% of positive poles ([`docs/dictionaries.md`](../dictionaries.md)).
- **The context length limits the baseline.** The source trained on contexts of 128 tokens. The repo's dictionary table already notes that these transcoders are only valid through position 128. That figure comes from the appendix summary, so we didn't check it against the source's own text.
- **No question for the maintainer.**
- **Related pages.** [`neurons-as-an-interpretable-basis.md`](neurons-as-an-interpretable-basis.md) and [`sparse-feature-circuits.md`](sparse-feature-circuits.md) review neuron and dictionary circuits.

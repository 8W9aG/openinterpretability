# Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers

**Citation.** Thomas Heap, Tim Lawson, Lucy Farnik and Laurence Aitchison, University of Bristol. "Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers". [arXiv:2501.17727](https://arxiv.org/abs/2501.17727). Version 1, of 29 January 2025, was titled "Sparse Autoencoders Can Interpret Randomly Initialized Transformers". Version 2, of 27 January 2026, has the current title. Preprint, no venue given. The text links no code of its own. It builds on EleutherAI's `sparsify` and `delphi`.

**Mismatch.** Version 2 hedges version 1. Version 1's abstract says random and trained transformers "produce similarly interpretable SAE latents". Version 2 says the metrics are "surprisingly similar" under certain conditions, and adds a limitations section.

**Status.** Planned. #5758 would run the repo's explanation scores on the trained and random-init GPT-2 units of `docs/analyses/untrained-null.md`. #5425 plans an untrained control for the concept map.

**Read.** On 2026-11-21 a `web-fetch` subagent read the abstract page and the HTML of both versions for this page's author, a model (Claude Opus 5.5). It had the whole text of each in the source's own words, through Appendix L of version 2. It saw the figure captions but not the figures, and the text gives few numbers.

## Summary

**Claim.** Sparse autoencoders trained on randomly initialised transformers give latents whose auto-interpretability scores are about as high as on trained ones. So these scores alone don't show that a model learned a feature.

**Method.**

- **Models.** Pythia from 70M to 6.9B parameters. Each comes trained, re-randomised with or without its embeddings, and at step 0. A control replaces the trained model's input embeddings with Gaussian noise per token.
- **SAEs.** TopK SAEs, expansion 64 and k = 32, on 100M tokens of RedPajama, at every layer of the small models.
- **Scores.** Fuzzing first, detection in an appendix, on 100 latents per SAE. One model, Llama-3.1-70B-Instruct, writes the explanations and makes the predictions.
- **Other measures.** Explained variance, cosine similarity, the cross-entropy loss recovered, and a new token-distribution entropy.

**Findings, as the source reports them.** The text gives the results in figures, with no AUROC values.

| Setting | Result |
| --- | --- |
| Fuzzing, trained against randomised Pythia (Figure 1) | "The trained model (gray line) and randomized variants (colored) overlap, whereas the control (black) is near chance" |
| Larger models (Section 3) | The gap between random and trained "was narrowed for larger models (e.g., Pythia-6.9b)" |
| Small SAEs on Pythia-1b, R = 2 and k = 4 (Appendix G) | Explained variance and loss separate them, but "the auto-interpretability scores of these SAEs remain similar" |
| Cross-entropy loss (Section 3) | For the randomised models "the loss is very poor", with or without the reconstruction |

## Where the claims go beyond the evidence

- **No numbers in the text.** The headline rests on overlapping lines in figures. No interval or test is given for the Pythia results.
- **One explainer, one scorer.** The same model writes and scores the explanations. The authors say other explainers "could yield valuable insights".
- **Latents, not neurons.** The units are SAE latents. The paper doesn't study the model's own neurons.
- **No held-out or adversarial test.** Fuzzing and detection score an explanation on the latent's own activating and non-activating texts. No test is designed to break the explanation.
- **The authors hold back.** They say "we do not claim that SAEs fail to capture information" beyond what a random model gives, and call the entropy measure "preliminary".

## Use in this repo

- **The protocol's verdict does separate the two.** `docs/analyses/untrained-null.md` audited 20 GPT-2 poles on trained and on random-init weights. 9 trained units VERIFIED against 2 random ones (one-sided Fisher p = 0.015), with mean probe balanced accuracy 0.87 against 0.62. The random units failed on specificity, because the reviewers' hard negatives fired too.
- **Why the results can differ.** The paper scores an explanation on the unit's own texts. The registry's reviewer writes hard negatives and holdouts after seeing the explanation. Our reading: a broad explanation can score well on its own texts and still fail adversarial probes, which is where the random units fell.
- **The repo's scores are untested on random weights.** `open-interpretability score` gives detection, fuzzing and embedding scores, like the paper's. The untrained-null study never ran them. #5758 would, so that we know whether a high score in the registry means more than a random model gives.
- **An untrained control for the maps.** #5425 plans one for the concept map. The paper's point applies there too: a region that random poles also fill says little about what the model learned.
- **No question for the maintainer.**
- **Related pages.** [`evaluating-neuron-explanations.md`](evaluating-neuron-explanations.md) reviews sanity tests for explanation metrics, and [`beyond-top-activations.md`](beyond-top-activations.md) reviews why scoring on top activations alone misleads.

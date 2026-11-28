# Plato's Key

**Citation.** Vivien Cheng, Jerry Liu, Morris Yau and Chris Ré. "Plato's Key". *Hazy Research blog*, 25 November 2026. [Post](https://hazyresearch.stanford.edu/blog/2026-11-25-platos-key). The page prints no affiliations, only links to the authors' profiles, three of them at Stanford. It is a blog post with no peer review and no arXiv version. It states no licence, funding or tool disclosure, and links no code or data.

**Status.** Planned. #7349 asks whether a fitted direction in another model matches the repo's VERIFIED poles that have no sister neuron there. That is this post's finding, a shared structure up to a rotation, carried to the sister search. No code, skill or protocol here depends on the post.

**Read.** On 2026-11-28 a `web-fetch` subagent read the post for this page's author, a model (Claude Opus 5.5). It returned the whole body, the captions, the figure labels that are in the page's text and the acknowledgements in one fetch, with no truncation. Quotes were capped at 125 characters, so the rest came back paraphrased. Figures 1, 3, 5 and 6 are images, and we saw only their captions. The numbers of Figures 2 and 4 come from their text labels, not from the plots.

## Summary

**Claim.** One language model can read facts from another model's hidden state, through a single rotation and scale fitted in closed form, with no gradient descent. This is a functional test of the Platonic representation hypothesis. The post says that its results are consistent with a shared space, and doesn't claim that they show one.

**Method.**

- **The key.** The hidden state at the last subject token. The post treats it as the key of a key-value lookup that the early and middle layers build.
- **The map.** An orthogonal Procrustes rotation, from a thin SVD, plus one global scale. It maps the donor's centred keys to the recipient's, and allows reflections and maps between different widths. It is fit on fewer than 5,000 facts that both models recall alone.
- **The swap test.** The donor's key for one subject ("France"), mapped into the recipient, replaces the recipient's key for another subject ("Germany") at one layer. The swap succeeds if the recipient then gives the donor's answer (Paris). The self-swap ceiling does the same within the recipient, with its own key.
- **Data and models.** 8,800 facts over 16 relations, split by subject. 12 base models from eight families, from 1.5B to 27B, among them Llama-3.1-8B, Gemma-2-27B, Qwen2.5-1.5B to 14B, Mistral-7B and DeepSeek-7B, in 20 directed pairs.
- **Layer choice.** The donor and recipient layers are chosen by transfer accuracy on a validation split, and the test split is reported. The alternative is the mutual nearest-neighbour similarity of the Platonic representation paper.
- **Controls.** Random and shuffled rotations. Ridge, whitened rotations, a trained MLP and Wasserstein Procrustes as other maps, compared by tests with the Holm adjustment.

**Findings, as the source reports them.**

| Result | Setting | Value |
| --- | --- | --- |
| Transfer accuracy of the rotation | 20 pairs, test facts | Median 41% (34–46%), against 55% for the self-swap. A median 73% of the self-swap (60–93%) |
| The best control | Random and shuffled rotations, each pair | 0.6%–2.2% |
| Across and within families | Share of the self-swap | 75% across families, 69% within |
| Other maps against the rotation | 20 pairs, $p$ < 0.05 after the Holm adjustment | Ridge, unscaled, the whitened rotation and the trained MLP beat it in 4 of 60 comparisons. Rescaled ridge beats it in 4 of 20 |
| Chains of rotations | 47 chains through a third model | Median 99% of the direct accuracy. None below 92% |
| Compression | Keys kept to their top principal components | 256 components keep a median 74% of the accuracy. 1,024 keep 90% |
| Layers chosen by mutual nearest-neighbour similarity | Same validation split | Median transfer 4.5%, about a tenth of the transfer-chosen layers. That similarity peaks in the first 5% of the recipient's depth, transfer at 22–38% |
| Dark memories | 1,463 facts that a recipient fails to recall but stores, by one of three tests | The rotated donor key recovers 267 (18%). No control recovers more than 1.1% |

**Where the claims go beyond the evidence.**

- **One task.** Every result is factual recall of a subject's attribute, in base models. The post names multi-hop reasoning, post-trained models and larger models as open.
- **Layers chosen on validation.** The main layers are picked to maximise transfer on a validation split. The test split is separate, so this is fair, but it is one choice per pair from many layer pairs.
- **No intervals.** Medians and ranges are given, and only the comparison of maps has a significance test. The post gives no intervals for the transfer or the dark-memory counts, and no split sizes.
- **A rotation of whole vectors.** The map acts on whole hidden states. The post makes no claim about single neurons or units.
- **No code.** Nobody can rerun the study as published.

## Use in this repo

- **The sister search compares units, the post compares whole spaces.** The sister search matches each VERIFIED pole with the single neuron of another model whose activity correlates best with it. It checks the match against a *random* rotation of that layer (`docs/analyses/sister-neurons.md`). The post fits a rotation of the whole space and finds that it carries function across families. The two answer different questions, and the post's result doesn't weaken the sister search's reference. A random rotation asks whether the neuron basis matters, and a fitted rotation asks whether the models share a space at all.
- **Few sisters across families may not mean few shared concepts.** Only 4% of Qwen2.5-0.5B's VERIFIED poles have a GPT-2 sister, and 12% of GPT-2's have a Qwen one, against 20% between Qwen2.5-0.5B and 1.5B. The post suggests that much is shared across families up to a rotation. So a concept with no sister may still exist in the other model as a direction that no single neuron carries. #7349 tests this, with a fitted map from the other model's MLP activations to the pole.
- **Layer choice.** The post finds that a geometric similarity score picks layers far too early for transfer. The sister search doesn't pick a layer by similarity. It searches every layer, and its sisters land at about the same relative depth, a median gap of 0.045. So that finding doesn't bear on it.
- **The shared concept map.** `papers/shared-concept-space/` places the poles of four models in one map by the meaning of their text. It finds that they cover the same concepts when all four read one corpus. That is a test of shared content. The post's swap is a test of shared function, and the two agree in direction.
- **Dark memories and fixing models.** Rescuing a stored fact by changing its key at inference time, not the weights, is a kind of fix that `fixing-models.md` doesn't yet cover. The fix campaign is the maintainer's, so this page records it as evidence only.
- **The cited works.** #7347 reviews Kornblith et al.'s CKA paper, which the post cites for geometric similarity. #7348 reviews *Inside-Out*, the source of the dark memories. *MLPs are Hebbians*, which the same lab wrote, is #7322.

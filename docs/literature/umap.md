# UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction

**Citation.** Leland McInnes and John Healy (Tutte Institute for Mathematics and Computing) and James Melville. "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction". [arXiv:1802.03426](https://arxiv.org/abs/1802.03426), version 1 of 9 February 2018, last revised as version 3 on 18 September 2020. Preprint, no venue given. The reference implementation is at [github.com/lmcinnes/umap](https://github.com/lmcinnes/umap), the `umap-learn` package.

**Mismatch.** None found.

**Status.** Used. `openinterpretability/concept_map.py` lays out the shared concept map with UMAP, and `scripts/literature_map.py` lays out the literature map. `papers/shared-concept-space` cites it for the layout.

**Read.** On 2026-11-21 a `web-fetch` subagent read the abstract page and the HTML of version 3 for this page's author, a model (Claude Opus 5.5). It had the abstract and Sections 1 and 2 in the source's own words. The page's author then took the text of the 63-page PDF with `pypdf`, and read Sections 5.3, 6 and 7 and Table 2 from it. Sections 3, 4 and the appendices weren't read word for word, and this page uses nothing from them.

## Summary

**Claim.** "UMAP (Uniform Manifold Approximation and Projection) is a novel manifold learning technique for dimension reduction". It rests on Riemannian geometry and algebraic topology, and the authors say it "arguably preserves more of the global structure" than t-SNE.

**Method.**

- **The graph.** Each point's k nearest neighbours define a local metric. In it, "a ball centered at X_i that contains exactly the k-nearest-neighbors of X_i should have approximately fixed volume". The local views are merged by a fuzzy union.
- **The layout.** "UMAP then optimizes the layout of the data representation in the low dimensional space", by fuzzy-set cross-entropy, with stochastic gradient descent.
- **Comparisons.** t-SNE, LargeVis, Laplacian eigenmaps and PCA, with a k-nearest-neighbour classifier on the embedding as one measure.

**Findings, as the source reports them.**

| Setting (Table 2, kNN classifier accuracy on the 2-D embedding) | UMAP | t-SNE |
| --- | --- | --- |
| MNIST, k = 100 | 0.967 | 0.967 |
| Shuttle, k = 3200 | 0.957 | 0.828 |
| Fashion-MNIST, k = 100 | 0.790 | 0.818 |

UMAP keeps more accuracy than t-SNE at large k, which reads wider neighbourhoods. t-SNE matches or beats it at small k.

## Where the claims go beyond the evidence

- **Global structure is argued, not measured.** The authors write that "there is a great deal of uncertainty surrounding the preservation of global structure", and that it lacks clear measures. They credit much of UMAP's global structure to its Laplacian eigenmaps start.
- **Layouts vary.** Section 5.3 says "the resulting embedding is necessarily different from run to run, and under sub-sampling of the data."
- **Structure in noise.** Section 6 says UMAP "can tend to find manifold structure within the noise of a dataset", and warns about small, noisy samples.
- **Small datasets.** Its approximations "can have an impact on the results of UMAP for small (less than 500 samples) dataset sizes".
- **Densities are evened out.** A loose region and a dense one are put "on an even footing", so sizes and gaps on the map don't show density.

## Use in this repo

- **Our citation checks out.** `papers/shared-concept-space` calls UMAP "a manifold-learning method for dimension reduction". The paper's own words are "a novel manifold learning technique for dimension reduction". The three authors and 2018 match.
- **Distances between regions mean little.** `concept_map.py` already says so, and that layouts kept about half of each point's neighbours. The paper agrees: it is built for local structure, evens out density, and is unsure about global structure. Our reading: the map's regions come from Leiden on the neighbour graph, not from the layout, which is the right way round.
- **Adding points is not in the paper.** `concept_map.py` places a new pole at the weighted mean of its nearest old poles, and says UMAP's `transform` starts a new point the same way. That describes the `umap-learn` library. The paper lists adding "new unseen data points into an existing embedding" as future work (Section 7). The comment cites no paper, so nothing needs to change.
- **The literature map is small.** It lays out about 300 points: some 180 reviews and 120 queued papers. That is under the 500 samples where the authors warn that approximations can give "suboptimal embeddings". Our reading: its layout may change more between runs than the concept map's, which has 2,031 poles. Its fixed seed keeps one build stable, but not a rebuild with more pages.
- **No question for the maintainer.**
- **Related pages.** [`platonic-representation-hypothesis.md`](platonic-representation-hypothesis.md) reviews the other work that `papers/shared-concept-space` builds on.

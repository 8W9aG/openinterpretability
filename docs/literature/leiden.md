# From Louvain to Leiden: Guaranteeing Well-Connected Communities

**Citation.** Vincent A. Traag, Ludo Waltman and Nees Jan van Eck, Centre for Science and Technology Studies, Leiden University. "From Louvain to Leiden: guaranteeing well-connected communities". *Scientific Reports* 9: 5233 (2019), [doi:10.1038/s41598-019-41695-z](https://doi.org/10.1038/s41598-019-41695-z). The preprint is [arXiv:1810.08473](https://arxiv.org/abs/1810.08473), version 1 of 19 October 2018, last revised as version 3 on 30 October 2019. Code is in Java at [github.com/CWTSLeiden/networkanalysis](https://github.com/CWTSLeiden/networkanalysis), and in Python at [github.com/vtraag/leidenalg](https://github.com/vtraag/leidenalg).

**Mismatch.** None found. The journal's page sent us to a login, so the journal details come from the arXiv listing.

**Status.** Used. `concept_map.clusters` and `scripts/literature_map.py` find the regions of both maps with `leidenalg`, the authors' Python package.

**Read.** On 2026-11-21 a `web-fetch` subagent read the arXiv abstract page and the HTML of version 3 for this page's author, a model (Claude Opus 5.5). It had the abstract, Sections I to III, the start of Section IV and Table 2 in the source's own words. The benchmarks, the discussion and the appendices came back only as a summary, and this page uses nothing from them.

## Summary

**Claim.** The Louvain algorithm "may yield arbitrarily badly connected communities", and even disconnected ones. The Leiden algorithm "yields communities that are guaranteed to be connected", and it is faster.

**Method.**

- **Phases.** "(1) local moving of nodes, (2) refinement of the partition and (3) aggregation of the network based on the refined partition". The refinement merges nodes at random, weighted by the gain in quality, with randomness set by θ.
- **Guarantees.** After each iteration, all communities are γ-separated and γ-connected. After a stable iteration, nodes are locally optimal and communities subpartition γ-dense. Iterated to convergence, communities are "uniformly γ-dense" and "subset optimal", the strongest guarantee.
- **Quality functions.** Modularity with a resolution γ, where "Higher resolutions lead to more communities", and the Constant Potts Model (CPM). The paper notes that CPM, unlike modularity, doesn't suffer from the resolution limit.
- **Tests.** Six empirical networks of 0.3M to 39M nodes, with modularity at γ = 1, and benchmark networks.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Louvain, first iteration (Section IV.1) | "on average respectively 23%, 16% and 14% badly connected communities" for Amazon, DBLP and Web UK |
| Louvain, more iterations (abstract) | "up to 25% of the communities are badly connected and up to 16% are disconnected" |
| Maximum modularity, Louvain against Leiden (Table 2) | DBLP 0.8262 against 0.8387, Amazon 0.9301 against 0.9341, Live Journal 0.7653 against 0.7739 |
| Speed (Section IV) | "up to 20 times faster than Louvain in empirical networks" |

## Where the claims go beyond the evidence

- **Quality is a proxy.** The paper shows that Leiden's partitions score higher and are connected. Whether they match true groups is tested only on benchmark networks, which we didn't read.
- **Randomness.** The refinement and the node order are random. The paper uses θ = 0.01 and says values from about 0.0005 to 0.1 "all provide reasonable results".
- **Two quality functions, no choice made.** The guarantees hold for "either modularity or CPM". The text we read doesn't say which one to use.

## Use in this repo

- **Our citation checks out.** `papers/shared-concept-space` says Leiden "finds communities that are guaranteed to be connected". That is the abstract's claim. The three authors, 2019, *Scientific Reports* 9 and article 5233 match the arXiv listing.
- **The maps get the connected guarantee, not the strongest.** `concept_map.clusters` and `scripts/literature_map.py` call `leidenalg.find_partition` with its default of two iterations. So every region is connected, which holds after each iteration. The stronger guarantees hold only once the algorithm stops changing the partition. Our reading: for maps of about 2,000 and 300 points, more iterations would cost little. It could also merge or split a few regions.
- **Modularity, with the resolution searched.** Both maps use `RBConfigurationVertexPartition`, which is modularity with a resolution γ, and search γ for a target count of regions. The paper says modularity has a resolution limit and may "hide" smaller communities, which CPM doesn't. Our reading: aiming at a fixed count sidesteps the choice of γ, but a small theme can still merge into a larger region.
- **A fixed seed.** Both maps fix Leiden's seed, so a run is repeatable. The paper's randomness means another seed could give other regions. Neither map measures that spread.
- **No question for the maintainer.**
- **Related pages.** [`umap.md`](umap.md) reviews the layout that the regions are drawn on.

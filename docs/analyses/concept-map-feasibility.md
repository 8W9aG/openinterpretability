# A concept map of a model's neurons: feasibility

Issue #3290 asked whether model pages could show a concept map, in the style of "2.5 million cancer research papers, mapped". Each neuron would be placed by the meaning of the text it fires on, then clustered and labelled by theme. This study tested that idea on the registry's own records, before anything is built. The numbers are in [concept-map-feasibility.jsonl](concept-map-feasibility.jsonl), from [scripts/concept_map_study.py](../../scripts/concept_map_study.py).

**The recommendation is to build it, in two steps.** First comes a map of the studied neurons, which is cheap and works with today's records. A full-model map waits until the first step shows that a reader or the planner uses it.

## Method

- **The text of a pole** is its newest record's top examples, up to 10 of them. Each is cut to ±20 tokens around the peak token, which is marked.
- **The embedding** is all-MiniLM-L6-v2, a local sentence-embedding model. all-mpnet-base-v2 is the check of the model choice.
- **A neuron's vector** is the mean embedding of its positive pole's texts minus the mean of its negative pole's, then L2-normalised. A missing pole counts as zero. The norm before normalising is kept as a confidence.
- **The test** is the neuron's own explanation. Each pole's `natural_language_explanation` is embedded and combined the same way. A good map puts a neuron's context vector close to its own explanation's vector.
- **The models** are the two with the most records: GPT-2 (821 neurons) and Qwen2.5-0.5B (846 neurons), from 8,390 and 9,500 texts.

## Results

1. **The contexts carry the meaning that the explanations name.** A neuron's context vector picks out its own explanation, among all of them, far more often than chance:

   | Model | Neurons | Own explanation first | In the top 10 | Chance (first / top 10) | Median rank |
   | --- | --- | --- | --- | --- | --- |
   | GPT-2 | 821 | 11.4% | 32.9% | 0.12% / 1.2% | 37 |
   | Qwen2.5-0.5B | 846 | 16.5% | 39.8% | 0.12% / 1.2% | 21 |

   Over random pairs of neurons, the context cosine and the explanation cosine correlate with Spearman ρ 0.61 (GPT-2) and 0.80 (Qwen2.5-0.5B). So neurons that the map puts close together mostly have explanations that say similar things.

2. **The smaller embedding model works better.** With all-mpnet-base-v2, a neuron finds its own explanation first 7.9% (GPT-2) and 11.1% (Qwen2.5-0.5B) of the time, against 11.4% and 16.5% with MiniLM. The two models agree on only 53–56% of each neuron's 10 nearest neighbours, so the choice of embedding changes the map's fine detail.

3. **Clusters group neurons with similar explanations across layers, at least on Qwen.** Leiden clustering on the 10-nearest-neighbour graph gives 8 clusters on GPT-2 and 5 on Qwen2.5-0.5B, with modularity 0.62–0.68. The test pairs are neurons in different layers whose explanations have a cosine above 0.6.

   | Model | Such pairs | Share in one cluster | Random pairs in one cluster |
   | --- | --- | --- | --- |
   | GPT-2 | 16,873 | 24% | 15% |
   | Qwen2.5-0.5B | 10,816 | 56% | 22% |

   The k of the graph (10 or 15) makes almost no difference. With so few studied neurons, the clusters are coarse themes, not fine concepts.

4. **One-pole neurons clump, and subtracting the corpus mean fixes it.** Most studied neurons have only one studied pole: 803 of 821 on GPT-2 and 742 of 846 on Qwen2.5-0.5B. The issue expected those to share a generic "web text" direction. On GPT-2 they did: the mean cosine among one-pole neurons was 0.088, against 0.016 among two-pole neurons. Subtracting the corpus-mean embedding from each pole's mean removed it, to 0.0004. On Qwen2.5-0.5B the clumping was already near zero (0.0002). Subtract the mean.

5. **Most two-pole neurons have unrelated poles.** For neurons with both poles studied, the median cosine between the two poles' mean texts is 0.20 (GPT-2, 18 neurons) and 0.19 (Qwen2.5-0.5B, 104 neurons). Most of them are below 0.3: 72% and 76%. So a neuron's point usually lies between two unrelated concepts. A map should draw each studied pole as its own point too, joined to its neuron, rather than only the difference.

6. **The 2-D layout is only moderately stable.** Across two UMAP seeds, a point keeps 52–58% of its 10 nearest 2-D neighbours. Across k = 10 and 15 it keeps 45–53%. So the local neighbourhoods hold, but the layout is not one fixed picture. As with any UMAP, distances between clusters mean little, and the page must say so.

7. **The cost is modest.** Embedding takes 1.8–2.2 ms per text on this machine's CPU. A full Qwen2.5-0.5B map, of 116,736 neurons × 2 poles × 10 texts, is 2.3 million texts, or about 1.2–1.4 hours of embedding. It also needs one pass over the corpus to collect the top contexts of every neuron, which is about one scan. UMAP on 116,736 points with a fixed `random_state` took 79 s here. That was on random vectors, a hard case for UMAP, and it ran single-threaded because the seed is fixed. With all-MiniLM-L6-v2's 384 numbers per neuron, the vectors are about 180 MB. The coordinates, cluster ids and norms to commit are a few MB.

## What this does not show

- **Only studied neurons were placed.** Every one of them passed a scan, and some were chosen, not drawn. The unstudied majority may map less cleanly. Dead neurons, and position or frequency neurons, would need their own regions or to be left out, as the issue said. This study had none of them.
- **The test is the registry's own explanations.** Those were written from the same top examples, so part of the agreement is shared input, not independent meaning. The Spearman correlation and the cross-layer clusters are the stronger evidence.
- **The labelling of clusters was not tested.** Naming a cluster from its members' VERIFIED explanations, or from an LLM summary marked as unverified, is a design choice for the build.

## Recommendation

1. **Build a map of the studied neurons now**, as an offline command (`open-interpretability map <model>`) with an optional dependency group: sentence-transformers, umap-learn and leidenalg.
   - Use MiniLM and centred pole means. Draw each studied pole as its own point, joined to its neuron.
   - Commit the coordinates, cluster ids and norms, so the site build stays light and deterministic. Render the map as SVG, since there are a few thousand points.
   - Show each point's status by shape as well as colour, so the map doesn't rest on colour alone.
   - Name clusters from their VERIFIED members only. Mark any other name as unverified.
2. **Hold the full-model map** until the first map shows use, for example a planner draw from studied clusters that have no VERIFIED member. A full map costs one corpus pass, about 1.3 hours of embedding and under 2 minutes of layout per model. It also needs a density image under an SVG overlay, because 117,000 points are too many for an SVG.

## Reproduce

```bash
uv run --with sentence-transformers --with umap-learn --with leidenalg \
    python scripts/concept_map_study.py
```

The registry's records and the two embedding models are its only inputs. The UMAP seeds are 0 and 1, and the Leiden seed is 0.

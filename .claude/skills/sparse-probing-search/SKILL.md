---
name: sparse-probing-search
description: Feature-first neuron search with sparse probing. Given a labelled concept (e.g. "is Python code", "French text", "inside a quotation"), find the neurons that encode it, then audit them. Use when the user wants neurons for a specific concept rather than random ones, or to cross-check a verified neuron against a labelled dataset.
---

# Sparse probing search

## Papers

- Gurnee, Nanda, Pauly, Harvey, Troitskii & Bertsimas, 2023, *Finding Neurons in a Haystack: Case Studies with Sparse Probing*, arXiv:2305.01610. Train k-sparse linear probes on neuron activations for labelled features, then sweep k. Features that a k=1 probe recovers well have a dedicated neuron; others are spread across neurons. Early layers show more superposition, middle layers more dedicated neurons.
- Bolukbasi et al., 2021, *An Interpretability Illusion for BERT*, arXiv:2104.07143. A concept found on one dataset may not transfer to another. Always validate on a second dataset.

## Procedure

1. **Build a labelled set.** Collect token-level or sequence-level examples with and without the concept from **two independent sources** (e.g. GitHub vs StackOverflow for "Python code"). Balance the classes. For concepts with human annotations, `openinterp concepts` writes them as spans, with a sidecar pinning the source:
   ```bash
   uv run openinterp concepts --dataset ud-upos                     # list the labels and counts
   uv run openinterp concepts --dataset ud-upos --label PROPN --out propn-ud.jsonl
   uv run openinterp concepts --dataset conll2003-pos --label NNP --out nnp-conll.jsonl  # second source
   ```
   Presets: `conll2003-ner` (PER, ORG, LOC, MISC), `conll2003-pos` (Penn tags), `ud-upos` and `ud-deprel` (`--subset <treebank>`, e.g. `en_gum`, `fr_gsd`), `language-identification` (20 languages) and `ag_news` (World, Sports, Business, Sci/Tech). Document classes are balanced already; word tags are not. The files are `probe-compare`'s `--train`/`--test` format. Pairs of sources: UD `en_ewt` vs `en_gum` or CoNLL Penn tags for parts of speech; `language-identification` vs a `wikipedia-multilingual` corpus for language. NER and topic have one preset each, so take their second source from elsewhere.
2. **Cache activations.**
   ```python
   _, cache = runner.run_with_cache(batch, layers)  # cache[layer]: [b, seq, d_mlp]
   ```
   Mask padding with `attention_mask`. For sequence-level labels, max-pool over tokens; pool per pole by applying `pole_strength` to each neuron first.
3. **Rank neurons (k=1).** For each neuron and pole, compute the AUROC separating the classes. That uses only torch and avoids adding a dependency:
   ```python
   def auroc(scores, labels):  # scores [n], labels bool [n]
       order = scores.argsort()
       ranks = torch.empty_like(order, dtype=torch.float)
       ranks[order] = torch.arange(len(scores), dtype=torch.float)
       pos = labels.sum()
       return (ranks[labels].sum() - pos * (pos - 1) / 2) / (pos * (~labels).sum())
   ```
4. **Hold out the second dataset.** Keep only neurons whose AUROC holds within about 0.05 on source B.
5. **Audit the survivors** with `openinterp audit --layer L --neuron N --pole P`. The probe's concept is a *prior*, not the hypothesis. Run the full scan, because the neuron may encode something narrower or broader.

## Fit with the protocol

- Record `provenance.selection = "manual"` and name the probe dataset in the explanation. Probe-selected neurons must never count towards random-sample registry statistics.
- The two-dataset check is the probing equivalent of the reviewer's positive holdouts. Reviewers should draw holdouts from yet another source.

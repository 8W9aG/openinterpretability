# Generalization through Memorization: Nearest Neighbor Language Models

**Citation.** Urvashi Khandelwal (Stanford University, work done while interning at Facebook AI Research), Omer Levy (FAIR), Dan Jurafsky (Stanford), Luke Zettlemoyer (FAIR) and Mike Lewis (FAIR). "Generalization through Memorization: Nearest Neighbor Language Models". [arXiv:1911.00172](https://arxiv.org/abs/1911.00172), version 2 of 15 February 2020. Version 1 is of 1 November 2019. The arXiv comment says "ICLR 2020". We didn't check the proceedings. The code is at [github.com/urvashik/knnlm](https://github.com/urvashik/knnlm). The text gives no funding statement and no disclosure.

**Mismatch.** Section 2 says the distance is squared L2 throughout. Section 3 says the runs other than Wikitext-103 use plain L2 on quantised keys. Table 4's caption claims a gain of "23 points", but the datastore's own gain in that table is 14.37. The 23 points is the gap to a model trained in the domain. Several figure and table numbers in the text point to the wrong figure.

**Status.** Planned. #5220 names it as the reference for arm H, a written side table beside a frozen model. The fix campaign is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 2 for this page's author, a model (Claude Opus 5.5). It read the source's own text in full, in two fetches: Sections 1 to 8, Tables 1 to 9, Appendix A and the references. We didn't see any figure as an image. So the curves of λ against datastore size and of perplexity against k appear here only as the text describes them. We didn't read version 1.

## Summary

**Claim.** Mix a frozen language model's next-token distribution with one from a nearest-neighbour search over its own hidden states. The model then improves. Retrieval over 3 billion tokens beats training on them.

**Method.**

- **The datastore.** One entry for each target token in the data. The key is the hidden state of the context before the token, and the value is the token.
- **The key.** The input to the last layer's feedforward block, after layer norm: "representations fed to the feedforward network in the final layer ... (after self-attention and layernorm)". It has 1,024 dimensions.
- **The search.** The $k = 1024$ nearest keys by L2 distance, with FAISS. It learns 4,096 centroids from 1 million keys and quantises keys to 64 bytes. Each query probes 32 centroids.
- **The kNN distribution.** A softmax over the negative distances of the neighbours, with the probability of each neighbour added to its token. A token that no neighbour holds has probability 0.
- **The mix.** $p(y \mid x) = \lambda\, p_{\mathrm{kNN}}(y \mid x) + (1 - \lambda)\, p_{\mathrm{LM}}(y \mid x)$, for every token. λ is tuned on validation data: 0.25 for Wikitext-103 and 0.65 for domain adaptation.
- **Data.** Wikitext-103 (103 million tokens), Books (0.7 billion), Wiki-3B (about 2.87 billion) and Wiki-100M, a 100-million-token subset of it.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Wikitext-103, test perplexity (Table 1, median of 3 seeds) | base model 18.65, kNN-LM 16.12, with a continuous cache 15.79 |
| Books, test (Table 2) | base 11.89, kNN-LM 10.89 |
| Wiki-100M model, Wiki-3B datastore, test (Table 3) | 13.73, against 15.17 for a model trained on Wiki-3B |
| Wiki-3B model on Books, test (Table 4) | 34.84 alone, 20.47 with a Books datastore, 11.89 for a model trained on Books |
| Key choice, Wikitext-103 dev (Table 5) | last FFN input after layer norm 16.06. Before layer norm 17.06. Attention input after layer norm 16.76. Model output 17.07. No datastore 17.96 |
| Cost (Section 3) | about 2 hours on one CPU for the Wikitext-103 store. About 25 minutes for the validation set with 1,024 neighbours |
| Where it helps (Section 6) | "examples where kNN-LM is most helpful typically contain rare patterns": names, facts and near-copies of training text |

## Where the claims go beyond the evidence

- **Single runs, except one table.** Only Table 1 gives a median of three seeds. No other result has a seed count, a variance or an interval.
- **The token analysis is a handful of examples.** Section 6 reads cases where the kNN term beat the model by hand. It doesn't say how many it read, and Appendix A shows four.
- **No case where it hurts is reported.** The mix applies to every token, so a wrong neighbour moves every prediction. The paper reports only averages, and never the tokens that got worse.
- **Domain adaptation is far from training in the domain.** The Books datastore takes perplexity from 34.84 to 20.47. A model trained on Books reaches 11.89. The caption's "approaching in-domain training" overstates this.
- **Memory is not reported.** The paper gives time costs but no size for the store.

The authors name future work themselves: learning the similarity function, and making the datastore smaller.

## Use in this repo

- **The issue's three questions, as the source answers them.**
  - *Which hidden state is the key:* the last layer's FFN input after layer norm. It beats the attention input, the output and the states before layer norm (Table 5). The layer before the last is "slightly worse."
  - *How the store is mixed in:* a fixed λ for every token. There is no gate, so the store acts at every position.
  - *Where it hurts:* the paper reports no token that becomes worse. Its weak case is a store from another domain, which closes only part of the gap to training in that domain.
- **A fixed λ is the opposite of a gated fix.** The fix plan measures interference as the change that a fix makes where it should do nothing. A kNN term with λ = 0.25 changes every prediction by design. A side table for arm H would need a gate or a distance threshold, so that it is silent on queries far from every key. The paper doesn't test a threshold.
- **The best key is late, as the gate study found.** The [copy-gate analysis](../analyses/copy-gate.md) for #5233 finds that only the state after the last layer separates wrong copies from right ones well. A gate there catches 58% of held-out failures. It opens on 5 of 97 right copies and on none of 18,000 corpus tokens. At layers 2 and 3 the same kind of gate catches 12% to 21%. kNN-LM's key is also the last layer's state. Arm H's table sits after layer 4 of pythia-70m's 6 layers. Our reading, not the paper's: both results suggest checking a key from the last layer before choosing a middle one.
- **Retrieval shows its evidence, and units don't.** Each kNN prediction comes with the training contexts that support it. That is a readable reason, but it is about data, not about any unit the registry explains. A side table keyed on a registry unit's activation would keep both.
- **Questions for the maintainer, recorded on this page only.**
  - Should a side table for arm H mix at a fixed weight, as kNN-LM does, or only when a key is near, as GRACE does?
  - Should its key come from the last layer, as both kNN-LM and the gate study suggest, rather than after layer 4?
- **Related pages.** [GRACE](aging-with-grace.md) keeps a codebook with a distance threshold for each key, the gated form of this idea. [SERAC](memory-based-model-editing.md) uses a trained scope classifier. [Transformer-Patcher](transformer-patcher.md) trains one key on the last FFN input, the same state that kNN-LM uses.

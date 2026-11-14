# Neurons Speak in Ranges

**Citation.** Muhammad Umair Haider (University of Kentucky), Hammad Rizwan (Dalhousie University), Hassan Sajjad (Dalhousie University), Peizhong Ju (University of Kentucky) and A.B. Siddique (University of Kentucky). "Neurons Speak in Ranges: Breaking Free from Discrete Neuronal Attribution". arXiv 2502.06809, v1 4 February 2025, v2 21 May 2025, v3 10 April 2026. [Paper](https://arxiv.org/abs/2502.06809). [Code](https://github.com/MuhammadUmairHaider/NeuronLens). *Transactions on Machine Learning Research*, 2026, [reviewed on OpenReview](https://openreview.net/forum?id=AukyIhfBuW). Funded in part by the US National Science Foundation (IIS-2401685) and the University of Kentucky.

**Status.** Not applicable. The paper's units are dimensions of the residual stream, not MLP neurons, and its concepts are the classes of labelled datasets. Its idea is that one unit carries different concepts in different ranges of its activation. That is a form of the modes that issue #3215 (polysemantic experiments) would test.

**Read.** The abstract page and the v3 HTML, on 2026-11-14. A model read the text and reported it to us, with short quotes. The body and most appendices came back as the paper's own text. Appendices H, N and O came back only as a summary or were not read, so their numbers are not here. We did not see the figures. We did not read v1 or v2.

## Summary

**Claim.** Even the most salient neurons for a concept are polysemantic. But each concept's activations in a neuron form a distinct, often Gaussian-like distribution, and those distributions overlap little. So a concept can be attributed to a *range* of a neuron's activation, not to the whole neuron. An intervention inside that range changes the concept with less damage to other concepts.

**Method.**

- **Models.** Fine-tuned BERT, GPT-2, and Llama-3.2-3B used few-shot.
- **Units.** A "neuron" is one component of a layer's hidden-state vector, the residual stream. The main experiments use the penultimate layer.
- **Concepts.** The classes of IMDB, SST-2, Emotions, AG-News and DBPedia-14. MMLU and Wikipedia perplexity measure general capability. An appendix adds token-level concepts from WikiText, such as pronouns and brackets.
- **Ranking.** Neurons are ranked per class by mean absolute activation, and the top 20–50% are kept.
- **Range.** For each kept neuron and concept: mean ± τ standard deviations of the concept's training activations, with τ = 2.5.
- **Intervention.** *Activation range masking* (ARM) sets an activation to zero only when it falls inside the concept's range. The baseline zeroes the whole neuron.

**Findings, as the text reports them.**

| Model, dataset | Whole neuron: target / other classes | Range only: target / other classes |
| --- | --- | --- |
| GPT-2, AG-News | −0.871 / −0.155 | −0.849 / −0.063 |
| GPT-2, DBPedia-14 | −0.810 / −0.154 | −0.782 / −0.015 |
| Llama-3.2-3B, AG-News | −0.934 / −0.660 | −0.935 / −0.484 |
| Llama-3.2-3B, DBPedia-14 | −0.992 / −0.912 | −0.986 / −0.663 |
| BERT, AG-News | −0.271 / +0.012 | −0.261 / +0.013 |

The cells are changes in accuracy. Range masking removes the target class about as well, and harms the other classes less. On Llama with AG-News, perplexity rises from 7.0 to 12.8 under neuron masking, and to 8.0 under range masking. MMLU stays at the base 0.53, where neuron masking drops it to 0.51. Concept-level distributions start to form "as early as layers 5-6". The paper also finds that over 60% of the top 30% neurons are shared across IMDB's classes.

## Where the claims go beyond the evidence

- **"Neurons" are residual-stream dimensions.** The title and the abstract say neurons, but the units are hidden-state components. Those are not a privileged basis in the way that MLP neurons are, so the results may not transfer to MLP neurons.
- **Concepts are classifier labels.** Range masking is judged on class accuracy, on datasets where each input has one label. A concept that is not a label, or a token-level feature, gets only an appendix.
- **The range is a Gaussian interval by assumption.** τ = 2.5 assumes a Gaussian, and the normality check uses a threshold on the KS statistic, not a test. Where two concepts' ranges overlap, the method can't separate them. The authors say so.
- **One layer.** The main results are at the penultimate layer. The authors list that as a limit.
- **No intervals or seeds.** Every table gives single point values. Table 3 reports trimmed means over classes.
- **A headline number we can't find.** The introduction says "-14 on Llama" for the gain on other classes. No row of Table 3 gives it. The largest Llama gap is about 25 points, on DBPedia-14.

## Use in this repo

**Different unit, different question.** The registry studies one pole of one MLP neuron and tests a free-text explanation of what makes it fire. This paper studies residual-stream dimensions and asks which dataset class a range of activation belongs to. Its interventions are on classifier outputs, which the registry doesn't measure.

**Its idea is a form of mode.** A pole that fires weakly on one concept and strongly on another would be polysemantic in a way that a single explanation can't capture. Our protocol counts any value above 0.25 × the pole's maximum as active (`docs/PROTOCOL.md` §4), so it treats both bands as one claim. Goodfire's mode split (`neurons-as-an-interpretable-basis.md`) groups a pole's firings by their inputs. This paper groups them by their size. Issue #3215 can test both, and a comment there records this one.

**Its result on the bulk agrees with Goodfire's.** Goodfire finds that ablating a neuron's middle activations costs almost no loss. This paper finds that zeroing a concept's range spares other concepts. Both suggest that much of a unit's function sits in specific ranges, and not in its whole output.

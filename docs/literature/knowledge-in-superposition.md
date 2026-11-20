# Knowledge in Superposition: Unveiling the Failures of Lifelong Knowledge Editing for Large Language Models

**Citation.** Chenhui Hu, Pengfei Cao, Yubo Chen, Kang Liu and Jun Zhao. The first affiliation is the Laboratory of Cognition and Decision Intelligence for Complex Systems, Institute of Automation, Chinese Academy of Sciences. The second is the School of Artificial Intelligence, University of Chinese Academy of Sciences. Both are in Beijing. "Knowledge in Superposition: Unveiling the Failures of Lifelong Knowledge Editing for Large Language Models". [arXiv:2408.07413](https://arxiv.org/abs/2408.07413), version 3 of 26 February 2025. Version 1 is of 14 August 2024, and version 2 of 12 January 2025. Venue: "To be published in AAAI 2025 (Oral)", as the arXiv comment states. The code is at [github.com/ChenhuiHu/knowledge_in_superposition](https://github.com/ChenhuiHu/knowledge_in_superposition). Funding: NSFC (U24A20335, 62176257, 62406321), the Beijing Natural Science Foundation (L243006), the Youth Innovation Promotion Association of CAS, and the China Postdoctoral Science Foundation (2024M753500).

**Mismatch.** The introduction says the interference "will accumulate linearly and eventually tend toward infinity". The text we read derives the interference as a sum but doesn't show that it diverges.

**Status.** Planned. It explains why the sequential edits of [MEMIT](memit.md) and [ROME](locating-editing-factual-associations.md) interfere. [AlphaEdit](alphaedit.md) is one answer to it. #5167 runs the registry's first rank-one edit. No code, skill or protocol here depends on it. The fix campaign (#5220) is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 2 for this page's author, a model (Claude Opus 5.5). It read the HTML in three chunks by offset. We have the source's own text for every section and appendix, and the figure captions. We didn't read version 3, which is the current one, so its text may differ. We didn't see any figure as an image. The paper gives its kurtosis values only in figures, so this page gives none.

## Summary

**Claim.** Lifelong knowledge editing fails because knowledge is stored in superposition. In a closed-form analysis of sequential edits, each edit disturbs other facts by an amount set by how much their keys overlap. The interference vanishes only if the keys are orthogonal in a whitened space. In real models they are nearly, but not exactly, orthogonal: their overlaps are heavy-tailed around zero. So interference builds up.

**Method.**

- **The edit.** As in ROME and MEMIT, the MLP's output matrix is a linear associative memory: keys $K$, values $V$, and $C = KK^\top$. A rank-one edit adds $\Lambda (C^{-1}k_e)^\top$ to the matrix.
- **The interference.** After $n$ sequential edits, an unedited fact $j$'s output moves by $\Delta o_j = \sum_i p(k_{e_i}, k_j)\, \delta(v_{e_i})$. Here $\delta(v_{e_i})$ is the change each edit makes to its own fact's value, and $p(k_i, k_j) = (C^{-1}k_i)^\top k_j / (C^{-1}k_i)^\top k_i$. Earlier edits are disturbed by later ones in the same way. Since $\delta \neq 0$, "we can only hope that $p = 0$". In whitened coordinates $C^{-1/2}k$, that means orthogonal keys.
- **The measure.** The key of a fact is the MLP's hidden activation at the subject's last token. That's 128 CounterFact facts that every model answers correctly, at every layer. $C$ is estimated from English Wikipedia. The statistic is the distribution of the off-diagonal $p$ values: its kernel density, its kurtosis, and the angles between whitened keys. Kurtosis settles by 128 facts (Appendix C).
- **Models.** Twelve: GPT-2 Small, Medium and Large, Pythia-1B, 2.8B and 6.9B, Llama2-7B and 13B, GPT-J-6B, Llama3-8B and Llama3.1-8B.
- **An editing check.** ROME on GPT-J, 16 sequential cases (Appendix H).

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Every model and layer (Figures 3, 4) | The $p$ values have "high kurtosis, zero mean, heavy-tailed" distributions, "concentrated around 0". No value is given in the text |
| Model size (Figure 5, GPT-2, Pythia, Llama2) | "As model size increases, kurtosis tends to rise". No fitted form |
| Depth | "Earlier layers exhibit higher kurtosis and thus lower degree of superposition" |
| Architecture | Pythia and GPT-2 "seem to encourage more orthogonal representations of knowledge". Llama2 "seems to encourage greater knowledge superposition" |
| Whitened against raw keys (Figure 6, Llama3-8B and 3.1-8B, last layer) | Whitened angles are "primarily concentrated around 90 degrees", with a long tail. Raw activations show "no orthogonal tendencies" |
| Most overlapping pairs, layer 0 (Tables 1 to 4) | The top pairs are near paraphrases, with $p = 1.00$. In GPT-2 Small, one is "University of Washington" and "Seattle". In GPT-J, one is "Windows 8.1" and "Mac OS X 10.1". In GPT-J, $p$ falls from 1 to 0.13 within the top 20 |
| Sequential ROME on GPT-J (Appendix H) | "continual editing introduces persistent interference to unrelated knowledge", shown on 16 cases, with no count |

## Where the claims go beyond the evidence

- **"Vanishes only if orthogonal" is shown as sufficient, not necessary.** If every $p$ is 0, the interference is 0. The text doesn't show the converse: a sum of nonzero terms could cancel. The claim that the interference tends to infinity has no derivation in the text.
- **The superposition evidence has no numbers.** "High kurtosis", the scaling law and the layer trend are read from figures. The text gives no kurtosis value, no fraction of near-orthogonal pairs and no fitted law. The scaling claim rests on 8 models from three families.
- **128 facts, and no spread.** Each model's distribution comes from 128 facts, chosen at random from those answered correctly. No seed, resample or interval is reported.
- **The editing evidence is 16 cases on one model**, with no numbers and no baseline. One figure appears to use the same image for two cases.
- **"Superposition" here is a property of keys, not of features.** The paper measures overlap between facts' whitened MLP activations. That's a different object from the superposition of features in Elhage et al. (2022), which the paper cites as its frame. The authors say that their whitening-space notion "is derived through rigorous mathematical reasoning", while the activation-space notion "is based on a heuristic perspective".
- **No limitations section.** The text has none. Some appendix figures were "deleted due to Arxiv upload restrictions".

## Use in this repo

- **The keys are neuron activations.** A fact's key is the vector of the layer's MLP neuron activations at the subject's last token. Two facts overlap when they share active neurons, after whitening by $C$. Our reading, not the paper's claim: VERIFIED poles that fire on both subjects are where two facts' keys overlap, so they predict which edits interfere. A pole whose explanation names a category ("cities in the US Pacific Northwest") would put that category's facts in each other's way. The paper's top pairs are near paraphrases, which fits this reading, but it measures no units.
- **Two of its models are close to the registry's.** It measures GPT-2 Small, which the registry studies. It also measures Pythia-1B to 6.9B, the larger siblings of the registry's pythia-70m. Computing its $P$ matrix on GPT-2 Small's layers costs one forward pass per fact and one covariance per layer. The registry could then compare a pair's $p$ with the number of VERIFIED poles that both subjects activate. This page proposes it as a question for the maintainer. It belongs with the traffic map (#5231), which predicts the units an edit will disturb.
- **It is the reason behind AlphaEdit's projection.** [`alphaedit.md`](alphaedit.md) cancels the overlap with preserved keys by projecting each update onto their null space. This paper explains why that is needed: real keys overlap, so a plain sequential edit spreads into other facts. [`editing-at-scale-forgetting.md`](editing-at-scale-forgetting.md) shows the collapse that this interference leads to.
- **The registry's superpositions are a different object.** The registry's superposition records (`openinterpretability/superposition_finders.py`) are groups of neurons that jointly carry a feature, after Elhage et al. ([`toy-models-of-superposition.md`](toy-models-of-superposition.md)). This paper's "knowledge superposition" is overlap between facts' keys. The two meet only through the neurons that a key is made of.
- **Not the same kurtosis as the registry's.** [`docs/analyses/weight-kurtosis.md`](../analyses/weight-kurtosis.md) measures the kurtosis of each neuron's weights through the vocabulary, as a prior for VERIFIED, and finds none. This paper's kurtosis is of the overlaps between facts' whitened keys. They share a word, not a measure.
- **Question for the maintainer.** Should the traffic map (#5231) include this paper's overlap $p$ between facts as a predictor of interference? It could be computed on GPT-2 Small, beside the count of shared VERIFIED poles. This page records the question and opens no issue.

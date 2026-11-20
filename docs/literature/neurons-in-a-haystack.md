# Finding Neurons in a Haystack: Case Studies with Sparse Probing

**Citation.** Wes Gurnee (MIT), Neel Nanda (independent), Matthew Pauly (Harvard), Katherine Harvey (Harvard), Dmitrii Troitskii (Northeastern) and Dimitris Bertsimas (MIT). "Finding Neurons in a Haystack: Case Studies with Sparse Probing". [arXiv:2305.01610](https://arxiv.org/abs/2305.01610), version 2 of 2 June 2023. Version 1 is of 2 May 2023. The arXiv page gives no venue, so we report it as a preprint. The code and data are at [github.com/wesg52/sparse-probing-paper](https://github.com/wesg52/sparse-probing-paper). The text gives no funding statement. It thanks SERI MATS and the MIT SuperCloud for computing.

**Mismatch.** The abstract says "over 100 unique features", and Table 2's feature counts add up to 157. Table 2 lists 167,959 compound-word sequences, but Appendix B.2 describes a probe set of 10,000. The paper calls one method "provably optimal," but Appendix B says its 60-second timeout is "frequently reached."

**Status.** Used. The `sparse-probing-search` skill takes its method from this paper: k-sparse probes on neuron activations, with k swept down to 1. The paper is also evidence for #4226 and for #5220's question about larger models. The fix campaign is the maintainer's, so this page reports evidence and records questions only.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 2 for this page's author, a model (Claude Opus 5.5). It read the source's own text in full, in four fetches: Sections 1 to 7, the references and Appendices A to D. We didn't see any figure as an image. The scale results are mostly in Figures 4, 7, 18 and 20, so this page gives them only as the text describes them.

## Summary

**Claim.** A sparse linear probe over a model's neurons shows where a feature lives. Early layers pack many features into few neurons. Middle layers have more neurons dedicated to one feature. As models grow, features become sparser on average, but in several different ways.

**Method.**

- **Probes.** A logistic regression on at most k post-GELU MLP neurons. Six ways to choose the neurons are compared, among them the mean difference and an optimal sparse probe. k is swept from 1 to 8, and from 256 down to 1 for the scale study.
- **Models.** Seven Pythia models, from 70M (6 layers, 12,288 neurons) to 6.9B (32 layers, 524,288 neurons).
- **Features.** 14 probe datasets: parts of speech, dependencies, morphology, code language, natural language, text features, Pile subset, compound words, LaTeX, and five Wikidata sets of facts.
- **Metric.** Out-of-sample F1. Baselines are random neurons, a random orthogonal projection and random Gaussian bases.
- **Checks.** Ablation, the neuron's output logits, average activation per token, precision against recall, and activations on a 10-billion-token sample.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Out-of-sample F1, averaged over models and features (Table 4) | mean difference 0.832 at k = 1 and 0.890 at k = 8. Random neurons 0.573 and 0.755 |
| Neuron-selection methods (Appendix B.5) | "The top methods are always within 1% of each other" |
| French context neuron ablated (Section 5.2) | loss rises by 8% per French sequence in Pythia-70M, against 0.2% in Pythia-6.9B |
| Pythia-70M L1/N111 (footnote 4) | clean on the probe set, but on 10 billion training tokens 16 of its top 20 examples are the "he" of German words starting "Schönhe" |
| Scale, by feature family (Section 5.3) | code languages and facts become sparser with scale. Parts of speech and compound words stay about the same. Some simple text features are sparser in the smallest models |
| Splitting with scale (Section 5.3) | ALL_CAPS splits into neurons for each reason a token is in capitals, so probes show "less sparsity, not more" |
| Layers (Section 1) | the first 25% of MLP layers show much more superposition than the rest, from weight statistics |

## Where the claims go beyond the evidence

- **The scale result has no numbers in the text.** The abstract says sparsity rises with scale "on average." The per-family results are in figures only, and the text names families that go the other way.
- **One model family and one dataset.** All seven models are Pythia, trained on the same data. The authors say this themselves, and that 6.9B is far from the frontier.
- **No seeds or intervals.** No result reports a seed, an interval or a test set size beyond Table 2's totals.
- **Probes are correlational.** A probe that finds a feature in one neuron doesn't show that the neuron is used for it. The authors say proving a neuron monosemantic is "extremely difficult." The L1/N111 example shows how a probe set can mislead.
- **The random baseline changes from table to table.** Random neurons in Table 4, a random projection in Figure 7, and random Gaussian bases in Figure 13.

The authors state limits themselves. Sparse probes can't find a feature built over several layers. They don't separate superposition from composition. Much of the analysis is by hand.

## Use in this repo

- **The issue's question: which features localise with scale, and which split.** By the text: code languages and facts become more local, often with a dedicated neuron only in larger models. Parts of speech and compound words don't change much. Some features split as the model grows, such as ALL_CAPS, so a probe sees less sparsity in a large model. Per-size numbers are only in the figures.
- **The skill already uses this, and agrees with its layers.** `sparse-probing-search` says that early layers show more superposition and middle layers more dedicated neurons. That is the paper's finding. The paper's Table 4 also supports the skill's cheap first step. The mean difference does as well as the slow methods: 0.832 against 0.831 at k = 1.
- **The random-neuron baseline belongs in the skill's report.** At k = 1, random neurons reach F1 0.573 on these balanced tasks. A single-neuron F1 means little without that floor. Our reading: the skill could report the random-neuron F1 beside each probe.
- **Pythia-70m is the paper's smallest model.** It is a configured model here. The French neuron's 8% loss increase shows that a single 70M neuron can matter a great deal. L1/N111, the paper's warning example, has no registry record yet. The registry's probes and causal checks are a way to test what the probe set missed.
- **For #4226 and #5220.** The paper's answer to "do larger models give cleaner neurons?" is "on average, for some features." Facts and code become more local, so a fix in a larger model may find a single neuron for them. Splitting features may need several neurons. #4226 holds the registry's own test of this.
- **Question for the maintainer, recorded on this page only.** Should #4226 use the paper's feature datasets, so that the registry's results compare with Figures 4 and 18 directly?
- **Related pages.** [`neurons-as-an-interpretable-basis.md`](neurons-as-an-interpretable-basis.md) and [`neuron-basis-circuits.md`](neuron-basis-circuits.md) argue that neurons are a useful basis. [`arithmetic-bag-of-heuristics.md`](arithmetic-bag-of-heuristics.md) finds arithmetic spread over many neurons, the opposite case.

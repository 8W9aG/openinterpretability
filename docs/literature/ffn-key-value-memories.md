# Transformer Feed-Forward Layers Are Key-Value Memories

**Citation.** Mor Geva, Roei Schuster, Jonathan Berant and Omer Levy. The HTML lists three affiliations: the Blavatnik School of Computer Science at Tel-Aviv University, the Allen Institute for Artificial Intelligence and Cornell Tech. The plain text does not map each author to an affiliation. The emails are at tau.ac.il for Geva, Berant and Levy, and at cornell.edu for Schuster. "Transformer Feed-Forward Layers Are Key-Value Memories". arXiv 2012.14913. arXiv dates version 1 to 29 December 2020 and version 2 (read here) to 5 September 2021. [Paper](https://arxiv.org/abs/2012.14913). [Code](https://github.com/mega002/ff-layers/). The arXiv comments field gives the venue as EMNLP 2021. We did not check the proceedings page. The model and data are public: the fairseq `transformer_lm.wiki103.adaptive` model and WikiText-103. Funding: the Yandex Initiative for Machine Learning, the Blavatnik Interdisciplinary Cyber Research Center, the Alon Scholarship and Intel Corporation.

**Status.** Used. The `vocab-projection-lookahead` skill names this paper as the source of its key-value view of a neuron. `openinterpretability/harness/jlens.py` (`vocabulary_lens`) is the repo's version of the paper's value projection.

**Read.** On 2026-11-18, from the v2 HTML at `https://arxiv.org/html/2012.14913` and the abstract page. A model fetched the HTML and changed it to plain text. Then the model read all of that text: the abstract, Sections 1–7, the acknowledgements and Appendices A and B. The formulas came through only as their LaTeX source, and we read the few that carry numbers (d = 1024, d_m = 4096, the 0.0004% baseline). We did not see Figures 2–11, so the per-layer values in them are not here. We did not read the PDF.

## Summary

**Claim.** In a transformer language model, a feed-forward layer acts as a key-value memory. Each key (a row of the first matrix) detects a pattern in the input prefix. Each value (a row of the second matrix) gives a distribution over the next token. In the upper layers, that distribution agrees with the pattern. A layer's output is a composition of hundreds of memories, and the residual stream refines the prediction from layer to layer.

**Method.**

- **Model.** One model: the 16-layer transformer of Baevski and Auli (2019), with 247M parameters, trained on WikiText-103. It has d = 1024 and 4096 memories per layer, with ReLU and adaptive softmax.
- **Keys.** The authors sample 10 keys per layer, 160 in all. For each key, they rank every sentence prefix in the training set by its memory coefficient. NLP graduate students annotate the top 25 prefixes for each key. A pattern must occur in at least 3 prefixes and is "shallow" (n-grams) or "semantic" (topic). One expert annotates each key.
- **Token removal.** For 1600 keys (100 per layer), the authors remove the first, the last or a random token from each of the top 50 prefixes. They measure the change in the memory coefficient.
- **Values.** The authors multiply each value vector by the output embedding matrix and apply a softmax. The paper says that this "assumes (naïvely) that all model's layers operate in the same embedding space". The *agreement rate* is the fraction of memories in a layer whose value's top token equals the next token after the key's top trigger prefix. The baseline is a uniform random token.
- **Composition.** On 4000 random validation prefixes, the authors count active memories, and the cases where the layer's top token differs from every memory's top token. Across layers, they compare the residual's top token with the final output and the feed-forward layer's own top token.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Patterns per key, 160 keys | At least one for every key, 3.6 on average |
| Top-25 prefixes that match a pattern | "65%-80%" |
| Shallow against semantic patterns | Layers 1–9 "dominated by shallow patterns". Layers 10–16 "characterized by more semantic patterns" |
| Removing the last token, against the first | The last token changes the coefficient more. The effect is smaller in upper layers |
| Agreement rate, layers 1–10 | "close to zero" |
| Agreement rate, from layer 11 | "quickly rises until 3.5%" |
| Random-token agreement rate | 0.0004% |
| The 100 values with the highest top-token probability | 97 are in layers 11–16. For 46, at least one of the key's top 50 prefixes agrees |
| Precision@50 of six example values (Table 2) | 4% to 92% |
| Active memories per prefix | "hundreds of memories per layer (10%-50% of 4096 dimensions)" |
| Layer's top token differs from every memory's top token | In "at least ~68%" of the 4000 examples, for any layer |
| Cases where one memory agrees with the layer | 60% have a common stop word as target. 43% have a prefix of fewer than 5 tokens |
| Residual's top token equals the final output | "roughly a third" in the bottom few layers, rising fast from layer 10 |
| When the feed-forward layer changes the residual's top token | It "rarely changes to the feed-forward layer's prediction". The result is a "compromise" |
| 100 last-layer changes, read by hand | 66 change to a semantically distant word, 34 to a related word |

The authors offer a conjecture: the feed-forward layer may "veto" the residual's top token and move mass to other tokens in the residual's head.

## Where the claims go beyond the evidence

**One model, one dataset.** Every number comes from one 247M-parameter model on Wikipedia text. The title and the conclusion speak of transformers in general. The authors say this themselves: they "leave verification of this for future work".

**A weak null for the agreement rate.** The paper compares the 3.5% peak with a uniform random token over about 270K types (0.0004%). Next tokens in text are far from uniform. A null that draws from token frequency, or that pairs each value with a random key's trigger, would be much higher. The paper does not report one, so "orders of magnitude" is against the weakest null.

**Top activations, one annotator.** The pattern analysis reads only the 25 prefixes with the highest coefficient for each key. One expert labels each key, and the paper reports no agreement between annotators. The 65%–80% coverage says the top of the distribution is coherent. It says nothing about what the key does at moderate coefficients.

**Observation, not intervention.** No experiment changes a key or a value and measures the output. That values "store information on how to directly predict the output" is an inference from top-token matches. Sections 5.1 and 5.2 compare top tokens only, under a projection that the authors call naive.

**The lower-layer gap is explained, not tested.** The authors put the near-zero agreement in layers 1–10 down to a different embedding space in those layers. The paper does not test that, and says so.

**Figure-only results, no intervals.** The composition and refinement results are mostly in Figures 8–11, with bounds such as "at least ~68%" in the text. No result has a seed, an interval or a sample size per layer. The study uses one trained model, so it cannot measure variation between seeds.

**ReLU only.** Memory coefficients are non-negative under ReLU. In GELU or SwiGLU models, a coefficient can be negative, and then the value pushes the other way. The key-value reading extends to those models only if each sign is read on its own.

## Use in this repo

**The skill already uses the key-value view.** `vocab-projection-lookahead` reads a neuron's input weights as a key and its output weights as a value. It does not use the paper's logit-lens projection for the value. It uses the Jacobian lens through `vocabulary_lens` in `openinterpretability/harness/jlens.py`. The skill reports that on Qwen2.5-0.5B, the logit lens and the Jacobian lens share 0 of their top-10 tokens up to the middle layers. That is the repo's own form of the paper's near-zero agreement in lower layers. It is also why the repo does not trust the naive projection there.

**Signed poles answer the ReLU point.** The registry studies each neuron as two poles, `+` and `-`. A negative pole reads the value in reverse. The VERIFIED answer route for #4781 includes three negative poles, L5/N2036-, L5/N575- and L5/N119-. Their stated roles push down layout and function tokens after the count word. Those are values read with a negative coefficient, which the paper's ReLU model cannot have.

**The paper's method is the step the protocol goes beyond.** Top-25 prefixes read by one person is close to what a proposer sees in scan evidence. The protocol adds an adversarial reviewer, hard negatives and causal tests before a verdict. The [Beyond Top Activations](beyond-top-activations.md) page discusses the same limit: top examples measure recall only.

**What #4781 already measures.** The shortlist in `docs/analyses/hallucination-shortlist.jsonl` gives, for "A week has seven days and a year has twelve":

- " days" at 0.7257, then " weeks" at 0.0796, " hours" at 0.0314 and " months" at 0.0295 (rank 4).
- A first-order contrast of logit(" days") − logit(" months") per unit. A negative value favours " months". L0/N1101+ is −1.53 and L1/N707+ is −1.01. L3/H1 is +0.81 and L4/H7 is +0.59.

L0/N1101+ is on the VERIFIED route as a time-unit neuron. By this first-order measure, its net effect favours " months", not " days". That is one linear attribution, not a patching result. The reversed prompt, "A year has twelve months and a week has seven", also fails: " months" 0.4549 against " days" 0.1503.

**What the paper predicts for the copy failure.** In the paper's terms, "a year has twelve → months" would be an upper-layer memory. Its key would be semantic, and its value's top token would be " months". Section 5.2 then predicts that such a memory does not flip the answer when the residual already holds " days". The feed-forward layer "rarely changes to the feed-forward layer's prediction". At most it lowers the top token. So the paper predicts one of two cases, and the measurements can tell them apart:

- *absent*: no layer-4 or layer-5 neuron with a " months"-promoting value is active at the last position.
- *present but outvoted*: such neurons fire and write a positive " months" − " days". But the copy heads L3/H3, L4/H0 and L4/H7 already put " days" on top.

Pythia-70m has six layers. The paper found agreement only from layer 11 of 16, so a 70M model may have few fact memories in this sense. That is a reason the *absent* case is likely, not evidence for it.

**What would change.** No new protocol work. The measurement above is question 1 of #5015 ("Why does nothing favour ' months'?"). It belongs in a comment on that issue, not a new issue. Measure each layer-4 and layer-5 neuron's direct write to " months" − " days" at the last position, with `vocabulary_lens`. Record each neuron's coefficient on the prompt too. The MQuAKE comment on #5015 already asks for a single-clause control, "A year has twelve". Both readings need that control to mean anything.

**A possible low-priority check.** The paper's agreement rate can be computed across the registry's VERIFIED lookahead poles. Use the Jacobian-lens top token, the token after each top example, and a frequency null in place of the uniform one. This would show whether the registry's lookahead neurons agree more than a frequency null predicts, by layer. It is a proposal only.

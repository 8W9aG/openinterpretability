# Massive Activations in Large Language Models

**Citation.** Mingjie Sun (Carnegie Mellon University), Xinlei Chen (Meta AI Research), J. Zico Kolter (Carnegie Mellon University and Bosch Center for AI) and Zhuang Liu (Meta AI Research). "Massive Activations in Large Language Models". *First Conference on Language Modeling (COLM 2024)*. [arXiv:2402.17762](https://arxiv.org/abs/2402.17762), version 1 of 27 February 2024 and version 2 of 14 August 2024. Code is at [github.com/locuslab/massive-activations](https://github.com/locuslab/massive-activations). Funding: the Bosch Center for Artificial Intelligence supported Mingjie Sun.

**Mismatch.** The abstract says massive activations are "100,000 times larger" than others, but Section 2 says LLaMA2-7B's largest is "about 10,000 times larger than the median". Section 1 counts "fewer than 10" among tens of millions, while Section 2 counts at most four of about 40,000 per hidden state.

**Status.** Used. `papers/default-on-poles` cites it for the first token's extreme values. The reviewer skill `interpretability-illusions-review` lists it as an illusion to probe for. `openinterpretability/core/partners.py` ranks partner correlations by Spearman's ρ partly so that heavy tails and massive activations don't dominate them.

**Read.** On 2026-11-20 a `web-fetch` subagent read the abstract page and the HTML of version 2 for this page's author, a model (Claude Opus 5.5). It read the whole text in the source's own words, in three fetches: Sections 1 to 7, the references and Appendices A to D. The figures are images, and no GPT-2 value on this page is read off a plot. We didn't read version 1.

## Summary

**Claim.** A few entries of a language model's hidden states are enormous, up to about 10,000 times the median. They sit in fixed dimensions at a few positions, mostly the first token and delimiters. They stay nearly constant across inputs, act as fixed biases, and draw attention to their tokens.

**Method.**

- **Where.** Massive activations are measured in each layer's hidden state, "the output of residual summations, not any intermediate states inside" a layer. So they are residual-stream values, not neuron activations.
- **Definition.** A magnitude over 100 that is "at least or around 1,000 times larger than the median magnitude of its hidden state".
- **Models.** LLaMA2 and LLaMA3, Mistral, Mixtral, Phi-2, MPT, Falcon, OPT and the GPT-2 family, and some vision Transformers.
- **Tests.** Set the massive activations at one layer to zero, or to their mean, and measure perplexity and zero-shot accuracy. Then retrain a small GPT-2 with an explicit learned attention bias.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Where they sit (Section 2.2) | Three groups. GPT-2, LLaMA2-13B and MPT: the starting token only. LLaMA2-7B: the starting token and the first delimiter. Others also use weak words such as "and" |
| LLaMA2-7B, WikiText perplexity (Table 3) | 5.47. Set to zero: infinite. Set to the mean: 5.47 |
| LLaMA2-13B, WikiText perplexity (Table 3) | 4.88. Set to zero: 5,729. Set to the mean: 4.88 |
| Variation across inputs (Table 2) | LLaMA2-7B top value 2,556.8 ± 141.0, so near-constant |
| Attention (Section 4.1) | From layer 3 on, LLaMA2-7B's attention is "mostly concentrated" on the two tokens with massive activations |
| GPT-2 with explicit attention biases (Section 4.3) | The massive activations disappear, at the same validation perplexity (3.04) |

## Where the claims go beyond the evidence

- **"Bias" is shown by a mean-replacement test.** Setting the values to their mean keeps performance, and setting them to zero breaks it. That shows they work as constants. It doesn't show how the model uses them beyond the attention analysis.
- **The GPT-2 experiments are small.** The authors say the retraining was "limited to a small GPT-2 model due to computational resource constraints". It is one run of each variant, with no seeds or intervals.
- **No GPT-2 numbers in the text.** GPT-2's dimensions and magnitudes are only in figures.
- **The text disagrees with itself on the ratio.** The abstract's 100,000× and Section 2's 10,000× differ by a factor of ten.

## Use in this repo

- **One citation checks out, and one needed a fix.** `papers/default-on-poles` cites the paper twice.
  - "Sun et al. (2024) report massive activations on GPT-2's starting token." Section 2.2 puts GPT-2 in the "starting token only" group. This matches.
  - The paper also said that its first token's "very large positive activations" agree with the massive activations. Its values are MLP neuron activations, and the source's are residual-stream values. This PR rewords the sentence to name the two quantities apart, and to mark the link as the paper's observation. No number changes.
  - The reference's authors, year and venue match.
- **The registry already guards the first token.** A neuron that fires at the first token only because that position's residual stream is extreme would have misleading top examples. Protocol 0.8 handles this ([`position-zero.md`](../analyses/position-zero.md)). Some units are active at the first position on at least half the vocabulary. Such a unit isn't measured there, in its scan, its top examples or its probes. The source gives a likely reason for such a floor, which the protocol didn't name.
- **The reviewer skill's summary is fair.** It says massive activations sit at "fixed positions (first token, delimiters) and act as" biases. For GPT-2, only the starting token is reported.
- **No question for the maintainer.**
- **Related pages.** [`gelus.md`](gelus.md) reviews GPT-2's activation, and [`gpt-2.md`](gpt-2.md) the model itself.

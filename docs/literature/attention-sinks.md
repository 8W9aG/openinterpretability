# Efficient Streaming Language Models with Attention Sinks

**Citation.** Guangxuan Xiao (MIT), Yuandong Tian (Meta AI), Beidi Chen (Carnegie Mellon University), Song Han (MIT and NVIDIA) and Mike Lewis (Meta AI). Xiao did part of the work during an internship at Meta AI. "Efficient Streaming Language Models with Attention Sinks". *ICLR 2024*. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453), version 1 of 29 September 2023, last revised as version 4 on 7 April 2024. Code is at [github.com/mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm). Funding: the MIT-IBM Watson AI Lab, Amazon, the MIT Science Hub and the National Science Foundation.

**Mismatch.** The introduction lists Pythia at 2.9B, but Section 4.1 tests Pythia-2.8B, 6.9B and 12B.

**Status.** Used. `papers/default-on-poles` cites it for attention sinks at the first token. Pythia, one of the families it tests, is a registry model family.

**Read.** On 2026-11-20 a `web-fetch` subagent read the source for this page's author, a model (Claude Opus 5.5). It read the abstract page and the HTML of version 4, in three fetches. It had the source's own text for the abstract, Sections 1 to 5, the references and Appendices A to E. The end of Appendix F and all of Appendices G to I came back only as a summary, and this page uses nothing from them.

## Summary

**Claim.** Language models put a large share of attention on the first few tokens "irrespective of their relevance to the language modeling task". The authors call these tokens "attention sinks". Keeping them in the cache lets a model with a sliding window run on very long streams.

**Method.**

- **Why sinks form.** Softmax makes attention sum to one, so a head with nothing useful to attend to must put its weight somewhere. The first tokens are "visible to all subsequent tokens", so they are trained to take it.
- **The test of position against meaning.** Replace the first four tokens with "\n". The model still attends to them and keeps its perplexity, so position matters more than meaning.
- **StreamingLLM.** Keep the keys and values of four initial tokens plus a rolling cache of recent tokens, with no fine-tuning.
- **Models.** Llama-2, MPT, Falcon and Pythia, and three 160M-parameter models trained from scratch with the Pythia code, with and without a dedicated sink token.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| Streaming length (abstract) | Stable language modelling "with up to 4 million tokens and more" on Llama-2, MPT, Falcon and Pythia |
| Speed (Section 4.5) | Up to 22.2× faster per token than recomputing a sliding window |
| Initial tokens to keep (Section 4.4) | Four are enough. More add little |
| 160M model, streaming perplexity with no sink kept, then one (Table 3) | Vanilla 27.87 then 18.49. With a learnable sink token: 1,235 with no sink kept, 18.01 with one |
| 160M model, zero-shot scores with a sink token (Table 4) | About the same as without, for example ARC-e 45.2 against 45.6 |

## Where the claims go beyond the evidence

- **Streaming is not long memory.** The authors say so: StreamingLLM "does not extend the models' context window or enhance their long-term memory capabilities". Accuracy falls to zero once an answer lies beyond the cache.
- **The sink-token test is small.** It uses three 160M models, one run each, with no seeds or intervals.
- **The cause is argued, not isolated.** The softmax explanation fits the results. The paper doesn't rule out other causes, such as the first token's large residual values that Sun et al. (2024) report.

## Use in this repo

- **One citation checks out, and one needed a fix.** `papers/default-on-poles` cites the paper twice.
  - Its related-work paragraph says the source names "the initial tokens that draw strong attention 'attention sinks'". That matches the introduction and Section 3.1. The paper's next sentence, that both sources "suggest why an intervention at the first token can reach every later token", is the paper's own step.
  - The paper said Xiao et al. "find attention sinks there", at GPT-2's starting token. The source tested Llama-2, MPT, Falcon and Pythia, and doesn't mention GPT-2. This PR names the models that it tested. No number changes.
  - The five authors, 2023 and ICLR 2024 match.
- **The protocol already names the first token "the attention sink".** `docs/PROTOCOL.md` says so three times, with no source. The first-token floor of protocol 0.8 leaves that position out for some units ([`position-zero.md`](../analyses/position-zero.md)). The dictionary fit leaves it out too. Input attribution keeps it, because replacing it sends the path off-distribution. This paper is the evidence for the name. Our reading: the protocol could cite it there.
- **Pythia shows sinks, and the registry studies Pythia.** The source shows sinks in Pythia-2.8B and larger, and trains its 160M models with the Pythia code. The registry studies pythia-70m. Our reading: a head of pythia-70m that attends to the first token may be acting as a sink, not reading the first token's content. The attention-head audits should check for this.
- **No question for the maintainer.**
- **Related pages.** [`massive-activations.md`](massive-activations.md) reviews the first token's large residual values, which Sun et al. tie to attention sinks.

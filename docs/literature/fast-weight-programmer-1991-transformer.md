# Neural Nets Learn to Program Neural Nets with Fast Weights (1991)

**Citation.** Jürgen Schmidhuber. "Neural nets learn to program neural nets with fast weights (1991)". Web page, IDSIA, 26 March 2021, updated November 2025. [Page](https://people.idsia.ch/~juergen/fast-weight-programmer-1991-transformer.html). A self-published history page, no venue. The page's own reference list gives a slightly different title ("…fast weights—like Transformer variants"), and the in-page header says "the first Transformer variants".

**Status.** Not applicable. The page is a history and a priority claim, with no method or measurement. The work it describes already has pages here: [`fast-weight-memories.md`](fast-weight-memories.md) (Schmidhuber, 1992) and [`linear-transformers-fast-weight-programmers.md`](linear-transformers-fast-weight-programmers.md) (Schlag, Irie and Schmidhuber, 2021).

**Read.** On 2026-11-21 a `web-fetch` subagent (Claude) read the whole page in the source's own words, in two parts, through the reference list and image captions. It did not read the linked PDFs, blog posts or papers. The reader was this page's author, a model (Claude Opus 5.5), through that subagent's report.

## Summary

**Claim.** In March 1991, a "slow" network learned by gradient descent to change the "fast weights" of another network. It did so by adding the outer product of FROM and TO activations. The page says that FROM and TO are today's key and value. So the 1991 system is "an unnormalised linear Transformer", "mathematically equivalent (apart from normalization)" to the linear Transformers of 2020–21. The page also says that the 1993 follow-up introduced the word attention.

**Method.** A history page. It cites the author's technical report FKI-147-91 (1991), the Neural Computation paper (1992), ICANN 1993, and later papers with Schlag and Irie (2017–2025).

**Findings, as the source reports them.** The page reports no experiment. Its claims are dates and attributions, with no table to give.

## Where the claims go beyond the evidence

- **The equivalence is asserted, not shown.** The page does not reproduce the 1991 update equation. It points to the 2021 paper by Schlag, Irie and Schmidhuber for the formal link. That paper's page here is the place to check it.
- **"First Transformer variant" is a framing.** The match is to linear attention without the softmax and normalisation, which the page itself sets aside. The 2017 Transformer's softmax attention is a different computation.
- **The author argues his own priority.** Several wider claims, about the 2017 Transformer paper and the 2024 Nobel Prize, rest on links to the author's own posts and reports.

The page asks readers to report errors, and it calls itself part of an ongoing priority dispute.

## Use in this repo

- **No bearing on the registry.** The page discusses no interpretability, single neurons or MLPs. Its technical point, that linear attention is a sum of key-value outer products, is already on [`linear-transformers-fast-weight-programmers.md`](linear-transformers-fast-weight-programmers.md).
- **No follow-up issue.**

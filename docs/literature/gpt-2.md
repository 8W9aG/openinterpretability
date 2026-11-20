# Language Models are Unsupervised Multitask Learners

**Citation.** Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever, of OpenAI. Radford and Wu contributed equally. "Language Models are Unsupervised Multitask Learners". A technical report, published as a [PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) with OpenAI's [blog post](https://openai.com/research/better-language-models) of 14 February 2019. No venue and no arXiv version. The paper gives no date of its own. It links code for the small model at [github.com/openai/gpt-2](https://github.com/openai/gpt-2).

**Mismatch.** The abstract claims the best results yet on "7 out of 8" language modelling datasets. Table 3 bolds a new best on 7 of its 8, and One Billion Word is the miss, so the count holds. The paper names its smallest model "117M". GPT-2's paper doesn't call it "small". That name comes from later use.

**Status.** Used. GPT-2 small is one of the registry's configured models, and six papers in `papers/` cite this source for it.

**Read.** On 2026-11-20 a `web-fetch` subagent couldn't extract text from the PDF, and read OpenAI's blog post instead. So this page's author, a model (Claude Opus 5.5), read the PDF's pages 1 to 10 as images. They hold Sections 1 to 7, Tables 1 to 6 and Figures 1 to 4. We didn't read the reference list past its first page, or the appendix of samples.

## Summary

**Claim.** A language model trained on a large and varied web corpus starts to do tasks with no task training. Examples are question answering, translation and summarisation. Its zero-shot performance grows log-linearly with model size.

**Method.**

- **Data.** WebText: the text of outbound Reddit links with at least 3 karma, created before December 2017. After de-duplication it has "slightly over 8 million documents for a total of 40 GB of text". Wikipedia articles were removed, to reduce overlap with evaluation sets.
- **Tokens.** Byte-level byte-pair encoding, with merges blocked across character categories. Section 2.3: "The vocabulary is expanded to 50,257."
- **Model.** A Transformer that "largely follows the details of the OpenAI GPT model". Layer norm moves to each sub-block's input, an extra layer norm is added at the end, and residual weights are scaled by 1/√N. Context 1,024 tokens, batch 512.
- **Sizes (Table 2).** 117M parameters with 12 layers and a width of 768, then 345M (24, 1,024), 762M (36, 1,280) and 1,542M (48, 1,600). The table gives no MLP width and the text names no activation function.
- **Evaluation.** Zero-shot, with invertible de-tokenizers, on language modelling, the Children's Book Test, LAMBADA, Winograd, CoQA, summarisation, translation and Natural Questions.

**Findings, as the source reports them.**

| Setting | 117M | 1,542M | Earlier best |
| --- | --- | --- | --- |
| LAMBADA perplexity (Table 3) | 35.13 | 8.63 | 99.8 |
| LAMBADA accuracy (Table 3) | 45.99 | 63.24 | 59.23 |
| WikiText-103 perplexity (Table 3) | 37.50 | 17.48 | 18.3 |
| One Billion Word perplexity (Table 3) | 75.20 | 42.16 | 21.8 |
| CoQA, 1,542M (Section 3.5) | | 55 F1 | 89 F1 human |
| Natural Questions, 1,542M (Section 3.8) | | 4.1% exact match | |
| Test 8-grams also in WebText train (Table 6) | 1–6% per set, mean 3.2% | | |

## Where the claims go beyond the evidence

- **Single runs.** Every number is one model, one run. The paper gives no seeds or intervals.
- **Hand-tuned zero-shot prompts.** Each task gets its own prompt format, such as `TL;DR:` for summaries and an `english sentence = french sentence` pattern for translation. The choice of format is part of the result.
- **Overlap is measured, not removed.** The authors check 8-gram overlap with WebText and find a small benefit on some sets. They don't retrain without the overlapping text.
- **"Unsupervised multitask learning" is a framing.** On summarisation, translation and question answering, the largest model is still far from supervised systems. The authors say so: its zero-shot performance "is still far from use-able" for practical applications.
- **Code and data.** Only the small model's code and weights were released at publication. WebText wasn't released.

## Use in this repo

- **Most of our citations check out.** The six papers cite it for GPT-2 small, its 12 layers and its vocabulary.
  - `papers/explanations-across-models` quotes "is expanded to 50,257" from Section 2.3. The words and the section match.
  - "12 layers" matches Table 2's smallest model.
  - The reference gives the six authors in order, and 2019, the blog's year.
- **The MLP width was not in the source.** `papers/native-vs-learned` and `papers/spd-input-projections` (both preprints) cited this paper for 3,072 neurons per MLP. The paper never states that width. It says the model "largely follows" GPT-1. This PR rewords both papers: the 12 layers stay with this paper, and the 3,072 neurons go with the released weights. No number changes.
- **GELU isn't in this paper either.** `papers/why-neurons-rest` says GPT-2 uses GELU and cites the [GELU paper](gelus.md) for it, which is right. This paper names no activation function.
- **The "117M" model is the registry's GPT-2 small.** The registry loads the released `openai-community/gpt2` weights, the size that the paper calls 117M. Counting the released weights gives about 124M. The paper doesn't say how it counted. Our papers give layers and widths, not the parameter count, so nothing depends on it.
- **WebText's overlap study is a precedent for ours.** The paper checks 8-gram overlap between training text and evaluation sets. The registry's held-out explanation scores rely on corpus windows that the proposer never saw, which is the same concern at a smaller scale.
- **No question for the maintainer.**
- **Related pages.** [`the-pile.md`](the-pile.md) reviews the corpus of the registry's scans, which is also web-heavy. [`gelus.md`](gelus.md) reviews GPT-2's activation function.

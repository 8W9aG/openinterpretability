# MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers

**Citation.** Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang and Ming Zhou, of Microsoft Research. "MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers". *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)* ([NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)). The preprint is [arXiv:2002.10957](https://arxiv.org/abs/2002.10957), version 1 of 25 February 2020, revised as version 2 on 6 April 2020. Code and models are at [github.com/microsoft/unilm/tree/master/minilm](https://github.com/microsoft/unilm/tree/master/minilm).

We also read the model card of [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), the model that the repo runs. Its licence tag is Apache 2.0.

**Mismatch.** Our reference called the paper a preprint, but it appeared at NeurIPS 2020. This review fixes the reference in `papers/shared-concept-space`. The HTML gives the number of fine-tuning runs as "44", which may be a rendering error.

**Status.** Used. `concept_map.EMBEDDER` and the literature map embed texts with `all-MiniLM-L6-v2`. `openinterpretability/harness/explanation_scores.py` uses it by default for the embedding score.

**Read.** On 2026-11-21 a `web-fetch` subagent read the source for this page's author, a model (Claude Opus 5.5). It read the arXiv abstract page, the NeurIPS abstract page and the HTML of version 2. It had the abstract, Sections 1 to 4 and Tables 1 to 4 in the source's own words. The rest came back only as a summary, and this page uses nothing from it. It read the whole model card.

## Summary

**Claim.** A small Transformer can learn from a large one by copying the self-attention of the teacher's last layer. Neither its depth nor its width needs to match the teacher's. The method keeps "more than 99% accuracy" on SQuAD 2.0 and several GLUE tasks with "50% of the Transformer parameters and computations".

**Method.**

- **What is copied.** The KL divergence between teacher and student self-attention in the last layer, and between their "value relation", the scaled dot product of the values.
- **Teacher assistant.** For a much smaller student, the teacher is first distilled into a mid-sized assistant, which then teaches the student.
- **Setting.** The teacher is BERT-Base, pre-trained on English Wikipedia and BookCorpus. The students are fine-tuned on each task.

**Findings, as the source reports them.**

| Setting | Result |
| --- | --- |
| 6 layers, width 768, 66M parameters (Table 2) | Average 80.4 against BERT-Base's 81.5. SQuAD 2.0 F1 76.4 against 76.8 |
| 6 layers, width 384, 22M parameters, with assistant (Table 3) | SQuAD 2.0 F1 72.7, MNLI-m 82.4, SST-2 91.2. TinyBERT of the same size: 71.6, 81.4, 90.2 |
| Speed (Table 4) | 6 × 768 is 2.0× faster than BERT-Base, and 6 × 384 is 5.3× faster |

**The model card.** `all-MiniLM-L6-v2` starts from `nreimers/MiniLM-L6-H384-uncased`, a 6-layer MiniLM model of width 384. It was then fine-tuned with a contrastive objective on over a billion sentence pairs, the largest share from Reddit comments. It "maps sentences & paragraphs to a 384 dimensional dense vector space". It is "intended to be used as a sentence and short paragraph encoder", and "input text longer than 256 word pieces is truncated".

## Where the claims go beyond the evidence

- **English and one teacher.** The public results compress BERT-Base, trained on English text.
- **No spread.** Results are averages over fine-tuning runs, with no standard deviations or tests in the text we read.
- **The repo's model is two steps away.** The paper tests task fine-tuning. `all-MiniLM-L6-v2` adds a contrastive stage on other data, which the paper doesn't cover.

## Use in this repo

- **Our citations now check out.** `papers/shared-concept-space` said `all-MiniLM-L6-v2` was "distilled with MiniLM". The model card says it starts from a 6-layer MiniLM model and was fine-tuned on sentence pairs. The sentence now says that. The reference now names NeurIPS 2020. Both are fixed in place, because the paper is a preprint.
- **Short texts suit it.** The model card names sentences and short paragraphs as its use. `concept_map.py` cuts each example to 20 tokens on each side of the peak, well under the 256-word-piece limit.
- **Reddit-heavy training.** Most of its training pairs are Reddit comments. Our reading: it may place code or unusual token runs less well than everyday text. That matters for poles whose examples are code.
- **No question for the maintainer.**
- **Related pages.** [`sentence-bert.md`](sentence-bert.md) reviews the library and the method that the model card builds on.

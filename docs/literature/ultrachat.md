# Enhancing Chat Language Models by Scaling High-quality Instructional Conversations

**Citation.** Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun and Bowen Zhou (Tsinghua University). "Enhancing Chat Language Models by Scaling High-quality Instructional Conversations". arXiv:2305.14233, v1 of 23 May 2023, the only version. [Paper](https://arxiv.org/abs/2305.14233), [code](https://github.com/thunlp/UltraChat), [data](https://huggingface.co/datasets/openbmb/UltraChat). The data is under the MIT licence, "intended solely for research and educational purposes". A changed text appeared at EMNLP 2023 (main conference, pages 3029–3051, [DOI 10.18653/v1/2023.emnlp-main.183](https://doi.org/10.18653/v1/2023.emnlp-main.183)). It has eight authors, without Zhi Zheng, and Bokai Xu is at The Chinese University of Hong Kong, Shenzhen. It also names its funding, two Chinese national grants, CAST and the Institute Guo Qiang at Tsinghua. The dataset cards and the code's README still cite only the arXiv text.

**Status.** Used. The repo reads HuggingFaceH4's filtered copy, UltraChat 200k, in two places. `scripts/base_instruct_patch.py` and `papers/instruct-difference/` take their user turns from its `test_sft` split. The `ultrachat` corpus preset (`openinterpretability/corpus_presets.py`) reads its `train_sft` split, and the `mixed` blend takes 34 of its conversations.

**Read.** On 2026-11-27 a subagent (Claude) downloaded the arXiv v1 PDF and the EMNLP PDF and turned them into text. It read both in full, with their appendices, and it checked three tables and figures as page images. It also read the code's README and licence, and the cards of `openbmb/UltraChat` and `HuggingFaceH4/ultrachat_200k`. It saw Figure 1 only as text labels, and it did not download the data. The reader of this page was a model (Claude Opus 5.5), through that subagent's report. This page gives the EMNLP numbers unless it names v1.

## Summary

**Claim.** Two models that talk to each other can write a large, varied set of multi-turn dialogues. A chat model trained on that set can match or beat other open chat models. UltraChat is that set, and UltraLM (UltraLLaMA in v1) is the model.

**Method.**

- **Writers:** two "ChatGPT Turbo APIs". One plays the user, from prompts that ask it to act like a person, and the other answers. No model snapshot is named. ChatGPT writes the opening lines, the user turns and the replies.
- **Three sectors.** Questions about the World starts from 30 topics that ChatGPT expanded, and from the 10,000 most frequent Wikidata entities. It has about 500,000 opening lines and 3 to 7 rounds. Creation and Writing has 20 types of text, with 2 to 4 rounds. Assistance on Existing Materials puts C4 web text into 7 templates with ChatGPT's instructions, with 500,000 opening lines and 2 to 4 rounds.
- **Filter:** the only filter the paper describes removes "excessively polite statements such as "Thank you"". It gives no criteria and no count of what it removed.
- **Model:** LLaMA-13B, fine-tuned on UltraChat alone, with the loss on the replies only, on 128 A100s with a batch of 512. No learning rate, epoch count or seed is given.
- **Judges:** GPT-4 in the EMNLP text, and ChatGPT in v1. The evaluation set is the Vicuna benchmark plus about 300 questions that GPT-4 wrote.

**UltraChat 200k.** HuggingFaceH4's card calls it "a heavily filtered version", used to train Zephyr-7B-β. It applies three steps:

- a subset "for faster supervised fine tuning".
- truecasing, because "around 5% of the data" had errors like "Hello. how are you?".
- removal of dialogues where the assistant says "I do not have emotions" or "I don't have opinions".

Its splits are `train_sft` (207,865), `test_sft` (23,110), `train_gen` (256,032) and `test_gen` (28,304). The card doesn't say how it chose the subset or the test splits. Its licence is MIT.

**Findings, as the source reports them.**

| Result | Setting |
| --- | --- |
| 1,468,352 dialogues, 3.8 turns and 1,467.4 tokens on average | The full UltraChat (Table 4) |
| Lexical diversity (MTLD) 74.3, the highest of eight sets. Coherence 9.06, tied with Baize | ChatGPT scores the coherence of 10,000 sampled dialogues |
| Helpful 0.960, honest 0.992, harmless 1.000 | Human scores of 500 dialogues, two annotators, Cohen's kappa 0.358. EMNLP only |
| A mean score of 9.00, against 8.95 for WizardLM-13B and 8.78 for Vicuna-13B | GPT-4 scores the curated evaluation set |
| A benchmark average of 60.95, against 60.19 for WizardLM-13B | ARC-Challenge, HellaSwag, MMLU and TruthfulQA, through the lm-evaluation-harness |
| AlpacaEval 76.09 (SE 1.50), against 75.31 (SE 1.51) for WizardLM-13B | Win rate against Text-Davinci-003, GPT-4 as judge |
| 102.8% of WizardLM's score, with Math at 79.3% | Evol-Instruct test set, 218 items, GPT-4 pairwise |
| A mean score of 9.023, against 8.961 for Vicuna-13B | v1 only, ChatGPT as judge |

## Where the claims go beyond the evidence

- **ChatGPT writes, scores and judges.** ChatGPT writes every turn and scores the coherence. In v1 it also judges a model distilled from it. GPT-4, from the same company, judges the EMNLP results and wrote 300 of the test questions. The paper doesn't discuss this. Its one human study scores the data, not the model.
- **Small gaps, no intervals.** UltraLM's lead over WizardLM on AlpacaEval is 0.78 points, inside one standard error. There is one training run, with no seeds. Apart from the AlpacaEval errors, the only intervals are unlabelled ± values in Table 1.
- **The filtering is not described.** The paper names one filter and no counts. UltraChat 200k doesn't say how it chose 207,865 training dialogues from about 1.4 million.
- **Inconsistencies.** v1 says 10,000 C4 pieces, and the EMNLP text says 100,000. Only 100,000 pieces with five instructions each give the stated 500,000 opening lines. The topic-diversity column is marked "lower is better", but the text reads a lower value as a weakness. The curated set has 831 items in Table 11, not the 380 that the text describes.
- **No contamination check.** Only the README discusses it, as a disclaimer. The evaluation set contains the Vicuna benchmark.
- **A privacy claim that Sector III contradicts.** The README says "we do not directly use any data available on the Internet as prompts". But Sector III opening lines contain C4 web text.
- **Limits the authors state.** The authors say the data is English only, that it has no data made for reasoning, and that automatic judging "may have biases". The math result agrees: "no mathematical problems are intentionally generated".

## Use in this repo

- **The repo reads the 200k copy, not the paper's set.** The paper's numbers describe 1.47 million dialogues. The repo reads HuggingFaceH4's filtered, truecased subset. `papers/instruct-difference/` pins it at commit `8049631c`. So none of the paper's statistics describes the text that the repo scans, and the card gives no statistics of its own.
- **The instruct-difference paper uses only the user turns.** `scripts/base_instruct_patch.py` takes the first user turn and lets the instruct model write its own reply. That keeps the reply on-policy, and it avoids ChatGPT's replies. The user turn is still ChatGPT acting as a user. So the paper's 128 conversations are model-written requests, not human ones. Its *Conversations* paragraph could say so.
- **The `mixed` blend scans ChatGPT's replies as assistant turns.** The `ultrachat` preset writes whole conversations with Qwen2.5's chat template, replies included. A Qwen2.5 instruct model then reads ChatGPT's replies as if it wrote them. That is text the model would not produce, so a pole that tracks the model's own style may fire differently on it. These are 34 documents of the blend, and each document records `source: ultrachat`. So a scan can tell them apart, and no change is needed. The point matters if a pole's top examples come mostly from this source.
- **The preset's licence note is right.** The preset's comment gives MIT, as the card does. The README adds "solely for research and educational purposes". Neither the paper nor the cards mention OpenAI's terms of use. The repo publishes neither the corpus nor the model outputs as a dataset, so the question doesn't arise here.
- **The citation.** `papers/instruct-difference/` cites the arXiv preprint with nine authors, as the dataset card does. The EMNLP text is a different version with eight authors. The preprint citation is accurate for what it names, so this page opens no issue.

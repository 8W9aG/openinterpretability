# Steering Llama 2 via Contrastive Activation Addition

**Citation.** Nina Panickssery, Meg Tong and Evan Hubinger (Anthropic), Nick Gabrieli (Harvard University), Julian Schulz (University of Göttingen). The sixth author is Alexander Matt Turner (Center for Human-Compatible AI). "Steering Llama 2 via Contrastive Activation Addition". arXiv 2312.06681, v1 9 December 2023, v4 5 July 2024. [Paper](https://arxiv.org/abs/2312.06681), [code](https://github.com/nrimsky/CAA) (MIT License). The venue is the Proceedings of the 62nd Annual Meeting of the ACL (Volume 1: Long Papers), August 2024, pages 15504–15522. Its Anthology ID is [2024.acl-long.828](https://aclanthology.org/2024.acl-long.828/), DOI 10.18653/v1/2024.acl-long.828. We checked the venue on the Anthology page, and it matches the issue's "venue from memory" note.

The first author's name changed. The arXiv record and the v4 HTML give "Nina Panickssery". The Anthology page gives "Nina Rimsky" and "Alexander Turner". Cite the ACL version under Rimsky and the arXiv version under Panickssery, and treat the two names as one person. The arXiv page has no comments or journal-ref field, and the v4 text does not name the venue. The acknowledgements thank the MATS team and several people. The text gives no funding statement. The paper uses GPT-4 to score text and to write two of its datasets.

**Status.** Not applicable now, and evidence for the fix campaign later. No code, skill or protocol rule depends on it, and no open issue would use it. The campaign belongs to the maintainer, so this page reports evidence and records questions only.

**Read.** We read the arXiv abstract page, the v4 HTML and the ACL Anthology page on 2026-11-18, through WebFetch. A model read the text for us in three passes. The tool let it quote only short excerpts. So the method and the numbers came back as that model's transcription, with a few verbatim fragments. We have verbatim text for the captions of Tables 3 to 5, the "3 out of 7" sentence and the §9.1 sentence on token positions. We did not read the PDF, the figures, the appendix prompts in full or Tables 9 to 13 cell by cell.

## Summary

**Claim.** The mean difference of residual-stream activations between contrast pairs gives a steering vector. Adding it during generation changes a behaviour of Llama 2 Chat in the expected direction. The effect adds to system prompts, and in some behaviours to finetuning. The cost to capabilities is small.

**Method.**

- **Models.** Llama 2 7B Chat and 13B Chat. The vector-similarity and transfer studies also use the 7B base model.
- **Vector.** Each contrast pair is one multiple-choice question with an "A" and a "B" answer. One answer shows the behaviour and one does not. The vector is the mean, over pairs, of the residual-stream difference at the answer-letter token (Eq. 1).
- **Intervention.** The method adds the vector times a multiplier "at every token position after the user's prompt" (§9.1). Layer sweeps use multipliers −1 and +1. Open-ended examples use ±2. The authors say larger multipliers make the text worse.
- **Layers.** Layer 13 is best for 7B, with similar results at nearby layers. Layers 14 or 15 are best for 13B.
- **Data.** Seven behaviours: AI coordination, corrigibility, hallucination, myopic reward, survival instinct, sycophancy and refusal. Most come from Anthropic's Advanced AI Risk set (Perez et al., 2022). GPT-4 wrote the hallucination and refusal sets. Each behaviour has 290 to 1,000 pairs to make the vector and 50 test items (Table 9).
- **Metrics.** Multiple choice: the mean probability of the answer that matches the behaviour. Open-ended: a GPT-4 score of the free text.
- **Baselines.** Positive and negative system prompts (Appendix K). Finetuning of 7B Chat for one epoch on the same multiple-choice data. The authors tried few-shot prompting early, and give no numbers for it.
- **Capability checks.** MMLU and TruthfulQA, with and without steering.
- **Statistics.** The text gives no seeds, intervals or significance tests. Each result is one run on 50 test items per behaviour.

**Findings, as the source reports them.** These numbers came through the reader model, as the *Read* section says. The model and layer come from the verbatim table captions.

| Setting | Result (multiplier −1 / 0 / +1) |
| --- | --- |
| 13B Chat, layer 13, multiple choice, no system prompt (Table 3): refusal | 0.56 / 0.78 / 0.86 |
| Same: corrigibility | 0.45 / 0.57 / 0.83 |
| Same: hallucination | 0.42 / 0.54 / 0.78 |
| Same: sycophancy | 0.56 / 0.63 / 0.60, a small effect |
| 7B Chat, layer 13, open-ended GPT-4 score, no finetuning (Table 4): corrigibility | 2.30 / 2.82 / 4.70 |
| Same: survival instinct | 4.28 / 4.75 / 5.66 |
| Same: sycophancy | 0.26 / 0.58 / 1.26 |
| Same: refusal | 4.60 / 4.62 / 4.62, no effect |
| Same, after positive refusal finetuning (Table 4) | 5.42 / 5.34 / 4.42: the positive vector lowers the score |
| CAA on top of finetuning | "for 3 out of 7 tested behaviors" it moves the behaviour further |
| 13B Chat, layer 14, MMLU (Table 5) | about 0.57 to 0.65, against 0.63 with no steering. The authors call the change not significant |
| 13B, TruthfulQA, sycophancy vector (Appendix H) | 0.57 added, 0.62 subtracted, 0.60 none |
| Cost | about 5 minutes on one L40 GPU to make a vector. Finetuning takes about 10 minutes per behaviour on two L40s |

The paper also reports these results, without numbers in the text that we read. Activations cluster by behaviour from about one third of the way through the layers (§3.2). A layer-13 vector still steers at other layers, and the effect falls sharply near layer 17 (§8.2). Vectors from the base model steer the chat model, mostly between layers 10 and 15 (§8.3).

## Where the claims go beyond the evidence

- **No random baseline.** The text has no random vector, no norm-matched control and no shuffled-pair vector. So the share of each effect that any vector of that norm would give is not measured.
- **Small test sets and no intervals.** Each cell is one run on 50 items. Small gaps, such as sycophancy 0.63 to 0.60, have no interval to say whether they differ from no effect.
- **The test format matches the vector's source.** The vector comes from A/B answer tokens, and the main test is A/B questions of the same kind. The open-ended test is the stronger check, and there refusal does not move at all.
- **A GPT-4 judge.** The open-ended scores depend on GPT-4. The authors say the scores can depend on prompt wording. They checked them only against their own spot scores. The source also gives two scales: 1–10 in the Figure 5 caption and 0–10 in the Appendix L prompts.
- **Layer choice and test layer.** The paper names layers 14 or 15 as best for 13B, but Table 3 steers 13B at layer 13. The text we read does not say why.
- **One model family.** All results are on Llama 2 Chat at two sizes, in English.
- **The finetuning baseline is weak.** The authors say they did not tune its hyperparameters. So "CAA steers beyond finetuning" can shrink with a better baseline.

The authors state the judge limit, the untuned finetuning, the partial system-prompt search and the untested vector normalisation themselves (§10).

## Use in this repo

- **The repo steers units, not directions.** `openinterpretability/harness/steering.py` clamps one neuron pole to 1, 2 and 4 × `max_activation`, or ablates it. It does this on five fixed neutral prompts, for 24 greedy tokens, at every position. A dictionary latent gets the same clamp along its decoder row. CAA adds a dense residual-stream direction found from contrast pairs. The repo has no mean-difference residual direction in code. The nearest thing is the class-mean difference that ranks neurons for k-sparse probes (`openinterpretability/harness/probing.py`).
- **The two scales match in form.** Both scale the intervention by a multiple, and both warn that large multiples break the text. The `neuron-steering-tests` skill treats drift only at 4× as weak, off-distribution evidence. CAA's remark that larger multipliers make the text worse supports that rule.
- **Both apply the change at every position.** Our steering hooks act on the prompt and the continuation. CAA acts on every position after the user's prompt. CAA names this as a limit on how far the model can move before the text breaks (§9.1). The same limit holds for our 4× samples.
- **Neither has a random control.** Our steering samples compare a pole with its own ablation and baseline, not with a random pole of the same norm. CAA has no random vector either. So neither can say how much drift any intervention of that size gives. This is a gap we share, not one the paper fixes.
- **What it says about #4781.** EleutherAI/pythia-70m gives " days" with p 0.7257 and " months" with p 0.0295 on "A week has seven days and a year has twelve" (`docs/analyses/hallucination-shortlist.jsonl`). The VERIFIED route copies the first clause's unit, and no verified unit supplies " months" (#5015). A CAA-style direction could push the answer toward " months" without a unit that explains it. That would move the token, but it would say nothing about the missing mechanism that #5015 asks about.
- **The shortlist already holds contrast pairs.** The same file has "A year has twelve months and a week has seven", where the model gives " months" (p 0.4549) over " days" (p 0.1503). That prompt and the #4781 prompt differ only in clause order. Such pairs are the input a CAA vector needs. This is evidence for the campaign, not a design for it.
- **Refusal is a warning for fixes.** Table 4 shows a vector that does nothing in open text, and one that reverses its effect after finetuning. A fix found by steering in one setting needs a check in the other setting before anyone trusts it.
- **Questions for the maintainer.** These are not issues, because the campaign is the maintainer's to design.
  1. Should the campaign consider dense residual directions, such as CAA vectors, as a fix? Or should it keep to units that an explanation names?
  2. If a direction fixes the #4781 answer, should a fix also need a VERIFIED unit that explains why it works?
  3. Should a steering fix count only with a norm-matched random direction as its null, and with free-decoded text as well as the answer token?

# How many VERIFIED explanations could a weight-only check test?

A weight-only check reads a neuron from its weights, with no forward pass. Such a check can test a rule that depends only on the current token. It can't test a rule that depends on the other tokens of the text. We labelled the explanation of each of the 826 standing VERIFIED neuron poles in four models as a token-set rule, a context rule or a token set gated by context. Only 147 are token-set rules: 17.8%, with a 95% interval of 15.3% to 20.6%. That share is the ceiling for any weight-only check of the registry. If a check could also test the token part of the 242 gated rules, the ceiling rises to 47.1%. Token-set rules sit mostly in the first third of each model. There they are 54% of Pythia-70m's and GPT-2's poles, and 24% of Qwen2.5-0.5B's. In the middle third they are 0% to 9%. A second rater agreed on 86% of a seeded fifth of the poles (Cohen's kappa 0.77, and 0.88 on token-set against the rest). The labels also agree with behaviour. The share of a pole's activation variance that a mean per token explains separates token-set poles from the rest with an AUROC of 0.91.

**Status:** Preprint. Work left: #3446 (a weight-only check on the token-set poles).

## Introduction

Some methods read a neuron from its weights alone. ROTATE disentangles MLP neurons in weight space with no forward pass ([`docs/literature/rotate.md`](../../docs/literature/rotate.md), #3446). A weight reading needs no corpus and no activations, so it is cheap. But a weight reading sees only what is fixed. An MLP neuron's input is the residual stream, and attention writes into it from the other tokens of the text. A rule that depends on those tokens is not in any one set of weights.

So a weight-only check can test only part of a registry of explanations. This paper measures that part for the registry's 826 VERIFIED neuron poles. Each explanation gets a label. A token-set rule says the pole fires on a set of tokens, decided by the current token alone. A context rule needs other tokens. A mixed rule is a token set that context gates. The share of token-set rules is the ceiling for any weight-only check.

The paper's contributions are these:

- A label for each VERIFIED explanation in four models, with a reason, and a second rater's labels on a seeded fifth.
- The ceiling, with its interval, overall, by model and by relative depth.
- A check of the labels against behaviour, from the `token_r2` that each record already stores.

The paper bears on research question 1, *Are native MLP neurons clean?* ([research questions](../../docs/research-questions.md)). A clean token-set neuron is the kind of unit that a weight reading could find.

## Related work

**Token rules and context rules.** RASP describes a transformer's computation as a program (Weiss et al., 2021). An elementwise operation on the tokens needs no attention. An aggregate step, which mixes values across positions, is what attention supplies. Our token-set label is a rule with no aggregate step. Our context label needs at least one.

**Fast weights.** Ba et al. (2016) keep a memory of the recent past in fast weights, which the activity of the current sequence writes. Schlag et al. (2021) show that linear attention is such a memory. Either way, the part of the computation that depends on the context is written at run time. A reading of the trained weights can't see it. Our review pages discuss both, and propose this count ([`fast-weights-recent-past.md`](../../docs/literature/fast-weights-recent-past.md), [`linear-transformers-fast-weight-programmers.md`](../../docs/literature/linear-transformers-fast-weight-programmers.md)).

## Method

**The population.** Every standing VERIFIED neuron pole of the four configured models, as `interference.keep.verified_records` gives them. That is 167 poles of Pythia-70m (Biderman et al., 2023), 222 of GPT-2 (Radford et al., 2019), 362 of Qwen2.5-0.5B and 75 of Qwen2.5-1.5B (Yang et al., 2024).

**The rubric.** Each explanation gets one of three labels:

- **token-set:** whether the pole fires can be decided from the current token alone. That covers a listed set of tokens, or a token class, such as digits, a subword or capitalised words. The label holds when the explanation also says where such tokens tend to occur, or that context changes the strength.
- **context:** deciding needs other tokens. Examples are the previous or next token, the topic, the language or format of the text, or a place in a structure, such as inside a quote.
- **mixed:** a token set that the explanation gates by context, such as "X, but only in code". It counts as context for the ceiling, because a weight-only check can't test the gate.

**The labels.** Eight subagents (Claude Opus 5.5) labelled the poles in batches of 103 or 104, shuffled across models with seed 5144. Each label has a reason of at most 15 words. Each agent saw the explanation, the model and the layer, but no activation and no `token_r2`.

**The second rater.** Two more agents relabelled a seeded random 20% of the poles (165) with the same rubric, without the first labels. We report Cohen's kappa (Cohen, 1960) on the three labels, and on token-set against the rest.

**The behavioural check.** Each record stores `token_r2`: the share of the pole's activation variance on its scan corpus that a mean per token id explains. A token-set pole should have a high `token_r2`. We report the AUROC of `token_r2` for a token-set label, with a 95% interval from 1,000 bootstrap draws over poles. This needs no model run.

**The intervals.** Each share has a 95% Wilson interval (Wilson, 1927). The depth of a pole is its layer over the model's last layer. The bands are thirds of that depth.

## What share of the explanations are token sets?

Only 147 of 826, or 17.8% (95% interval 15.3% to 20.6%, Table 1). The share falls with the size of the model, from 27.5% in Pythia-70m to 9.3% in Qwen2.5-1.5B. The models differ in more than size, though. Their poles came from different scan corpora and proposers.

**Table 1.** The labels of each model's VERIFIED explanations, and the token-set share with its 95% Wilson interval.

| Model | Poles | token-set | mixed | context | Share | 95% interval |
| --- | --- | --- | --- | --- | --- | --- |
| All | 826 | 147 | 242 | 437 | 17.8% | 15.3%–20.6% |
| Pythia-70m | 167 | 46 | 49 | 72 | 27.5% | 21.3%–34.8% |
| GPT-2 | 222 | 52 | 59 | 111 | 23.4% | 18.3%–29.4% |
| Qwen2.5-0.5B | 362 | 42 | 108 | 212 | 11.6% | 8.7%–15.3% |
| Qwen2.5-1.5B | 75 | 7 | 26 | 42 | 9.3% | 4.6%–18.0% |

## Where are the token-set rules?

Mostly in the first third of each model (Figure 1, Table 2). There they are 54% of Pythia-70m's and GPT-2's poles. In the middle third, the share is 0% to 9% in every model. Layer 0 holds 68 of the 147 token-set poles, and 68 of its 74 poles are token sets. A layer-0 MLP sees the token embedding and only one attention layer's output, so its rules can mostly read the current token. In every model the late third has slightly more token-set poles than the middle third. We did not test why.

![Only 18% of VERIFIED explanations are token sets, and most are in the first third](media/share.svg)

**Figure 1.** The token-set share of each model's VERIFIED poles in each third of its layers. Each dot has its 95% Wilson interval and its count. The grey line is the share over all four models.

**Table 2.** Token-set poles out of all poles in each third of each model's layers.

| Model | Early | Middle | Late |
| --- | --- | --- | --- |
| Pythia-70m | 34 of 63 (54.0%) | 4 of 47 (8.5%) | 8 of 57 (14.0%) |
| GPT-2 | 37 of 68 (54.4%) | 8 of 92 (8.7%) | 7 of 62 (11.3%) |
| Qwen2.5-0.5B | 31 of 131 (23.7%) | 1 of 102 (1.0%) | 10 of 129 (7.8%) |
| Qwen2.5-1.5B | 5 of 32 (15.6%) | 0 of 19 (0.0%) | 2 of 24 (8.3%) |

## Are the labels reliable?

The two raters agree on 142 of 165 poles (86.1%, kappa 0.77, Table 3). Most disagreements, 17 of 23, are between mixed and context. That line doesn't move the ceiling. The split that sets the ceiling, token-set against the rest, has a kappa of 0.88. On this subset, the first raters call 34 poles token sets and the second raters call 30. So the ceiling moves by about two points between raters, well inside its interval.

**Table 3.** The first and second raters' labels of the same 165 poles.

| | Second: token-set | Second: mixed | Second: context |
| --- | --- | --- | --- |
| **First: token-set** | 29 | 5 | 0 |
| **First: mixed** | 0 | 35 | 14 |
| **First: context** | 1 | 3 | 78 |

## Do the labels match behaviour?

Yes. `token_r2` gives an AUROC of 0.91 (95% interval 0.88 to 0.93) for a token-set label (Figure 2). The median is 0.86 for token-set poles and 0.34 for the rest. Of the 100 poles with `token_r2` of at least 0.8, 82 are token sets.

![The token mean explains far more of a token-set pole's activity (AUROC 0.91)](media/token-r2.svg)

**Figure 2.** Each VERIFIED pole's `token_r2` by the label of its explanation. The ink line and value are each label's median.

The overlap is where the labels and the activations disagree. A context label with a high `token_r2` belongs to a pole whose explanation names a context, but whose activity on the scan corpus a token mean mostly explains. `token_r2` alone can't replace the labels, because a rare token gets a mean of its own and inflates it.

## Discussion

A weight-only check of this registry can test at most about one explanation in six. Those explanations are mostly at layer 0 and in the first third of the small models. In Qwen2.5, the models that the registry covers most, a weight-only check can test about one in ten. So the registry can check a weight reading such as ROTATE, but only on a narrow, early slice of it. A result on that slice says little about the middle layers, where almost every explanation needs context.

The ceiling is on what the explanations claim, not on what the neurons do. A neuron whose activity a token mean mostly explains, but whose explanation adds a context, counts as context here. A weight-only check could still predict much of such a neuron's activity. It could not check the explanation as written.

## Limitations

- **The labels read the explanation, not the neuron.** An explanation that adds a context condition to a token-set neuron makes it context here. So the ceiling is a ceiling on what the registry's explanations claim.
- **Agents of one model gave every label.** The raters used one rubric. A human rater might draw the mixed line in another place. Most disagreements are on that line, and it doesn't move the ceiling.
- **The registry is not a random sample.** The draw, the proposers and the reviewers decide which poles are VERIFIED. The shares describe the registry, not the models.
- **Mixed is a choice.** Counting mixed rules against the ceiling assumes that a weight-only check can't test a gate. A check that tests only the token part would reach up to 47.1%.

## Conclusion

Only 17.8% of the registry's VERIFIED explanations are token-set rules, which a reading of the weights alone could test. They sit mostly in the first third of each model. The next step is to run a weight-only check on these 147 poles (#3446).

## References

- Ba, J., Hinton, G., Mnih, V., Leibo, J. Z., & Ionescu, C. (2016). *Using fast weights to attend to the recent past*. [arXiv:1610.06258](https://arxiv.org/abs/1610.06258).
- Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O'Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S. S., Raff, E., Skowron, A., Sutawika, L., & van der Wal, O. (2023). *Pythia: A suite for analyzing large language models across training and scaling*. ICML 2023. [arXiv:2304.01373](https://arxiv.org/abs/2304.01373).
- Cohen, J. (1960). *A coefficient of agreement for nominal scales*. Educational and Psychological Measurement, 20(1), 37–46. [doi:10.1177/001316446002000104](https://doi.org/10.1177/001316446002000104).
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language models are unsupervised multitask learners*. [OpenAI](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
- Schlag, I., Irie, K., & Schmidhuber, J. (2021). *Linear transformers are secretly fast weight programmers*. ICML 2021. [arXiv:2102.11174](https://arxiv.org/abs/2102.11174).
- Weiss, G., Goldberg, Y., & Yahav, E. (2021). *Thinking like transformers*. ICML 2021. [arXiv:2106.06981](https://arxiv.org/abs/2106.06981).
- Wilson, E. B. (1927). *Probable inference, the law of succession, and statistical inference*. Journal of the American Statistical Association, 22(158), 209–212. [doi:10.2307/2276774](https://doi.org/10.2307/2276774).
- Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., . . . Qiu, Z. (2024). *Qwen2.5 technical report*. Preprint. [arXiv:2412.15115](https://arxiv.org/abs/2412.15115).

## Appendices

### A. Reproduce

The labels are in `labels/`: `labels-*.json` for the first raters and `second-labels-*.json` for the second. Each row has a slug, a label and a reason. The script joins them with the registry's records and runs no model. The bootstrap uses seed 5144 and 1,000 draws.

    uv run python scripts/token_set_ceiling.py --labels papers/token-set-ceiling/labels \
        --out papers/token-set-ceiling/results.jsonl
    uv run python scripts/paper_figures.py

`results.jsonl` holds one `pole` row for each pole, with its label, reason, depth and `token_r2`. It also holds the `share` rows, the `agreement` row and the `token_r2` row. The poles depend on the registry on the day the script runs.

### B. Who did the work

A model, Claude Opus 5.5 in Claude Code, wrote the plan, the code, the analysis and the text. Ten subagents of the same model gave the labels. The registry's agents wrote and reviewed the 826 explanations.

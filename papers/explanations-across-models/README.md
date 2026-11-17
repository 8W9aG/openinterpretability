# Do explanations carry across models?

When a neuron pole in one model has a VERIFIED explanation, does the same explanation describe a matched pole in another model? The registry has 80 sister studies that test this. Each audits a pole of one model with another model's VERIFIED explanation, unchanged. The explanation verifies on 8 of 50 sisters (16%), 1 of 26 other close matches and 0 of 4 random poles. Most sisters fail only in part: their median balanced accuracy is 0.78, against 0.56 for other matches and 0.45 for random poles. The explanation is near chance (REJECTED) on 24% of sisters and 54% of other matches. So matching finds poles where an explanation largely carries over, but it seldom carries over whole. Sisters across tokenizers, GPT-2 to Qwen, carry about as well as sisters within the Qwen family.

**Why we wrote it.** If a VERIFIED explanation carried over to a matched pole in another model, the registry could explain new models without auditing every pole again. The registry has 80 sister studies that test this, and they had not yet been read together. **The question.** Does a VERIFIED explanation describe a matched pole in another model? It is research question 4, *Do explanations carry across models?* ([research questions](../../docs/research-questions.md)).

![A pole's explanation carries better to its sister than to another match or a random pole](media/transfer.svg)

## Results

1. **An explanation seldom verifies whole on another model's pole.** 8 of 50 sisters (16%, 95% CI 8–29%) and 1 of 26 other top matches (4%, 1–19%). The difference is not significant (Fisher p = 0.15). *Limit:* 80 studies, by agents that the planner drew. They are not a designed experiment.
2. **On a sister, it mostly carries in part.** The median balanced accuracy is 0.78 on sisters, 0.56 on other matches and 0.45 on random poles. Sisters are REJECTED (near chance) 12 of 50 times, other matches 14 of 26 (Fisher p = 0.012). Most sister studies end NEEDS_REVISION: the explanation passes many probes and misses some. *Limit:* only 4 random-pole controls, so the random baseline is barely measured.
3. **The tokenizer matters little for sisters.** 4 of 28 sisters verify within the Qwen family, which shares a tokenizer. 4 of 22 verify across tokenizers, GPT-2 to Qwen. Their median balanced accuracy is 0.75 and 0.79. Other matches across tokenizers do worst: none of 13 verify, and 8 are REJECTED. *Limit:* each cell has 13 to 28 studies.
4. **A higher correlation goes with carrying over.** The median correlation r between the two poles is 0.49 for studies that verified, and 0.38 for the rest. *Limit:* 9 VERIFIED studies.

**What this does not show.** Whether two poles that share an explanation represent the same feature in the same way. A sister study tests one explanation's predictions on new probes. It doesn't compare the two poles' roles in their models.

## Method

- **Matching.** `open-interpretability sisters` correlates each VERIFIED pole's activation with every pole of another model, over `corpus.jsonl`, at the positions where both tokenizations end a token (`docs/PROTOCOL.md` §4, *Sister neurons*).
  - **Sister:** a match whose own best partner is the pole: a *mutual* best match. Its correlation must also beat the best that a random rotation of the layer reaches.
  - **Our steps:** correlating across models against a rotated baseline follows Gurnee et al. (2024), who take the maximum correlation over the other model. The mutual step, and its use to pick poles for a study, are this registry's.
- **The models:** GPT-2 small (Radford et al., 2019), and [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) and [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B).
- **Sister studies.** `open-interpretability audit --sister-of <record>` takes the source record's explanation as the hypothesis for the tested pole. A reviewer writes new probes against it, and the pole is judged by the protocol of the time. The planner draws a source's sister, its other top matches and, at a low weight, random poles of the other model as controls (`sister` draws).
- **Classes.** `scripts/explanation_transfer.py` reads every study with a `sister_of`. It classifies the tested pole against `docs/analyses/sister-neurons.jsonl`: the source's sister, another of its 5 top matches, or a random pole. Each tested pole counts once, by its first study. That is 80 poles from 89 studies.
- **Intervals and tests.** 95% Wilson intervals, and two-sided Fisher exact tests (`openinterpretability/builder/learned_baseline.py`).

## How far each explanation carries

| Tested pole | Studies | VERIFIED (95% CI) | REJECTED (95% CI) | Median balanced accuracy |
| --- | --- | --- | --- | --- |
| The source's sister | 50 | 8 (8–29%) | 12 (14–37%) | 0.78 |
| Another top match | 26 | 1 (1–19%) | 14 (35–71%) | 0.56 |
| A random pole | 4 | 0 (0–49%) | 4 (51–100%) | 0.45 |

The left panel of the figure shows each study. Sisters crowd between 0.75 and 0.95, and other matches spread from 0.4 to 0.9.

## Within a tokenizer and across tokenizers

| Tested pole | Tokenizer | Studies | VERIFIED | REJECTED | Median balanced accuracy |
| --- | --- | --- | --- | --- | --- |
| Sister | the same (Qwen ↔ Qwen) | 28 | 4 | 6 | 0.75 |
| Sister | across (GPT-2 ↔ Qwen) | 22 | 4 | 6 | 0.79 |
| Other match | the same | 13 | 1 | 6 | 0.64 |
| Other match | across | 13 | 0 | 8 | 0.53 |

A sister found across tokenizers is as good a target as one within a family.

## Caveats

- **The random control is four studies.** The planner draws random poles at a low weight, so few ran. The contrast that matters most, sisters against random poles, rests on 4 controls. Another match is a weaker control: it already correlates with the source.
- **Mostly protocol 0.8.** 88 studies ran under protocol 0.8 and one under 0.9, with the agents that each draw had.
- **One reviewer per study.** First readings of one hypothesis agree on VERIFIED about 71% of the time (`docs/analyses/registry-stats.md`), so a study's verdict is noisy.

## Reproduce

    uv run open-interpretability build --output-dir build/site
    uv run python scripts/explanation_transfer.py --site-dir build/site \
        --out papers/explanations-across-models/results.jsonl
    uv run python scripts/paper_figures.py

## Next

- About 30 random-pole sister studies, so that sisters can be compared with a real control.
- The same poles under one protocol and one agent pair.
- A study that compares what two sisters do, not only what fires them, for example by patching one model's pole and reading the other's.

## References

- Gurnee, W., Horsley, T., Guo, Z. C., Kheirkhah, T. R., Sun, Q., Hathaway, W., Nanda, N., & Bertsimas, D. (2024). *Universal neurons in GPT2 language models*. [arXiv:2401.12181](https://arxiv.org/abs/2401.12181).
- Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language models are unsupervised multitask learners*. [OpenAI](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

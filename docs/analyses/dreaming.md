# Fluent dreaming against VERIFIED explanations

Issue #3228. A reviewer writes probes that it thinks will break a hypothesis, so it finds the failures it can imagine. An optimiser can find others. This page optimises prompts to drive each of 20 VERIFIED GPT-2 small poles as high as it goes. Then it asks a judge whether the pole's explanation predicts those prompts. Run on 2026-11-14 with `scripts/dream_audit.py`. The rows are in `dreaming.jsonl`. Reported, not gated.

**Answer: mostly not.** Of 60 distinct optimised prompts that fire their pole, the explanation misses 44: 73% (95% CI 61–83%). 13 of the 19 poles that got a firing prompt have at least one miss, and for 12 of them the explanation misses every one. The judges called 6 of 61 silent controls active (10%). So the misses are not a judge that lists nothing.

## Method

- **Dreaming.** EPO, from Thompson, Straznickas & Sklar (2024), §2, in `openinterpretability/harness/dreaming.py`.
  - A population of 12-token prompts has one member per λ, with eight λ log-spaced from 0.1 to 10, as in the paper.
  - Member i maximises the pole's strength minus λ times the prompt's mean cross-entropy under GPT-2 itself, so a larger λ asks for more fluent text.
  - Each step takes the gradient of that objective with respect to a one-hot encoding of the tokens. It keeps the 128 tokens with the largest gradient at each position. Each member then has 32 children, each with one random position swapped for one of those tokens. Each λ keeps the best of the pooled prompts.
  - The run is 150 steps, from a random start with seed 0.
  - The strength is the pole's largest value over the prompt, first token left out.
  - The paper uses k = 512, 300 steps and restarts. We used fewer for cost on a laptop GPU. The population often converges, so the eight members give fewer distinct prompts.
- **Poles.** 20 standing-VERIFIED GPT-2 small positive poles, drawn with seed 0 from all of them, with errata applied.
- **Dreams kept.** Each distinct final prompt that clears the record's own threshold, 0.25 times the record's scale. That gives 60 prompts from 160. One pole, L6/N62, has none.
- **Controls.** For each pole, as many 24-token windows of `corpus.jsonl` where the pole stays below its threshold.
- **The judge.** One detection prompt per pole, built the way `score` builds one, with the record's own explanation. The dreams and controls are shuffled together. The judges were `explanation-judge` agents on Claude Haiku 5.5.
- **Misprediction.** A dream that fires but that the judge, applying the explanation, calls inactive.

## Results

| | Count | Share (95% CI) |
| --- | --- | --- |
| Firing dreams the explanation misses | 44 of 60 | 0.73 (0.61–0.83) |
| Poles with at least one miss | 13 of 19 | |
| Poles where every dream is missed | 12 of 19 | |
| Silent controls called active | 6 of 61 | 0.10 |

The dreams go past the scan's range. Their strongest is a median 1.35 times the record's scale, which is the pole's maximum over 200,000 scanned tokens.

**What the misses look like.** The optimiser finds token strings that no corpus text contains, and the explanation, written from corpus examples, doesn't cover them:

- **L3/N2620+** detects the partitive "of" after a quantity ("most of", "a third of"). Its dreams contain no "of": "SecuritiesMaterials Troutfinfin Industries Bigil Nitrate</p" reaches 0.80 of its scale.
- **L8/N2851+** detects the vocabulary of formal institutions. Its dreams are a citation-like template, "ESA/Des Gurery e-journal FEMS.", at up to 1.35 times its scale.
- **L4/N78+** detects verbs of construal ("interpret", "regard"). "Favor Dayton Police Yemen Recommend Print Edatively Library HB 51." reaches 1.17 times its scale.

**What the hits look like.** Where the explanation is a word family, the optimiser often finds the family:

- **L3/N2427+** is a nearness family, and every dream ends on "nearest" or "closest".
- **L11/N1579+** is a regulatory frame, and its dreams say "Brexit Regulations".

## Reading it

A VERIFIED explanation predicts the pole on text like the corpus. Its probes are natural sentences, written by a reviewer. An optimiser leaves that distribution. There, most of these poles fire on things their explanation doesn't name, often more strongly than on any scanned text. The paper names this out-of-distribution use as dreaming's value. It also warns that the highest-activating prompts may not show typical behaviour.

So a misprediction here is not evidence that the explanation is wrong on real text. It shows what the explanation leaves out. That is the kind of false negative an LLM reviewer wouldn't think to write. As a reviewer arm, the dreams could become hard holdouts, and a hypothesis would have to cover them or say why it doesn't.

## Limits

- **A small run.** 20 poles, one seed, and EPO with fewer steps and a smaller k than the paper. Near-duplicate dreams of one pole count separately, so the per-pole numbers are the safer ones.
- **One judge per pole.** The 10% false-positive rate on controls comes mostly from one pole, L0/N754 ("starts with h"), where 3 of 4 controls fit the explanation literally.
- **Only one direction.** This finds inputs the explanation misses: false negatives. The opposite arm finds false positives: inputs the explanation predicts active that stay silent. It needs an objective that holds a hypothesis's trigger fixed while driving the pole down. #3779 holds it, with the "exactly hit" arm of the issue.
- **Positive poles of GPT-2 small only.**

## Three arms and two models (#3779)

Run on 2026-11-21 with `scripts/dream_arms.py`. The rows are in `dreaming-arms.jsonl`. It takes 20 VERIFIED positive poles each of GPT-2 small and pythia-70m (seed 0). EPO runs at the paper's k = 512 and 300 steps, from seeds 0 and 1. `dream()` gained a goal and frozen positions for the new arms (`openinterpretability/harness/dreaming.py`).

- **max**: as above. A miss is a firing dream that the judge calls inactive.
- **min**: a window of 24 tokens around the peak of each of up to two of the record's positive holdouts that fired. The peak token is frozen, and EPO drives the pole down. A false positive is a silenced prompt that the judge calls active.
- **hit**: dreams aimed at the record's active line, kept within 10% of it. The share that the judge calls active shows how sharp the explanation's boundary is.
- **review**: each GPT-2 pole's firing max-dreams are added to its record's own probes. Each one expects what the judge said the explanation predicts. The verdict is then measured again. Reported, not gated.

One `explanation-judge` (Claude Haiku 5.5) prompt per pole holds every arm's texts and the silent controls, shuffled.

| Arm (95% Wilson interval) | GPT-2 small | pythia-70m |
| --- | --- | --- |
| max: firing dreams missed | 87 of 148, 59% (51–66%) | 85 of 156, 54% (47–62%) |
| min: silenced prompts called active | 48 of 75, 64% (53–74%) | 46 of 73, 63% (52–73%) |
| hit: dreams at the line called active | 12 of 56, 21% (13–34%) | 9 of 60, 15% (8–26%) |
| control: silent windows called active | 21 of 148, 14% (9–21%) | 10 of 156, 6% (4–11%) |

- **Misses hold at the paper's settings, on both models.** Over half of the firing dreams fall outside the explanation, against 73% in the smaller run above.
- **False positives are as common.** With the trigger token kept, the context can silence the pole, and the explanation still predicts it active about two times in three. So the explanations name the trigger but not what turns it off.
- **The boundary is soft.** At the active line, the judge calls only 15–21% active. An explanation states when a pole fires strongly, not where it starts to fire.
- **Verdicts would move.** On GPT-2, 18 of the 20 poles re-measure VERIFIED on their own probes. With their dreams added, 9 do. Two NEEDS_REVISION poles go to REJECTED.

The same limits apply as above: one judge model, one corpus, and dreams that may be more unusual than any probe a reviewer writes. The review arm is a counterfactual, and no record changes.

## References

- Thompson, T. B., Straznickas, Z. & Sklar, M. (2024). *Fluent dreaming for language models*. [arXiv:2402.01702](https://arxiv.org/abs/2402.01702).

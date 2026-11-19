# Inference-Time Intervention

**Citation.** Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister and Martin Wattenberg (Harvard University). "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model". arXiv 2306.03341. Version 1 is from 6 June 2023. We read version 6, from 26 June 2024. [Paper](https://arxiv.org/abs/2306.03341), [code](https://github.com/likenneth/honest_llama). The arXiv comment gives the venue as "NeurIPS 2023 spotlight". We did not check the proceedings page. The paper uses the public TruthfulQA dataset. The acknowledgements say: "This work was partially supported by NSF grant IIS-1901030 and NSF grant NCS-FO-2124179."

**Status.** Planned. The fix campaign that #4784 prepares for, with #4781 and #5015 as its first case, would use it. ITI is a fix at inference time that changes a few attention heads, and our steering harness gives heads no samples.

**Read.** On 2026-11-18, through WebFetch. A model read the abstract page and the v6 HTML for us, in four parts by offset. It returned verbatim text for Tables 1 to 5, the table captions and short quotes from Sections 1 to 6. The rest of the main text came back as its summary. Of the appendices, we saw only the per-category examples in Appendix A, with their flip counts. The text stopped partway through Appendix B, so we did not see Appendices C to F, which hold the seed discussion. We did not see the figures or the PDF. The abstract came back as a summary.

## Summary

**Claim.** A model can know more than it says. The authors write that "LLMs sometimes "know" more than they "say"". Adding a fixed direction to the outputs of a few attention heads at inference time makes LLaMA-7B and two finetuned models more truthful on TruthfulQA.

**Method.**

- **Models.** LLaMA-7B is the main model. Alpaca and Vicuna, both finetuned from LLaMA, are the other two.
- **Units.** The outputs of single attention heads. The text says "we only intervene on the results of the top K heads so as to be minimally invasive".
- **Selection.** The authors train a linear probe for "true answer" on each head's activations. The probing data is split "into training and validation sets by 4:1". They rank the heads by validation accuracy, and "We take the top-K heads as the targeted set."
- **Direction.** The default is the mass mean shift, the difference of the class means of true and false answers at each head. The other candidates are the probe's weight vector, a CCS direction and a random direction.
- **Strength.** The shift is "equivalent to shifting activations along the truthful directions for α times the standard deviation". The authors write: "We choose the optimal hyperparameters K=48 and α=15 by considering multiple scores."
- **Cost.** The shift is a constant vector, so it can go into the output projection's bias. The text says the authors can "bake ITI into a pretrained LLM by an offline editing of its bias terms".
- **Data and splits.** TruthfulQA. "Unless otherwise specified, we use 2-fold cross-validation for our results." Table 1 uses 5% of the questions for training and validation.
- **Metrics.** GPT-judge, a finetuned GPT-3 judge, labels each answer as true or not and informative or not. The main score is True\*Info, their product. The others are multiple-choice accuracy (MC), the pre-training cross-entropy (CE) and the KL divergence of next-token distributions before and after the intervention.
- **Baselines.** Supervised finetuning and few-shot prompting, and the two finetuned models without ITI.

**Findings, as the source reports them.**

| Setting | True\*Info (%) | True (%) | MC (%) | CE | KL |
| --- | --- | --- | --- | --- | --- |
| LLaMA-7B, 5% of questions (Table 1), baseline | 30.5 | 31.6 | 25.7 | 2.16 | 0.0 |
| Same, supervised finetuning | 36.1 | 47.1 | 24.2 | 2.10 | 0.01 |
| Same, few-shot prompting | 49.5 | 49.5 | 32.5 | – | – |
| Same, baseline + ITI | 43.5 | 49.1 | 25.9 | 2.48 | 0.40 |
| Same, few-shot prompting + ITI | 51.4 | 53.5 | 32.5 | – | – |
| Alpaca, 2-fold CV (Table 2) | 32.5 | 32.7 | 27.8 | 2.56 | 0.0 |
| Alpaca + ITI | 65.1 | 66.6 | 31.9 | 2.92 | 0.61 |
| Vicuna | 51.5 | 55.6 | 33.3 | 2.63 | 0.0 |
| Vicuna + ITI | 74.0 | 88.6 | 38.9 | 3.36 | 1.41 |
| LLaMA-7B, directions (Table 3), random direction, α = 20 | 31.2 | 32.3 | 25.8 | 2.19 | 0.02 |
| Same, CCS direction, α = 5 | 33.4 | 34.7 | 26.2 | 2.21 | 0.06 |
| Same, probe weight direction, α = 15 | 34.8 | 36.3 | 27.0 | 2.21 | 0.06 |
| Same, mass mean shift, α = 20 | 42.3 | 45.1 | 28.8 | 2.41 | 0.27 |
| LLaMA-7B, selection (Table 5), without selection, α = 5 | 35.4 | 37.1 | 28.3 | 2.19 | 0.08 |
| Same, point-wise selection, α = 15 | 39.2 | 55.3 | 28.7 | 4.01 | 1.95 |
| Same, head-wise selection, α = 20 | 42.3 | 45.1 | 28.8 | 2.41 | 0.27 |

Table 4 tests transfer to other datasets with no new training. Natural Questions goes from 46.6 to 51.3, TriviaQA from 89.6 to 91.1 and MMLU from 35.71 to 40.16.

The best single head's probe "has a validation accuracy of 83.3%". It is "the 18th head in the 14th layer". Section 2 says that "ITI uses as few as 40 samples to locate and find truthful heads and directions". The abstract instead gives "only few hundred examples". The Figure 4 caption says the metrics "have been averaged over 55 random seeds". The figure itself we did not see.

The authors name the trade-off. Truthfulness goes up and informativeness goes down as α grows, because "it is trivial to attain a perfect truthfulness score simply by answering "no comment."" On the human check, "truthfulness is slightly overestimated by GPT-judge and the opposite for informativeness". The text that we read gives no numbers for that check.

## Where the claims go beyond the evidence

**One benchmark stands in for "truthful".** All selection, tuning and the main scores use TruthfulQA. The authors state this limit: "this dataset does not cover the full range of meanings of the word "truth"". Table 4 is the only test away from TruthfulQA, and its gains are smaller.

**The judge is a model.** True and informative are GPT-judge labels. The authors say that it overestimates truthfulness. They give no numbers for their human check in the text that we read.

**The random control is one direction at the chosen heads.** Table 3's random direction at α = 20 gives 31.2 against 30.5 for the baseline. That is a null for the direction. We found no null for the heads, such as a mean shift at K random heads. Table 5 compares selection rules, but not a random set of the same size.

**The direction comes from the same data that selects the heads.** The probes, the class means and the hyperparameters all come from TruthfulQA folds. The 2-fold split keeps test questions out of training. But α and K were chosen "by considering multiple scores", and we did not see on which split.

**A part of the gain is fewer claims, not truer ones.** CE rises from 2.16 to 2.48 and KL to 0.40 in Table 1. The authors' own "no comment" sentence shows how True can rise without new knowledge. True\*Info guards against this, but only as far as GPT-judge does.

**Few intervals.** Table 1 gives the mean of three runs, and the text that we read gives no spread. The "55 random seeds" in the Figure 4 caption may be a typo for 5. The seed discussion is in an appendix that we did not see.

**The mechanism stays open.** The authors write: "We make no claims about mechanistically understanding what ITI does to the model's internal representations". They also say that LLMs "may have an internal representation of the likelihood of something being true". That claim rests on probe accuracy of about 83%, not on a causal test.

**Credit.** The paper compares four directions and three selection rules, reports CE and KL as side effects, and states the helpfulness trade-off itself. It also says: "We do not claim that ITI by itself is anywhere near sufficient for ensuring truthful answers from LLMs."

## Use in this repo

**Our steering harness skips heads, and ITI shows a way to steer one.** `openinterpretability/harness/steering.py` says: "A head writes a subspace with no "high" setting, so heads get no samples." ITI gives a head a "high" setting. It adds α standard deviations of a class-mean direction to the head's output. That would let the copy heads L3/H3, L4/H0 and L4/H7 of the #4781 route have steering samples. The follow-up below proposes it.

**The repo already computes the same direction for other units.** `openinterpretability/core/probing.py` fits mass-mean probes (Marks and Tegmark, 2023) on MLP hidden activations and the residual stream. `docs/analyses/probe-comparison.md` finds that "Every cosine between a dense probe's direction and the pole's basis vector is at most 0.19 on Qwen for mass-mean". So a mass-mean direction and a VERIFIED pole are different objects. An ITI-style fix along a concept's direction is not the same as a fix at the explained unit.

**ITI selects units by probe accuracy, and our protocol does not.** ITI's head ranking is observational: a head is chosen because a probe can read "true" from it. Our causal scores rank a unit against matched same-layer units under minimal-pair patching (`docs/PROTOCOL.md` §4). The `activation-patching-practice` skill explains why a readable unit need not be a causal one. ITI's own result fits this warning. The probe weight direction reads better but steers worse than the mass mean shift (34.8 against 42.3).

**The paper's nulls are thinner than ours.** Our answer experiments compare against matched random sets. For #4781 the patching score is 0.75 against −0.004 for random sets (`changelog.d/4781.added.md`). ITI has a random direction but no random heads. If the fix campaign uses an ITI-style shift, the same-size random-head null is the check to add.

**Scale differs a lot.** ITI picks 48 heads in LLaMA-7B. EleutherAI/pythia-70m has 6 layers of 8 heads, so 48 heads in all. A top-K choice that leaves "irrelevant heads out" in LLaMA would take every head in pythia-70m. A port would need K near the route's three copy heads, and the paper gives no evidence at that size.

**The side-effect metrics carry over.** ITI reports CE on pre-training text and KL of next-token distributions beside the target score. A fix of #4781 could report the same two numbers, with the other five prompts of `docs/analyses/hallucination-shortlist.jsonl` as the target set. All six copy the first clause's unit. How to weigh these is a decision for the fix campaign, which the maintainer designs. This page only records the metrics.

**Steering as a gate is still open.** #3209 asks whether steering predictions should become gates. ITI's α sweep is a model of a pre-registered strength range. Our skill already treats drift only at 4× as weak evidence. The choice of format stays with #3209.

**Follow-up.** Steering samples for attention heads by a class-mean shift, as ITI does it, with a same-size random-head null (area: pipeline, priority: medium). The page proposes this issue. It does not open it.
